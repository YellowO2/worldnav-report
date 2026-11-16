"""experiments/section7_per_robot/later_starts.py with one more arm.

    python experiments/section5_first_model/later_speed.py recon --arms 7
"""
import gc
import json
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
import siglip_steps as ss  # noqa: E402
from curves import log_curve  # noqa: E402
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import recording, scored  # noqa: E402
from later_feats import starts  # noqa: E402
from path_labels import resample, start_frame  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load, score  # noqa: E402
from step_oracle import at, length, moves_from  # noqa: E402
from tartan_speed_head import SpeedRoute  # noqa: E402
from math import comb  # noqa: E402


class CurveRoute(ss.GridRoute):
    """GridRoute whose output is a Bezier curve: the network gives the free control points."""
    def __init__(self, cells, globs, parse_dim, degree):
        super().__init__(cells, globs, parse_dim)
        K = self.route.out_features // 2
        self.route = nn.Linear(256, 2 * degree)
        t = torch.arange(1, K + 1, dtype=torch.float32) / K
        self.register_buffer('basis', torch.stack([comb(degree, j) * t ** j * (1 - t) ** (degree - j) for j in range(1, degree + 1)], 1))

    def run(self, b):
        ctrl = super().run(b)                                 # (B, degree, 2)
        return torch.einsum('kj,bjc->bkc', self.basis.to(ctrl.dtype), ctrl)

class ActionRoute(ss.GridRoute):
    """GridRoute whose output is actions: for each of the K route segments a turn."""
    def __init__(self, cells, globs, parse_dim):
        super().__init__(cells, globs, parse_dim)
        nn.init.zeros_(self.route.weight)
        nn.init.zeros_(self.route.bias)

    def run(self, b):
        out = super().run(b)                                  # (B, K, 2)
        heading = torch.cumsum(out[..., 0], 1)
        length = nn.functional.softplus(out[..., 1] - 0.5)    # about 0.47 m per segment at the start
        return torch.cumsum(torch.stack([length * torch.cos(heading), length * torch.sin(heading)], -1), 1)


class AnchorRoute(ss.GridRoute):
    """GridRoute that picks one of A typical routes."""
    def __init__(self, cells, globs, parse_dim, anchors):
        super().__init__(cells, globs, parse_dim)
        self.register_buffer('anchors', anchors)               # (A, K, 2)
        A, K, _ = anchors.shape
        self.route = nn.Linear(256, A + A * K * 2)
        nn.init.zeros_(self.route.weight[A:])
        nn.init.zeros_(self.route.bias[A:])

    def heads(self, b):
        out = super().run(b).flatten(1)
        A, K, _ = self.anchors.shape
        return out[:, :A], out[:, A:].view(-1, A, K, 2)

    def run(self, b):
        logits, res = self.heads(b)
        k = logits.argmax(1)
        ar = torch.arange(len(k), device=k.device)
        return self.anchors[k] + res[ar, k]

    def train_loss(self, b, y):
        logits, res = self.heads(b)
        k = ((y[:, None] - self.anchors[None]).norm(dim=-1).mean(-1)).argmin(1)
        ar = torch.arange(len(k), device=k.device)
        return nn.functional.cross_entropy(logits.float(), k) + score_loss((self.anchors[k] + res[ar, k]).float(), y)


def kmeans_routes(y, A=64, iters=25, seed=0):
    """k-means on routes (n, K, 2) by mean point distance."""
    g = torch.Generator().manual_seed(seed)
    x = y.flatten(1).to(DEV)
    c = x[torch.randperm(len(x), generator=g)[:A].to(DEV)].clone()
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        for j in range(A):
            m = lab == j
            c[j] = x[m].mean(0) if m.any() else x[torch.randint(len(x), (1,), generator=g).item()]
    return c.view(A, *y.shape[1:])


DEV = 'cuda'
ALL_ROBOTS_HELD = 43.98            # RECON design trained on every non-TartanDrive robot
Q_TABLES = ('q_cells', 'q_last', 'q_text')
IMG_TABLES = ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls')


def later_rows(group, tag, rows):
    """The named rows of the later-photo features (chunks of 2_features/later_feats.py), in the given order."""
    import os
    cdir = f'features/later_{group}_{tag}_chunks'
    rows = torch.as_tensor(rows)
    out = {k: [] for k in ('cells', 'last', 'text_mean')}
    off = 0
    for c in sorted(os.listdir(cdir)):
        d = torch.load(f'{cdir}/{c}')
        n = len(d['last'])
        m = (rows >= off) & (rows < off + n)
        assert (rows[m][1:] >= rows[m][:-1]).all()       # rows are sorted, so chunk order = row order
        for k in out:
            out[k].append(d[k][rows[m] - off])
        off += n
    return {k: torch.cat(v) for k, v in out.items()}


def main(group, arms):
    final.setup()
    steps = pathnet.typical_steps()
    g = GROUPS.index(group)
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = tr['q_last'].shape[-1], tr['q_cells'].shape[-1]
    tr_eps, va_eps = load('data/lcvn/train.jsonl'), load('data/lcvn/val_seen.jsonl')
    sizes = {k: int(max(tr[f'seg_{k}'].max(), va[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    ti = torch.where((tr['group'] == g) & ok)[0]
    vi = torch.where(va['group'] == g)[0]
    recs = sorted({recording(tr_eps[i]['trajectory']) for i in ti.tolist()})
    random.seed(0)
    held = set(random.sample(recs, len(recs) // 10))
    is_h = torch.tensor([recording(tr_eps[i]['trajectory']) in held for i in ti.tolist()])
    fit, hold = ti[~is_h], ti[is_h]
    print(f'{group}: fit {len(fit)} episodes, held-out {len(hold)}, val {len(vi)}', flush=True)
    pdim = ss.parse_vec(tr, fit[:2], sizes).shape[1]

    # the extra examples: (source episode, later-feature row, target)
    key_row = {tuple(k): j for j, k in enumerate(torch.load(f'features/later_{group}.pt')['keys'])}
    src, lrow, target, left, later_logv = [], [], [], [], []
    mean_step = lambda p: float(np.log(max(np.linalg.norm(np.diff(np.array(p), axis=0), axis=1).mean(), 1e-3)))
    logv_tr = torch.tensor([mean_step(x['position']) for x in tr_eps]).float()
    logv_va = torch.tensor([mean_step(x['position']) for x in va_eps]).float()
    for i in fit.tolist():
        e = tr_eps[i]
        for k in starts(len(e['position']) - 1):
            pts, _ = resample(start_frame(e['position'][k:], e['yaw'][k]))
            if np.isfinite(pts).all():
                src.append(i)
                lrow.append(key_row[(e['episode'][:-2], k)])
                target.append(pts)
                left.append(len(e['position']) - 1 - k)
                later_logv.append(mean_step(e['position'][k:]))
    e = tr_eps[int(fit[0])]                              # the same code gives the stored first-photo target
    assert np.allclose(resample(start_frame(e['position'], e['yaw'][0]))[0], tr['floor'][fit[0]].numpy(), atol=1e-4)
    src = torch.tensor(src)
    used = sorted(set(lrow))
    pos = {r: j for j, r in enumerate(used)}
    n = len(tr['episodes'])
    tartan = GROUPS.index('tartan')
    others = torch.where((tr['group'] != tartan) & (tr['group'] != g) & ok)[0]      # the other robots' first-photo examples
    print(f'extra examples: {len(src)} episodes from {len(used)} later photos; moves left 10/50/90%: '
          f'{np.percentile(left, [10, 50, 90])}; other robots: {len(others)} first-photo episodes', flush=True)

    def build(orig, tag):
        """Feature dict of the original episodes `orig` followed by the later-start examples."""
        sel = torch.cat([orig, src])
        f = {k: (v[sel] if torch.is_tensor(v) and len(v) == n and k not in Q_TABLES + IMG_TABLES else v) for k, v in tr.items()}
        f['floor'] = torch.cat([tr['floor'][orig], torch.tensor(np.stack(target), dtype=tr['floor'].dtype)])
        f['partial'] = torch.cat([torch.zeros(len(orig)), torch.ones(len(src))])
        f['logv'] = torch.cat([logv_tr[orig], torch.tensor(later_logv).float()])
        uniq, inv = torch.unique(tr['q_index'][orig], return_inverse=True)
        f['q_index'] = torch.cat([inv, len(uniq) + torch.tensor([pos[r] for r in lrow])])
        d = later_rows(group, tag, used)
        for k, kk in zip(Q_TABLES, ('cells', 'last', 'text_mean')):
            f[k] = torch.cat([tr[k][uniq], d[kk]])
        return f

    def make(f, idx, flag, blank):
        b = batch(f, idx, DEV)
        parse = ss.parse_vec(f, idx, sizes)
        if flag:
            p = f['partial'][idx] if 'partial' in f else torch.zeros(len(idx))
            if blank:
                parse = parse * (1 - p)[:, None]
            parse = torch.cat([parse, p[:, None]], 1)
        return dict(camera=b['camera'], parse=parse.to(DEV), qwen=b['q_cells'], q_last=b['q_last'], q_text=b['q_text'])

    @torch.no_grad()
    def predict(model, f, idx, flag):
        model.eval()
        r = np.concatenate([model.run(make(f, idx[i:i + 1024], flag, False)).float().cpu().numpy() for i in range(0, len(idx), 1024)])
        model.train()
        return r

    def official_n(eps_, idx, routes, lv):
        e = [eps_[i] for i in idx.tolist()]
        acts = []
        for x, v in zip(routes, np.exp(lv)):
            pts = np.concatenate([[[0.0, 0.0]], x])
            L = length(pts)
            nn_ = int(np.clip(round(L / max(v, 1e-3)), 1, 64))
            acts.append(moves_from(at(pts, np.linspace(0, L, nn_ + 1)[1:])))
        return score(e, dict(zip([x['episode'] for x in e], acts)), workers=8)[0]['score']

    @torch.no_grad()
    def predict2(model, f, idx):
        model.eval()
        out = [model.run2(make(f, idx[i:i + 1024], False, False)) for i in range(0, len(idx), 1024)]
        model.train()
        return np.concatenate([o[0].float().cpu().numpy() for o in out]), np.concatenate([o[1].float().cpu().numpy() for o in out])

    def run_speed(name, f, train_idx):
        ph, pv = {ep: [] for ep in ss.CHECKPOINTS}, {ep: [] for ep in ss.CHECKPOINTS}
        losses, sl = np.zeros(ss.EPOCHS), np.zeros(ss.EPOCHS)
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            model = SpeedRoute(['qwen'], ['q_last', 'q_text'], pdim).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=ss.LR, weight_decay=0.05)
            m = len(train_idx)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, ss.LR, total_steps=ss.EPOCHS * (m // ss.BS), pct_start=0.05)
            for ep in range(1, ss.EPOCHS + 1):
                perm = train_idx[torch.randperm(m)]
                tot, ts = 0.0, 0.0
                for k in range(m // ss.BS):
                    idx = perm[k * ss.BS:(k + 1) * ss.BS]
                    r, sp = model.run2(make(f, idx, False, False))
                    lp = score_loss(r.float(), f['floor'][idx].to(DEV))
                    lsp = (sp.float() - f['logv'][idx].to(DEV)).abs().mean()
                    opt.zero_grad()
                    (lp + 0.3 * lsp).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tot, ts = tot + float(lp), ts + float(lsp)
                losses[ep - 1] += tot / (m // ss.BS) / 3
                sl[ep - 1] += ts / (m // ss.BS) / 3
                if ep in ss.CHECKPOINTS:
                    ph[ep].append(predict2(model, tr, hold))
                    pv[ep].append(predict2(model, va, vi))
            print(f'  {name}: seed {seed} done; single-seed held-out by checkpoint '
                  + ' '.join(f'{scored(tr_eps, tr, hold, ph[ep][-1][0], steps):.2f}' for ep in ss.CHECKPOINTS), flush=True)
        avg = lambda d, ep: (np.mean([x[0] for x in d[ep]], 0), np.mean([x[1] for x in d[ep]], 0))
        hs = {ep: scored(tr_eps, tr, hold, avg(ph, ep)[0], steps) for ep in ss.CHECKPOINTS}
        vs = {ep: scored(va_eps, va, vi, avg(pv, ep)[0], steps) for ep in ss.CHECKPOINTS}
        tag = f'later_{group}_arm7'
        for ep in range(1, ss.EPOCHS + 1):
            kv = dict(epoch=ep, train_loss=float(losses[ep - 1]), train_speed_l1=float(sl[ep - 1]))
            if ep in ss.CHECKPOINTS:
                kv.update(held=hs[ep], val=vs[ep])
            log_curve(tag, **kv)
        best = max(hs, key=hs.get)
        (rh, lh), (rv, lv) = avg(ph, best), avg(pv, best)
        eh, ev = lh - logv_tr[hold].numpy(), lv - logv_va[vi].numpy()
        hn, vn = official_n(tr_eps, hold, rh, lh), official_n(va_eps, vi, rv, lv)
        print(f'{name}: train examples {len(train_idx)}; typical step: held-out {hs[best]:.2f} (epoch {best}), val {vs[best]:.2f}; moves at the predicted step: held-out {hn:.2f}, val {vn:.2f}; '
              f'predicted step error sd held {eh.std():.3f} (corr {np.corrcoef(lh, logv_tr[hold].numpy())[0, 1]:.2f}), val {ev.std():.3f} (corr {np.corrcoef(lv, logv_va[vi].numpy())[0, 1]:.2f}); '
              f'no information: {logv_va[vi].numpy().std():.3f}; by checkpoint held ' + ' '.join(f'{hs[ep]:.2f}' for ep in ss.CHECKPOINTS) + ' val ' + ' '.join(f'{vs[ep]:.2f}' for ep in ss.CHECKPOINTS), flush=True)
        json.dump(dict(name=name, train=len(train_idx), held=hs[best], epoch=best, val=vs[best], held_speed_moves=hn, val_speed_moves=vn, held_by=hs, val_by=vs),
                  open(f'runs/later_speed_{group}.json', 'w'), indent=1)

    def run(name, f, train_idx, flag=False, blank=False, degree=0, anchors=None, actions=False):
        ph, pv = {ep: [] for ep in ss.CHECKPOINTS}, {ep: [] for ep in ss.CHECKPOINTS}
        losses = np.zeros(ss.EPOCHS)
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            model = (CurveRoute(['qwen'], ['q_last', 'q_text'], pdim, degree) if degree else
                     AnchorRoute(['qwen'], ['q_last', 'q_text'], pdim, anchors) if anchors is not None else
                     ActionRoute(['qwen'], ['q_last', 'q_text'], pdim) if actions else
                     ss.GridRoute(['qwen'], ['q_last', 'q_text'], pdim + int(flag))).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=ss.LR, weight_decay=0.05)
            m = len(train_idx)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, ss.LR, total_steps=ss.EPOCHS * (m // ss.BS), pct_start=0.05)
            for ep in range(1, ss.EPOCHS + 1):
                perm = train_idx[torch.randperm(m)]
                tot = 0.0
                for k in range(m // ss.BS):
                    idx = perm[k * ss.BS:(k + 1) * ss.BS]
                    if anchors is not None:
                        loss = model.train_loss(make(f, idx, flag, blank), f['floor'][idx].to(DEV).float())
                    else:
                        loss = score_loss(model.run(make(f, idx, flag, blank)).float(), f['floor'][idx].to(DEV))
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tot += float(loss)
                losses[ep - 1] += tot / (m // ss.BS) / 3
                if ep in ss.CHECKPOINTS:
                    ph[ep].append(predict(model, tr, hold, flag))
                    pv[ep].append(predict(model, va, vi, flag))
            print(f'  {name}: seed {seed} done; single-seed held-out by checkpoint '
                  + ' '.join(f'{scored(tr_eps, tr, hold, ph[ep][-1], steps):.2f}' for ep in ss.CHECKPOINTS), flush=True)
        hs = {ep: scored(tr_eps, tr, hold, np.mean(ph[ep], 0), steps) for ep in ss.CHECKPOINTS}
        vs = {ep: scored(va_eps, va, vi, np.mean(pv[ep], 0), steps) for ep in ss.CHECKPOINTS}
        tag = f'later_{group}_{name.split(":")[0].replace(" ", "")}'
        for ep in range(1, ss.EPOCHS + 1):
            kv = dict(epoch=ep, train_loss=float(losses[ep - 1]))
            if ep in ss.CHECKPOINTS:
                kv.update(held=hs[ep], val=vs[ep])
            log_curve(tag, **kv)
        best = max(hs, key=hs.get)
        print(f'{name:58s} train examples {len(train_idx):6d}   held-out {hs[best]:.2f} (epoch {best})   val {vs[best]:.2f}   '
              f'by checkpoint held ' + ' '.join(f'{hs[ep]:.2f}' for ep in ss.CHECKPOINTS)
              + ' val ' + ' '.join(f'{vs[ep]:.2f}' for ep in ss.CHECKPOINTS), flush=True)
        log.append(dict(name=name, train=len(train_idx), held=hs[best], epoch=best, val=vs[best], held_by=hs, val_by=vs))
        json.dump(log, open(f'runs/later_starts_{group}{"" if 0 in arms else "_arms" + "".join(map(str, arms))}.json', 'w'), indent=1)
        return hs[best]

    log = []
    if 7 in arms:
        f2 = build(fit, 'qwen2')
        run_speed('arm 7: + later starts + step-size output', f2, torch.arange(len(f2['partial'])))
        del f2
        gc.collect()
    if 0 in arms:
        run('arm 0: baseline (first photos only)', tr, fit)
    if 8 in arms:
        run('arm 8: first photos, Bezier degree 4 output', tr, fit, degree=4)
    if 9 in arms:
        run('arm 9: first photos, Bezier degree 6 output', tr, fit, degree=6)
    if 10 in arms:
        anc = kmeans_routes(tr['floor'][fit].float())
        run('arm 10: first photos, pick 1 of 64 typical routes + correction', tr, fit, anchors=anc)
    if 11 in arms:
        run('arm 11: first photos, actions output (turn + length per segment)', tr, fit, actions=True)
    if 1 in arms or 2 in arms:
        f2 = build(fit, 'qwen2')
        every = torch.arange(len(f2['partial']))
        if 1 in arms:
            run('arm 1: + later starts, no marker', f2, every)
        if 2 in arms:
            run('arm 2: + later starts + partial flag', f2, every, flag=True)
        del f2
        gc.collect()
    if 3 in arms:
        f2 = build(fit, 'qwen2_blank')
        run('arm 3: + later starts + flag, instruction left out', f2, torch.arange(len(f2['partial'])), flag=True, blank=True)
        del f2
        gc.collect()
    if 6 in arms:                                        # no later starts: is the Stanford data worth having for this robot?
        stanford = torch.isin(tr['group'][others], torch.tensor([GROUPS.index('stanford_real'), GROUPS.index('stanford_sim')]))
        h6 = run('arm 6: first photos, all robots but Stanford', tr, torch.cat([fit, others[~stanford]]))
        if 45 in arms:                                   # then the in-between recipe once
            arms = arms + [5 if h6 > ALL_ROBOTS_HELD else 4]
            print(f'no Stanford {h6:.2f} vs all robots {ALL_ROBOTS_HELD:.2f} held-out -> arm {arms[-1]}', flush=True)
    if 4 in arms:
        f2 = build(torch.cat([fit, others]), 'qwen2')
        run('arm 4: + later starts + other robots first photos', f2, torch.arange(len(f2['partial'])))
        del f2
        gc.collect()
    if 5 in arms:
        stanford = torch.isin(tr['group'][others], torch.tensor([GROUPS.index('stanford_real'), GROUPS.index('stanford_sim')]))
        f2 = build(torch.cat([fit, others[~stanford]]), 'qwen2')
        run('arm 5: + later starts + other robots first photos, no Stanford', f2, torch.arange(len(f2['partial'])))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], [int(x) for x in a[a.index('--arms') + 1:]] if '--arms' in a else [0, 1, 2, 3])
