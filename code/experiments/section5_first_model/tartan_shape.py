"""TartanDrive: other route outputs on the same backbone.

    python experiments/section5_first_model/tartan_shape.py [anchor action curve4]
"""
import json
from math import comb
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
from routenet import score_loss  # noqa: E402
from score import load, score  # noqa: E402
from step_oracle import at, length, moves_from  # noqa: E402
from later_feats import starts  # noqa: E402
from later_starts import later_rows  # noqa: E402
from lookup_routes import piece  # noqa: E402
from path_labels import resample, start_frame  # noqa: E402

W = 0.3
CELLS, GLOBS = ['qwen'], ['q_last', 'q_text']


class SpeedRoute(ss.GridRoute):
    def __init__(self, cells, globs, pdim):
        super().__init__(cells, globs, pdim)
        self.speed = nn.Linear(256, 1)

    def run2(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse'])]
        if self.tilt:
            toks.append(self.camera(b['camera']))
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))[:, 0]
        return self.route(q).view(B, -1, 2), self.speed(q)[:, 0]


class ShapeRoute(SpeedRoute):
    def __init__(self, cells, globs, pdim, kind, anchors=None):
        super().__init__(cells, globs, pdim)
        self.kind = kind
        K = self.route.out_features // 2
        if kind == 'curve4':
            self.route = nn.Linear(256, 8)
            t = torch.arange(1, K + 1, dtype=torch.float32) / K
            self.register_buffer('basis', torch.stack([comb(4, j) * t ** j * (1 - t) ** (4 - j) for j in range(1, 5)], 1))
        elif kind == 'action':
            nn.init.zeros_(self.route.weight)
            nn.init.zeros_(self.route.bias)
        elif kind == 'anchor':
            self.register_buffer('anchors', anchors)
            A = len(anchors)
            self.route = nn.Linear(256, A + A * K * 2)
            nn.init.zeros_(self.route.weight[A:])
            nn.init.zeros_(self.route.bias[A:])

    def heads(self, b):
        raw, s = super().run2(b)
        A, K, _ = self.anchors.shape
        raw = raw.flatten(1)
        return raw[:, :A], raw[:, A:].view(-1, A, K, 2), s

    def run2(self, b):
        if self.kind == 'anchor':
            logits, res, s = self.heads(b)
            k = logits.argmax(1)
            return self.anchors[k] + res[torch.arange(len(k), device=k.device), k], s
        raw, s = super().run2(b)
        if self.kind == 'curve4':
            return torch.einsum('kj,bjc->bkc', self.basis.to(raw.dtype), raw), s
        heading = torch.cumsum(raw[..., 0], 1)
        length_ = nn.functional.softplus(raw[..., 1] - 0.5)
        return torch.cumsum(torch.stack([length_ * torch.cos(heading), length_ * torch.sin(heading)], -1), 1), s

    def path_loss(self, b, y):
        if self.kind != 'anchor':
            r, s = self.run2(b)
            return score_loss(r.float(), y), s
        logits, res, s = self.heads(b)
        k = (y[:, None] - self.anchors[None]).norm(dim=-1).mean(-1).argmin(1)
        r = self.anchors[k] + res[torch.arange(len(k), device=k.device), k]
        return nn.functional.cross_entropy(logits.float(), k) + score_loss(r.float(), y), s


def kmeans_routes(y, A, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = y.flatten(1).to(ss.DEV)
    c = x[torch.randperm(len(x), generator=g)[:A].to(ss.DEV)].clone()
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        for j in range(A):
            m = lab == j
            c[j] = x[m].mean(0) if m.any() else x[torch.randint(len(x), (1,), generator=g).item()]
    return c.view(A, *y.shape[1:])


def main():
    final.setup()
    steps = pathnet.typical_steps()
    vu, vu_eps = final.load_labelled('val_unseen'), load('data/lcvn/val_unseen.jsonl')
    tr = final.load_labelled('train')
    data = final.merge([tr, vu])
    eps = load('data/lcvn/train.jsonl') + vu_eps
    off = len(tr['episodes'])
    del tr
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = data['q_last'].shape[-1], data['q_cells'].shape[-1]
    sizes = {k: int(data[f'seg_{k}'].max()) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(data['stop_kind'].max()) + 1
    logv = torch.tensor([np.log(max(np.linalg.norm(np.diff(np.array(e['position']), axis=0), axis=1).mean(), 1e-3)) for e in eps]).float()
    day = [e['trajectory'].split('/')[-1].split('_')[0] for e in vu_eps]
    val = torch.tensor([off + i for i, r in enumerate(day) if r in {'20210826', '20210828', '20210910'}])
    hold = torch.tensor([off + i for i, r in enumerate(day) if r == '20210829'])
    fit_t = torch.tensor([off + i for i, r in enumerate(day) if r in {'20210902', '20210903'}])
    ok = torch.isfinite(data['floor']).flatten(1).all(1)
    every = torch.cat([torch.where(ok[:off] & (data['group'][:off] != GROUPS.index('tartan')))[0], fit_t])
    print(f'tartan: fit {len(fit_t)} episodes, pick {len(hold)}, report {len(val)}; all robots {len(every)}', flush=True)
    pdim = ss.parse_vec(data, fit_t[:2], sizes).shape[1]

    def rule_cut(pos):
        """Positions from a start -> the part a TartanDrive piece would keep: last photo within 10 m."""
        d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.array(pos), axis=0), axis=1))])
        return min(int((d <= 10.0).sum()) - 1, 19)
    by_piece = {piece(e['trajectory']): e for e in vu_eps}
    same = np.mean([rule_cut(e['position'] + by_piece[(r, k + 1)]['position']) == len(e['position']) - 1
                    for (r, k), e in by_piece.items() if (r, k + 1) in by_piece])
    key_row = {tuple(k): j for j, k in enumerate(torch.load('features/later_tartan_val_unseen.pt')['keys'])}
    src, lrow, target, lv_new, moves_new = [], [], [], [], []
    for i in fit_t.tolist():
        e = vu_eps[i - off]
        r, k0 = piece(e['trajectory'])
        nxt = by_piece.get((r, k0 + 1))
        if nxt is None:
            continue
        for k in starts(len(e['position']) - 1):
            pos = e['position'][k:] + nxt['position']
            m = rule_cut(pos)
            if m < 5:
                continue
            pts, _ = resample(start_frame(pos[:m + 1], e['yaw'][k]))
            if np.isfinite(pts).all():
                src.append(i), lrow.append(key_row[(e['episode'][:-2], k)]), target.append(pts), moves_new.append(m)
                lv_new.append(float(np.log(max(np.linalg.norm(np.diff(np.array(pos[:m + 1]), axis=0), axis=1).mean(), 1e-3))))
    n0 = len(data['episodes'])
    src = torch.tensor(src)
    used = sorted(set(lrow))
    posr = {r: j for j, r in enumerate(used)}
    Q = {'q_cells': 'cells', 'q_last': 'last', 'q_text': 'text_mean'}
    base = len(data['q_last'])
    for kk, v in list(data.items()):
        if torch.is_tensor(v) and len(v) == n0 and kk not in Q and kk not in ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls'):
            data[kk] = torch.cat([v, v[src]])
    data['floor'][n0:] = torch.tensor(np.stack(target), dtype=data['floor'].dtype)
    data['q_index'][n0:] = base + torch.tensor([posr[r] for r in lrow])
    d = later_rows('tartan_val_unseen', 'qwen2', used)
    for kk, k2 in Q.items():
        data[kk] = torch.cat([data[kk], d[k2].to(data[kk].dtype)])
    logv = torch.cat([logv, torch.tensor(lv_new).float()])
    extra = torch.arange(n0, n0 + len(src))
    tl = np.array([len(vu_eps[i - off]['position']) - 1 for i in fit_t.tolist()])
    print(f'the rule reproduces the stored piece end for {same:.0%} of pieces with a next piece; cross-piece later starts: {len(src)} episodes '
          f'from {len(used)} later photos (fit days {len(fit_t)} episodes); moves 10/50/90%: {np.percentile(moves_new, [10, 50, 90])} '
          f'(original pieces {np.percentile(tl, [10, 50, 90])})', flush=True)
    dev = ss.DEV

    def make(idx):
        b = batch(data, idx, dev)
        o = dict(camera=b['camera'], parse=ss.parse_vec(data, idx, sizes).to(dev), qwen=b['q_cells'], q_last=b['q_last'], q_text=b['q_text'])
        return o

    @torch.no_grad()
    def predict(m, idx):
        m.eval()
        out = [m.run2(make(idx[i:i + 1024])) for i in range(0, len(idx), 1024)]
        m.train()
        return np.concatenate([o[0].float().cpu().numpy() for o in out]), np.concatenate([o[1].float().cpu().numpy() for o in out])

    def official(idx, routes, Ls, ns):
        e = [eps[i] for i in idx.tolist()]
        acts = []
        for x, l, n in zip(routes, Ls, ns):
            pts = np.concatenate([[[0.0, 0.0]], x])
            pts = pts * (l / max(length(pts), 1e-6))
            n = int(np.clip(n, 1, 64))
            acts.append(moves_from(at(pts, np.linspace(0, l, n + 1)[1:])))
        return score(e, dict(zip([x['episode'] for x in e], acts)), workers=8)[0]

    def variants(idx, routes, lv, with_speed):
        L = np.array([length(np.concatenate([[[0.0, 0.0]], x])) for x in routes])
        out = {'typical': official(idx, routes, L, np.round(L / steps['tartan'])),
               'rule': official(idx, routes, L, np.where(L < 8.0, 19, np.round(L / 0.6)))}
        if with_speed:
            v = np.exp(lv)
            Lv = np.minimum(10.0, 19 * v)
            Lh = np.exp(0.5 * np.log(Lv) + 0.5 * np.log(L))
            out['speed count'] = official(idx, routes, L, np.round(L / v))
            out['speed route'] = official(idx, routes, Lv, np.round(Lv / v))
            out['speed route, half-way length'] = official(idx, routes, Lh, np.round(Lh / v))
        return out

    log = []
    kinds = [a for a in sys.argv[1:]] or ['anchor', 'action', 'curve4']
    anchors = torch.cat([kmeans_routes(data['floor'][every].float(), 64), kmeans_routes(data['floor'][fit_t].float(), 32)])
    for arm in kinds:
        train_set = every
        tag = 'tartan_shape_' + arm
        ph, pv = {e: [] for e in ss.CHECKPOINTS}, {e: [] for e in ss.CHECKPOINTS}
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            m = ShapeRoute(CELLS, GLOBS, pdim, arm, anchors if arm == 'anchor' else None).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=ss.LR, weight_decay=0.05)
            n = len(train_set)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, ss.LR, total_steps=ss.EPOCHS * (n // ss.BS), pct_start=0.05)
            for ep in range(1, ss.EPOCHS + 1):
                perm = train_set[torch.randperm(n)]
                tot, ts = 0.0, 0.0
                for k in range(n // ss.BS):
                    idx = perm[k * ss.BS:(k + 1) * ss.BS]
                    l_path, s = m.path_loss(make(idx), data['floor'][idx].to(dev).float())
                    l_speed = (s.float() - logv[idx].to(dev)).abs().mean()
                    loss = l_path + W * l_speed
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    tot, ts = tot + float(l_path), ts + float(l_speed)
                if ep in ss.CHECKPOINTS:
                    ph[ep].append(predict(m, hold))
                    pv[ep].append(predict(m, val))
                    hr, hl = ph[ep][-1]
                    Lh = np.array([length(np.concatenate([[[0.0, 0.0]], x])) for x in hr])
                    hs = official(hold, hr, Lh, np.round(Lh / steps['tartan']))['score']
                    err = float(np.std(hl - logv[hold].numpy()))
                    print(f'  {arm} seed {seed} epoch {ep}: path loss {tot / (n // ss.BS):.3f}, speed L1 {ts / (n // ss.BS):.3f}, pick day typical {hs:.2f}, pick day speed error sd {err:.3f}', flush=True)
                    log_curve(tag, seed=seed, epoch=ep, train_loss=tot / (n // ss.BS), train_speed_l1=ts / (n // ss.BS), held=hs, held_speed_err=err)
        avg = lambda d, e: (np.mean([x[0] for x in d[e]], 0), np.mean([x[1] for x in d[e]], 0))
        hs = {}
        for e in ss.CHECKPOINTS:
            r, _ = avg(ph, e)
            L = np.array([length(np.concatenate([[[0.0, 0.0]], x])) for x in r])
            hs[e] = official(hold, r, L, np.round(L / steps['tartan']))['score']
        best = max(hs, key=hs.get)
        r, lv = avg(pv, best)
        rh, lvh = avg(ph, best)
        err = lv - logv[val].numpy()
        print(f'\n{arm}: epoch {best} (pick day typical {hs[best]:.2f})' + (f'; report days: predicted step error sd {err.std():.3f}, bias {err.mean():+.3f}, corr {np.corrcoef(lv, logv[val].numpy())[0, 1]:.2f}' if True else ''), flush=True)
        res = {'pick': variants(hold, rh, lvh, True), 'report': variants(val, r, lv, True)}
        for name in res['report']:
            a, b = res['pick'][name], res['report'][name]
            print(f'  {name:30s} pick day {a["score"]:6.2f} | report days {b["score"]:6.2f}  SR {b["SR"]:.3f}  ATE {b["ATE"]:.3f}  RPE {b["RPE"]:.3f}  F {b["F"]:.3f}  H {b["H"]:.3f}', flush=True)
        log.append(dict(arm=arm, epoch=best, held=hs, pick={k: v['score'] for k, v in res['pick'].items()}, report={k: v for k, v in res['report'].items()}))
        json.dump(log, open('runs/tartan_shape.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
