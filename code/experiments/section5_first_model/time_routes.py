"""Does predicting the route per move.

    python experiments/section5_first_model/time_routes.py
"""
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
from ladder import recording  # noqa: E402
from path_labels import start_frame  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load, score  # noqa: E402

N_MAX, EPOCHS, LR, BS = 30, 12, 3e-4, 256
TEMPS = (1.0, 2.0, 4.0)          # count-head temperatures (higher = less confident)
ALPHAS = (0.0, 0.5, 1.0)         # speed head: 0 = the typical step (baseline)
CHECKPOINTS = (4, 8, 12)


def time_targets(eps):
    """(n_ep, N_MAX, 2) positions after each move (start frame, forward/right), padded with the last."""
    T = np.zeros((len(eps), N_MAX, 2), np.float32)
    n = np.zeros(len(eps), np.int64)
    for i, e in enumerate(eps):
        p = start_frame(e['position'], e['yaw'][0])[1:N_MAX + 1]
        n[i] = len(p)
        T[i, :len(p)] = p
        T[i, len(p):] = p[-1]
    return torch.from_numpy(T), torch.from_numpy(n)


def seq_score(P, T, n, c=None):
    """Smooth benchmark score index by index."""
    B, N, _ = T.shape
    k = torch.arange(N, device=T.device)[None]
    m = (k < n[:, None]).float()
    c = n if c is None else c
    mf = (k < torch.maximum(n, c)[:, None]).float()
    mc = (k < c[:, None]).float()
    zero = torch.zeros_like(P[:, :1])
    Pp, Tp = torch.cat([zero, P], 1), torch.cat([zero, T], 1)
    gap = (P - T).norm(dim=-1)
    span = (T.norm(dim=-1) * m).max(1).values.clamp(min=0.5)
    soft_max = ((gap.pow(8) * mf).sum(1) / mf.sum(1) + 1e-9).pow(1 / 8)
    dp, dt = Pp.diff(dim=1), Tp.diff(dim=1)
    lp, lt = (dp.norm(dim=-1) * mc).sum(1), (dt.norm(dim=-1) * m).sum(1).clamp(min=0.1)
    F_ = (1 - soft_max / span).clamp(min=0) * (lt / torch.maximum(lp, lt))
    cos = (dp * dt).sum(-1) / (dp.norm(dim=-1) * dt.norm(dim=-1) + 1e-6)
    H_ = 1 - (((1 - cos) / 2) * m).sum(1) / m.sum(1)
    end = gap.gather(1, (n - 1).clamp(min=0)[:, None])[:, 0]
    SR = torch.sigmoid((1.5 - end) / 0.2)
    ATE = ((gap.pow(2) * m).sum(1) / m.sum(1) + 1e-9).sqrt()
    RPE = (((dp - dt).norm(dim=-1).pow(2) * m).sum(1) / m.sum(1) + 1e-9).sqrt()
    return F_ * (1 + H_) / 2 * (0.7 + 0.1 * SR + 0.1 / (1 + ATE) + 0.1 / (1 + RPE)), gap, span, m


def truncate(P, nh):
    """P (B, N, 2) -> the first nh moves, then standing still (as the scorer pads a short prediction)."""
    k = torch.arange(P.shape[1], device=P.device)[None, :, None]
    last = P.gather(1, (nh - 1).clamp(min=0)[:, None, None].expand(-1, 1, 2))
    return torch.where(k < nh[:, None, None], P, last)


def choose_n(P, logits):
    """Expected-score count: for each candidate c."""
    p = logits.softmax(-1)
    B, N = p.shape
    cand = torch.arange(1, N + 1, device=P.device)
    best = torch.zeros(B, dtype=torch.long, device=P.device)
    best_v = torch.full((B,), -1.0, device=P.device)
    refs = [truncate(P, torch.full((B,), j, device=P.device)) for j in cand.tolist()]
    for c in cand.tolist():
        cc = torch.full((B,), c, device=P.device)
        pc = truncate(P, cc)
        v = sum(p[:, j - 1] * seq_score(pc, refs[j - 1], torch.full((B,), j, device=P.device), cc)[0] for j in cand.tolist())
        upd = v > best_v
        best[upd], best_v[upd] = c, v[upd]
    return best, p.argmax(-1) + 1


def moves_from(points, n):
    """Positions after each move (forward, right) -> n moves [dx, dy (left), dyaw], facing along each move."""
    xy = np.concatenate([[[0.0, 0.0]], points[:n]])
    xy = np.stack([xy[:, 0], -xy[:, 1]], 1)
    acts, yaw = [], 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a
        h = float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-4 else yaw
        c, s = np.cos(yaw), np.sin(yaw)
        acts.append([float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1]), h - yaw])
        yaw = h
    return acts


class PathCount(ss.GridRoute):
    """The path head (16 points along the path) plus a move-count head on the same backbone."""
    def __init__(self, pdim):
        super().__init__(['qwen'], ['q_last', 'q_text'], pdim)
        self.count = nn.Linear(256, N_MAX)

    def run2(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse']), self.camera(b['camera'])]
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))[:, 0]
        return self.route(q).view(B, -1, 2), self.count(q)


class PathSpeed(PathCount):
    """The path head plus a speed head: log of the route's mean step (metres per move)."""
    def __init__(self, pdim):
        super().__init__(pdim)
        self.speed = nn.Linear(256, 1)

    def run2(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse']), self.camera(b['camera'])]
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))[:, 0]
        return self.route(q).view(B, -1, 2), self.speed(q)[:, 0]


class PathSchedule(PathCount):
    """The path head plus a learned schedule: where along our path each of the N_MAX moves ends."""
    def __init__(self, pdim):
        super().__init__(pdim)
        self.sched = nn.Linear(256, N_MAX)
        nn.init.zeros_(self.sched.weight)
        nn.init.zeros_(self.sched.bias)

    def run2(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse']), self.camera(b['camera'])]
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))[:, 0]
        return self.route(q).view(B, -1, 2), self.sched(q)


def along(route, arc):
    """Points of a 16-point path (B, 16, 2) at arc lengths arc (B, N) metres (clamped to the path's end)."""
    pts = torch.cat([torch.zeros_like(route[:, :1]), route], 1)
    seg = pts.diff(dim=1).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
    s = torch.minimum(arc, cum[:, -1:])
    idx = torch.searchsorted(cum.contiguous(), s.contiguous()).clamp(1, pts.shape[1] - 1)
    a, b_ = cum.gather(1, idx - 1), cum.gather(1, idx)
    w = ((s - a) / (b_ - a).clamp(min=1e-6))[..., None]
    pa = pts.gather(1, (idx - 1)[..., None].expand(-1, -1, 2))
    pb = pts.gather(1, idx[..., None].expand(-1, -1, 2))
    return pa + w * (pb - pa)


def schedule_arc(raw, step0):
    """Raw schedule outputs (B, N) -> cumulative arc lengths: increments step0 x exp(raw)."""
    return (step0[:, None] * raw.clamp(-3, 3).exp()).cumsum(1)


def cut(route, c):
    """A 16-point path (B, 16, 2) cut into c equal moves -> positions (B, N_MAX, 2), padded with the end."""
    B = len(route)
    pts = torch.cat([torch.zeros_like(route[:, :1]), route], 1)
    seg = pts.diff(dim=1).norm(dim=-1)
    cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(1)], 1)
    k = torch.arange(1, N_MAX + 1, device=route.device)[None].float()
    s = (k / c[:, None].float()).clamp(max=1.0) * cum[:, -1:]
    idx = torch.searchsorted(cum.contiguous(), s.contiguous()).clamp(1, pts.shape[1] - 1)
    a, b_ = cum.gather(1, idx - 1), cum.gather(1, idx)
    w = ((s - a) / (b_ - a).clamp(min=1e-6))[..., None]
    pa = pts.gather(1, (idx - 1)[..., None].expand(-1, -1, 2))
    pb = pts.gather(1, idx[..., None].expand(-1, -1, 2))
    return pa + w * (pb - pa)


def choose_cut(route, logits, temp):
    """Expected-score move count for the path: candidate c = the path in c moves."""
    p = (logits / temp).softmax(-1)
    B = len(route)
    best = torch.zeros(B, dtype=torch.long, device=route.device)
    best_v = torch.full((B,), -1.0, device=route.device)
    refs = {j: cut(route, torch.full((B,), j, device=route.device)) for j in range(1, N_MAX + 1)}
    for c in range(1, N_MAX + 1):
        cc = torch.full((B,), c, device=route.device)
        v = sum(p[:, j - 1] * seq_score(refs[c], refs[j], torch.full((B,), j, device=route.device), cc)[0] for j in range(1, N_MAX + 1))
        upd = v > best_v
        best[upd], best_v[upd] = c, v[upd]
    return best


class TimeRoute(ss.GridRoute):
    def __init__(self, pdim):
        super().__init__(['qwen'], ['q_last', 'q_text'], pdim)
        self.pts = nn.Linear(256, N_MAX * 2)
        self.count = nn.Linear(256, N_MAX)

    def run2(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse']), self.camera(b['camera'])]
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))[:, 0]
        return self.pts(q).view(B, N_MAX, 2).cumsum(1), self.count(q)      # positions as summed moves


def main():
    final.setup()
    steps = pathnet.typical_steps()
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = tr['q_last'].shape[-1], tr['q_cells'].shape[-1]
    tr_eps, va_eps = load('data/lcvn/train.jsonl'), load('data/lcvn/val_seen.jsonl')
    Ttr, ntr = time_targets(tr_eps)
    Tva, nva = time_targets(va_eps)
    sizes = {s: int(max(tr[f'seg_{s}'].max(), va[f'seg_{s}'].max())) + 1 for s in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1) & (tr['group'] != GROUPS.index('tartan'))
    ti = torch.where(ok)[0]
    recs = sorted({recording(tr_eps[i]['trajectory']) for i in ti.tolist()})
    random.seed(0)
    held = set(random.sample(recs, len(recs) // 10))
    is_h = torch.tensor([recording(tr_eps[i]['trajectory']) in held for i in ti.tolist()])
    fit, hold = ti[~is_h], ti[is_h]
    pdim = ss.parse_vec(tr, fit[:2], sizes).shape[1]
    print(f'fit {len(fit)}, held-out {len(hold)}, val {len(va_eps)} episodes', flush=True)

    def make(f, idx):
        b = batch(f, idx, 'cuda')
        return dict(camera=b['camera'], parse=ss.parse_vec(f, idx, sizes).cuda(), qwen=b['q_cells'], q_last=b['q_last'], q_text=b['q_text'])

    def official(eps, f, idx, acts):
        e = [eps[i] for i in idx.tolist()]
        rows = score(e, dict(zip([x['episode'] for x in e], acts)))[1]
        g = f['group'][idx].numpy()
        out = {'all': float(np.mean([r['score'] for r in rows]))}
        for k, gn in enumerate(GROUPS):
            if (g == k).sum() >= 30:
                out[gn] = float(np.mean([r['score'] for r, gg in zip(rows, g) if gg == k]))
        return out

    heads = ('pathcount',) if '--pathcount' in sys.argv else ('pathspeed',) if '--pathspeed' in sys.argv else \
        ('pathschedule',) if '--pathschedule' in sys.argv else ('path', 'time')
    true_speed_tr = torch.tensor([float(np.linalg.norm(np.diff(np.array(e['position']), axis=0), axis=1).mean()) for e in tr_eps]).clamp(min=0.01)
    true_speed_va = torch.tensor([float(np.linalg.norm(np.diff(np.array(e['position']), axis=0), axis=1).mean()) for e in va_eps]).clamp(min=0.01)
    typ = torch.tensor([steps[g] for g in GROUPS])
    for head in heads:
        torch.manual_seed(0)
        model = (TimeRoute(pdim) if head == 'time' else PathCount(pdim) if head == 'pathcount'
                 else PathSpeed(pdim) if head == 'pathspeed' else PathSchedule(pdim) if head == 'pathschedule'
                 else ss.GridRoute(['qwen'], ['q_last', 'q_text'], pdim)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        n = len(fit)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (n // BS), pct_start=0.05)
        best = None
        for ep in range(1, EPOCHS + 1):
            perm = fit[torch.randperm(n)]
            tot = 0.0
            for j in range(n // BS):
                idx = perm[j * BS:(j + 1) * BS]
                b = make(tr, idx)
                if head == 'path':
                    loss = score_loss(model.run(b).float(), tr['floor'][idx].cuda())
                elif head == 'pathschedule':
                    r, raw = model.run2(b)
                    step0 = typ[tr['group'][idx].long()].cuda()
                    P = along(r.float().detach(), schedule_arc(raw.float(), step0))   # the path held fixed for the schedule
                    sq = seq_score(P, Ttr[idx].cuda(), ntr[idx].cuda())[0]
                    loss = score_loss(r.float(), tr['floor'][idx].cuda()) - sq.mean()
                elif head == 'pathspeed':
                    r, ls = model.run2(b)
                    target = (true_speed_tr[idx] / typ[tr['group'][idx].long()]).log().cuda()   # log speed vs the robot's typical
                    loss = score_loss(r.float(), tr['floor'][idx].cuda()) + 0.3 * (ls.float() - target).abs().mean()
                elif head == 'pathcount':
                    r, lg = model.run2(b)
                    loss = score_loss(r.float(), tr['floor'][idx].cuda()) + 0.3 * nn.functional.cross_entropy(lg.float(), ntr[idx].cuda() - 1)
                else:
                    P, lg = model.run2(b)
                    T, nt = Ttr[idx].cuda(), ntr[idx].cuda()
                    s, gap, span, m = seq_score(P.float(), T, nt)
                    loss = -s.mean() + 0.2 * ((gap * m).sum(1) / m.sum(1) / span).mean() + 0.5 * nn.functional.cross_entropy(lg.float(), nt - 1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() / (n // BS)
            log_curve(f'time_routes_{head}', epoch=ep, train_loss=tot)
            if ep not in CHECKPOINTS:
                continue
            model.eval()
            res = {}
            with torch.no_grad():
                for name, f, eps, idx in (('held', tr, tr_eps, hold), ('val', va, va_eps, torch.where(va['group'] != GROUPS.index('tartan'))[0])):
                    acts_e, acts_m, acts_t = [], [], {}
                    pred_speed, true_sp, grp = [], [], []
                    for i in range(0, len(idx), 512):
                        j = idx[i:i + 512]
                        b = make(f, j)
                        if head == 'path':
                            r = model.run(b).float().cpu().numpy()
                            acts_e += final.to_moves(r, f['group'][j], steps)
                        elif head == 'pathschedule':
                            r, raw = model.run2(b)
                            r = r.float()
                            acts_m += final.to_moves(r.cpu().numpy(), f['group'][j], steps)        # typical step (baseline)
                            step0 = typ[f['group'][j].long()].cuda()
                            arc = schedule_arc(raw.float(), step0)
                            L = torch.cat([torch.zeros_like(r[:, :1]), r], 1).diff(dim=1).norm(dim=-1).sum(1)
                            nmv = ((arc < L[:, None] - 1e-3).sum(1) + 1).clamp(max=N_MAX)      # moves until the end is reached
                            P = along(r, arc).cpu().numpy()
                            acts_t.setdefault('sched', []).extend(moves_from(p, int(c)) for p, c in zip(P, nmv.tolist()))
                        elif head == 'pathspeed':
                            r, ls = model.run2(b)
                            rn = r.float().cpu().numpy()
                            gg = f['group'][j].long()
                            for a_ in ALPHAS:                    # step = typical x exp(a x predicted log ratio)
                                stp = (typ[gg] * (a_ * ls.float().cpu()).exp()).numpy()
                                from manual_path_score import moves_along
                                acts_t.setdefault(a_, []).extend(moves_along(np.concatenate([[[0.0, 0.0]], x]), float(st)) for x, st in zip(rn, stp))
                            pred_speed.append(ls.float().cpu())
                            true_sp.append(((true_speed_tr if f is tr else true_speed_va)[j] / typ[gg]).log())
                            grp.append(gg)
                        elif head == 'pathcount':
                            r, lg = model.run2(b)
                            r = r.float()
                            acts_m += final.to_moves(r.cpu().numpy(), f['group'][j], steps)        # typical step (baseline)
                            for T_ in TEMPS:
                                cs = choose_cut(r, lg.float(), T_)
                                P = cut(r, cs).cpu().numpy()
                                acts_t.setdefault(T_, []).extend(moves_from(p, int(c)) for p, c in zip(P, cs.tolist()))
                        else:
                            P, lg = model.run2(b)
                            ne, nm = choose_n(P.float(), lg.float())
                            Pn = P.float().cpu().numpy()
                            acts_e += [moves_from(p, int(c)) for p, c in zip(Pn, ne.tolist())]
                            acts_m += [moves_from(p, int(c)) for p, c in zip(Pn, nm.tolist())]
                    if head == 'pathschedule':
                        res[name] = official(eps, f, idx, acts_m)
                        res[name + '_sched'] = official(eps, f, idx, acts_t['sched'])
                        continue
                    if head == 'pathspeed':
                        for a_, a2 in acts_t.items():
                            res[f'{name}_a{a_}'] = official(eps, f, idx, a2)
                        ps, ts, gs = torch.cat(pred_speed).numpy(), torch.cat(true_sp).numpy(), torch.cat(grp).numpy()
                        res[f'{name}_speed_corr'] = {GROUPS[k]: float(np.corrcoef(ps[gs == k], ts[gs == k])[0, 1]) for k in range(len(GROUPS)) if (gs == k).sum() > 30}
                        continue
                    if head == 'pathcount':
                        res[name] = official(eps, f, idx, acts_m)
                        for T_, a_ in acts_t.items():
                            res[f'{name}_count_T{T_}'] = official(eps, f, idx, a_)
                        continue
                    res[name] = official(eps, f, idx, acts_e)
                    if head == 'time':
                        res[name + '_most_likely_n'] = official(eps, f, idx, acts_m)['all']
            model.train()
            log_curve(f'time_routes_{head}', epoch=ep, **{k: (v.get('all', v) if isinstance(v, dict) else v) for k, v in res.items()})
            if head == 'pathschedule':
                print(f'pathschedule epoch {ep}: typical step held-out {res["held"]["all"]:.2f} val {res["val"]["all"]:.2f} '
                      + ' '.join(f'{g}:{v:.2f}' for g, v in res['val'].items() if g != 'all') + f' | learned schedule held-out '
                      f'{res["held_sched"]["all"]:.2f} val {res["val_sched"]["all"]:.2f} '
                      + ' '.join(f'{g}:{v:.2f}' for g, v in res['val_sched'].items() if g != 'all'), flush=True)
                continue
            if head == 'pathspeed':
                print(f'pathspeed epoch {ep}: speed corr (val, log, per robot) ' + ' '.join(f'{g}:{c:.2f}' for g, c in res['val_speed_corr'].items()), flush=True)
                for a_ in ALPHAS:
                    print(f'   alpha {a_}: held-out {res[f"held_a{a_}"]["all"]:.2f} val {res[f"val_a{a_}"]["all"]:.2f}  '
                          + ' '.join(f'{g}:{v:.2f}' for g, v in res[f"val_a{a_}"].items() if g != 'all'), flush=True)
                continue
            if head == 'pathcount':
                print(f'pathcount epoch {ep}: typical step held-out {res["held"]["all"]:.2f} val {res["val"]["all"]:.2f} | '
                      + ' | '.join(f'count T{T_}: held-out {res[f"held_count_T{T_}"]["all"]:.2f} val {res[f"val_count_T{T_}"]["all"]:.2f} '
                                   + ' '.join(f'{g}:{v:.1f}' for g, v in res[f"val_count_T{T_}"].items() if g != 'all') for T_ in TEMPS), flush=True)
                continue
            print(f'{head} epoch {ep}: held-out {res["held"]["all"]:.2f}, val {res["val"]["all"]:.2f} '
                  + ' '.join(f'{g}:{v:.1f}' for g, v in res['val'].items() if g != 'all')
                  + (f' | most likely n: held-out {res["held_most_likely_n"]:.2f}, val {res["val_most_likely_n"]:.2f}' if head == 'time' else ''), flush=True)
            if best is None or res['held']['all'] > best[0]:
                best = (res['held']['all'], ep, res['val'])
        if best is None:
            continue
        print(f'== {head}: best epoch {best[1]} (held-out {best[0]:.2f}): val {json.dumps({k: round(v, 2) for k, v in best[2].items()})}', flush=True)


if __name__ == '__main__':
    main()
