"""Retrieval model, route part.

    python experiments/section9_data_split/retrieval_route.py
"""
import os
import random
import sys
import zlib

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
from curves import log_curve  # noqa: E402
from floor_geometry import group_of  # noqa: E402
from lookup_final import ROBOTS, SPLITS, rebuild  # noqa: E402
from lookup_routes import hermite, local, piece  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load, score  # noqa: E402
from step_oracle import at, length, moves_from  # noqa: E402

K = 16


def anchors(P, N):
    """The seam geometry lookup_final.rebuild uses ->."""
    pp, py = np.array(P['position']), np.unwrap(np.array(P['yaw']))
    pn, ny = np.array(N['position']), np.unwrap(np.array(N['yaw']))
    v_prev = np.linalg.norm(np.diff(pp, axis=0), axis=1)[-3:].mean()
    v_next = np.linalg.norm(np.diff(pn, axis=0), axis=1)[:3].mean()
    turn = float(np.mean(np.diff(py)[-3:])) if len(py) > 3 else 0.0
    turn_n = float(np.mean(np.diff(ny)[:3])) if len(ny) > 3 else 0.0
    yaw0 = py[-1] + turn
    start = pp[-1] + v_prev * np.array([np.cos(py[-1] + 0.5 * turn), np.sin(py[-1] + 0.5 * turn)])
    end = local(pn[0] - v_next * np.array([np.cos(ny[0]), np.sin(ny[0])]), start, yaw0)
    dy = float(np.arctan2(np.sin(ny[0] - yaw0), np.cos(ny[0] - yaw0)))
    d = float(np.linalg.norm(end))
    curve = hermite(end, np.array([np.cos(dy), -np.sin(dy)]))
    L = length(curve)
    c16 = at(curve, np.linspace(0, L, K + 1)[1:]) if L > 1e-3 else np.zeros((K, 2))
    vbar = max(0.5 * (v_prev + v_next), 1e-3)
    a = float(np.arctan2(end[1], end[0]))
    x = [np.cos(a), np.sin(a), np.cos(dy), np.sin(dy), np.log(max(d, 0.02)), v_prev / max(d, 0.02), v_next / max(d, 0.02),
         np.log(max(v_prev, 1e-3) / max(v_next, 1e-3)), turn, turn_n, np.log(vbar)]
    return np.array(x, np.float32), c16.astype(np.float32), vbar, d


def truth(E):
    p, y = np.array(E['position']), np.array(E['yaw'])
    pts = local(p, p[0], y[0])
    L = length(pts)
    return (at(pts, np.linspace(0, L, K + 1)[1:]) if L > 1e-3 else np.zeros((K, 2))).astype(np.float32), L / (len(p) - 1)


class RouteFromNeighbours(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.path, self.step = nn.Linear(256, 2 * K), nn.Linear(256, 1)
        nn.init.zeros_(self.path.weight), nn.init.zeros_(self.path.bias), nn.init.zeros_(self.step.weight), nn.init.zeros_(self.step.bias)

    def forward(self, x, curve, d):
        h = self.net(x)
        return curve + d[:, None, None] * self.path(h).view(-1, K, 2), self.step(h)[:, 0]


class Rebuilder:
    """The trained models (3 seeds averaged) as a drop-in for lookup_final.rebuild: (P, N, robot) -> moves."""
    def __init__(self):
        self.models = []
        for seed in (0, 1, 2):
            d = torch.load(f'runs/retrieval_route/model_s{seed}.pt', weights_only=False)
            m = RouteFromNeighbours(len(d['mu']))
            m.load_state_dict(d['state'])
            self.models.append((m.eval(), d['mu'], d['sd']))
        self.robots = list(d['robots'])

    @torch.no_grad()
    def __call__(self, P, N, robot):
        x, c, vbar, d = anchors(P, N)
        if d <= 0.05:
            return rebuild(P, N)
        one = np.zeros(len(self.robots), np.float32)
        one[self.robots.index(robot)] = 1
        x, c, dd = torch.tensor(np.concatenate([x, one]))[None], torch.tensor(c)[None], torch.tensor([d]).float()
        out = [m((x - mu) / sd, c, dd) for m, mu, sd in self.models]
        path = np.mean([o[0][0].numpy() for o in out], 0)
        step = float(np.exp(np.mean([float(o[1]) for o in out])) * vbar)
        pts = np.concatenate([[[0.0, 0.0]], path])
        L = length(pts)
        n = int(np.clip(round(L / max(step, 1e-3)), 1, 64))
        return moves_from(at(pts, np.linspace(0, L, n + 1)[1:]))


def build():
    """-> rows: dict."""
    eps, all_eps = {}, {}
    for s in SPLITS:
        for e in load(f'data/lcvn/{s}.jsonl'):
            eps.setdefault(e['trajectory'], (s, e))
            all_eps.setdefault(e['trajectory'], []).append(e)
    by = {}
    for t, (s, e) in eps.items():
        by[piece(t)] = t
    rows = []
    for (rec, k), t in by.items():
        s = eps[t][0]
        g = group_of(t, s)
        tp, tn = by.get((rec, k - 1)), by.get((rec, k + 1))
        if tp is None or tn is None:
            continue
        if s != 'val_unseen' and (eps[tp][0] != 'train' or eps[tn][0] != 'train'):
            continue                                     # neighbours must be training pieces (as at test time)
        rows.append(dict(robot=g, split=s, rec=rec, t=t, P=eps[tp][1], N=eps[tn][1], E=eps[t][1]))
    return rows, all_eps


def main():
    rows, all_eps = build()
    rb = {g: i for i, g in enumerate(ROBOTS)}
    recs = sorted({r['rec'] for r in rows if r['split'] != 'val_seen'})
    random.seed(0)
    held = set(random.sample(recs, len(recs) // 10))
    part = lambda r: ('report' if r['split'] == 'val_seen' or (r['split'] == 'val_unseen' and zlib.crc32(r['rec'].encode()) % 10 < 3)
                      else 'held' if r['rec'] in held else 'fit')
    X, C, D, V, T, S, G, W = [], [], [], [], [], [], [], []
    for r in rows:
        x, c, vbar, d = anchors(r['P'], r['N'])
        tr, st = truth(r['E'])
        one = np.zeros(len(ROBOTS), np.float32)
        one[rb[r['robot']]] = 1
        X.append(np.concatenate([x, one])), C.append(c), D.append(d), V.append(vbar), T.append(tr), S.append(max(st, 1e-3)), G.append(r['robot']), W.append(part(r))
    X, C, T = torch.tensor(np.stack(X)), torch.tensor(np.stack(C)), torch.tensor(np.stack(T))
    D, V, S = torch.tensor(D).float(), torch.tensor(V).float(), torch.tensor(S).float()
    W, G = np.array(W), np.array(G)
    ok = (D > 0.05).numpy() & (T.norm(dim=-1).max(1).values > 0.05).numpy()
    fit, hold, rep = np.where((W == 'fit') & ok)[0], np.where((W == 'held') & ok)[0], np.where(W == 'report')[0]
    print(f'pieces with both neighbours: fit {len(fit)}, held-out {len(hold)}, report {len(rep)}; fit by robot ' +
          ', '.join(f'{g} {int(((W == "fit") & (G == g)).sum())}' for g in ROBOTS), flush=True)
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    sd[-len(ROBOTS):], mu[-len(ROBOTS):] = 1.0, 0.0
    Xn = (X - mu) / sd

    def loss_of(m, idx):
        p, s = m(Xn[idx], C[idx], D[idx])
        return score_loss(p, T[idx]), (s + V[idx].log() - S[idx].log()).abs().mean()

    def moves(path, step):
        pts = np.concatenate([[[0.0, 0.0]], path])
        L = length(pts)
        n = int(np.clip(round(L / max(step, 1e-3)), 1, 64))
        return moves_from(at(pts, np.linspace(0, L, n + 1)[1:]))

    best, state = 1e9, None
    preds = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = RouteFromNeighbours(X.shape[1])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        EP = 150
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EP * (len(fit) // 512 + 1), pct_start=0.05)
        best_s, keep = 1e9, None
        for ep in range(1, EP + 1):
            perm = np.random.default_rng(1000 * seed + ep).permutation(fit)
            tot = 0.0
            for i in range(0, len(perm), 512):
                lp, ls = loss_of(m, perm[i:i + 512])
                opt.zero_grad()
                (lp + 0.3 * ls).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(lp) * len(perm[i:i + 512])
            if ep % 10 == 0:
                m.eval()
                with torch.no_grad():
                    hp, hs_ = loss_of(m, hold)
                m.train()
                log_curve('retrieval_route', seed=seed, epoch=ep, train_loss=tot / len(fit), held_loss=float(hp), held_step_l1=float(hs_))
                if float(hp) + 0.3 * float(hs_) < best_s:
                    best_s, keep = float(hp) + 0.3 * float(hs_), ({k: v.clone() for k, v in m.state_dict().items()}, ep)
                if ep % 50 == 0:
                    print(f'  seed {seed} epoch {ep}: train path loss {tot / len(fit):.4f}, held-out path loss {float(hp):.4f}, step L1 {float(hs_):.3f}', flush=True)
        m.load_state_dict(keep[0])
        m.eval()
        with torch.no_grad():
            z0 = torch.zeros_like(C[hold])
            base = score_loss(C[hold], T[hold])
            p, s = m(Xn, C, D)
        print(f'seed {seed}: picked epoch {keep[1]}; held-out path loss {loss_of(m, hold)[0]:.4f} (smooth curve {float(base):.4f})', flush=True)
        preds.append((p.numpy(), s.numpy()))
        os.makedirs('runs/retrieval_route', exist_ok=True)
        torch.save(dict(state=keep[0], mu=mu, sd=sd, robots=ROBOTS), f'runs/retrieval_route/model_s{seed}.pt')
    P = np.mean([p for p, _ in preds], 0)
    St = np.exp(np.mean([s for _, s in preds], 0)) * V.numpy()
    print('\nreport pieces (both neighbours labelled), official score:')
    for g in ROBOTS:
        idx = [i for i in rep if G[i] == g]
        if not idx:
            continue
        ep_l, a_curve, a_path, a_both = [], [], [], []
        for i in idx:
            r = rows[i]
            mv0 = rebuild(r['P'], r['N'])
            use = bool(ok[i]) or D[i] > 0.05
            mv1 = moves(P[i], float(V[i])) if use else mv0
            mv2 = moves(P[i], float(St[i])) if use else mv0
            for e in all_eps[r['t']]:
                ep_l.append(e), a_curve.append(mv0), a_path.append(mv1), a_both.append(mv2)
        ids = [e['episode'] for e in ep_l]
        out = [score(ep_l, dict(zip(ids, a)), workers=16)[0] for a in (a_curve, a_path, a_both)]
        print(f'  {g:14s} n {len(idx):4d} pieces | smooth curve {out[0]["score"]:6.2f} (F {out[0]["F"]:.3f}, H {out[0]["H"]:.3f}, RPE {out[0]["RPE"]:.3f}) | learned path {out[1]["score"]:6.2f} '
              f'| learned path + step {out[2]["score"]:6.2f} (F {out[2]["F"]:.3f}, H {out[2]["H"]:.3f}, RPE {out[2]["RPE"]:.3f})', flush=True)


if __name__ == '__main__':
    main()
