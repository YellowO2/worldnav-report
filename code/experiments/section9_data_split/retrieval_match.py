"""Retrieval model, matching part.

    python experiments/section9_data_split/retrieval_match.py --embed
"""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
from curves import log_curve  # noqa: E402
from lookup_final import ROBOTS, Photos, test_photos  # noqa: E402
from retrieval_route import build  # noqa: E402


def keys_of(r):
    split = lambda e: 'val_unseen' if e['trajectory'].startswith('val_unseen') else e['trajectory'].split('/')[0]
    return ((split(r['P']), r['P']['trajectory'], -1), (split(r['N']), r['N']['trajectory'], 0), (split(r['E']), r['E']['trajectory'], 0))


class Matcher(nn.Module):
    def __init__(self, dim=768, k=128):
        super().__init__()
        self.A, self.B = nn.Linear(dim, k, bias=False), nn.Linear(dim, k, bias=False)
        self.w = nn.Parameter(torch.tensor([[1.0, 0.5, 1.0, 0.0]]).repeat(len(ROBOTS), 1))   # the fixed sum, per robot
        self.g = nn.Parameter(torch.zeros(len(ROBOTS), 2))                                   # weight of the learned maps
        self.temp = nn.Parameter(torch.tensor(3.0))

    def forward(self, ea, eb, ga, gb, ec, gc, robot):
        """Holes (n) against candidates (m) -> (n, m) scores."""
        z = lambda s: (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6)
        f = [z(ea @ ec.T), z(eb @ ec.T), z(ga @ gc.T), z(gb @ gc.T)]
        la = z(F.normalize(self.A(ea), dim=-1) @ F.normalize(self.A(ec), dim=-1).T)
        lb = z(F.normalize(self.B(eb), dim=-1) @ F.normalize(self.B(ec), dim=-1).T)
        w, g = self.w[robot], self.g[robot]
        return sum(w[i] * f[i] for i in range(4)) + g[0] * la + g[1] * lb


def main(embed):
    rows, _ = build()
    photos = Photos()
    if embed:
        ks = [k for r in rows for k in keys_of(r)]
        todo = [k for k in dict.fromkeys(ks) if k not in photos.cache]
        print(f'{len(ks)} photos needed, {len(todo)} to embed', flush=True)
        for i in range(0, len(todo), 2048):
            photos.get(todo[i:i + 2048])
            print(f'  {min(i + 2048, len(todo))} / {len(todo)}', flush=True)
        return
    rows = [r for r in rows if all(k in photos.cache for k in keys_of(r))]
    recs = sorted({r['rec'] for r in rows if r['split'] == 'train'})
    random.seed(0)
    held = set(random.sample(recs, len(recs) // 10))
    T = lambda ks: tuple(torch.tensor(x) for x in photos.get(ks))
    data = {}
    for g in ROBOTS:
        for part in ('fit', 'held', 'val'):
            rs = [r for r in rows if r['robot'] == g and (('val' if r['split'] != 'train' else 'held' if r['rec'] in held else 'fit') == part)]
            if len(rs) < 8:
                continue
            (ea, ga), (eb, gb), (ec, gc) = (T([keys_of(r)[j] for r in rs]) for j in range(3))
            data[(g, part)] = dict(ea=ea, eb=eb, ga=ga, gb=gb, ec=ec, gc=gc)
    _, t_by = test_photos()
    extra = {g: T([('test', p) for p, _ in t_by[g].values()]) for g in ROBOTS}
    print({k: len(v['ea']) for k, v in data.items()}, flush=True)

    def accuracy(m, g, part, with_test):
        d = data[(g, part)]
        ec, gc = (torch.cat([d['ec'], extra[g][0]]), torch.cat([d['gc'], extra[g][1]])) if with_test else (d['ec'], d['gc'])
        with torch.no_grad():
            s = m(d['ea'], d['eb'], d['ga'], d['gb'], ec, gc, ROBOTS.index(g))
        return float((s.argmax(1) == torch.arange(len(s))).float().mean())

    torch.manual_seed(0)
    m = Matcher()
    base = {k: accuracy(m, k[0], k[1], k[1] == 'val') for k in data if k[1] != 'fit'}
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    fit_robots = [g for g in ROBOTS if (g, 'fit') in data]
    best, keep = -1, None
    for ep in range(1, 61):
        tot, n = 0.0, 0
        for _ in range(40):
            g = random.choice(fit_robots)
            d = data[(g, 'fit')]
            idx = torch.randperm(len(d['ea']))[:256]
            s = m(d['ea'][idx], d['eb'][idx], d['ga'][idx], d['gb'][idx], d['ec'][idx], d['gc'][idx], ROBOTS.index(g)) * self_temp(m)
            y = torch.arange(len(idx))
            loss = F.cross_entropy(s, y) + F.cross_entropy(s.T, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + float(loss), n + 1
        h = np.mean([accuracy(m, g, 'held', False) for g in ROBOTS if (g, 'held') in data])
        log_curve('retrieval_match', epoch=ep, train_loss=tot / n, held=float(h))
        if h > best:
            best, keep = h, ({k: v.clone() for k, v in m.state_dict().items()}, ep)
        if ep % 10 == 0:
            print(f'  epoch {ep}: loss {tot / n:.3f}, held-out best-match accuracy {h:.3f}', flush=True)
    m.load_state_dict(keep[0])
    os.makedirs('runs/retrieval_match', exist_ok=True)
    torch.save(keep[0], 'runs/retrieval_match/model.pt')
    print(f'picked epoch {keep[1]}. Best-match accuracy, fixed sum -> learned (val: candidates include the test photos):')
    for k in sorted(base):
        print(f'  {k[0]:14s} {k[1]:5s} n {len(data[k]["ea"]):4d}: {base[k]:.1%} -> {accuracy(m, k[0], k[1], k[1] == "val"):.1%}   weights {[round(float(x), 2) for x in m.w[ROBOTS.index(k[0])]]} learned maps {[round(float(x), 2) for x in m.g[ROBOTS.index(k[0])]]}', flush=True)


def self_temp(m):
    return m.temp.clamp(0.5, 20)


if __name__ == '__main__':
    main('--embed' in sys.argv)
