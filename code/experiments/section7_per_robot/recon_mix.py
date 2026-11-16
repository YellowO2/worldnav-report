"""Ensembles and training data for one robot's val_seen routes, all with the parse side fix.

    python experiments/section7_per_robot/recon_mix.py recon
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
from floor_geometry import GROUPS  # noqa: E402
from ladder import inputs, recording  # noqa: E402
from post_turn import CACHE, net_turn  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load, score  # noqa: E402

EPOCHS = 4


def side_fix(routes, side):
    r = routes.copy()
    wrong = np.array([(s == 'left' and net_turn(x) > 0.1) or (s == 'right' and net_turn(x) < -0.1) for x, s in zip(r, side)])
    r[wrong, :, 1] *= -1
    return r


def train_net(X, Y, seed, K, dev='cuda'):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.GELU(),
                        nn.Dropout(0.2), nn.Linear(512, 2 * K)).to(dev)
    with torch.no_grad():
        net[-1].bias.copy_(Y.mean(0))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
    Xd, Yd = X.to(dev), Y.to(dev)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Xd), device=dev)
        for k in range(0, len(Xd) - 255, 256):
            b = perm[k:k + 256]
            loss = score_loss(net(Xd[b]).view(-1, K, 2), Yd[b].view(-1, K, 2))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def main(group):
    final.setup()
    steps = pathnet.typical_steps()
    g = GROUPS.index(group)
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
    eps = load('data/lcvn/val_seen.jsonl')
    sizes = {k: int(max(tr[f'seg_{k}'].max(), va[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    K = tr['floor'].shape[1]
    vi = torch.where(va['group'] == g)[0]
    parse = {json.loads(l)['episode']: json.loads(l) for l in open('features/val_seen_parsed2.jsonl')}
    side = []
    for i in vi.tolist():
        t = [s for s in parse[eps[i]['episode']]['parsed']['segments'] if s['move'] in ('turn', 'curve')]
        side.append(t[0]['side'] if t else 'none')
    eg = [eps[i] for i in vi.tolist()]

    def run(routes):
        r = side_fix(routes, side)
        return float(np.mean([x['score'] for x in score(eg, dict(zip([e['episode'] for e in eg],
                                                                       final.to_moves(r, va['group'][vi], steps))))[1]]))
    res = {'r3': np.load(CACHE)[vi.numpy()]}
    ti = torch.where((tr['group'] == g) & ok)[0]
    Y = tr['floor'][ti].float().flatten(1)
    families = {}
    for which in ('base without DINOv3', 'base', 'Qwen summaries', 'base without parse'):
        X, Xv = inputs(tr, ti, sizes, which), inputs(va, vi, sizes, which)
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
        X, Xv = (X - mu) / sd, (Xv - mu) / sd
        runs = []
        for seed in (0, 1, 2):
            net = train_net(X, Y, seed, K)
            with torch.no_grad():
                runs.append(net(Xv.cuda()).view(-1, K, 2).cpu().numpy())
        families[which] = np.mean(runs, 0)
        print(f'{"net on " + which + ", 3 seeds":44s} {run(families[which]):.2f}', flush=True)
        # linear model on the same inputs (ridge 1e4), a different kind of member
        X1 = torch.cat([X, torch.ones(len(X), 1)], 1).double().cuda()
        W = torch.linalg.solve(X1.T @ X1 + 1e4 * torch.eye(X1.shape[1], dtype=torch.float64, device='cuda'),
                               X1.T @ Y.double().cuda()).float().cpu()
        families['linear on ' + which] = (torch.cat([Xv, torch.ones(len(Xv), 1)], 1) @ W).view(-1, K, 2).numpy()
        print(f'{"linear on " + which:44s} {run(families["linear on " + which]):.2f}', flush=True)
    print(f'{"r3":44s} {run(res["r3"]):.2f}')
    nets = ['base without DINOv3', 'base', 'Qwen summaries']
    print(f'{"r3 + net(base)":44s} {run((res["r3"] + families["base without DINOv3"]) / 2):.2f}')
    print(f'{"r3 + 3 nets (equal)":44s} {run((res["r3"] + sum(families[n] for n in nets)) / 4):.2f}')
    print(f'{"r3 + 3 nets + 3 linears (equal)":44s} {run((res["r3"] + sum(families[n] for n in nets) + sum(families["linear on " + n] for n in nets)) / 7):.2f}')
    for w in (0.3, 0.4, 0.5, 0.6):
        mixed = w * res['r3'] + (1 - w) * sum(families[n] for n in nets) / 3
        print(f'{"r3 weight " + str(w) + ", 3 nets the rest":44s} {run(mixed):.2f}')


if __name__ == '__main__':
    main(sys.argv[1])
