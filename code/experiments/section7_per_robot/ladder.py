"""The simplest models first, for one robot.

    python experiments/section7_per_robot/ladder.py recon
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
from routenet import score_loss  # noqa: E402
from score import load, score  # noqa: E402


def recording(t):
    return '_'.join(t.split('/')[-1].split('_')[:-2])


def one_hot(x, n):
    return torch.nn.functional.one_hot(x.long().clamp(min=0), n).flatten(1).float()


PCA = None


def inputs(f, idx, sizes, which):
    im = f['image_index'][idx]
    base = [f['sig_pooled'][im].float(), f['dino_cls'][im].float(), f['sentence'][idx].float(),
            one_hot(f['seg_move'][idx], sizes['move']), one_hot(f['seg_side'][idx], sizes['side']),
            one_hot(f['seg_degree'][idx], sizes['degree']), one_hot(f['stop_kind'][idx][:, None], sizes['stop']),
            f['gravity'][idx].float()]
    maps = [f['seg_map'][idx].float().flatten(1), f['has_phrase'][idx].float()]   # keyword maps
    q = f['q_index'][idx]
    qwen = [f['q_last'][q].float().flatten(1), f['q_text'][q].float().flatten(1)]
    if which == 'SigLIP pooled + maps':                 # SigLIP2 pooled vector + keyword maps + parse
        return torch.cat([base[0]] + maps + base[3:], 1)
    if which == 'base without DINOv3':                  # = base: SigLIP2 pooled + sentence + parse + tilt
        return torch.cat([base[0]] + base[2:], 1)
    if which == 'base without parse':                   # SigLIP2 pooled + sentence + tilt
        return torch.cat([base[0], base[2], base[-1]], 1)
    if which == 'base without sentence':                # SigLIP2 pooled + parse + tilt
        return torch.cat([base[0]] + base[3:], 1)
    if which == 'SigLIP only':                          # SigLIP 8 x 8 grid (16 numbers per cell) + keyword maps + parse
        grid = ((f['sig_tokens'][im].float() - PCA[0]) @ PCA[1]).flatten(1)
        return torch.cat([grid] + maps + base[3:], 1)
    parts = {'base': base, 'base + keyword maps': base + maps, 'Qwen summaries': qwen,
             'base + maps + Qwen': base + maps + qwen}[which]
    return torch.cat(parts, 1)


def scored(eps, f, idx, routes, steps):
    e = [eps[i] for i in idx]
    rows = score([x for x in e], dict(zip([x['episode'] for x in e], final.to_moves(routes, f['group'][idx], steps))))[1]
    return float(np.mean([r['score'] for r in rows]))


SETS = ('base', 'base + keyword maps', 'Qwen summaries', 'base + maps + Qwen')


def main(group):
    final.setup()
    steps = pathnet.typical_steps()
    g = GROUPS.index(group)
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
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
    print(f'{group}: fit on {len(fit)} training episodes, held-out {len(hold)} (other recordings), val {len(vi)}', flush=True)

    K = tr['floor'].shape[1]
    Yf = tr['floor'][fit].float().flatten(1)
    mean = Yf.mean(0)
    print(f'0  average route: val {scored(va_eps, va, vi, mean.view(1, K, 2).expand(len(vi), -1, -1).numpy(), steps):.1f}', flush=True)
    global PCA
    cells = tr['sig_tokens'][tr['image_index'][fit].unique()].float().reshape(-1, 768)
    cells = cells[torch.randperm(len(cells))[:200000]]
    m = cells.mean(0)
    PCA = (m, torch.pca_lowrank(cells - m, q=16, center=False)[2])
    summary = []
    for which in SETS:
        Xf, Xh, Xv = inputs(tr, fit, sizes, which), inputs(tr, hold, sizes, which), inputs(va, vi, sizes, which)
        mu, sd = Xf.mean(0), Xf.std(0).clamp(min=1e-4)
        Xf, Xh, Xv = (Xf - mu) / sd, (Xh - mu) / sd, (Xv - mu) / sd
        best = None
        X1 = torch.cat([Xf, torch.ones(len(Xf), 1)], 1).double().cuda()
        G, B = X1.T @ X1, X1.T @ Yf.double().cuda()
        for lam in (1e3, 1e4, 3e4, 1e5, 3e5, 1e6):
            W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=torch.float64, device='cuda'), B).float().cpu()
            lin = lambda X: (torch.cat([X, torch.ones(len(X), 1)], 1) @ W).view(-1, K, 2).numpy()  # noqa: E731
            h = scored(tr_eps, tr, hold, lin(Xh), steps)
            if best is None or h > best[0]:
                best = (h, lam, lin(Xv))
        v_lin = scored(va_eps, va, vi, best[2], steps)
        print(f'1  {which} ({Xf.shape[1]} inputs): linear (ridge {best[1]:.0e}, held-out {best[0]:.1f}) val {v_lin:.1f}', flush=True)
        dev = 'cuda'
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(Xf.shape[1], 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 512), nn.GELU(),
                            nn.Dropout(0.2), nn.Linear(512, 2 * K)).to(dev)
        with torch.no_grad():
            net[-1].bias.copy_(mean)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.05)
        Xd, Yd = Xf.to(dev), Yf.to(dev)
        best = None
        for ep in range(1, 16):
            net.train()
            perm = torch.randperm(len(Xd), device=dev)
            for k in range(0, len(Xd) - 255, 256):
                b = perm[k:k + 256]
                loss = score_loss(net(Xd[b]).view(-1, K, 2), Yd[b].view(-1, K, 2))
                opt.zero_grad()
                loss.backward()
                opt.step()
            if ep % 3 == 0:
                net.eval()
                with torch.no_grad():
                    ph, pv = net(Xh.to(dev)).view(-1, K, 2).cpu().numpy(), net(Xv.to(dev)).view(-1, K, 2).cpu().numpy()
                h = scored(tr_eps, tr, hold, ph, steps)
                if best is None or h > best[0]:
                    best = (h, ep, pv)
        v_net = scored(va_eps, va, vi, best[2], steps)
        print(f'2  {which}: network, score loss (epoch {best[1]}, held-out {best[0]:.1f}) val {v_net:.1f}', flush=True)
        summary.append((which, v_lin, v_net))
    print('\nval on ' + group + ' (r3, the Qwen transformer: see loss_table.py)')
    for which, a_, b_ in summary:
        print(f'{which:24s} linear {a_:5.1f}   network {b_:5.1f}')
    print(json.dumps(dict(note='compare: r3 (Qwen route model, all robots) val on this robot, from loss_table.py')))


if __name__ == '__main__':
    if len(sys.argv) > 2:                                # e.g
        SETS = tuple(sys.argv[2:])
    main(sys.argv[1])
