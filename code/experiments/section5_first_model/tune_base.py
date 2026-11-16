"""Step 2 of the per-robot study: tune the simple pipeline's network on one robot.

    python experiments/section5_first_model/tune_base.py recon
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
from ladder import inputs, recording, scored  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load  # noqa: E402

WHICH = 'base without DINOv3'
CHECKPOINTS = (2, 3, 4, 6, 8, 12)
DEFAULT = dict(width=512, depth=2, dropout=0.2, lr=1e-3, wd=0.05, bs=256)
SEARCH = [('width', (256, 512, 1024, 2048)), ('depth', (1, 2, 3, 4)), ('dropout', (0.0, 0.1, 0.2, 0.4, 0.6)),
          ('lr', (1e-4, 3e-4, 1e-3, 3e-3)), ('wd', (0.0, 0.05, 0.2)), ('bs', (128, 256, 512))]


def make_net(d_in, K, width, depth, dropout, mean):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]
        d = width
    layers.append(nn.Linear(d, 2 * K))
    net = nn.Sequential(*layers)
    with torch.no_grad():
        net[-1].bias.copy_(mean)
    return net


def train(X, Y, Xs, cfg, seed, K, dev='cuda'):
    """Predictions for each X in Xs at every checkpoint epoch: {epoch: [pred per X]}."""
    torch.manual_seed(seed)
    net = make_net(X.shape[1], K, cfg['width'], cfg['depth'], cfg['dropout'], Y.mean(0)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), cfg['lr'], weight_decay=cfg['wd'])
    Xd, Yd, bs, out = X.to(dev), Y.to(dev), cfg['bs'], {}
    for ep in range(1, max(CHECKPOINTS) + 1):
        net.train()
        perm = torch.randperm(len(Xd), device=dev)
        for k in range(0, len(Xd) - bs + 1, bs):
            b = perm[k:k + bs]
            loss = score_loss(net(Xd[b]).view(-1, K, 2), Yd[b].view(-1, K, 2))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in CHECKPOINTS:
            net.eval()
            with torch.no_grad():
                out[ep] = [net(x.to(dev)).view(-1, K, 2).cpu().numpy() for x in Xs]
    return out


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
    X, Xh, Xv = inputs(tr, fit, sizes, WHICH), inputs(tr, hold, sizes, WHICH), inputs(va, vi, sizes, WHICH)
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
    X, Xh, Xv = (X - mu) / sd, (Xh - mu) / sd, (Xv - mu) / sd
    Y = tr['floor'][fit].float().flatten(1)
    K = tr['floor'].shape[1]
    print(f'{group}: fit {len(fit)} episodes, held-out {len(hold)} (other recordings), val {len(vi)}', flush=True)

    def evaluate(cfg, with_val=False):
        runs = [train(X, Y, [Xh, Xv] if with_val else [Xh], cfg, s, K) for s in (0, 1, 2)]
        best = None
        for ep in CHECKPOINTS:
            h = scored(tr_eps, tr, hold, np.mean([r[ep][0] for r in runs], 0), steps)
            if best is None or h > best[0]:
                best = (h, ep)
        v = scored(va_eps, va, vi, np.mean([r[best[1]][1] for r in runs], 0), steps) if with_val else None
        return best[0], best[1], v

    log = []
    h, ep, v = evaluate(DEFAULT, with_val=True)
    print(f'default {DEFAULT}: held-out {h:.2f} (epoch {ep}), val {v:.2f}', flush=True)
    log.append(dict(step='default', cfg=dict(DEFAULT), held=h, epoch=ep, val=v))
    cfg = dict(DEFAULT)
    best_h = h
    for key, values in SEARCH:
        results = {}
        for val_ in values:
            c = dict(cfg, **{key: val_})
            if val_ == cfg[key]:
                results[val_] = (best_h, None)
                continue
            hh, ee, _ = evaluate(c)
            results[val_] = (hh, ee)
            print(f'  {key} = {val_}: held-out {hh:.2f} (epoch {ee})', flush=True)
        pick = max(results, key=lambda k: results[k][0])
        if results[pick][0] > best_h:
            cfg[key], best_h = pick, results[pick][0]
        print(f'{key}: keep {cfg[key]} (held-out {best_h:.2f})', flush=True)
        log.append(dict(step=key, results={str(k): r[0] for k, r in results.items()}, keep=cfg[key], held=best_h))
    h, ep, v = evaluate(cfg, with_val=True)
    print(f'\nfinal {cfg}: held-out {h:.2f} (epoch {ep}), val {v:.2f}', flush=True)
    log.append(dict(step='final', cfg=cfg, held=h, epoch=ep, val=v))
    json.dump(log, open(f'runs/tune_base_{group}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
