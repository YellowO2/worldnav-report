"""Stage 2: from the X to the route.

    python 3_models/routenet.py --name r1 --x x1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
import endnet  # noqa: E402
import pathnet  # noqa: E402
from endnet import EndNet, add_qwen, batch, drop  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from manual_path_score import moves_along  # noqa: E402
from path_labels import K  # noqa: E402
from score import load, score  # noqa: E402

UNUSED = ['parse', 'sentence']
NOISE = 0.08                     # (arbitrary, about stage 1's typical miss) noise on the training X, image units


class RouteNet(EndNet):
    """EndNet's layers (Qwen3.5 inputs) plus the X as a token and a K-point route head."""
    def __init__(self, qwen_dim, qwen_last_dim):
        super().__init__(False, 1, qwen_dim, qwen_last_dim)
        self.xin = nn.Sequential(nn.Linear(4, 256), nn.GELU(), nn.Linear(256, 256))   # X (u, v) + its floor point
        self.route = nn.Linear(256, 2 * K)
        self.da3in = nn.Sequential(nn.Linear(3, 256), nn.GELU(), nn.Linear(256, 256))   # depth per cell + has-depth

def forward_route(model, b):
    """Qwen3.5 cells, two Qwen3.5 summary tokens, the X token and the camera token through the encoder."""
    B = len(b['q_cells'])
    pos = model.cell_pos
    cells = model.qcell(model.qnorm(b['q_cells'])) + pos
    if 'da3' in b:                                       # single-photo DA3 depth per cell (da3_tartan.py), if given
        cells = cells + model.da3in(b['da3'])
    x = model.xin(torch.cat([b['x_uv'], b['x_floor'] / 10], -1))
    glob = torch.stack([model.qlast(model.qnorm_last(b['q_last'])), model.qtext(model.qnorm(b['q_text'])),
                        x, model.camera(b['camera'])], 1)
    h = model.norm_in(torch.cat([cells, glob], 1))
    mem = model.encoder(h)
    q = model.decoder(model.queries[:, -1:].expand(B, -1, -1), mem)
    return model.route(q[:, 0]).view(B, K, 2)


def lift(uv, b):
    return pathnet.lift_torch(uv[:, None], b['cam'], b['down'])[:, 0]


def fidelity_loss(pred, true):
    """Smooth stand-in for route fidelity: both (B, K, 2)."""
    span = true.norm(dim=-1).max(1).values.clamp(min=0.5)
    gap = (pred - true).norm(dim=-1)
    soft_max = (gap.pow(8).mean(1) + 1e-9).pow(1 / 8)
    zero = torch.zeros_like(pred[:, :1])
    lp = torch.cat([zero, pred], 1).diff(dim=1).norm(dim=-1).sum(1)
    lt = torch.cat([zero, true], 1).diff(dim=1).norm(dim=-1).sum(1).clamp(min=0.1)
    over = (lp / lt - 1).clamp(min=0)
    return (soft_max / span + over + 0.5 * gap.mean(1) / span).mean()


def score_loss(pred, true):
    """Minus a smooth stand-in for the benchmark score of one route, score = F (1 + H) / 2."""
    zero = torch.zeros_like(pred[:, :1])
    P, T = torch.cat([zero, pred], 1), torch.cat([zero, true], 1)
    span = true.norm(dim=-1).max(1).values.clamp(min=0.5)
    gap = (pred - true).norm(dim=-1)
    soft_max = (gap.pow(8).mean(1) + 1e-9).pow(1 / 8)
    dp, dt = P.diff(dim=1), T.diff(dim=1)
    lp, lt = dp.norm(dim=-1).sum(1), dt.norm(dim=-1).sum(1).clamp(min=0.1)
    F_ = (1 - soft_max / span).clamp(min=0) * (lt / torch.maximum(lp, lt))
    cos = (dp * dt).sum(-1) / (dp.norm(dim=-1) * dt.norm(dim=-1) + 1e-6)
    H_ = 1 - ((1 - cos) / 2).mean(1)
    SR = torch.sigmoid((1.5 - gap[:, -1]) / 0.2)
    ATE = gap.pow(2).mean(1).add(1e-9).sqrt()
    RPE = (dp - dt).norm(dim=-1).pow(2).mean(1).add(1e-9).sqrt()
    score = F_ * (1 + H_) / 2 * (0.7 + 0.1 * SR + 0.1 / (1 + ATE) + 0.1 / (1 + RPE))
    return (-score + 0.2 * gap.mean(1) / span).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--x', required=True, help='stage 1 run (xnet.py) whose X is used for validation')
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--loss', default='fidelity', choices=['fidelity', 'score'],
                    help='fidelity: the first stand-in (gap and overshoot); score: the smooth benchmark score')
    ap.add_argument('--no-x', action='store_true', help='no stage-1 X: the route straight from Qwen3.5 + camera')
    ap.add_argument('--extra', action='store_true',
                    help='also train on the extra views a third of the way along (extra_frames.py), end point only')
    args = ap.parse_args()
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, 'v3'
    endnet.QWEN_TAG = 'qwen2'
    dev, out = 'cuda', f'runs/{args.name}'
    os.makedirs(out, exist_ok=True)
    steps = pathnet.typical_steps()
    vals = {s: (load(f'data/lcvn/{s}.jsonl'), pathnet.load_split(s)) for s in ('val_seen', 'val_unseen')}
    for s, (_, f) in vals.items():
        add_qwen(f, s)
        xs = torch.load(f'runs/{args.x}/{s}_x.pt')
        assert xs['episodes'] == f['episodes']
        f['x_pred'] = xs['x']
    tr = pathnet.load_split('train')
    qd = add_qwen(tr, 'train')
    model = RouteNet(qd, tr['q_last'].shape[-1]).to(dev)

    def make(f, idx, noisy):
        b = batch(f, idx, dev)
        if noisy:
            uv = f['uv'][idx, -1].to(dev) + NOISE * torch.randn(len(idx), 2, device=dev)
        else:
            uv = f['x_pred'][idx].to(dev).float()
        b['x_uv'], b['x_floor'] = uv, lift(uv, b)
        if args.no_x:                                    # the X token carries nothing
            b['x_uv'], b['x_floor'] = torch.zeros_like(uv), torch.zeros_like(uv)
        return b

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for split, (eps, f) in vals.items():
            routes = []
            for i in range(0, len(eps), 1024):
                idx = torch.arange(i, min(i + 1024, len(eps)))
                routes.append(forward_route(model, make(f, idx, False)).float().cpu().numpy())
            routes = np.concatenate(routes)
            groups = [GROUPS[int(g)] for g in f['group']]
            preds = {e['episode']: moves_along(np.concatenate([[(0.0, 0.0)], r]), steps[g])
                     for e, g, r in zip(eps, groups, routes)}
            means, rows = score(eps, preds)
            res[split] = dict(score=round(means['score'], 2), F=round(means['F'], 3),
                              groups={g: round(float(np.mean([r['score'] for r, gg in zip(rows, groups) if gg == g])), 1)
                                      for g in GROUPS if g in groups})
        model.train()
        return res

    if args.extra:                                       # extra views: Qwen3.5 features + remaining route
        ex = torch.load('features/train_extra.pt')
        parts = [torch.load(f'features/train_extra_qwen2_chunks/{c}') for c in ex['chunks']]
        ex.update(q_cells=torch.cat([p['cells'] for p in parts]), q_last=torch.cat([p['last'] for p in parts]),
                  q_text=torch.cat([p['text_mean'] for p in parts]))
        print(f'extra views: {len(ex["route"])}', flush=True)

        def extra_batch(idx):
            cam = pathnet.TABLE[ex['group'][idx].long()]
            down = ex['gravity'][idx].float()
            return dict(q_cells=ex['q_cells'][idx].to(dev).float(), q_last=ex['q_last'][idx].to(dev).float(),
                        q_text=ex['q_text'][idx].to(dev).float(), cam=cam.to(dev), down=down.to(dev),
                        camera=torch.cat([cam[:, 11:13], cam[:, :1], down], 1).to(dev),
                        x_uv=torch.zeros(len(idx), 2, device=dev), x_floor=torch.zeros(len(idx), 2, device=dev))

    keep = torch.where(tr['visible'][:, -1])[0]
    n = len(keep)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * (n // args.bs), pct_start=0.05)
    log, best = [], -1.0
    for ep in range(args.epochs):
        t0, perm, tot = time.time(), keep[torch.randperm(n)], 0.0
        for k in range(n // args.bs):
            idx = perm[k * args.bs:(k + 1) * args.bs]
            pred = forward_route(model, make(tr, idx, True)).float()
            loss = (score_loss if args.loss == 'score' else fidelity_loss)(pred, tr['floor'][idx].to(dev))
            if args.extra:                               # end point of the remaining route, relative to its size
                eidx = torch.randint(0, len(ex['route']), (args.bs // 2,))
                end_pred = forward_route(model, extra_batch(eidx)).float()[:, -1]
                et = ex['floor'][eidx].to(dev)
                size = et.norm(dim=-1).max(1).values.clamp(min=0.5)
                loss = loss + ((end_pred - et[:, -1]).norm(dim=-1) / size).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() / (n // args.bs)
        row = dict(epoch=ep + 1, sec=round(time.time() - t0), train_loss=round(tot, 4))
        if (ep + 1) % 4 == 0:
            row.update(evaluate())
            row['phase1'] = round((row['val_seen']['score'] + row['val_unseen']['score']) / 2, 2)
            if row['phase1'] > best:
                best = row['phase1']
                torch.save(model.state_dict(), f'{out}/model.pt')
        log.append(row)
        print(json.dumps(row), flush=True)
        json.dump(dict(args=vars(args), epochs=log), open(f'{out}/log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
