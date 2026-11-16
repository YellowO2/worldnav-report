"""A different family: anchors + refinement + choosing by expected score.

    python experiments/section5_first_model/anchor_routes.py recon [--k 64]
"""
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
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import recording, scored  # noqa: E402
from score import load  # noqa: E402
from curves import log_curve  # noqa: E402

EPOCHS, LR, BS = 16, 3e-4, 256
CHECKPOINTS = (4, 8, 12, 16)


def smooth_score(pred, true):
    """routenet.score_loss's score without the mean: pred, true (..., K, 2) -> (...)."""
    zero = torch.zeros_like(pred[..., :1, :])
    P, T = torch.cat([zero, pred], -2), torch.cat([zero, true], -2)
    span = true.norm(dim=-1).max(-1).values.clamp(min=0.5)
    gap = (pred - true).norm(dim=-1)
    soft_max = (gap.pow(8).mean(-1) + 1e-9).pow(1 / 8)
    dp, dt = P.diff(dim=-2), T.diff(dim=-2)
    lp, lt = dp.norm(dim=-1).sum(-1), dt.norm(dim=-1).sum(-1).clamp(min=0.1)
    F_ = (1 - soft_max / span).clamp(min=0) * (lt / torch.maximum(lp, lt))
    cos = (dp * dt).sum(-1) / (dp.norm(dim=-1) * dt.norm(dim=-1) + 1e-6)
    H_ = 1 - ((1 - cos) / 2).mean(-1)
    SR = torch.sigmoid((1.5 - gap[..., -1]) / 0.2)
    ATE = gap.pow(2).mean(-1).add(1e-9).sqrt()
    RPE = (dp - dt).norm(dim=-1).pow(2).mean(-1).add(1e-9).sqrt()
    return F_ * (1 + H_) / 2 * (0.7 + 0.1 * SR + 0.1 / (1 + ATE) + 0.1 / (1 + RPE))


def kmeans(x, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = a == j
            c[j] = x[m].mean(0) if m.any() else x[torch.randint(len(x), (1,), generator=g)]
    return c


class AnchorRoute(ss.GridRoute):
    def __init__(self, pdim, anchors):
        super().__init__(['qwen'], ['q_last', 'q_text'], pdim)
        self.register_buffer('anchors', anchors)                 # (K, 16, 2)
        K = len(anchors)
        self.logits = nn.Linear(256, K)
        self.delta = nn.Linear(256, K * 32)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def run2(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse']), self.camera(b['camera'])]
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))[:, 0]
        routes = self.anchors[None] + self.delta(q).view(B, -1, 16, 2)
        return self.logits(q), routes


def choose(logits, routes):
    """Expected-score choice and the probability-weighted mean."""
    p = logits.softmax(-1)
    B, K = p.shape
    s = smooth_score(routes[:, :, None].expand(B, K, K, 16, 2), routes[:, None].expand(B, K, K, 16, 2))   # (B, c, j)
    best = (s * p[:, None]).sum(-1).argmax(-1)
    return routes[torch.arange(B), best], (p[..., None, None] * routes).sum(1)


def main(group, k=64):
    final.setup()
    steps = pathnet.typical_steps()
    g = GROUPS.index(group)
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = tr['q_last'].shape[-1], tr['q_cells'].shape[-1]
    tr_eps, va_eps = load('data/lcvn/train.jsonl'), load('data/lcvn/val_seen.jsonl')
    sizes = {s: int(max(tr[f'seg_{s}'].max(), va[f'seg_{s}'].max())) + 1 for s in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    ti = torch.where((tr['group'] == g) & ok)[0]
    vi = torch.where(va['group'] == g)[0]
    recs = sorted({recording(tr_eps[i]['trajectory']) for i in ti.tolist()})
    random.seed(0)
    held = set(random.sample(recs, len(recs) // 10))
    is_h = torch.tensor([recording(tr_eps[i]['trajectory']) in held for i in ti.tolist()])
    fit, hold = ti[~is_h], ti[is_h]
    anchors = kmeans(tr['floor'][fit].float().flatten(1), k).view(k, 16, 2)
    pdim = ss.parse_vec(tr, fit[:2], sizes).shape[1]
    torch.manual_seed(0)
    model = AnchorRoute(pdim, anchors).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    n = len(fit)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (n // BS), pct_start=0.05)

    def make(f, idx):
        b = batch(f, idx, 'cuda')
        return dict(camera=b['camera'], parse=ss.parse_vec(f, idx, sizes).cuda(), qwen=b['q_cells'], q_last=b['q_last'], q_text=b['q_text'])

    @torch.no_grad()
    def predict(f, idx):
        model.eval()
        e, m = [], []
        for i in range(0, len(idx), 256):
            lg, rt = model.run2(make(f, idx[i:i + 256]))
            a, b = choose(lg.float(), rt.float())
            e.append(a.cpu().numpy())
            m.append(b.cpu().numpy())
        model.train()
        return np.concatenate(e), np.concatenate(m)
    print(f'{group}: fit {len(fit)}, held-out {len(hold)}, val {len(vi)}; {k} anchors', flush=True)
    for ep in range(1, EPOCHS + 1):
        perm = fit[torch.randperm(n)]
        for j in range(n // BS):
            idx = perm[j * BS:(j + 1) * BS]
            lg, rt = model.run2(make(tr, idx))
            true = tr['floor'][idx].cuda().float()
            near = torch.cdist(true.flatten(1), model.anchors.flatten(1)).argmin(1)
            chosen = rt[torch.arange(len(idx)), near].float()
            loss = nn.functional.cross_entropy(lg.float(), near) - smooth_score(chosen, true).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        log_curve(f'anchor_{group}_k{k}', epoch=ep, train_loss=float(loss))
        if ep in CHECKPOINTS:
            he, hm = predict(tr, hold)
            ve, vm = predict(va, vi)
            log_curve(f'anchor_{group}_k{k}', epoch=ep, held_expected=scored(tr_eps, tr, hold, he, steps), val_expected=scored(va_eps, va, vi, ve, steps),
                      held_mean=scored(tr_eps, tr, hold, hm, steps), val_mean=scored(va_eps, va, vi, vm, steps))
            print(f'epoch {ep}: expected-score choice held-out {scored(tr_eps, tr, hold, he, steps):.2f} val {scored(va_eps, va, vi, ve, steps):.2f} | '
                  f'weighted mean held-out {scored(tr_eps, tr, hold, hm, steps):.2f} val {scored(va_eps, va, vi, vm, steps):.2f}   '
                  f'(frozen regression, 1 seed: 43.58 / 42.93)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], k=int(a[a.index('--k') + 1]) if '--k' in a else 64)
