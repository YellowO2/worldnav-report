"""SigLIP pipeline, one change at a time, for one robot (the per-robot study, steps 1-6).

    python 3_models/siglip_steps.py recon
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import one_hot, recording, scored  # noqa: E402
from routenet import RouteNet, score_loss  # noqa: E402
from score import load  # noqa: E402
from tune_base import train as train_mlp  # noqa: E402
from curves import log_curve  # noqa: E402

EPOCHS, LR, BS = 16, 3e-4, 256
CHECKPOINTS = (2, 4, 6, 8, 12, 16)
MLP = dict(width=1024, depth=2, dropout=0.0, lr=1e-3, wd=0.05, bs=256)
DEV = 'cuda'


def parse_vec(f, idx, sizes):
    return torch.cat([one_hot(f['seg_move'][idx], sizes['move']), one_hot(f['seg_side'][idx], sizes['side']),
                      one_hot(f['seg_degree'][idx], sizes['degree']), one_hot(f['stop_kind'][idx][:, None], sizes['stop'])], 1)


def maps8(f, idx):
    """Keyword maps (7 phrases x 16 x 16) averaged to the 8 x 8 cells -> (n, 64, 7)."""
    m = f['seg_map'][idx].float().view(-1, 7, 16, 16)
    return F.avg_pool2d(m, 2).flatten(2).transpose(1, 2)


CELL_DIMS = {'sig': 768, 'dino': 1024, 'maps': 7, 'qwen': 2560}
GLOB_DIMS = {'sentence': 768, 'pooled': 768, 'q_last': None, 'q_text': 2560}      # q_last set from the data


class GridRoute(RouteNet):
    """routenet's transformer with configurable inputs: per-cell grids."""
    def __init__(self, cells, globs, parse_dim, tilt=True):
        super().__init__(768, 768)
        self.tilt = tilt
        self.cells_in = nn.ModuleDict({c: nn.Sequential(nn.LayerNorm(CELL_DIMS[c]), nn.Linear(CELL_DIMS[c], 256)) for c in cells})
        self.globs_in = nn.ModuleDict({g: nn.Sequential(nn.LayerNorm(GLOB_DIMS[g]), nn.Linear(GLOB_DIMS[g], 256)) for g in globs})
        self.parsein = nn.Sequential(nn.Linear(parse_dim, 256), nn.GELU(), nn.Linear(256, 256))

    def run(self, b):
        B = len(b['camera'])
        cells = sum(self.cells_in[c](b[c]) for c in self.cells_in) + self.cell_pos
        toks = [self.globs_in[g](b[g]) for g in self.globs_in] + [self.parsein(b['parse'])]
        if self.tilt:                                    # camera numbers + the photo's lean (one robot: lean only)
            toks.append(self.camera(b['camera']))
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))
        return self.route(q[:, 0]).view(B, -1, 2)


def main(group):
    final.setup()
    steps = pathnet.typical_steps()
    g = GROUPS.index(group)
    tr, va = (final.load_labelled('train'), final.load_labelled('val_seen')) if QWEN else \
        (pathnet.load_split('train'), pathnet.load_split('val_seen'))
    if QWEN:
        GLOB_DIMS['q_last'], CELL_DIMS['qwen'] = tr['q_last'].shape[-1], tr['q_cells'].shape[-1]
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
    tartan = GROUPS.index('tartan')
    print(f'{group}: fit {len(fit)} episodes, held-out {len(hold)}, val {len(vi)}', flush=True)
    log = []

    def report(name, held_scores, preds_val, keep=None):
        best = max(held_scores, key=held_scores.get)
        v = scored(va_eps, va, vi, preds_val[best], steps)
        print(f'{name:44s} held-out {held_scores[best]:.2f} (epoch {best})   val {v:.2f}'
              + ('' if keep is None else f'   {"KEEP" if keep else "drop"}'), flush=True)
        log.append(dict(name=name, held=held_scores[best], epoch=best, val=v, keep=keep))
        path = f'runs/siglip_steps_{group}{"_only" if ONLY else ""}{"_qwen" if QWEN else ""}.json'
        old = json.load(open(path)) if ONLY and os.path.exists(path) else []
        json.dump(old + log if ONLY else log, open(path, 'w'), indent=1)
        return held_scores[best]

    # steps 1-2: small network on vectors
    def vec(f, idx, sentence):
        im = f['image_index'][idx]
        parts = [f['sig_pooled'][im].float(), parse_vec(f, idx, sizes), f['gravity'][idx].float()]
        if sentence:
            parts.insert(1, f['sentence'][idx].float())
        return torch.cat(parts, 1)
    K = tr['floor'].shape[1]
    for sentence in (False, True):
        X, Xh, Xv = vec(tr, fit, sentence), vec(tr, hold, sentence), vec(va, vi, sentence)
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
        X, Xh, Xv = (X - mu) / sd, (Xh - mu) / sd, (Xv - mu) / sd
        runs = [train_mlp(X, tr['floor'][fit].float().flatten(1), [Xh, Xv], MLP, s, K) for s in (0, 1, 2)]
        hs = {ep: scored(tr_eps, tr, hold, np.mean([r[ep][0] for r in runs], 0), steps) for ep in runs[0]}
        report(f'step {2 if sentence else 1}: small net, pooled + parse + tilt' + (' + sentence' if sentence else ''),
               hs, {ep: np.mean([r[ep][1] for r in runs], 0) for ep in runs[0]})

    # step 3 on: the transformer
    def make(f, idx, cells, globs):
        b = batch(f, idx, DEV)
        im = f['image_index'][idx]
        out = dict(camera=b['camera'], parse=parse_vec(f, idx, sizes).to(DEV))
        if 'sig' in cells:
            out['sig'] = f['sig_tokens'][im].to(DEV).float()
        if 'dino' in cells:
            out['dino'] = f['dino_tokens'][im].to(DEV).float()
        if 'maps' in cells:
            out['maps'] = maps8(f, idx).to(DEV)
        if 'qwen' in cells or 'q_last' in globs or 'q_text' in globs:
            out['qwen'], out['q_last'], out['q_text'] = b['q_cells'], b['q_last'], b['q_text']
        if 'sentence' in globs:
            out['sentence'] = f['sentence'][idx].to(DEV).float()
        if 'pooled' in globs:
            out['pooled'] = f['sig_pooled'][im].to(DEV).float()
        return out

    @torch.no_grad()
    def predict(model, f, idx, cells, globs):
        model.eval()
        r = np.concatenate([model.run(make(f, idx[i:i + 1024], cells, globs)).float().cpu().numpy()
                            for i in range(0, len(idx), 1024)])
        model.train()
        return r

    def transformer(cells, globs, train_idx, tilt=True, save=None):
        ph, pv = {ep: [] for ep in CHECKPOINTS}, {ep: [] for ep in CHECKPOINTS}
        pdim = parse_vec(tr, fit[:2], sizes).shape[1]
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = GridRoute(cells, globs, pdim, tilt).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
            n = len(train_idx)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (n // BS), pct_start=0.05)
            for ep in range(1, EPOCHS + 1):
                perm = train_idx[torch.randperm(n)]
                for k in range(n // BS):
                    idx = perm[k * BS:(k + 1) * BS]
                    loss = score_loss(model.run(make(tr, idx, cells, globs)).float(), tr['floor'][idx].to(DEV))
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                if ep in CHECKPOINTS:
                    ph[ep].append(predict(model, tr, hold, cells, globs))
                    pv[ep].append(predict(model, va, vi, cells, globs))
                    if save:
                        os.makedirs(save, exist_ok=True)
                        torch.save(model.state_dict(), f'{save}/seed{seed}_epoch{ep}.pt')
        hs = {ep: scored(tr_eps, tr, hold, np.mean(ph[ep], 0), steps) for ep in CHECKPOINTS}
        for ep in CHECKPOINTS:                           # the curve of the seed average
            log_curve(f'steps_{group}_{"+".join(cells + globs)}_{len(train_idx)}', epoch=ep, held=hs[ep],
                      val=scored(va_eps, va, vi, np.mean(pv[ep], 0), steps))
        if save:
            best_ep = max(hs, key=hs.get)
            np.save(f'{save}/val_routes.npy', np.mean(pv[best_ep], 0))
            np.save(f'{save}/val_index.npy', vi.numpy())
            json.dump(dict(epoch=best_ep, held=hs[best_ep], cells=cells, globs=globs, train=len(train_idx)),
                      open(f'{save}/picked.json', 'w'))
            for f_ in os.listdir(save):                  # keep only the picked epoch's models
                if f_.endswith('.pt') and not f_.endswith(f'_epoch{best_ep}.pt'):
                    os.remove(f'{save}/{f_}')
        return hs, {ep: np.mean(pv[ep], 0) for ep in CHECKPOINTS}

    if ONLY:                                             # one setting: --only cells globs [--no-tilt]
        c, gl = ONLY
        tag = f'only: {"+".join(c + gl)}{", no tilt" if NO_TILT else ""}, {EPOCHS} epochs'
        save = f'runs/qsteps_{group}_e{EPOCHS}' if '--save' in sys.argv else None
        report(tag, *transformer(c, gl, fit, tilt=not NO_TILT, save=save))
        return
    if QWEN:
        cells, globs = ['qwen'], ['q_last', 'q_text']
        best = report('Qwen: grid + summaries + parse + tilt', *transformer(cells, globs, fit))
        for kind, item, label in (('cell', 'dino', '+ DINOv3 grid per cell'), ('cell', 'sig', '+ SigLIP grid per cell'),
                                  ('glob', 'sentence', '+ sentence token')):
            c, gl = (cells + [item], globs) if kind == 'cell' else (cells, globs + [item])
            hs, pv = transformer(c, gl, fit)
            h = max(hs.values())
            keep = h > best
            report(f'  {label}', hs, pv, keep)
            if keep:
                cells, globs, best = c, gl, h
        every = torch.where((tr['group'] != tartan) & ok)[0]
        every = every[~torch.isin(every, hold)]
        h_all = report(f'Qwen best ({"+".join(cells + globs)}), trained on all robots', *transformer(cells, globs, every))
        use = every if h_all > best else fit
        report(f'Qwen best, final models saved ({"all robots" if h_all > best else "this robot"})',
               *transformer(cells, globs, use, save=f'runs/qsteps_{group}'))
        return
    cells, globs = ['sig'], []
    best = report('step 3: transformer, SigLIP grid + parse + tilt', *transformer(cells, globs, fit))
    for kind, item, label in (('glob', 'sentence', '+ sentence token'), ('glob', 'pooled', '+ pooled photo token'),
                              ('cell', 'maps', '+ keyword maps per cell'), ('cell', 'dino', '+ DINOv3 grid per cell')):
        c, gl = (cells + [item], globs) if kind == 'cell' else (cells, globs + [item])
        hs, pv = transformer(c, gl, fit)
        h = max(hs.values())
        keep = h > best
        report(f'  {label}', hs, pv, keep)
        if keep:
            cells, globs, best = c, gl, h
    every = torch.where((tr['group'] != tartan) & ok)[0]
    every = every[~torch.isin(every, hold)]
    report(f'step 6: best ({"+".join(cells + globs)}), trained on all robots', *transformer(cells, globs, every))


ONLY, NO_TILT, QWEN = None, False, '--qwen' in sys.argv
SEEDS = (0,) if '--one-seed' in sys.argv else (0, 1, 2)
if '--epochs' in sys.argv:                               # longer training: checkpoints every 4 epochs up to it
    EPOCHS = int(sys.argv[sys.argv.index('--epochs') + 1])
    CHECKPOINTS = tuple(e for e in (2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48) if e <= EPOCHS)

if __name__ == '__main__':
    # python 3_models/siglip_steps.py recon --only sig,dino sentence --no-tilt
    if '--only' in sys.argv:
        k = sys.argv.index('--only')
        ONLY = (sys.argv[k + 1].split(','), [x for x in sys.argv[k + 2].split(',') if x and x != '-'])
    NO_TILT = '--no-tilt' in sys.argv
    main(sys.argv[1])
