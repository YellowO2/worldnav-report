"""Step 3 of the per-robot study: the Qwen pipeline.

    python experiments/section5_first_model/tune_qwen.py recon [--parse-only]
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import one_hot, recording, scored  # noqa: E402
from routenet import RouteNet, score_loss  # noqa: E402
from score import load  # noqa: E402

EPOCHS, LR, BS = 16, 3e-4, 256
CHECKPOINTS = (2, 4, 6, 8, 12, 16)


def parse_vec(f, idx, sizes):
    return torch.cat([one_hot(f['seg_move'][idx], sizes['move']), one_hot(f['seg_side'][idx], sizes['side']),
                      one_hot(f['seg_degree'][idx], sizes['degree']), one_hot(f['stop_kind'][idx][:, None], sizes['stop'])], 1)


class QwenRoute(RouteNet):
    def __init__(self, qd, qld, parse_dim):
        super().__init__(qd, qld)
        self.parsein = nn.Sequential(nn.Linear(parse_dim, 256), nn.GELU(), nn.Linear(256, 256)) if parse_dim else None

    def run(self, b):
        """routenet.forward_route with the parse as one more token (when given)."""
        B = len(b['q_cells'])
        cells = self.qcell(self.qnorm(b['q_cells'])) + self.cell_pos
        toks = [self.qlast(self.qnorm_last(b['q_last'])), self.qtext(self.qnorm(b['q_text'])), self.camera(b['camera'])]
        if self.parsein is not None:
            toks.append(self.parsein(b['parse']))
        h = self.norm_in(torch.cat([cells, torch.stack(toks, 1)], 1))
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), self.encoder(h))
        return self.route(q[:, 0]).view(B, -1, 2)


def main(group):
    final.setup()
    steps = pathnet.typical_steps()
    dev = 'cuda'
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
    print(f'{group}: fit {len(fit)} episodes, held-out {len(hold)}, val {len(vi)}', flush=True)

    def make(f, idx, with_parse):
        b = batch(f, idx, dev)
        if with_parse:
            b['parse'] = parse_vec(f, idx, sizes).to(dev)
        return b

    @torch.no_grad()
    def predict(model, f, idx, with_parse):
        model.eval()
        out = [model.run(make(f, idx[i:i + 1024], with_parse)).float().cpu().numpy() for i in range(0, len(idx), 1024)]
        model.train()
        return np.concatenate(out)

    parse_dim = parse_vec(tr, fit[:2], sizes).shape[1]
    log = []
    out_dir = f'runs/qparse_{group}'
    os.makedirs(out_dir, exist_ok=True)
    for with_parse in ((True,) if PARSE_ONLY else (False, True)):
        preds = {ep: ([], []) for ep in CHECKPOINTS}
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            model = QwenRoute(tr['q_cells'].shape[-1], tr['q_last'].shape[-1], parse_dim if with_parse else 0).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
            n = len(fit)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (n // BS), pct_start=0.05)
            for ep in range(1, EPOCHS + 1):
                perm = fit[torch.randperm(n)]
                for k in range(n // BS):
                    idx = perm[k * BS:(k + 1) * BS]
                    loss = score_loss(model.run(make(tr, idx, with_parse)).float(), tr['floor'][idx].to(dev))
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                if ep in CHECKPOINTS:
                    preds[ep][0].append(predict(model, tr, hold, with_parse))
                    preds[ep][1].append(predict(model, va, vi, with_parse))
                    if with_parse:                       # every checkpoint kept
                        torch.save(model.state_dict(), f'{out_dir}/seed{seed}_epoch{ep}.pt')
            print(f'  parse {with_parse}: seed {seed} trained', flush=True)
        held_scores = {ep: scored(tr_eps, tr, hold, np.mean(preds[ep][0], 0), steps) for ep in CHECKPOINTS}
        best = max(held_scores, key=held_scores.get)
        v = scored(va_eps, va, vi, np.mean(preds[best][1], 0), steps)
        name = 'Qwen pipeline + parse' if with_parse else 'Qwen pipeline (no parse)'
        print(f'{name}: held-out by epoch {json.dumps({k: round(s, 2) for k, s in held_scores.items()})}; '
              f'picked epoch {best}: held-out {held_scores[best]:.2f}, val {v:.2f}', flush=True)
        log.append(dict(name=name, held=held_scores, epoch=best, val=v))
        if with_parse:
            np.save(f'{out_dir}/val_routes.npy', np.mean(preds[best][1], 0))
            np.save(f'{out_dir}/val_index.npy', vi.numpy())
            json.dump(dict(epoch=best, held=held_scores[best], val=v), open(f'{out_dir}/picked.json', 'w'))
    json.dump(log, open(f'runs/tune_qwen_{group}.json', 'w'), indent=1)


PARSE_ONLY = '--parse-only' in sys.argv

if __name__ == '__main__':
    main(sys.argv[1])
