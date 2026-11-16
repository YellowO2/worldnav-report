"""TartanDrive, one change at a time (the per-robot study for the robot that is 45% of the test).

    python experiments/section7_per_robot/tartan_steps.py
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
import siglip_steps as ss  # noqa: E402
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import scored  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load  # noqa: E402
from tartan import add_da3, recording, split_routes  # noqa: E402

ss.CELL_DIMS['da3'] = 3


def main():
    final.setup()
    steps = pathnet.typical_steps()
    vu, vu_eps = final.load_labelled('val_unseen'), load('data/lcvn/val_unseen.jsonl')
    add_da3(vu, vu_eps)
    tr = final.load_labelled('train')
    tr['da3'], tr['has_da3'] = torch.zeros(len(tr['episodes']), 64, 2, dtype=torch.float16), torch.zeros(len(tr['episodes']))
    data = final.merge([tr, vu])                          # val_unseen rows come after the training rows
    eps = load('data/lcvn/train.jsonl') + vu_eps
    off = len(tr['episodes'])
    del tr
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = data['q_last'].shape[-1], data['q_cells'].shape[-1]
    sizes = {k: int(data[f'seg_{k}'].max()) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(data['stop_kind'].max()) + 1
    if BY_DAY:                                           # whole recording days held out (test is likely other days)
        rec = [e['trajectory'].split('/')[-1].split('_')[0] for e in vu_eps]
        report_recs, pick_recs = {'20210826', '20210828', '20210910'}, {'20210829'}
    else:
        report_recs = split_routes()
        others = sorted({recording(e['trajectory']) for e in vu_eps} - report_recs)
        random.seed(0)
        pick_recs = set(random.sample(others, len(others) // 8))
        rec = [recording(e['trajectory']) for e in vu_eps]
    val = torch.tensor([off + i for i, r in enumerate(rec) if r in report_recs])
    hold = torch.tensor([off + i for i, r in enumerate(rec) if r in pick_recs])
    fit_t = torch.tensor([off + i for i, r in enumerate(rec) if r not in report_recs and r not in pick_recs])
    ok = torch.isfinite(data['floor']).flatten(1).all(1)
    every = torch.cat([torch.where(ok[:off] & (data['group'][:off] != GROUPS.index('tartan')))[0], fit_t])
    print(f'tartan: fit {len(fit_t)} episodes, pick (held-out) {len(hold)}, report {len(val)}; all robots {len(every)}', flush=True)
    pdim = ss.parse_vec(data, fit_t[:2], sizes).shape[1]
    dev = ss.DEV
    log = []

    def make(idx, cells, globs):
        b = batch(data, idx, dev)
        im = data['image_index'][idx]
        o = dict(camera=b['camera'], parse=ss.parse_vec(data, idx, sizes).to(dev))
        o['qwen'], o['q_last'], o['q_text'] = b['q_cells'], b['q_last'], b['q_text']
        if 'sig' in cells:
            o['sig'] = data['sig_tokens'][im].to(dev).float()
        if 'dino' in cells:
            o['dino'] = data['dino_tokens'][im].to(dev).float()
        if 'da3' in cells:
            o['da3'] = torch.cat([data['da3'][idx].float(), data['has_da3'][idx].float()[:, None, None].expand(-1, 64, 1)], -1).to(dev)
        if 'sentence' in globs:
            o['sentence'] = data['sentence'][idx].to(dev).float()
        return o

    @torch.no_grad()
    def predict(m, idx, cells, globs):
        m.eval()
        r = np.concatenate([m.run(make(idx[i:i + 1024], cells, globs)).float().cpu().numpy() for i in range(0, len(idx), 1024)])
        m.train()
        return r

    def transformer(cells, globs, train_idx, save=None):
        ph, pv = {e: [] for e in ss.CHECKPOINTS}, {e: [] for e in ss.CHECKPOINTS}
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            m = ss.GridRoute(cells, globs, pdim).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=ss.LR, weight_decay=0.05)
            n = len(train_idx)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, ss.LR, total_steps=ss.EPOCHS * max(n // ss.BS, 1), pct_start=0.05)
            for ep in range(1, ss.EPOCHS + 1):
                perm = train_idx[torch.randperm(n)]
                for k in range(max(n // ss.BS, 1)):
                    idx = perm[k * ss.BS:(k + 1) * ss.BS]
                    loss = score_loss(m.run(make(idx, cells, globs)).float(), data['floor'][idx].to(dev))
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    sched.step()
                if ep in ss.CHECKPOINTS:
                    ph[ep].append(predict(m, hold, cells, globs))
                    pv[ep].append(predict(m, val, cells, globs))
                    if save:
                        os.makedirs(save, exist_ok=True)
                        torch.save(m.state_dict(), f'{save}/seed{seed}_epoch{ep}.pt')
        hs = {e: scored(eps, data, hold, np.mean(ph[e], 0), steps) for e in ss.CHECKPOINTS}
        best = max(hs, key=hs.get)
        if save:
            json.dump(dict(epoch=best, held=hs[best], cells=cells, globs=globs, train=len(train_idx)), open(f'{save}/picked.json', 'w'))
            for f_ in os.listdir(save):
                if f_.endswith('.pt') and not f_.endswith(f'_epoch{best}.pt'):
                    os.remove(f'{save}/{f_}')
        return hs, best, scored(eps, data, val, np.mean(pv[best], 0), steps)

    def report(name, res, keep=None):
        hs, best, v = res
        print(f'{name:48s} held-out {hs[best]:.2f} (epoch {best})   report {v:.2f}' + ('' if keep is None else f'   {"KEEP" if keep else "drop"}'), flush=True)
        log.append(dict(name=name, held=hs[best], epoch=best, report=v, keep=keep))
        path = 'runs/tartan_steps_by_day.json' if BY_DAY else 'runs/tartan_steps.json'
        old = json.load(open(path)) if '--epochs' in sys.argv and os.path.exists(path) else []
        json.dump(old + log, open(path, 'w'), indent=1)
        return hs[best]

    if BY_DAY:                                           # the f5-like design first: the baseline to beat
        report('f5-like: Qwen + DA3 + parse, all robots + TartanDrive', transformer(['qwen', 'da3'], ['q_last', 'q_text'], every))
    if '--epochs' in sys.argv:                           # the kept design, all robots + TartanDrive, trained longer
        ss.EPOCHS = int(sys.argv[sys.argv.index('--epochs') + 1])
        ss.CHECKPOINTS = tuple(e for e in (8, 12, 16, 20, 24, 28, 32) if e <= ss.EPOCHS)
        report(f'qwen+dino+q_last+q_text, all robots + TartanDrive, {ss.EPOCHS} epochs',
               transformer(['qwen', 'dino'], ['q_last', 'q_text'], every))
        return
    cells, globs = ['qwen'], ['q_last', 'q_text']
    best = report('Qwen grid + summaries + parse + tilt, TartanDrive only', transformer(cells, globs, fit_t))
    for kind, item, label in (('cell', 'dino', '+ DINOv3 grid'), ('cell', 'sig', '+ SigLIP grid'),
                              ('glob', 'sentence', '+ sentence'), ('cell', 'da3', '+ DA3 depth cells')):
        c, gl = (cells + [item], globs) if kind == 'cell' else (cells, globs + [item])
        res = transformer(c, gl, fit_t)
        keep = max(res[0].values()) > best
        report(f'  {label}', res, keep)
        if keep:
            cells, globs, best = c, gl, max(res[0].values())
    h_all = report(f'best ({"+".join(cells + globs)}), all robots + TartanDrive', transformer(cells, globs, every))
    use = every if h_all > best else fit_t
    if not BY_DAY:
        report(f'final models saved ({"all robots" if h_all > best else "TartanDrive only"})',
               transformer(cells, globs, use, save='runs/qsteps_tartan'))


BY_DAY = '--by-day' in sys.argv

if __name__ == '__main__':
    main()
