"""Qwen features (qwen_features.py --v2, frozen) for later photos of each training route of one robot.

    python 2_features/later_feats.py recon [val_seen] [--blank]
"""
import json
import os
import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import qwen_features as qf  # noqa: E402
from floor_geometry import group_of  # noqa: E402

MIN_LEFT, CHUNK, BS = 5, 2048, 16


def starts(n_moves):
    """Later starting photos of a route with n_moves moves: at 1/4 and 1/2, at least MIN_LEFT moves left."""
    ks = []
    for fr in (0.25, 0.5):
        k = round(fr * n_moves)
        if k >= 1 and n_moves - k >= MIN_LEFT and k not in ks:
            ks.append(k)
    return ks


def rows_of(group, split='train'):
    """(route key, start k, photo item, instruction) for every route of the robot in the split."""
    eps = [json.loads(line) for line in open(f'data/lcvn/{split}.jsonl')]
    frames = json.load(open(f'data/lcvn/{split}.frames.json'))
    by = {}
    for e in eps:
        if group_of(e['trajectory'], split) == group:
            by.setdefault(e['episode'][:-2], {})[e['episode'][-1]] = e
    rows = []
    for key, d in by.items():
        e = d.get('2') or d.get('0') or next(iter(d.values()))      # the same instruction choice as qf.routes_of
        for k in starts(len(e['position']) - 1):
            rows.append((key, k, frames[e['trajectory']][k], e['instruction']))
    return rows


def name_of(group, split='train'):
    return f'later_{group}' + ('' if split == 'train' else f'_{split}')


def main(group, split='train', with_blank=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    qf.V2 = True
    rows = rows_of(group, split)
    base = name_of(group, split)
    print(f'{group} {split}: {len(rows)} later photos from {len({r[0] for r in rows})} routes', flush=True)
    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    model16 = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda').eval()
    model32 = None
    tar = open(f'data/lcvn/{split}.tar', 'rb')
    for tag, blank in (('qwen2', False),) + ((('qwen2_blank', True),) if with_blank else ()):
        cdir = f'features/{base}_{tag}_chunks'
        os.makedirs(cdir, exist_ok=True)
        redo = 0
        for c0 in range(0, len(rows), CHUNK):
            cpath = f'{cdir}/{c0:06d}.pt'
            if os.path.exists(cpath):
                continue
            C, L, T = [], [], []
            for i in range(c0, min(c0 + CHUNK, len(rows)), BS):
                batch = [(qf.load_photo(split, r[2], tar), '' if blank else r[3]) for r in rows[i:min(i + BS, c0 + CHUNK)]]
                cells, last, text_mean = qf.run(model16, proc, batch, torch.float16)
                if any(not torch.isfinite(c).all() for c in cells) or not torch.isfinite(last).all():
                    redo += 1
                    if model32 is None:
                        model32 = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float32).to('cuda').eval()
                    cells, last, text_mean = qf.run(model32, proc, batch, torch.float32)
                assert all(c.shape[0] == 64 for c in cells)
                C.append(torch.stack(cells).half().cpu())
                L.append(last.half().cpu())
                T.append(text_mean.half().cpu())
            torch.save(dict(cells=torch.cat(C), last=torch.cat(L), text_mean=torch.cat(T)), cpath)
            print(f'{tag}: {min(c0 + CHUNK, len(rows))}/{len(rows)}, batches redone in fp32: {redo}', flush=True)
    torch.save(dict(keys=[(r[0], r[1]) for r in rows]), f'features/{base}.pt')
    print('saved', flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    main(a[0], a[1] if len(a) > 1 else 'train', '--blank' in sys.argv)
