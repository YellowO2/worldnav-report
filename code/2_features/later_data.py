"""Later-start training examples for the final models."""
import os

import numpy as np
import torch

from floor_geometry import GROUPS
from later_feats import name_of, starts
from path_labels import resample, start_frame
from score import load

Q_TABLES = {'q_cells': 'cells', 'q_last': 'last', 'q_text': 'text_mean'}
IMG_TABLES = ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls')


def drop_images(f):
    for k in IMG_TABLES:
        f[k] = torch.zeros((1,) + tuple(f[k].shape[1:]), dtype=f[k].dtype)
    f['image_index'] = torch.zeros_like(f['image_index'])


def add_later(tr, splits, robots):
    n = len(tr['episodes'])
    per_episode = [k for k, v in tr.items() if torch.is_tensor(v) and len(v) == n and k not in Q_TABLES and k not in IMG_TABLES]
    src, qrow, target, files = [], [], [], []
    base, off, row0 = len(tr['q_last']), 0, 0
    for split in splits:
        eps = load(f'data/lcvn/{split}.jsonl')
        assert tr['episodes'][off:off + len(eps)] == [e['episode'] for e in eps]
        for g in robots:
            name = name_of(g, split)
            keys = torch.load(f'features/{name}.pt')['keys']
            key_row = {tuple(k): j for j, k in enumerate(keys)}
            gi = GROUPS.index(g)
            for i, e in enumerate(eps):
                if int(tr['group'][off + i]) != gi or not torch.isfinite(tr['floor'][off + i]).all():
                    continue
                for k in starts(len(e['position']) - 1):
                    pts, _ = resample(start_frame(e['position'][k:], e['yaw'][k]))
                    if np.isfinite(pts).all():
                        src.append(off + i)
                        qrow.append(base + row0 + key_row[(e['episode'][:-2], k)])
                        target.append(pts)
            files.append((f'features/{name}_qwen2_chunks', len(keys)))
            row0 += len(keys)
        off += len(eps)
    src = torch.tensor(src)
    for k in per_episode:
        tr[k] = torch.cat([tr[k], tr[k][src]])
    tr['floor'][n:] = torch.tensor(np.stack(target), dtype=tr['floor'].dtype)
    tr['q_index'][n:] = torch.tensor(qrow)
    tr['episodes'] = tr['episodes'] + [tr['episodes'][i] for i in src.tolist()]
    for k, kk in Q_TABLES.items():                       # one allocation per table, filled chunk by chunk
        new = torch.empty((base + row0,) + tuple(tr[k].shape[1:]), dtype=tr[k].dtype)
        new[:base] = tr[k]
        tr[k] = None
        at = base
        for cdir, rows in files:
            got = 0
            for c in sorted(os.listdir(cdir)):
                d = torch.load(f'{cdir}/{c}')[kk]
                new[at:at + len(d)] = d
                at, got = at + len(d), got + len(d)
            assert got == rows, (cdir, got, rows)
        tr[k] = new
    print(f'later starts: {len(src)} extra examples from {row0} later photos ({", ".join(robots)}; {", ".join(splits)}) '
          f'on top of {n}', flush=True)
    return n
