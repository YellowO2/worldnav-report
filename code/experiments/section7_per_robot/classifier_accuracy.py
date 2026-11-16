import sys, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'eval')
import pathnet
from floor_geometry import GROUPS
from group_classifier import predict_groups
pathnet.GRAVITY, pathnet.FEATURES = True, 'v3'
for split in ('val_seen', 'val_unseen'):
    f = pathnet.load_split(split)
    p = predict_groups(f).cpu().numpy(); t = f['group'].numpy()
    print(f'{split}: accuracy {np.mean(p == t):.4f} on {len(t)} episodes')
    for k, g in enumerate(GROUPS):
        m = t == k
        if m.sum():
            wrong = {GROUPS[j]: int(((p == j) & m).sum()) for j in range(len(GROUPS)) if j != k and ((p == j) & m).sum()}
            print(f'   {g:14s} {m.sum():5d}  correct {np.mean(p[m] == k):.3f}  mistaken for {wrong}')
