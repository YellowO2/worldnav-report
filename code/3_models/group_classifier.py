"""Which camera took a photo? A linear classifier on the frozen image embeddings.

    python 3_models/group_classifier.py
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
from floor_geometry import GROUPS  # noqa: E402

PATH = 'runs/group_classifier.pt'


def image_features(f):
    x = torch.cat([f['sig_pooled'].float(), f['dino_cls'].float()], 1)
    return x[f['image_index']]                       # per episode


def main():
    dev = 'cuda'
    xs, ys = [], []
    for split in ('train', 'val_unseen'):
        f, lab = torch.load(f'features/{split}_v2.pt'), torch.load(f'features/{split}_paths.pt')
        xs.append(image_features(f))
        ys.append(lab['group'])
    x, y = torch.cat(xs).to(dev), torch.cat(ys).to(dev)
    mean, std = x.mean(0), x.std(0) + 1e-4
    clf = nn.Linear(x.shape[1], len(GROUPS)).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-3)
    counts = torch.bincount(y, minlength=len(GROUPS)).float()
    weight = (counts.sum() / counts.clamp_min(1)).to(dev)           # rare groups (SCAND Jackal) count as much
    for step in range(600):
        idx = torch.randint(0, len(x), (4096,), device=dev)
        loss = nn.functional.cross_entropy(clf((x[idx] - mean) / std), y[idx], weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(dict(state=clf.state_dict(), mean=mean.cpu(), std=std.cpu()), PATH)
    f, lab = torch.load('features/val_seen_v2.pt'), torch.load('features/val_seen_paths.pt')
    pred = predict_groups(f)
    acc = (pred == lab['group']).float()
    print('val_seen accuracy %.3f;' % acc.mean(), {GROUPS[g]: round(float(acc[lab['group'] == g].mean()), 3)
                                                  for g in lab['group'].unique().tolist()})


def predict_groups(f):
    ck = torch.load(PATH)
    clf = nn.Linear(ck['mean'].shape[0], len(GROUPS))
    clf.load_state_dict(ck['state'])
    with torch.no_grad():
        return clf((image_features(f) - ck['mean']) / ck['std']).argmax(1)


if __name__ == '__main__':
    main()
