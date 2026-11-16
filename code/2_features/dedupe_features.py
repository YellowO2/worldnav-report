"""Store the image features once.

    python 2_features/dedupe_features.py val_seen val_unseen test train
"""
import os
import sys

import torch

IMAGE = ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls')


def main(split):
    img_path = f'features/{split}_img.pt'
    for v in ('v2', 'v3'):
        path = f'features/{split}_{v}.pt'
        if not os.path.exists(path):
            continue
        f = torch.load(path)
        img = {k: f[k] for k in IMAGE}
        if os.path.exists(img_path):
            old = torch.load(img_path)
            assert all(torch.equal(old[k], img[k]) for k in IMAGE), f'{path}: image part differs from {img_path}'
        else:
            torch.save(img, img_path)
        text = {k: x for k, x in f.items() if k not in IMAGE}
        torch.save(text, f'features/{split}_{v}_text.pt')
        back = torch.load(f'features/{split}_{v}_text.pt')
        back.update(torch.load(img_path))
        assert back.keys() == f.keys()
        for k in f:
            a, b = f[k], back[k]
            assert (torch.equal(a, b) if torch.is_tensor(a) else a == b), f'{path}: {k} differs'
        os.remove(path)
        print(f'{path} -> {img_path} + {split}_{v}_text.pt (checked)', flush=True)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        main(s)
