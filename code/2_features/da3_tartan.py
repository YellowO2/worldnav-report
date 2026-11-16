"""Single-photo depth for TartanDrive.

    python 2_features/da3_tartan.py starts
"""
import io
import json
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, 'eval')
from floor_geometry import load_cameras  # noqa: E402

MODEL = 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'
W, H = 1024, 544
FAR = 40.0


def cells(depth):
    """Depth map (h, w) metres -> (64, 2): per 8 x 8 cell, median log depth and share beyond FAR."""
    d = torch.as_tensor(depth, dtype=torch.float32)
    h, w = d.shape
    out = []
    for r in range(8):
        for c in range(8):
            blk = d[r * h // 8:(r + 1) * h // 8, c * w // 8:(c + 1) * w // 8].reshape(-1)
            out.append((blk.clamp(0.1, FAR).log().median(), (blk > FAR).float().mean()))
    return torch.tensor(out)


@torch.no_grad()
def run(items, from_files=False):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(MODEL).to('cuda')
    cam = load_cameras()['tartan']
    tar = open('data/lcvn/val_unseen.tar', 'rb')
    out = []
    for k, item in enumerate(items):
        if from_files:
            img = Image.open(item)
        else:
            tar.seek(item[0])
            img = Image.open(io.BytesIO(tar.read(item[1])))
        img = img.convert('RGB').resize((256, 256), Image.BICUBIC).resize((W, H), Image.BICUBIC)
        K = np.array([[cam['fx'], 0, cam['cx']], [0, cam['fy'], cam['cy']], [0, 0, 1.0]])
        pred = model.inference([img], intrinsics=K[None], process_res=504)
        out.append(cells(np.asarray(pred.depth[0])))
        if k % 500 == 0:
            print(f'{k}/{len(items)} photos; this one: depth at the bottom centre {float(np.asarray(pred.depth[0])[-5, pred.depth[0].shape[1] // 2]):.1f} m',
                  flush=True)
    return torch.stack(out).half()


def main(which):
    if which == 'test':
        import pathnet
        from floor_geometry import GROUPS
        from group_classifier import predict_groups
        pathnet.FEATURES = 'v3'
        test = pathnet.load_split('test', with_labels=False)
        groups = predict_groups(test).cpu()
        eps = [json.loads(l) for l in open('data/test/test.jsonl')]
        is_t = groups == GROUPS.index('tartan')
        import hashlib
        key = {e['image']: hashlib.md5(open(e['image'], 'rb').read()).hexdigest()
               for e, t in zip(eps, is_t.tolist()) if t}          # each photo appears 3 times (once per instruction)
        first = {}
        for f_, h in key.items():
            first.setdefault(h, f_)
        print(f'{int(is_t.sum())} of {len(eps)} test episodes look like TartanDrive; {len(first)} distinct photos', flush=True)
        d = dict(zip(first, run(list(first.values()), from_files=True)))
        da3 = torch.zeros(len(eps), 64, 2, dtype=torch.float16)
        for k, (e, t) in enumerate(zip(eps, is_t.tolist())):
            if t:
                da3[k] = d[key[e['image']]]
        torch.save(dict(episodes=[e['episode'] for e in eps], da3=da3, has_da3=is_t.float()), 'features/test_da3.pt')
        print('saved test')
        return
    if which == 'starts':
        frames = json.load(open('data/lcvn/val_unseen.frames.json'))
        eps = [json.loads(l) for l in open('data/lcvn/val_unseen.jsonl')]
        order, seen = [], set()
        for e in eps:                                   # one photo per trajectory, in the order of val_unseen_img.pt
            if e['trajectory'] not in seen:
                seen.add(e['trajectory'])
                order.append(e['trajectory'])
        torch.save(dict(trajectories=order, da3=run([frames[t][0] for t in order])), 'features/val_unseen_da3.pt')
    else:
        tag = 'tartan_held_views' if which == 'held_views' else 'tartan_views'
        ex = torch.load(f'features/{tag}.pt')
        frames = json.load(open('data/lcvn/val_unseen.frames.json'))
        eps = {json.loads(l)['episode'][:-2]: json.loads(l) for l in open('data/lcvn/val_unseen.jsonl')}
        items = [frames[eps[r]['trajectory']][k] for r, k in zip(ex['route'], ex['frame'].tolist())]
        torch.save(dict(da3=run(items)), f'features/{tag}_da3.pt')
    print('saved', which)


if __name__ == '__main__':
    main(sys.argv[1])
