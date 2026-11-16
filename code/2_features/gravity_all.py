"""Each first photo's own lean: GeoCalib gravity direction, with our known focal length as a prior.

    python 2_features/gravity_all.py train|val_seen|val_unseen|test [--shift]
"""
import io
import json
import math
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, 'eval')
from anycalib_lens import undistort_kb  # noqa: E402
from floor_geometry import GROUPS, NATIVE_SIZE, group_of, load_cameras, start_photo  # noqa: E402

FLAT_W = 320


class Photos(torch.utils.data.Dataset):
    def __init__(self, split, items, groups):
        self.tar_path, self.tar, self.items, self.groups = f'data/lcvn/{split}.tar', None, items, groups
        self.cams = load_cameras()
        self.lens = json.load(open('config/fisheye_anycalib.json'))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item, g = self.items[i], GROUPS[self.groups[i]]
        if isinstance(item, str):
            img = Image.open(item)
        else:
            if self.tar is None:
                self.tar = open(self.tar_path, 'rb')
            self.tar.seek(item[0])
            img = Image.open(io.BytesIO(self.tar.read(item[1])))
        img = img.convert('RGB').resize((256, 256), Image.BICUBIC).resize(NATIVE_SIZE[g], Image.BICUBIC)
        cam = self.cams[g]
        if cam['lens'] == 'kb':
            img, f = undistort_kb(img, self.lens['stanford' if g.startswith('stanford') else g], FLAT_W)
        else:
            s = FLAT_W / img.size[0]
            img, f = img.resize((FLAT_W, round(img.size[1] * s)), Image.BICUBIC), cam['fx'] * s
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255), float(f)


@torch.no_grad()
def main(split, batch=32, shift=False):
    from geocalib import GeoCalib
    dev = 'cuda'
    geo = GeoCalib().to(dev)
    if split == 'test':
        eps = [json.loads(line) for line in open('data/test/test.jsonl')]
        from group_classifier import predict_groups
        groups = predict_groups(torch.load('features/test_v2.pt')).tolist()
        items, index = [e['image'] for e in eps], list(range(len(eps)))
    else:
        eps = [json.loads(line) for line in open(f'data/lcvn/{split}.jsonl')]
        frames = json.load(open(f'data/lcvn/{split}.frames.json'))
        rows, items, groups = {}, [], []
        for e in eps:
            if e['trajectory'] not in rows:
                rows[e['trajectory']] = len(items)
                items.append(start_photo(frames[e['trajectory']], group_of(e['trajectory'], split), shift))
                groups.append(GROUPS.index(group_of(e['trajectory'], split)))
        index = [rows[e['trajectory']] for e in eps]
    photos = Photos(split, items, groups)
    n = len(items)
    grav, rp = np.zeros((n, 3), np.float32), np.zeros((n, 2), np.float32)
    t0 = time.time()
    for gi in sorted(set(groups)):                     # one camera group at a time: same image size within a batch
        idx = [i for i in range(n) if groups[i] == gi]
        sub = torch.utils.data.Subset(photos, idx)
        done = 0
        for x, f in torch.utils.data.DataLoader(sub, batch_size=batch, num_workers=8):
            res = geo.calibrate(x.to(dev), priors={'focal': f.float().to(dev)})
            g = res['gravity'].vec3d.reshape(len(x), 3).cpu().numpy()
            g *= np.where(g[:, 1:2] < 0, -1.0, 1.0)         # pointing down the image (y down)
            r = np.degrees(res['gravity'].rp.reshape(len(x), 2).cpu().numpy())
            rows_ = idx[done:done + len(x)]
            grav[rows_], rp[rows_] = g, r
            done += len(x)
        print(f'{GROUPS[gi]}: {len(idx)} photos, pitch median {np.median(rp[idx, 1]):+.1f} '
              f'(middle half {np.percentile(rp[idx, 1], 25):+.1f} to {np.percentile(rp[idx, 1], 75):+.1f}), '
              f'roll middle half {np.percentile(rp[idx, 0], 25):+.1f} to {np.percentile(rp[idx, 0], 75):+.1f}; '
              f'{time.time() - t0:.0f} s', flush=True)
    index = torch.tensor(index)
    torch.save(dict(episodes=[e['episode'] for e in eps], gravity=torch.from_numpy(grav)[index],
                    roll=torch.from_numpy(rp[:, 0])[index], pitch=torch.from_numpy(rp[:, 1])[index],
                    group=torch.tensor(groups)[index]), f'features/{split}_gravity{"_shift" if shift else ""}.pt')


if __name__ == '__main__':
    main(sys.argv[1], shift='--shift' in sys.argv)
