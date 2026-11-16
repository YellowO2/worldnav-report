"""Training targets for the path-drawing model: each true route drawn into its first photo.

    python 2_features/path_labels.py train|val_seen|val_unseen [--gravity]
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, 'eval')
from floor_geometry import GROUPS, NATIVE_SIZE, group_of, load_cameras, project, with_gravity  # noqa: E402
from score import load  # noqa: E402

K = 16


def start_frame(position, yaw0):
    p = np.array(position, dtype=float)
    c, s = math.cos(yaw0), math.sin(yaw0)
    d = p - p[0]
    return np.stack([c * d[:, 0] + s * d[:, 1], -(-s * d[:, 0] + c * d[:, 1])], 1)      # (forward, right)


def resample(route, k=K):
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.repeat(route[-1:], k, 0), 0.0
    s = np.linspace(cum[-1] / k, cum[-1], k)
    return np.stack([np.interp(s, cum, route[:, 0]), np.interp(s, cum, route[:, 1])], 1), float(cum[-1])


def main(split, gravity=False, shift=False):
    eps = load(f'data/lcvn/{split}.jsonl')
    tag = '_shift' if shift else ''      # --shift: the lean of the shifted photo
    grav = torch.load(f'features/{split}_gravity{tag}.pt')['gravity'].numpy() if gravity else None
    cams = load_cameras()
    n = len(eps)
    uv, vis = np.zeros((n, K, 2), np.float32), np.zeros((n, K), bool)
    floor, length, groups = np.zeros((n, K, 2), np.float32), np.zeros(n, np.float32), []
    for i, e in enumerate(eps):
        g = group_of(e['trajectory'], split)
        cam, (W, H) = cams[g], NATIVE_SIZE[g]
        if grav is not None:
            cam = with_gravity(cam, grav[i])
        pts, length[i] = resample(start_frame(e['position'], e['yaw'][0]))
        floor[i] = pts
        for j, (f, r) in enumerate(pts):
            q = project(f, r, cam)
            if q is not None and f > 0.05:
                u, v = q[0] / W, q[1] / H
                if -1.0 <= u <= 2.0 and -1.0 <= v <= 2.0:
                    uv[i, j], vis[i, j] = (u, v), True
        groups.append(GROUPS.index(g))
    torch.save(dict(episodes=[e['episode'] for e in eps], uv=torch.from_numpy(uv), visible=torch.from_numpy(vis),
                    floor=torch.from_numpy(floor), length=torch.from_numpy(length), group=torch.tensor(groups)),
               f'features/{split}_paths{"_grav" if gravity else ""}{tag}.pt')
    print(f'{split}: {n} episodes; visible points {vis.mean():.0%}; groups',
          {GROUPS[k]: groups.count(k) for k in sorted(set(groups))})


if __name__ == '__main__':
    main(sys.argv[1], '--gravity' in sys.argv, '--shift' in sys.argv)
