"""Unit check with a drawn path: dots along the floor from the robot to where it should stop.

    python common/manual_path.py
"""
import io
import json
import math
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import source  # noqa: E402
from manual_metric import GRID, IMAGENET_MEAN, IMAGENET_STD, load_nested  # noqa: E402

# case -> dots (column, row) on the 20x20 grid (cell units, 0 = left/top edge)
PATHS = {
    2: [(10, 18.5), (9.5, 16), (9, 14), (8.5, 12.5), (8, 11.5), (7.5, 10.5)],
    4: [(10, 18.5), (10.2, 17), (10.3, 15.5), (10.5, 14.5)],
    8: [(10, 18.5), (10.5, 16), (11, 14), (11.7, 12.5), (12.5, 11.5)],
    9: [(10, 18.5), (9.8, 16), (9.7, 14), (9.5, 12.5), (9.5, 11.5)],
    10: [(10, 18.5), (9.8, 16), (9.7, 13.5), (9.6, 11.5), (9.5, 10.5)],
    12: [(10, 18), (10.5, 16), (11, 14), (11.5, 12.5)],
    14: [(10, 18), (8.5, 16), (6.5, 14), (4.5, 12.5)],
    20: [(10, 18), (10.2, 15.5), (10.5, 12.5)],
    21: [(10, 18), (9, 16), (7.5, 14), (6.5, 12.5)],
    22: [(10, 18), (9.8, 16), (9.6, 14), (9.5, 12.5)],
    23: [(10, 18), (11, 16), (12, 14), (13, 12.3), (13.5, 11.5)],
    24: [(10, 18.5), (10.8, 16), (11.5, 14), (12, 12.5), (12.5, 11.5)],
    25: [(10, 18.5), (10.8, 16), (11.5, 14), (12, 12.5), (12.5, 11.5)],
    27: [(10, 18.5), (9, 16), (8.2, 14), (7.5, 12.5)],
    29: [(10, 18.5), (9.8, 16), (9.6, 13.5), (9.5, 11.5)],
    30: [(10, 18.5), (11.5, 17), (12.5, 15.5), (13.5, 14.5)],
    31: [(10, 18.5), (9.5, 16), (9, 14), (8.5, 12.5)],
    34: [(10, 18.5), (11.5, 16.5), (13, 14.5), (14.5, 12.5)],
    36: [(10, 18.5), (12, 16), (14, 13.5), (15.5, 11.5)],
    37: [(10, 18.5), (10.2, 15.5), (10.4, 12.5), (10.5, 10.5)],
    38: [(10, 18.5), (10.2, 15), (10.4, 12), (10.5, 9.5)],
    42: [(10, 18.5), (9.5, 15.5), (9, 12.5), (8.5, 10.5)],
    43: [(10, 18.5), (10.2, 16), (10.4, 13.5), (10.5, 11.5)],
    44: [(10, 18.5), (8, 16), (6, 13.5), (3.5, 11.5)],
    45: [(10, 18.5), (10.2, 15.5), (10.4, 12.5), (10.5, 10.5)],
}


# second batch: config/cases2.json (half with turns)
PATHS2 = {
    0: [(10, 18.5), (9.8, 16), (9.6, 13.5), (9.4, 11.5), (9.3, 10.5)],
    1: [(10, 18.5), (10, 16), (9.8, 13), (9.6, 10.5), (9.5, 9)],
    2: [(10, 18.5), (9.8, 16), (9.6, 13.5), (9.4, 11.5), (9.3, 10.5)],
    3: [(10, 18.5), (10, 16), (10, 14), (10.5, 12.5), (12, 12)],
    4: [(10, 18.5), (10, 16.5), (10, 15), (11.5, 14.8), (13, 14.7)],
    5: [(10, 18.5), (10, 16), (10, 13), (9.5, 11), (7, 10.5), (5, 10.2)],
    6: [(9, 18.5), (9, 16), (9.5, 14), (10.5, 12.5), (12, 11.5), (13.5, 11)],
    7: [(10, 18), (10.2, 16), (10.5, 14), (10.8, 12.5), (11, 11.5)],
    8: [(10, 18), (10, 16), (10, 14), (10, 12.5)],
    9: [(10, 18), (11, 16), (12.5, 14), (14, 12.5), (15.5, 11.3)],
    10: [(10, 18), (10, 15.5), (9.5, 13), (8, 11.5), (5.5, 10.8), (4, 10.5)],
    11: [(10, 18), (10.5, 15.5), (11, 13), (11.5, 11.5), (10, 11), (8, 10.8)],
    12: [(10, 18), (10.5, 15.5), (11, 13.5), (11.5, 12), (11.5, 11)],
    13: [(10, 18), (9, 15.5), (8, 13.5), (7, 11.5), (6.5, 10.8)],
    14: [(10, 18.5), (10, 16), (10.2, 14), (10.5, 12.5), (10.5, 11.5)],
    15: [(10, 18.5), (11, 16), (12, 14), (13, 12.5), (13.5, 12)],
    16: [(10, 18.5), (9.5, 16.5), (9, 15), (8.5, 13.8), (8, 13)],
    17: [(10, 18.5), (9, 17), (7, 16), (4, 15.5)],
    18: [(10, 18.5), (10.5, 16), (11, 14), (12, 12.8), (13.5, 12.3)],
    19: [(10, 18.5), (11, 16.5), (12.5, 15), (14, 13.5), (15, 12.5)],
    20: [(10, 18.5), (10.5, 16), (11, 14), (11, 12.5), (9.5, 12.7)],
    21: [(10, 18.5), (9.8, 16), (9.5, 13.5), (9.3, 11.5)],
    22: [(10, 18.5), (9.3, 16), (8.5, 13.5), (8, 11.5), (7.5, 10.5)],
    23: [(10, 18.5), (10.3, 16), (10.7, 13.5), (11, 11.5), (11.3, 10.5)],
    24: [(10, 18.5), (10.2, 16), (10.4, 13.5), (10.5, 11), (9.5, 10), (8.5, 9.5)],
    25: [(9, 18.5), (8.5, 16), (8.3, 14), (8.8, 12), (9.6, 11)],
    26: [(10, 18.5), (9, 16), (7.5, 13.5), (6, 11.5), (4.5, 10)],
    27: [(10, 18.5), (9.5, 16.5), (9, 15), (7, 13.5), (5, 12.5)],
}
LOW = {3, 4, 8, 17, 20, 25}


def true_route(c):
    p, y0 = c['position'], c['yaw'][0]
    y0 = y0[0] if isinstance(y0, list) else y0
    cs, sn = math.cos(y0), math.sin(y0)
    return np.array([(cs * (x - p[0][0]) + sn * (y - p[0][1]), -(-sn * (x - p[0][0]) + cs * (y - p[0][1])))
                     for x, y in p])                                   # (forward, right)


def floor_plane(depth, K, rng, iters=300):
    """Plane n.P + d = 0."""
    H, W = depth.shape
    v, u = np.mgrid[int(H * 0.55):H:2, 0:W:2]
    z = depth[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0)]
    tol = 0.03 * np.median(pts[:, 2])
    best, best_count = None, -1
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.7:                                          # floor: normal roughly along the image's up/down
            continue
        count = (np.abs(pts @ n - n @ a) < tol).sum()
        if count > best_count:
            best, best_count = (n, -n @ a), count
    n, d = best
    inl = pts[np.abs(pts @ n + d) < tol]
    centre = inl.mean(0)                                              # least-squares refit on the inliers
    n = np.linalg.svd(inl - centre)[2][-1]
    n = n if n[1] < 0 else -n                                         # points from the floor up towards the camera
    return n, -n @ centre, len(inl) / len(pts)


def snap(u, v, K, n, d):
    """Where the viewing ray through pixel (u, v) meets the floor, as."""
    ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    t = -d / (n @ ray)
    if t <= 0:
        return None                                                   # at or above the horizon: never meets the floor
    foot = -d * n                                                     # floor point below the camera
    fwd = np.array([0, 0, 1.0]) - n[2] * n
    right = np.array([1.0, 0, 0]) - n[0] * n
    g = t * ray - foot
    return float(g @ fwd / np.linalg.norm(fwd)), float(g @ right / np.linalg.norm(right))


def length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


@torch.no_grad()
def main():
    dev = 'cuda'
    batch2 = '--batch2' in sys.argv
    cases = json.load(open('config/cases2.json' if batch2 else 'config/cases.json'))
    paths = PATHS2 if batch2 else PATHS
    tag = '2' if batch2 else ''
    if not batch2:                                                   # cases.json has no yaw
        want = {c['episode']: c for c in cases}
        for line in open('data/lcvn/train.jsonl'):
            e = json.loads(line)
            if e['episode'] in want:
                want[e['episode']]['yaw'] = e['yaw']
    frames = json.load(open('data/lcvn/train.frames.json'))
    tar = open('data/lcvn/train.tar', 'rb')
    nested = load_nested().to(dev)
    rng = np.random.default_rng(0)
    rows, drawn = [], []
    for k, dots in paths.items():
        c = cases[k]
        o, s = frames[c['trajectory']][0]
        tar.seek(o)
        photo = Image.open(io.BytesIO(tar.read(s))).convert('RGB')
        w, h = photo.size
        side = 504 / max(w, h)
        W, H = round(w * side / 14) * 14, round(h * side / 14) * 14
        x = torch.from_numpy(np.array(photo.resize((W, H), Image.BICUBIC))).permute(2, 0, 1).float() / 255
        with torch.autocast('cuda', dtype=torch.float16):
            out = nested(((x - IMAGENET_MEAN) / IMAGENET_STD)[None, None].to(dev))
        depth = out.depth[0, 0].float().cpu().numpy()                 # (H, W) metres
        K = out.intrinsics[0, 0].float().cpu().numpy()                # at the network's input size
        n, dist, inliers = floor_plane(depth, K, rng)
        lifted = [(0.0, 0.0)]                                          # the point below the robot's camera
        for col, row in dots:
            g = snap(col / GRID * W, row / GRID * H, K, n, dist)
            if g is not None:
                lifted.append(g)
        lifted = np.array(lifted)
        gt = true_route(c)
        r = dict(case=k, source=source(c), phrase=c.get('phrase', ''), low_confidence=batch2 and k in LOW, gt_end=float(np.linalg.norm(gt[-1])),
                 gt_len=length(gt), lift_end=float(np.linalg.norm(lifted[-1])), lift_len=length(lifted),
                 focal=float(K[0, 0]) / side, camera_height=float(dist), floor_inliers=float(inliers),
                 tilt_deg=float(np.degrees(np.arcsin(abs(n[2])))))
        r['ratio_end'], r['ratio_len'] = r['gt_end'] / r['lift_end'], r['gt_len'] / r['lift_len']
        r['lifted'] = lifted.tolist()                                   # (forward, right) metres, robot first
        rows.append(r)
        drawn.append((k, c, photo, dots, lifted, gt, r))
        print(f'#{k:2d} {r["source"]:8s} {r["phrase"][:26]:26s} true end {r["gt_end"]:5.2f} len {r["gt_len"]:5.2f} | '
              f'drawn end {r["lift_end"]:5.2f}m len {r["lift_len"]:5.2f}m | ratio end {r["ratio_end"]:4.2f} len {r["ratio_len"]:4.2f} | '
              f'camera {r["camera_height"]:4.2f}m up, tilt {r["tilt_deg"]:4.1f} deg, floor fit {r["floor_inliers"]:.0%}',
              flush=True)
    json.dump(rows, open(f'config/path{tag}.json', 'w'), indent=1)
    print()
    for src in ('scand', 'recon', 'stanford', 'huron', None):
        R = [r for r in rows if src is None or r['source'] == src]
        e, l_ = np.array([r['ratio_end'] for r in R]), np.array([r['ratio_len'] for r in R])
        corr = np.corrcoef([r['lift_end'] for r in R], [r['gt_end'] for r in R])[0, 1]
        print(f'{src or "all":8s} n={len(R):2d} true/drawn end median {np.median(e):.2f} '
              f'(IQR {np.percentile(e, 25):.2f}-{np.percentile(e, 75):.2f}), length median {np.median(l_):.2f}, corr {corr:.2f}, '
              f'camera height median {np.median([r["camera_height"] for r in R]):.2f}m '
              f'(range {min(r["camera_height"] for r in R):.2f}-{max(r["camera_height"] for r in R):.2f})')

    for page in range(0, len(drawn), 9):
        chunk = drawn[page:page + 9]
        fig, ax = plt.subplots(len(chunk), 2, figsize=(13, 4.6 * len(chunk)), squeeze=False)
        for i, (k, c, photo, dots, lifted, gt, r) in enumerate(chunk):
            w, h = photo.size
            ax[i, 0].imshow(photo)
            us, vs = [d[0] / GRID * w for d in dots], [d[1] / GRID * h for d in dots]
            ax[i, 0].plot(us, vs, 'o-', color='red', ms=5, lw=2)
            ax[i, 0].axis('off')
            ax[i, 0].set_title(f'#{k} {r["source"]} {w}x{h}{" (low confidence)" if r["low_confidence"] else ""}\n' + '\n'.join(textwrap.wrap(c['instruction'], 70)) +
                               f'\nred: path drawn by eye; camera {r["camera_height"]:.2f} m above the fitted floor', fontsize=8, loc='left')
            ax[i, 1].plot(gt[:, 1], gt[:, 0], 'o-', ms=2, label=f'true route (units): end {r["gt_end"]:.2f}')
            ax[i, 1].plot(lifted[:, 1], lifted[:, 0], 's-', color='red', ms=3,
                          label=f'drawn path lifted (m): end {r["lift_end"]:.2f}')
            ax[i, 1].plot(0, 0, 'g^', ms=9)
            ax[i, 1].set(aspect='equal', xlabel='right', ylabel='forward',
                         title=f'true / drawn: end {r["ratio_end"]:.2f}, length {r["ratio_len"]:.2f}')
            ax[i, 1].grid(alpha=.3)
            ax[i, 1].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(f'runs/path{tag}_{page // 9 + 1}.png' if tag else f'runs/path{page // 9 + 1}.png', dpi=70)
        plt.close(fig)
    print('saved runs/path*.png')


if __name__ == '__main__':
    main()
