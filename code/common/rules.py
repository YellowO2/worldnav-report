"""Rules that turn a hand-drawn path (dots on the first photo) into the route the dataset expects.

    python common/rules.py
"""
import collections
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import source  # noqa: E402
from manual_path import GRID, LOW, PATHS, PATHS2, true_route  # noqa: E402
from manual_path_score import moves_along, n_med, step  # noqa: E402
from score import episode_metrics  # noqa: E402
from floor_geometry import group, lift_path, load_cameras  # noqa: E402

SIZE = {'scand': (320, 240), 'recon': (640, 480), 'huron': (160, 120), 'stanford': (160, 120)}
START = {  # lens, focal length (px), camera height (m), tilt down (deg)
    'scand': ('pinhole', 200.0, 1.10, 0.0),
    'recon': ('fisheye', 216.0, 0.95, 0.0),
    'huron': ('fisheye', 57.0, 0.50, 0.0),
    'stanford': ('fisheye', 57.0, 0.82, 0.0),
}


def cut(points, max_len):
    """The polyline up to max_len along it."""
    pts = np.array(points)
    out, run = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        seg = float(np.linalg.norm(b - a))
        if run + seg >= max_len:
            out.append(a + (b - a) * (max_len - run) / max(seg, 1e-9))
            break
        out.append(b)
        run += seg
    return np.array(out)


def lift(dots, src, lens, f, h, tilt_deg, cx=None, cy=None):
    """Grid dots -> (forward, right) floor points in metres, robot first."""
    W, H = SIZE[src]
    cx = W / 2 if cx is None else cx
    cy = H / 2 if cy is None else cy
    t = math.radians(tilt_deg)
    out = [(0.0, 0.0)]
    for col, row in dots:
        x, y = (col / GRID * W - cx) / f, (row / GRID * H - cy) / f
        if lens == 'fisheye':
            r = math.hypot(x, y)
            s = math.sin(r) / r if r > 1e-9 else 1.0
            ray = (x * s, y * s, math.cos(r))
        else:
            ray = (x, y, 1.0)
        down = ray[1] * math.cos(t) + ray[2] * math.sin(t)
        fwd = -ray[1] * math.sin(t) + ray[2] * math.cos(t)
        if down <= 1e-6:
            continue
        k = h / down
        out.append((k * fwd, k * ray[0]))
    return np.array(out)


def true_len(x):
    p = np.array(x['c']['position'])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def exact(points, length):
    """Cut at length, or extend straight along the last segment until the path is that long."""
    pts = cut(points, length)
    have = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if have < length - 1e-6 and len(pts) >= 2:
        d = pts[-1] - pts[-2]
        d = d / max(float(np.linalg.norm(d)), 1e-9)
        pts = np.vstack([pts, pts[-1] + d * (length - have)])
    return pts


def scaled(points, length):
    """The whole path shrunk or stretched about the robot so its length matches."""
    pts = np.array(points)
    have = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return pts * (length / max(have, 1e-9))


def load_cases():
    cases = []
    lifted_da3 = {}
    for tag, paths, name in (('', PATHS, 'cases'), ('2', PATHS2, 'cases2')):
        cs = json.load(open(f'config/{name}.json'))
        for r in json.load(open(f'config/path{tag}.json')):
            lifted_da3[(tag, r['case'])] = np.array(r['lifted'])
        for k, dots in paths.items():
            cases.append(dict(tag=tag, k=k, c=cs[k], dots=dots, low=bool(tag) and k in LOW))
    want = {x['c']['episode'] for x in cases}
    yaw = {}
    for line in open('data/lcvn/train.jsonl'):
        e = json.loads(line)
        if e['episode'] in want:
            yaw[e['episode']] = np.asarray(e['yaw'], dtype=float).reshape(-1).tolist()
    for x in cases:
        x['c']['yaw'] = yaw[x['c']['episode']]
        x['src'] = source(x['c'])
        x['da3'] = lifted_da3[(x['tag'], x['k'])]
    return cases


def score_route(x, pts):
    e = x['c']
    return episode_metrics(moves_along(pts, step[x['src']]), e['position'], e['yaw'])['score']


REACHED = [  # (batch, case, floor mark at the landmark on the 20 x 20 grid): the robot visibly stopped there
    ('', 14, (4, 12)), ('', 20, (10, 12)), ('', 21, (6, 12)), ('', 22, (9, 12)), ('', 30, (13, 14)), ('', 34, (14, 12)),
    ('', 36, (15, 11)), ('', 38, (10, 9)), ('', 42, (8, 10)), ('', 43, (10, 11)), ('', 44, (3, 11))]


def main():
    cases = load_cases()
    typical = {s: step[s] * n_med[s] for s in step}
    by_src = collections.defaultdict(list)
    for x in cases:
        by_src[x['src']].append(x)
    fit, test = [], []
    for s, xs in by_src.items():                        # alternate within each robot, so both halves cover all four
        fit += xs[0::2]
        test += xs[1::2]

    fitted = {}
    for s, (lens, f0, h0, t0) in START.items():
        xs = [x for x in fit if x['src'] == s]
        best = None
        for fm, h, t in itertools.product((0.6, 0.8, 1.0, 1.25, 1.6), (0.3, 0.5, 0.7, 0.95, 1.2, 1.5), (-5, 0, 5, 10, 15)):
            sc = np.mean([score_route(x, cut(lift(x['dots'], s, lens, f0 * fm, h, t), typical[s])) for x in xs])
            if best is None or sc > best[0]:
                best = (sc, f0 * fm, h, t)
        fitted[s] = (lens,) + best[1:]
        print(f'{s:9s} fitted on {len(xs)} cases: focal {best[1]:.0f}px, height {best[2]:.2f}m, tilt {best[3]} deg '
              f'(fit-half score {best[0]:.1f})')

    cams = json.load(open('config/cameras.json'))                # eval/calib_video.py: per-robot camera from video
    video = {s: ('pinhole', c['focal'], c['height'], -c['tilt_deg']) for s, c in cams.items()}
    # fixed constants: measured lens
    cu, fl = json.load(open('config/cameras_undistorted.json')), json.load(open('config/fisheye.json'))
    fixed = {s: (('fisheye', fl[s]['f_fisheye'], c['height'], -c['tilt_deg'], fl[s]['cx'], fl[s]['cy']) if s in fl
                 else ('pinhole', c['focal'], c['height'], -c['tilt_deg'])) for s, c in cu.items()}
    geo = load_cameras()

    def floor(x, max_jump=3.0):   # fixed per-camera floor geometry, jumps cut back
        W, H = SIZE[x['src']]
        return lift_path([(c / GRID * W, r / GRID * H) for c, r in x['dots']], geo[group(x['c']['trajectory'])],
                         max_jump=max_jump)

    variants = {
        'straight': lambda x: np.array([(i * step[x['src']], 0.0) for i in range(n_med[x['src']] + 1)]),
        'DA3 lift': lambda x: x['da3'],
        'camera lift': lambda x: lift(x['dots'], x['src'], *START[x['src']]),
        'camera lift + cap': lambda x: cut(lift(x['dots'], x['src'], *START[x['src']]), typical[x['src']]),
        'DA3 lift + cap': lambda x: cut(x['da3'], typical[x['src']]),
        'fitted camera + cap': lambda x: cut(lift(x['dots'], x['src'], *fitted[x['src']]), typical[x['src']]),
        'video camera': lambda x: lift(x['dots'], x['src'], *video[x['src']]),
        'video camera + cap': lambda x: cut(lift(x['dots'], x['src'], *video[x['src']]), typical[x['src']]),
        'floor geometry': floor,
        'straight, true length': lambda x: np.array([(0.0, 0.0), (true_len(x), 0.0)]),
        'floor, cut at true length': lambda x: cut(floor(x, 1e9), true_len(x)),
        'floor, exactly true length': lambda x: exact(floor(x, 1e9), true_len(x)),
        'floor, scaled to true length': lambda x: scaled(floor(x, 1e9), true_len(x)),
        'floor, trim 2x': lambda x: floor(x, 2.0),
        'floor, trim 1.5x': lambda x: floor(x, 1.5),
        'floor geometry + cap': lambda x: cut(floor(x), typical[x['src']]),
        'fixed constants': lambda x: lift(x['dots'], x['src'], *fixed[x['src']]),
        'fixed constants + cap': lambda x: cut(lift(x['dots'], x['src'], *fixed[x['src']]), typical[x['src']]),
    }
    print()
    print(f'{"variant":22s}{"fit half":>10s}{"test half":>11s}{"test, confident drawings":>27s}')
    rows = {}
    for name, make in variants.items():
        f_sc = np.mean([score_route(x, make(x)) for x in fit])
        t_all = [score_route(x, make(x)) for x in test]
        t_conf = [s for s, x in zip(t_all, test) if not x['low']]
        rows[name] = t_all
        print(f'{name:22s}{f_sc:10.1f}{np.mean(t_all):11.1f}{np.mean(t_conf):27.1f}')
    print('\nreached-landmark cases: true end distance / distance of the marked floor spot with the fixed constants')
    for tag, k, (col, row) in REACHED:
        x = next(c for c in cases if c['tag'] == tag and c['k'] == k)
        d = lift([(col + 0.5, row + 0.5)], x['src'], *fixed[x['src']])
        dist = float(np.linalg.norm(d[-1])) if len(d) > 1 else float('nan')
        true = float(np.linalg.norm(np.array(x['c']['position'][-1]) - np.array(x['c']['position'][0])))
        print(f'  #{k:2d} {x["src"]:9s} true {true:5.2f}  fixed-constants {dist:5.2f}  ratio {true / dist:4.2f}')
    print(f'\ntest half: {len(test)} cases ({sum(not x["low"] for x in test)} confident); per robot (test half):')
    for s in START:
        idx = [i for i, x in enumerate(test) if x['src'] == s]
        print(f'  {s:9s} n={len(idx)}  ' + '  '.join(f'{n} {np.mean([rows[n][i] for i in idx]):.1f}' for n in rows))


if __name__ == '__main__':
    main()
