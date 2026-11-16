"""Official score of the hand-drawn, floor-lifted paths (common/manual_path.py) on their 25 cases.

    python common/manual_path_score.py
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import local_actions, source  # noqa: E402
from score import episode_metrics  # noqa: E402

eps = {}
steps, lengths = collections.defaultdict(list), collections.defaultdict(list)
for line in open('data/lcvn/train.jsonl'):
    e = json.loads(line)
    e['yaw'] = np.asarray(e['yaw'], dtype=float).reshape(-1).tolist()
    eps[e['episode']] = e
    if e['episode'].endswith('_0'):
        p = np.array(e['position'])
        steps[source(e)].append(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())
        lengths[source(e)].append(len(p) - 1)
step = {s: float(np.median(v)) for s, v in steps.items()}
n_med = {s: int(np.median(v)) for s, v in lengths.items()}


def moves_along(points, step_len):
    """Resample a (forward, right) polyline every step_len and face along it -> local [dx, dy(left), dyaw]."""
    pts = np.array(points)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(1, round(cum[-1] / step_len))
    s = np.linspace(0, cum[-1], n + 1)
    xy = np.stack([np.interp(s, cum, pts[:, 0]), -np.interp(s, cum, pts[:, 1])], 1)   # (forward, left)
    head = [math.atan2(*(xy[i + 1] - xy[i])[::-1]) for i in range(n)]
    acts, yaw = [], 0.0
    for i in range(n):
        d = xy[i + 1] - xy[i]
        c, sn = math.cos(yaw), math.sin(yaw)
        acts.append([c * d[0] + sn * d[1], -sn * d[0] + c * d[1], head[i] - yaw])
        yaw = head[i]
    return acts[:64]




def run():
    rows = json.load(open('config/path.json'))
    cases = json.load(open('config/cases.json'))
    heights = json.load(open('config/robot_heights.json'))
    true_height = {s: v['published'] or v['da3_median'] for s, v in heights.items()}
    typical = {s: step[s] * n_med[s] for s in step}                         # the robot's typical route length (m)


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


    VARIANTS = ('straight', 'drawn', 'depth fix', 'stop rule', 'depth fix + stop rule')
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        e = eps[cases[r['case']]['episode']]
        src = r['source']
        path = np.array(r['lifted'])
        fixed = path * (true_height[src] / r['camera_height'])                # rescale so the floor sits at the robot's height
        routes = {'drawn': path, 'depth fix': fixed, 'stop rule': cut(path, typical[src]),
                  'depth fix + stop rule': cut(fixed, typical[src])}
        sc = {'straight': episode_metrics([[step[src], 0.0, 0.0]] * n_med[src], e['position'], e['yaw'])['score']}
        for name, pts in routes.items():
            sc[name] = episode_metrics(moves_along(pts, step[src]), e['position'], e['yaw'])['score']
        for name in VARIANTS:
            out[src][name].append(sc[name])
        print(f'#{r["case"]:2d} {src:8s} true end {r["gt_end"]:5.2f} | drawn end {r["lift_end"]:5.2f} -> depth-fixed '
              f'{np.linalg.norm(fixed[-1]):5.2f} | ' + '  '.join(f'{n} {sc[n]:5.1f}' for n in VARIANTS))
    print()
    print(f'{"robot":9s}' + ''.join(f'{n:>24s}' for n in VARIANTS))
    for s in list(out) + ['all']:
        vals = {n: [v for src in out for v in out[src][n]] if s == 'all' else out[s][n] for n in VARIANTS}
        print(f'{s:9s}' + ''.join(f'{np.mean(vals[n]):24.1f}' for n in VARIANTS))
    print('camera heights used:', {s: round(h, 2) for s, h in true_height.items()}, ' typical lengths (m):',
          {s: round(v, 2) for s, v in typical.items()})


    def self_check():
        """The true route pushed through the same conversion (manual_path.true_route -> moves_along -> score)."""
        from manual_path import true_route
        scores = []
        for r in rows:
            e = eps[cases[r['case']]['episode']]
            gt = true_route(e)                                             # (forward, right) metres, start first
            scores.append(episode_metrics(moves_along(gt, step[r['source']]), e['position'], e['yaw'])['score'])
            exact = episode_metrics(local_actions(e), e['position'], e['yaw'])['score']
            print(f'self-check #{r["case"]:2d}: true route through my conversion {scores[-1]:5.1f}  (exact moves {exact:5.1f})')
        print(f'self-check mean {np.mean(scores):.1f}')


    if '--self-check' in sys.argv:
        self_check()


if __name__ == '__main__':
    run()
