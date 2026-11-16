"""Scores trivial predictors on a validation split.

    python experiments/section4_baseline/baselines.py data/lcvn/train.jsonl data/lcvn/val_seen.jsonl
"""
import collections
import json
import math
import sys

import numpy as np

from score import load, score


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def local_actions(e):
    """Ground-truth actions in the agent's frame, using the recorded yaw at each step."""
    p, yaw = e['position'], e['yaw']
    out = []
    for t in range(len(p) - 1):
        dX, dY = p[t + 1][0] - p[t][0], p[t + 1][1] - p[t][1]
        c, s = math.cos(yaw[t]), math.sin(yaw[t])
        out.append([c * dX + s * dY, -s * dX + c * dY, wrap(yaw[t + 1] - yaw[t])])
    return out


def source(e):
    name = e['trajectory'].split('/', 1)[1]
    for prefix, src in (('jackal_', 'recon'), ('random_mdps_', 'scand'), ('no', 'stanford'), ('sim', 'stanford')):
        if name.startswith(prefix):
            return src
    if name[:4] in ('Jan-', 'Feb-', 'Mar-', 'Apr-', 'May-', 'Jun-', 'Jul-', 'Aug-', 'Sep-', 'Oct-', 'Nov-', 'Dec-'):
        return 'huron'
    return 'other'


def main(train_path, val_path):
    train, val = load(train_path), load(val_path)
    by_src = collections.defaultdict(list)
    for e in train:
        by_src[source(e)].append(local_actions(e))
    every = [a for v in by_src.values() for a in v]

    def straight(acts_list):
        n = int(np.median([len(a) for a in acts_list]))
        step = float(np.median([math.hypot(x, y) for a in acts_list for x, y, _ in a]))
        return [[step, 0.0, 0.0]] * n

    def mean_route(acts_list):
        n = int(np.median([len(a) for a in acts_list]))
        same = [a for a in acts_list if len(a) >= n]
        return np.mean([np.array(a[:n]) for a in same], axis=0).tolist()

    srcs = set(by_src) | {'other'}
    straight_all = straight(every)
    straight_src = {s: straight(by_src.get(s, every)) for s in srcs}
    mean_src = {s: mean_route(by_src.get(s, every)) for s in srcs}
    preds = {
        'oracle': {e['episode']: local_actions(e) for e in val},
        'straight, all data': {e['episode']: straight_all for e in val},
        'straight, per source': {e['episode']: straight_src[source(e)] for e in val},
        'mean route, per source': {e['episode']: mean_src[source(e)] for e in val},
    }
    print('val sources:', dict(collections.Counter(source(e) for e in val)))
    print('train sources:', {k: len(v) for k, v in by_src.items()})
    for name, p in preds.items():
        means, _ = score(val, p)
        print(f'{name:24s}', json.dumps({k: round(v, 3) for k, v in means.items()}))


if __name__ == '__main__':
    main(*sys.argv[1:3])
