"""How many route points are enough? Each true val_seen route.

    python experiments/section5_first_model/k_ceiling.py
"""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
from floor_geometry import GROUPS, group_of  # noqa: E402
from manual_path_score import moves_along  # noqa: E402
from path_labels import start_frame  # noqa: E402
from score import load, score  # noqa: E402

KS = [2, 4, 8, 16, 32, 64]


def resample(route, k):
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return route[[0, -1]]
    t = np.linspace(0, 1, k + 1) * cum[-1]
    return np.stack([np.interp(t, cum, route[:, 0]), np.interp(t, cum, route[:, 1])], 1)


def main():
    steps = pathnet.typical_steps()
    eps = load('data/lcvn/val_seen.jsonl')
    groups = [group_of(e['trajectory'], 'val_seen') for e in eps]
    res = {}
    for k in KS + ['full']:
        preds = {}
        for e, g in zip(eps, groups):
            r = start_frame(e['position'], e['yaw'][0])
            preds[e['episode']] = moves_along(r if k == 'full' else resample(r, k), steps[g])
        _, rows = score(eps, preds)
        for g in set(groups):
            res.setdefault(g, {})[k] = np.mean([x['score'] for x, gg in zip(rows, groups) if gg == g])
        res.setdefault('all', {})[k] = np.mean([x['score'] for x in rows])
    cols = KS + ['full']
    print('| Camera group | ' + ' | '.join(f'K={k}' if k != 'full' else 'full route' for k in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for g in [x for x in GROUPS if x in res] + ['all']:
        print(f'| {g} | ' + ' | '.join(f'{res[g][k]:.1f}' for k in cols) + ' |')


if __name__ == '__main__':
    main()
