"""Ceiling of better steps: keep each model's predicted val paths and cut them into moves in different ways, some using the true answer (oracles), and score with the official metric.

    python experiments/section6_limits/step_oracle.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
from path_labels import start_frame  # noqa: E402
from score import load, score  # noqa: E402

ROBOTS = ('recon', 'huron', 'stanford_real', 'scand_spot')


def moves_from(points):
    """Positions after each move (forward, right) -> moves [dx, dy (left), dyaw], facing along each move."""
    xy = np.concatenate([[[0.0, 0.0]], points])
    xy = np.stack([xy[:, 0], -xy[:, 1]], 1)
    acts, yaw = [], 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a
        h = float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 1e-4 else yaw
        c, s = np.cos(yaw), np.sin(yaw)
        acts.append([float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1]), h - yaw])
        yaw = h
    return acts[:64]


def at(pts, arcs):
    """Points of polyline pts (starting at the origin) at arc lengths arcs."""
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return np.stack([np.interp(arcs, cum, pts[:, 0]), np.interp(arcs, cum, pts[:, 1])], 1)


def length(pts):
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def main():
    steps = pathnet.typical_steps()
    eps = load('data/lcvn/val_seen.jsonl')
    out = {}
    for g in ROBOTS:
        routes, idx = np.load(f'runs/qsteps_{g}/val_routes.npy'), np.load(f'runs/qsteps_{g}/val_index.npy')
        e = [eps[i] for i in idx]
        acts = {k: [] for k in ('typical', 'true count', 'true step', 'true schedule', 'scaled schedule', 'len typical',
                                'len schedule', 'truth typical', 'truth')}
        counts = {n: [] for n in range(1, 41)}
        for x, r in zip(e, routes):
            true = np.concatenate([[[0.0, 0.0]], start_frame(x['position'], x['yaw'][0])[1:]])
            arc = np.cumsum(np.linalg.norm(np.diff(true, axis=0), axis=1))
            n, Lt = len(arc), float(arc[-1])
            pred = np.concatenate([[[0.0, 0.0]], r])
            L = max(length(pred), 1e-3)
            uni = lambda p, k: moves_from(at(p, np.linspace(0, length(p), k + 1)[1:]))   # noqa: E731
            acts['typical'].append(uni(pred, max(1, round(L / steps[g]))))
            acts['true count'].append(uni(pred, n))
            acts['true step'].append(uni(pred, max(1, round(L / max(Lt / n, 1e-3)))))
            keep = max(1, int((arc < L - 1e-3).sum()) + (1 if Lt > L else 0))
            acts['true schedule'].append(moves_from(at(pred, np.minimum(arc[:keep], L))))
            acts['scaled schedule'].append(moves_from(at(pred, arc * L / max(Lt, 1e-3))))
            stretched = pred * Lt / L
            acts['len typical'].append(uni(stretched, max(1, round(Lt / steps[g]))))
            acts['len schedule'].append(moves_from(at(stretched, arc)))
            acts['truth typical'].append(uni(true, max(1, round(Lt / steps[g]))))
            acts['truth'].append(moves_from(true[1:]))
            for k in counts:
                counts[k].append(uni(pred, k))
        names = [x['episode'] for x in e]
        res = {k: score(e, dict(zip(names, v)), workers=32) for k, v in acts.items()}
        per = np.stack([[r['score'] for r in score(e, dict(zip(names, counts[k])), workers=32)[1]] for k in counts])
        row = {k: v[0] for k, v in res.items()}
        row['best count'] = {'score': float(per.max(0).mean())}
        best_fixed = max(counts, key=lambda k: per[k - 1].mean())
        row['best single count'] = {'score': float(per[best_fixed - 1].mean()), 'count': best_fixed}
        out[g] = row
        print(f'\n{g}: {len(e)} val episodes, typical step {steps[g]:.2f} m', flush=True)
        for k, v in row.items():
            extra = '  '.join(f'{m} {v[m]:.3f}' for m in v if m != 'score')
            print(f'  {k:18s} {v["score"]:6.2f}   {extra}', flush=True)
        json.dump(out, open('runs/step_oracle.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
