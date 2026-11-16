"""Where do points go between a route (points in metres) and the moves we submit? On val_seen.

    python experiments/section5_first_model/conversion_check.py
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from manual_path_score import moves_along  # noqa: E402
from path_labels import start_frame  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402


def every_record(e):
    """Recorded positions in order, each move facing along its own travel direction."""
    p = start_frame(e['position'], e['yaw'][0])
    xy = np.stack([p[:, 0], -p[:, 1]], 1)                        # (forward, left)
    acts, yaw = [], 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        d = b - a
        if np.hypot(*d) < 1e-6:
            acts.append([0.0, 0.0, 0.0])
            continue
        h = math.atan2(d[1], d[0])
        c, s = math.cos(yaw), math.sin(yaw)
        acts.append([c * d[0] + s * d[1], -s * d[0] + c * d[1], h - yaw])
        yaw = h
    return acts[:64]


def main():
    final.setup()
    dev, steps = 'cuda', pathnet.typical_steps()
    eps, f = load('data/lcvn/val_seen.jsonl'), final.load_labelled('val_seen')
    g = f['group'].numpy()
    names = [e['episode'] for e in eps]
    true16 = f['floor'].numpy()

    def run(acts):
        return np.array([r['score'] for r in score(eps, dict(zip(names, acts)))[1]])

    def own_step(e):
        p = np.array(e['position'])
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())

    rows = {
        'recorded moves': run([e['delta'][:64] for e in eps]),
        'every record': run([every_record(e) for e in eps]),
        'full route, step': run([moves_along(np.concatenate([[[0, 0]], start_frame(e['position'], e['yaw'][0])[1:]]),
                                             steps[GROUPS[int(k)]]) for e, k in zip(eps, g)]),
        '16 points, step': run([moves_along(np.concatenate([[[0, 0]], t]), steps[GROUPS[int(k)]]) for t, k in zip(true16, g)]),
        '16 points, own step': run([moves_along(np.concatenate([[[0, 0]], t]), own_step(e)) for t, e in zip(true16, eps)]),
    }
    model = RouteNet(f['q_cells'].shape[-1], f['q_last'].shape[-1]).to(dev)
    model.load_state_dict(torch.load('runs/r3/model.pt'), strict=False)
    ours = final.routes_of(model, f, dev)
    for m in (0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5):
        rows[f'ours, step x{m}'] = run([moves_along(np.concatenate([[[0, 0]], r]), steps[GROUPS[int(k)]] * m) for r, k in zip(ours, g)])
    groups = [(n, g == k) for k, n in enumerate(GROUPS) if (g == k).sum() >= 30]
    print(f'{"":22s} {"all":>6s} ' + ' '.join(f'{n[:12]:>12s}' for n, _ in groups))
    for name, s in rows.items():
        print(f'{name:22s} {s.mean():6.1f} ' + ' '.join(f'{s[m].mean():12.1f}' for _, m in groups))


if __name__ == '__main__':
    main()
