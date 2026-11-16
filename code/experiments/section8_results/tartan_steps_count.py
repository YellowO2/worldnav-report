"""TartanDrive pieces end at 10 m or 20 photos, whichever first: a slow.

    python experiments/section8_results/tartan_steps_count.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
from final import load_labelled, routes_of, setup  # noqa: E402
from manual_path_score import moves_along  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402
from tartan import recording, split_routes, subset  # noqa: E402


def main():
    setup()
    steps = pathnet.typical_steps()
    held = split_routes()
    eps, f = load('data/lcvn/val_unseen.jsonl'), load_labelled('val_unseen')
    hi = [i for i, e in enumerate(eps) if recording(e['trajectory']) in held]
    g = subset(f, hi)
    m = RouteNet(f['q_cells'].shape[-1], f['q_last'].shape[-1])
    m.load_state_dict(torch.load('runs/t1/model.pt', map_location='cpu'), strict=False)
    r = routes_of(m, g, 'cpu')
    he = [eps[i] for i in hi]
    n_true = np.array([len(e['position']) - 1 for e in he])
    L = np.linalg.norm(np.diff(np.concatenate([np.zeros((len(r), 1, 2)), r], 1), axis=1), axis=2).sum(1)
    print(f'{len(he)} held-out episodes; true moves: median {np.median(n_true):.0f}, 19+ for {np.mean(n_true >= 19):.0%}; '
          f'typical step {steps["tartan"]:.3f} m')

    def run(name, nmoves):
        acts = []
        for x, n in zip(r, nmoves):
            pts = np.concatenate([[[0.0, 0.0]], x])
            l = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            acts.append(moves_along(pts, max(l / max(int(n), 1), 1e-4)))
        mt = score(he, dict(zip([e['episode'] for e in he], acts)))[0]
        print(f'  {name:28s} score {mt["score"]:6.2f}  SR {mt["SR"]:.3f}  ATE {mt["ATE"]:.3f}  RPE {mt["RPE"]:.3f}  F {mt["F"]:.3f}  H {mt["H"]:.3f}', flush=True)
    run('typical step', np.maximum(1, np.round(L / steps['tartan'])))
    run('19 moves', np.full(len(r), 19))
    for cut in (8.0, 8.5, 9.0):
        for fast in (0.55, 0.6):
            run(f'rule: <{cut} m 19 moves, else /{fast}', np.where(L < cut, 19, np.maximum(1, np.round(L / fast))))
    run('oracle: true number of moves', n_true)


if __name__ == '__main__':
    main()
