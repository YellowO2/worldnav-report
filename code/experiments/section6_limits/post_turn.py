"""Post-processing our routes (r3) with the parse, on one robot's val_seen routes.

    python experiments/section6_limits/post_turn.py recon
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402

CACHE = 'runs/r3/val_seen_routes.npy'


def our_routes(f):
    if os.path.exists(CACHE):
        return np.load(CACHE)
    model = RouteNet(f['q_cells'].shape[-1], f['q_last'].shape[-1]).to('cuda')
    model.load_state_dict(torch.load('runs/r3/model.pt'), strict=False)
    r = final.routes_of(model, f, 'cuda')
    np.save(CACHE, r)
    return r


def amplify(r, a):
    """r (K, 2) forward, right."""
    p = np.concatenate([[[0, 0]], r])
    d = np.diff(p, axis=0)
    step = np.linalg.norm(d, axis=1)
    head = np.unwrap(np.arctan2(d[:, 1], d[:, 0])) * a
    return np.cumsum(np.stack([step * np.cos(head), step * np.sin(head)], 1), 0)


def net_turn(r):
    p = np.concatenate([[[0, 0]], r])
    d = np.diff(p, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-3]
    if len(d) < 2:
        return 0.0
    return float(np.arctan2(d[-1, 1], d[-1, 0]))              # right positive, relative to straight ahead


def main(group):
    final.setup()
    steps = pathnet.typical_steps()
    eps, f = load('data/lcvn/val_seen.jsonl'), final.load_labelled('val_seen')
    ours = our_routes(f)
    parse = {json.loads(l)['episode']: json.loads(l) for l in open('features/val_seen_parsed2.jsonl')}
    gi = np.where(f['group'].numpy() == GROUPS.index(group))[0]
    eg = [eps[i] for i in gi]
    side = []
    for e in eg:
        t = [s for s in parse[e['episode']]['parsed']['segments'] if s['move'] in ('turn', 'curve')]
        side.append(t[0]['side'] if t else 'none')
    side = np.array(side)
    turns = np.isin(side, ['left', 'right'])

    def run(routes):
        return np.array([x['score'] for x in score(eg, dict(zip([e['episode'] for e in eg],
                                                                 final.to_moves(routes, f['group'][gi], steps))))[1]])
    base = ours[gi]
    s0 = run(base)
    print(f'{group}: {len(gi)} episodes; ours {s0.mean():.2f} (turn routes {s0[turns].mean():.2f}, others {s0[~turns].mean():.2f})')
    flipped = base.copy()
    wrong = np.array([(s == 'left' and net_turn(r) > 0.1) or (s == 'right' and net_turn(r) < -0.1) for r, s in zip(base, side)])
    flipped[wrong, :, 1] *= -1
    s1 = run(flipped)
    print(f'side fixed by the parse: {wrong.sum()} routes mirrored; score {s1.mean():.2f} (mirrored ones {s0[wrong].mean():.1f} -> {s1[wrong].mean():.1f})')
    for which, sel in (('turn routes', turns), ('all routes', np.ones(len(gi), bool))):
        for a in (1.2, 1.4, 1.6, 2.0):
            r = flipped.copy()
            r[sel] = np.stack([amplify(x, a) for x in flipped[sel]])
            s = run(r)
            print(f'side fixed + amplify x{a} on {which}: {s.mean():.2f} (turn routes {s[turns].mean():.2f}, others {s[~turns].mean():.2f})', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
