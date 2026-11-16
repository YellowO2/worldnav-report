"""If a hidden piece's neighbours in its recording are known.

    python experiments/section9_data_split/lookup_routes.py
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
from floor_geometry import group_of  # noqa: E402
from path_labels import start_frame  # noqa: E402
from score import load, score  # noqa: E402
from step_oracle import at, length, moves_from  # noqa: E402


def piece(t):
    p = t.split('/')[-1].split('_')
    if t.startswith('val_unseen'):                       # TartanDrive: run_piece
        return '_'.join(p[:-1]), int(p[-1])
    return '_'.join(p[:-2]) + '|' + p[-1], int(p[-2])


def warp(route, end):
    """A model route (16 points) turned and scaled about the start so that it ends at `end`."""
    a, b = complex(*route[-1]), complex(*end)
    if abs(a) < 0.05:
        return None
    c = (route[:, 0] + 1j * route[:, 1]) * (b / a)
    return np.concatenate([[[0.0, 0.0]], np.stack([c.real, c.imag], 1)])


def local(p, origin, yaw):
    """Fixed-frame points -> (forward, right) in the frame at origin facing yaw."""
    d = np.asarray(p, float) - origin
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * d[..., 0] + s * d[..., 1], s * d[..., 0] - c * d[..., 1]], -1)


def hermite(end, h_end, n=40):
    """Smooth curve from the origin heading forward to end, arriving along direction h_end."""
    d = np.linalg.norm(end)
    t = np.linspace(0, 1, n + 1)[:, None]
    m0, m1 = np.array([d, 0.0]), d * h_end
    return (t ** 3 - 2 * t ** 2 + t) * m0 + (-2 * t ** 3 + 3 * t ** 2) * end + (t ** 3 - t ** 2) * m1


def main():
    steps = pathnet.typical_steps()
    eps = {}
    for s in ('train', 'val_seen', 'val_unseen'):
        for e in load(f'data/lcvn/{s}.jsonl'):
            if e['trajectory'] not in eps:
                eps[e['trajectory']] = (s, e)
    rec = defaultdict(dict)
    for t, (s, e) in eps.items():
        key, k = piece(t)
        rec[(group_of(t, s), key)][k] = t
    seam = defaultdict(list)
    for (g, key), d in rec.items():
        for k in d:
            if k + 1 in d:
                a, b = eps[d[k]][1], eps[d[k + 1]][1]
                pa, pb = np.array(a['position']), np.array(b['position'])
                st = np.linalg.norm(np.diff(pa, axis=0), axis=1)[-3:].mean()
                if st > 0.02:
                    seam[g].append((np.linalg.norm(pb[0] - pa[-1]) / st, abs((b['yaw'][0] - a['yaw'][-1] + np.pi) % (2 * np.pi) - np.pi)))
    for g, v in seam.items():
        v = np.array(v)
        print(f'seam {g:14s} n {len(v):5d}: gap in steps 10/50/90% {np.percentile(v[:, 0], [10, 50, 90]).round(2)}, heading change (deg) 50/90% {np.degrees(np.percentile(v[:, 1], [50, 90])).round(1)}')
    gap = {g: float(np.median(np.array(v)[:, 0])) for g, v in seam.items()}
    import os
    model = {}                                           # trajectory -> our model's saved val route
    val_list = load('data/lcvn/val_seen.jsonl')
    for g in ('recon', 'huron', 'stanford_real', 'scand_spot'):
        if os.path.exists(f'runs/qsteps_{g}/val_routes.npy'):
            by = defaultdict(list)
            for i, r in zip(np.load(f'runs/qsteps_{g}/val_index.npy'), np.load(f'runs/qsteps_{g}/val_routes.npy')):
                by[val_list[i]['trajectory']].append(r)
            model.update({t: np.mean(v, 0) for t, v in by.items()})
    out = defaultdict(lambda: defaultdict(list))
    val_count = defaultdict(int)
    for (g, key), d in rec.items():
        for k, t in d.items():
            s, e = eps[t]
            if s == 'train':
                continue
            val_count[g] += 1
            P = eps[d[k - 1]][1] if k - 1 in d and eps[d[k - 1]][0] in ('train', 'val_unseen') else None
            N = eps[d[k + 1]][1] if k + 1 in d and eps[d[k + 1]][0] in ('train', 'val_unseen') else None
            if P is None:
                continue
            pp, py = np.array(P['position']), np.array(P['yaw'])
            v_prev = np.linalg.norm(np.diff(pp, axis=0), axis=1)[-3:].mean()
            turn = float(np.mean(np.diff(np.unwrap(py))[-3:])) if len(py) > 3 else 0.0
            yaw0 = py[-1] + gap[g] * turn
            start = pp[-1] + gap[g] * v_prev * np.array([np.cos(py[-1] + 0.5 * gap[g] * turn), np.sin(py[-1] + 0.5 * gap[g] * turn)])
            n_med = 11
            if N is None:
                arc = v_prev * n_med
                th = turn * np.arange(1, n_med + 1)
                pts = np.concatenate([[[0.0, 0.0]], np.cumsum(v_prev * np.stack([np.cos(th), -np.sin(th)], 1), 0)])
                out[g]['before only: carry on'].append((e, moves_from(pts[1:])))
                out[g]['before only: straight'].append((e, moves_from(at(np.array([[0.0, 0.0], [arc, 0.0]]), np.arange(1, n_med + 1) * v_prev))))
                continue
            np_, ny = np.array(N['position']), np.array(N['yaw'])
            v_next = np.linalg.norm(np.diff(np_, axis=0), axis=1)[:3].mean()
            end_raw = local(np_[0], pp[-1], py[-1])
            line = np.array([[0.0, 0.0], end_raw])
            n = max(1, round(length(line) / steps[g]))
            out[g]['both: line'].append((e, moves_from(at(line, np.linspace(0, length(line), n + 1)[1:]))))
            end_fix = np_[0] - gap[g] * v_next * np.array([np.cos(ny[0]), np.sin(ny[0])])
            end = local(end_fix, start, yaw0)
            dy = ny[0] - yaw0
            curve = hermite(end, np.array([np.cos(dy), -np.sin(dy)]))
            L = length(curve)
            for name, st in (('typical step', steps[g]), ('neighbour step', max(0.5 * (v_prev + v_next), 1e-3))):
                n = int(np.clip(round(L / st), 1, 64))
                out[g][f'both: curve, {name}'].append((e, moves_from(at(curve, np.linspace(0, L, n + 1)[1:]))))
            if t in model:
                st = max(0.5 * (v_prev + v_next), 1e-3)
                mr = np.concatenate([[[0.0, 0.0]], model[t]])
                n = int(np.clip(round(length(mr) / steps[g]), 1, 64))
                out[g]['both: (our model, same pieces)'].append((e, moves_from(at(mr, np.linspace(0, length(mr), n + 1)[1:]))))
                w = warp(model[t], end)
                if w is not None:
                    n = int(np.clip(round(length(w) / st), 1, 64))
                    out[g]['both: model path warped to the end'].append((e, moves_from(at(w, np.linspace(0, length(w), n + 1)[1:]))))
                    k2 = int(np.clip(round(L / st), 1, 64))
                    mix = 0.5 * (at(curve, np.linspace(0, L, k2 + 1)[1:]) + at(w, np.linspace(0, length(w), k2 + 1)[1:]))
                    out[g]['both: curve and warped model, averaged'].append((e, moves_from(mix)))
            line2 = np.array([[0.0, 0.0], end])
            n = int(np.clip(round(length(line2) / max(0.5 * (v_prev + v_next), 1e-3)), 1, 64))
            out[g]['both: line, fixed seams, neighbour step'].append((e, moves_from(at(line2, np.linspace(0, length(line2), n + 1)[1:]))))
    res = {}
    for g, d in out.items():
        print(f'\n{g}: {val_count[g]} val pieces', flush=True)
        res[g] = {}
        for name, v in sorted(d.items()):
            e = [x[0] for x in v]
            m = score(e, dict(zip([x['episode'] for x in e], [x[1] for x in v])), workers=16)[0]
            res[g][name] = dict(n=len(v), **m)
            print(f'  {name:42s} n {len(v):4d} ({len(v) / val_count[g]:.0%})  score {m["score"]:6.2f}  ' + '  '.join(f'{k} {m[k]:.3f}' for k in m if k != 'score'), flush=True)
    json.dump(res, open('runs/lookup_routes.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
