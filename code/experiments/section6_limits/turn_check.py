"""How do one robot's routes really turn, against what we draw (r3).

    python experiments/section6_limits/turn_check.py recon
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from path_labels import start_frame  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load  # noqa: E402


def turn_stats(pts):
    """pts (n, 2) forward, right from the start (start included)."""
    d = np.diff(pts, axis=0)
    step = np.linalg.norm(d, axis=1)
    keep = step > 0.02
    if keep.sum() < 2:
        return 0.0, np.nan, np.nan
    d, step = d[keep], step[keep]
    head = np.unwrap(np.arctan2(-d[:, 1], d[:, 0]))             # left positive
    change = head - head[0]
    total = change[-1]
    cum = np.cumsum(step)
    if abs(total) < np.radians(20):
        return float(np.degrees(total)), np.nan, np.nan
    half = np.argmax(np.abs(change) >= abs(total) / 2)
    ninety = np.argmax(np.abs(change) >= 0.9 * abs(total))
    return float(np.degrees(total)), float(cum[half] / cum[-1]), float(cum[-1] - cum[ninety])


def main(group):
    final.setup()
    eps, f = load('data/lcvn/val_seen.jsonl'), final.load_labelled('val_seen')
    model = RouteNet(f['q_cells'].shape[-1], f['q_last'].shape[-1]).to('cuda')
    model.load_state_dict(torch.load('runs/r3/model.pt'), strict=False)
    ours = final.routes_of(model, f, 'cuda')
    parse = {json.loads(l)['episode']: json.loads(l) for l in open('features/val_seen_parsed2.jsonl')}
    k = GROUPS.index(group)
    rows = []
    for i, e in enumerate(eps):
        if int(f['group'][i]) != k or not e['episode'].endswith('_2'):
            continue
        p = start_frame(e['position'], e['yaw'][0])
        t = turn_stats(p)
        o = turn_stats(np.concatenate([[[0, 0]], ours[i]]))
        yaw = np.unwrap(np.array(e['yaw']))
        mv = np.linalg.norm(np.diff(np.array(e['position']), axis=0), axis=1)
        inplace = float(np.degrees(np.abs(np.diff(yaw))[mv < 0.05].sum()))
        segs = parse[e['episode']]['parsed']['segments']
        first = segs[0]
        turns = [s for s in segs if s['move'] in ('turn', 'curve')]
        rows.append(dict(t=t, o=o, inplace=inplace, starts_turn=first['move'] in ('turn', 'curve'),
                         n_turns=len(turns), degree=turns[0]['degree'] if turns else 'none',
                         side=turns[0]['side'] if turns else 'none', length=float(mv.sum())))

    def show(name, sel):
        r = [x for x in rows if sel(x)]
        if len(r) < 15:
            return
        tt = np.array([x['t'][0] for x in r])
        oo = np.array([x['o'][0] for x in r])
        tw = np.array([x['t'][1] for x in r])
        ow = np.array([x['o'][1] for x in r])
        ta = np.array([x['t'][2] for x in r])
        oa = np.array([x['o'][2] for x in r])
        big = np.abs(tt) >= 60
        print(f'{name:26s} {len(r):4d} | |turn| true {np.median(np.abs(tt)):5.0f} ours {np.median(np.abs(oo)):5.0f} | '
              f'turns 60+: true {big.mean():4.0%} ours {(np.abs(oo) >= 60).mean():4.0%} | '
              f'where true {np.nanmedian(tw):.2f} ours {np.nanmedian(ow):.2f} | after true {np.nanmedian(ta):.1f} m ours {np.nanmedian(oa):.1f} m | '
              f'in place {np.median([x["inplace"] for x in r]):4.0f} deg | length {np.median([x["length"] for x in r]):.1f} m')
    print(f'{group}: {len(rows)} val routes (landmark-style instruction)\n')
    show('all', lambda x: True)
    show('straight only', lambda x: x['n_turns'] == 0)
    show('starts with a turn', lambda x: x['starts_turn'])
    show('turn later', lambda x: x['n_turns'] > 0 and not x['starts_turn'])
    for d in sorted({x['degree'] for x in rows}):
        show(f'first turn degree: {d}', lambda x, d=d: x['degree'] == d and x['n_turns'] > 0)
    tt = np.array([x['t'][0] for x in rows])
    sign_ok = [np.sign(x['t'][0]) == (1 if x['side'] == 'left' else -1) for x in rows if x['side'] in ('left', 'right') and abs(x['t'][0]) >= 30]
    print(f'\nturn side in the parse matches the true turn (true turn 30+ deg): {np.mean(sign_ok):.0%} of {len(sign_ok)}')
    print('true |turn| percentiles 25/50/75/90:', np.percentile(np.abs(tt), [25, 50, 75, 90]).round(0))


if __name__ == '__main__':
    main(sys.argv[1])
