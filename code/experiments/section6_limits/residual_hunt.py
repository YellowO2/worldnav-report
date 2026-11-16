"""Residual hunt: is there anything that predicts where a saved model is wrong? One row per val route.

    python experiments/section6_limits/residual_hunt.py qsteps_recon
"""
import json
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
from path_labels import start_frame  # noqa: E402
from post_turn import net_turn  # noqa: E402
from score import load  # noqa: E402

MIN = 30


def length(r):
    return float(np.linalg.norm(np.diff(np.concatenate([[[0, 0]], r]), axis=0), axis=1).sum())


def gap(err, has):
    a, b = err[has], err[~has]
    se = np.sqrt(a.var() / len(a) + b.var() / len(b)) + 1e-9
    return a.mean(), b.mean(), (a.mean() - b.mean()) / se


def main(run):
    eps = load('data/lcvn/val_seen.jsonl')
    idx = np.load(f'runs/{run}/val_index.npy')
    ours = np.load(f'runs/{run}/val_routes.npy')
    parse = {json.loads(l)['episode']: json.loads(l) for l in open('features/val_seen_parsed2.jsonl')}
    by = defaultdict(list)
    for k, i in enumerate(idx):
        by[eps[i]['episode'][:-2]].append((k, eps[i]))
    rows = []
    for key, items in by.items():
        k, e = items[0]
        true = start_frame(e['position'], e['yaw'][0])
        tl, ol = length(true[1:]), length(ours[k])
        if tl < 0.3 or ol < 0.1:
            continue
        text = ' '.join(x['instruction'].lower() for _, x in items)
        words = re.findall(r"[a-z']+", text)
        toks = set(words) | {f'{a} {b}' for a, b in zip(words, words[1:])}
        p = parse[e['episode']]
        segs = p['parsed']['segments']
        turns = [s for s in segs if s['move'] in ('turn', 'curve')]
        name = e['trajectory'].split('/')[-1]
        nums = re.findall(r'\d+', name)
        facts = {f'parse: starts with a turn': segs[0]['move'] in ('turn', 'curve'),
                 f'parse: {min(len(turns), 2)} turns': True, f'parse: stop kind {p["stop_kind"]}': True,
                 'parse: passes something': bool(p.get('passing')), f'parse: {min(len(segs), 4)} segments': True,
                 f'recording: session {name[:17]}': True, f'recording: piece {min(int(nums[-2]), 10) if len(nums) >= 2 else -1}': True}
        rows.append(dict(len_err=np.log(tl / ol), turn_err=abs(np.degrees(net_turn(true[1:]))) - abs(np.degrees(net_turn(ours[k]))),
                         toks=toks, facts=facts, tl=tl, ol=ol))
    n = len(rows)
    print(f'{run}: {n} val routes; our length / true median {np.median([r["ol"] / r["tl"] for r in rows]):.2f}')
    clues = defaultdict(lambda: np.zeros(n, bool))
    for j, r in enumerate(rows):
        for t in r['toks']:
            clues['word: ' + t][j] = True
        for f, v in r['facts'].items():
            if v:
                clues[f][j] = True
    clues = {c: m for c, m in clues.items() if MIN <= m.sum() <= n - MIN}
    for err_name, unit in (('len_err', 'length: log(true / ours)'), ('turn_err', 'turn: |true| - |ours| deg')):
        err = np.array([r[err_name] for r in rows])
        res = sorted(((c,) + gap(err, m) + (m.sum(),) for c, m in clues.items()), key=lambda x: -abs(x[3]))
        print(f'\n{unit}: overall mean {err.mean():+.2f}; {len(clues)} clues tested. Top 25 by |t|:')
        for c, a, b, t, k in res[:25]:
            print(f'  {c:40s} n {k:4d}   with {a:+.2f}  without {b:+.2f}   t {t:+.1f}')
        print(f'  (with {len(clues)} clues tested, |t| around 3.5 can happen by chance once)')


if __name__ == '__main__':
    main(sys.argv[1])
