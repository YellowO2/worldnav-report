"""g7: g6 with the later-start models.

    python 4_final/final_g7.py [--w 0.4] [--name g7] [--later L1]
"""
import argparse
import json
import os
import zipfile

import numpy as np

import final_g6
from floor_geometry import GROUPS


def main(w, name, later):
    test, steps = final_g6.load_test()
    out = np.load('runs/g6/test_routes.npy')
    g6 = out.copy()
    tartan = GROUPS.index('tartan')
    for gn in GROUPS:
        p = f'runs/{later}/routes_{gn}.npy'
        if gn == 'tartan' or not os.path.exists(p):
            print(f'{gn}: no later-start routes, g6 kept', flush=True)
            continue
        d = np.load(p, allow_pickle=True).item()
        assert (test['group'][d['index']] != tartan).all()
        out[d['index']] = (1 - w) * g6[d['index']] + w * d['routes']
        gap = np.linalg.norm(d['routes'][:, -1] - g6[d['index'], -1], axis=1)
        print(f'{gn}: {len(d["index"])} episodes, weight {w:.2f}; end points of L1 vs g6 apart by a median {np.median(gap):.2f} m', flush=True)
    assert np.isfinite(out).all()
    final_g6.save(name, out, test, steps)
    sub = json.loads(zipfile.ZipFile(f'runs/{name}/submission_test.zip').read('test.json'))
    ref = json.loads(zipfile.ZipFile('runs/g6/submission_test.zip').read('test.json'))
    assert [e['episode_id'] for e in sub['episodes']] == [e['episode_id'] for e in ref['episodes']] and len(sub['episodes']) == 15303
    assert all(np.isfinite(np.array(e['actions'], dtype=float)).all() and len(e['actions']) > 0 for e in sub['episodes'])
    changed = sum(a['actions'] != b['actions'] for a, b in zip(sub['episodes'], ref['episodes']))
    print(f'checked: 15303 episodes, same ids as g6, no NaN; {changed} episodes differ from g6', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=float, default=0.4)
    ap.add_argument('--name', default='g7')
    ap.add_argument('--later', default='L1')
    a = ap.parse_args()
    main(a.w, a.name, a.later)
