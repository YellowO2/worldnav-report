"""g6: g5's test routes with the test-time-averaged LoRA models added (lora_final2.py).

    python 4_final/final_g6.py
"""
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from group_classifier import predict_groups  # noqa: E402
from manual_path_score import moves_along  # noqa: E402

CUT, FAST = 8.0, 0.6


def load_test():
    final.setup()
    steps = pathnet.typical_steps()
    test = pathnet.load_split('test', with_labels=False)
    if os.path.exists('runs/test_group_audit.npy'):      # the camera classifier's labels saved by experiments/section7_per_robot/test_group_audit.py
        import torch
        test['group'] = torch.from_numpy(np.load('runs/test_group_audit.npy', allow_pickle=True).item()['cls'])
    else:
        test['group'] = predict_groups(test).cpu()
    return test, steps


def main():
    test, steps = load_test()
    out = np.load('runs/g5/test_routes.npy')
    for src, w in (('runs/lora_final2', 1 / 3), ('runs/lora_tartan2', 1 / 5)):
        if not os.path.exists(f'{src}/test_routes.npy'):
            print(f'{src}: missing, skipped', flush=True)
            continue
        d = np.load(f'{src}/test_routes.npy', allow_pickle=True).item()
        assert [test['episodes'][k] for k in d['index']] == d['episodes']
        out[d['index']] = (1 - w) * out[d['index']] + w * d['routes']
        print(f'{src}: {len(d["index"])} episodes, weight {w:.2f}', flush=True)
    save('g6', out, test, steps)


def save(name, out, test, steps):
    """Routes -> moves (TartanDrive by the move rule, the others at their typical step) -> runs/{name}/."""
    tartan = GROUPS.index('tartan')
    moves = []
    for r, g in zip(out, test['group'].tolist()):
        pts = np.concatenate([[[0.0, 0.0]], r])
        if g == tartan:
            L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            n = 19 if L < CUT else max(1, round(L / FAST))
            moves.append(moves_along(pts, max(L / n, 1e-4)))
        else:
            moves.append(moves_along(pts, steps[GROUPS[g]]))
    os.makedirs(f'runs/{name}', exist_ok=True)
    np.save(f'runs/{name}/test_routes.npy', out)
    with zipfile.ZipFile(f'runs/{name}/submission_test.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('test.json', json.dumps({'split': 'test', 'episodes': [{'episode_id': ep, 'actions': a}
                                                                          for ep, a in zip(test['episodes'], moves)]}))
    print(f'saved runs/{name}/submission_test.zip: {len(moves)} test episodes')


if __name__ == '__main__':
    main()
