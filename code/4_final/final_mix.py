"""A submission from f3 plus, for each robot other than TartanDrive, the simple base network.

    python 4_final/final_mix.py --name f4
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from group_classifier import predict_groups  # noqa: E402
from ladder import inputs  # noqa: E402
from recon_mix import side_fix, train_net  # noqa: E402

WHICH = 'base without DINOv3'
WHICH_FOR = {'huron': 'base'}           # HuRON ladder (12 Nov): DINOv3 helps there (+1.2)
NO_MIX = {'tartan', 'scand_jackal'}       # val (experiments/section7_per_robot/group_mix.py, 12 Nov)
NO_FIX = {'huron'}                        # the side fix cost HuRON 0.55 on val


def sides(path, episodes):
    parse = {json.loads(l)['episode']: json.loads(l) for l in open(path)}
    out = []
    for e in episodes:
        t = [s for s in parse[e]['parsed']['segments'] if s['move'] in ('turn', 'curve')]
        out.append(t[0]['side'] if t else 'none')
    return np.array(out)


def main(name, base_run='f3', mix=0.5):
    final.setup()
    steps = pathnet.typical_steps()
    tr = final.merge([final.load_labelled('train'), final.load_labelled('val_seen')])
    for k in ('sentence', 'seg_move', 'seg_side', 'seg_degree', 'stop_kind', 'gravity'):
        assert k in tr, k
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test).cpu()
    final.add_qwen(test, 'test')
    sizes = {k: int(max(tr[f'seg_{k}'].max(), test[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), test['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    K = tr['floor'].shape[1]
    routes = np.load(f'runs/{base_run}/test_routes.npy')
    assert len(routes) == len(test['episodes'])
    side = sides('features/test_parsed2.jsonl', test['episodes'])
    out = routes.copy()
    for g, gn in enumerate(GROUPS):
        if gn in NO_MIX:
            v = torch.where(test['group'] == g)[0].numpy()
            if gn not in NO_FIX and gn != 'tartan' and len(v):
                out[v] = side_fix(routes[v], side[v])
            continue
        ti = torch.where((tr['group'] == g) & ok)[0]
        vi = torch.where(test['group'] == g)[0]
        if len(vi) == 0:
            continue
        which = WHICH_FOR.get(gn, WHICH)
        X, Xt = inputs(tr, ti, sizes, which), inputs(test, vi, sizes, which)
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
        X, Xt = (X - mu) / sd, (Xt - mu) / sd
        Y = tr['floor'][ti].float().flatten(1)
        preds = []
        for seed in (0, 1, 2):
            net = train_net(X, Y, seed, K)
            with torch.no_grad():
                preds.append(net(Xt.cuda()).view(-1, K, 2).cpu().numpy())
        v = vi.numpy()
        mixed = mix * routes[v] + (1 - mix) * np.mean(preds, 0)
        out[v] = mixed if gn in NO_FIX else side_fix(mixed, side[v])
        print(f'{gn}: trained on {len(ti)} episodes, {len(v)} test episodes mixed', flush=True)
    os.makedirs(f'runs/{name}', exist_ok=True)
    np.save(f'runs/{name}/test_routes.npy', out)
    moves = final.to_moves(out, test['group'], steps)
    with zipfile.ZipFile(f'runs/{name}/submission_test.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('test.json', json.dumps({'split': 'test', 'episodes': [{'episode_id': ep, 'actions': a}
                                                                          for ep, a in zip(test['episodes'], moves)]}))
    print(f'saved runs/{name}/submission_test.zip: {len(moves)} test episodes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--base-run', default='f3')
    a = ap.parse_args()
    main(a.name, a.base_run)
