"""Fair ensemble check for one or more robots on val_seen, all with the parse side fix.

    python experiments/section7_per_robot/group_mix.py --runs r3 r3b r3c --groups recon huron
"""
import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import inputs  # noqa: E402
from recon_mix import side_fix, train_net  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402


def main(runs, groups, which='base without DINOv3'):
    final.setup()
    steps = pathnet.typical_steps()
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
    eps = load('data/lcvn/val_seen.jsonl')
    sizes = {k: int(max(tr[f'seg_{k}'].max(), va[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    K = tr['floor'].shape[1]
    r_routes = {}
    for name in runs:
        model = RouteNet(va['q_cells'].shape[-1], va['q_last'].shape[-1]).to('cuda')
        model.load_state_dict(torch.load(f'runs/{name}/model.pt'), strict=False)
        r_routes[name] = final.routes_of(model, va, 'cuda')
    parse = {json.loads(l)['episode']: json.loads(l) for l in open('features/val_seen_parsed2.jsonl')}
    for group in groups:
        g = GROUPS.index(group)
        vi = torch.where(va['group'] == g)[0]
        eg = [eps[i] for i in vi.tolist()]
        side = []
        for e in eg:
            t = [s for s in parse[e['episode']]['parsed']['segments'] if s['move'] in ('turn', 'curve')]
            side.append(t[0]['side'] if t else 'none')

        def run(routes, fix=True):
            r = side_fix(routes, side) if fix else routes
            return float(np.mean([x['score'] for x in score(eg, dict(zip([e['episode'] for e in eg],
                                                                           final.to_moves(r, va['group'][vi], steps))))[1]]))
        ti = torch.where((tr['group'] == g) & ok)[0]
        X, Xv = inputs(tr, ti, sizes, which), inputs(va, vi, sizes, which)
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
        X, Xv = (X - mu) / sd, (Xv - mu) / sd
        Y = tr['floor'][ti].float().flatten(1)
        base = []
        for seed in (0, 1, 2):
            net = train_net(X, Y, seed, K)
            with torch.no_grad():
                base.append(net(Xv.cuda()).view(-1, K, 2).cpu().numpy())
        base = np.mean(base, 0)
        rs = [r_routes[n][vi.numpy()] for n in runs]
        r_avg = np.mean(rs, 0)
        print(f'\n{group}: {len(vi)} val episodes')
        print(f'  {runs[0]} alone, no side fix         {run(rs[0], fix=False):.2f}')
        for n, r in zip(runs, rs):
            print(f'  {n} alone                      {run(r):.2f}')
        print(f'  r runs averaged ({len(runs)})            {run(r_avg):.2f}')
        print(f'  base nets (3 seeds)             {run(base):.2f}')
        print(f'  r avg + base nets, half/half    {run((r_avg + base) / 2):.2f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=['r3'])
    ap.add_argument('--groups', nargs='+', default=['recon'])
    a = ap.parse_args()
    main(a.runs, a.groups)
