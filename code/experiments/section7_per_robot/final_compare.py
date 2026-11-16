"""Per robot, on val_seen: the new per-robot models.

    python experiments/section7_per_robot/final_compare.py
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
import siglip_steps as ss  # noqa: E402
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import inputs  # noqa: E402
from recon_mix import side_fix, train_net  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402

R_RUNS = ['r3', 'r3b', 'r3c', 'r3d']
OWN = {'recon': 'qsteps_recon', 'huron': 'qsteps_huron', 'stanford_real': 'qsteps_stanford_real', 'scand_spot': 'qsteps_scand_spot'}


def main():
    final.setup()
    steps = pathnet.typical_steps()
    tr, va = final.load_labelled('train'), final.load_labelled('val_seen')
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = tr['q_last'].shape[-1], tr['q_cells'].shape[-1]
    eps = load('data/lcvn/val_seen.jsonl')
    sizes = {k: int(max(tr[f'seg_{k}'].max(), va[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    K = tr['floor'].shape[1]
    pdim = ss.parse_vec(va, torch.arange(2), sizes).shape[1]
    parse = {json.loads(l)['episode']: json.loads(l) for l in open('features/val_seen_parsed2.jsonl')}

    r_all = []
    for name in R_RUNS:
        m = RouteNet(va['q_cells'].shape[-1], va['q_last'].shape[-1]).cuda()
        m.load_state_dict(torch.load(f'runs/{name}/model.pt'), strict=False)
        r_all.append(final.routes_of(m, va, 'cuda'))
    r_all = np.mean(r_all, 0)

    @torch.no_grad()
    def saved(run, idx):
        pk = json.load(open(f'runs/{run}/picked.json'))
        cells, globs = pk['cells'], pk['globs']
        out = []
        for seed in (0, 1, 2):
            m = ss.GridRoute(cells, globs, pdim).cuda()
            m.load_state_dict(torch.load(f'runs/{run}/seed{seed}_epoch{pk["epoch"]}.pt'))
            m.eval()
            rs = []
            for i in range(0, len(idx), 1024):
                j = idx[i:i + 1024]
                b = batch(va, j, 'cuda')
                o = dict(camera=b['camera'], parse=ss.parse_vec(va, j, sizes).cuda(), qwen=b['q_cells'], q_last=b['q_last'], q_text=b['q_text'])
                im = va['image_index'][j]
                if 'sig' in cells:
                    o['sig'] = va['sig_tokens'][im].cuda().float()
                if 'dino' in cells:
                    o['dino'] = va['dino_tokens'][im].cuda().float()
                if 'sentence' in globs:
                    o['sentence'] = va['sentence'][j].cuda().float()
                rs.append(m.run(o).float().cpu().numpy())
            out.append(np.concatenate(rs))
        return np.mean(out, 0), pk
    all_robot_runs = [r for r in OWN.values() if os.path.exists(f'runs/{r}/picked.json')
                      and json.load(open(f'runs/{r}/picked.json'))['train'] > 50000]
    table = {}
    for g, gn in enumerate(GROUPS):
        vi = torch.where(va['group'] == g)[0]
        if gn == 'tartan' or len(vi) < 30:
            continue
        eg = [eps[i] for i in vi.tolist()]
        side = []
        for e in eg:
            t = [s for s in parse[e['episode']]['parsed']['segments'] if s['move'] in ('turn', 'curve')]
            side.append(t[0]['side'] if t else 'none')
        side = np.array(side)

        def run(r):
            return float(np.mean([x['score'] for x in score(eg, dict(zip([e['episode'] for e in eg], final.to_moves(r, va['group'][vi], steps))))[1]]))
        ti = torch.where((tr['group'] == g) & ok)[0]
        X, Xv = inputs(tr, ti, sizes, 'base without DINOv3'), inputs(va, vi, sizes, 'base without DINOv3')
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-4)
        X, Xv = (X - mu) / sd, (Xv - mu) / sd
        base = []
        for seed in (0, 1, 2):
            net = train_net(X, tr['floor'][ti].float().flatten(1), seed, K)
            with torch.no_grad():
                base.append(net(Xv.cuda()).view(-1, K, 2).cpu().numpy())
        base = np.mean(base, 0)
        r = r_all[vi.numpy()]
        old = r if gn == 'scand_jackal' else (r + base) / 2
        old = old if gn == 'huron' else side_fix(old, side)
        cands = [OWN[gn]] if gn in OWN else all_robot_runs
        row = {'old (f6-style)': run(old)}
        for c in cands:
            new, _ = saved(c, vi)
            row[f'new {c}'] = run(new)
            row[f'new {c} + old'] = run((new + old) / 2)
        table[gn] = row
        print(gn, ' | '.join(f'{k} {v:.2f}' for k, v in row.items()), flush=True)
    json.dump(table, open('runs/final_compare.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
