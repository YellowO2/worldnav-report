"""TartanDrive report set (tartan.py's held-out 20% of recordings): the new model.

    python experiments/section7_per_robot/tartan_avg_check.py [t1|t4]
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
import siglip_steps as ss  # noqa: E402
from endnet import batch  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402
from tartan import recording, split_routes, subset  # noqa: E402


def main():
    final.setup()
    steps = pathnet.typical_steps()
    eps, f = load('data/lcvn/val_unseen.jsonl'), final.load_labelled('val_unseen')
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = f['q_last'].shape[-1], f['q_cells'].shape[-1]
    held = split_routes()
    hi = torch.tensor([i for i, e in enumerate(eps) if recording(e['trajectory']) in held])
    sizes = {'move': 7, 'side': int(f['seg_side'].max()) + 1, 'degree': int(f['seg_degree'].max()) + 1, 'stop': 5}
    tr_text = torch.load('features/train_v3_text.pt')          # the sizes the saved models were built with
    for k in ('move', 'side', 'degree'):
        sizes[k] = max(sizes[k], int(tr_text[f'seg_{k}'].max()) + 1)
    sizes['stop'] = max(sizes['stop'], int(tr_text['stop_kind'].max()) + 1)
    pk = json.load(open('runs/qsteps_tartan/picked.json'))
    pdim = ss.parse_vec(f, hi[:2], sizes).shape[1]
    new = []
    with torch.no_grad():
        for seed in (0, 1, 2):
            m = ss.GridRoute(pk['cells'], pk['globs'], pdim).cuda()
            m.load_state_dict(torch.load(f'runs/qsteps_tartan/seed{seed}_epoch{pk["epoch"]}.pt'))
            m.eval()
            rs = []
            for i in range(0, len(hi), 1024):
                j = hi[i:i + 1024]
                b = batch(f, j, 'cuda')
                o = dict(camera=b['camera'], parse=ss.parse_vec(f, j, sizes).cuda(), qwen=b['q_cells'], q_last=b['q_last'],
                         q_text=b['q_text'], dino=f['dino_tokens'][f['image_index'][j]].cuda().float())
                rs.append(m.run(o).float().cpu().numpy())
            new.append(np.concatenate(rs))
    new = np.mean(new, 0)
    from tartan import add_da3
    old_name = sys.argv[1] if len(sys.argv) > 1 else 't1'
    if old_name == 't4':                                 # t4 was trained with DA3 depth cells
        add_da3(f, eps)
    g = subset(f, hi.tolist())
    old_m = RouteNet(f['q_cells'].shape[-1], f['q_last'].shape[-1]).cuda()
    old_m.load_state_dict(torch.load(f'runs/{old_name}/model.pt'), strict=False)
    old = final.routes_of(old_m, g, 'cuda')
    he = [eps[i] for i in hi.tolist()]

    def run(r):
        return float(np.mean([x['score'] for x in score(he, dict(zip([e['episode'] for e in he], final.to_moves(r, g['group'], steps))))[1]]))
    L = lambda r: np.linalg.norm(np.diff(np.concatenate([np.zeros((len(r), 1, 2)), r], 1), axis=1), axis=2).sum(1)  # noqa: E731
    print(f'TartanDrive report set ({len(hi)} episodes): new {run(new):.2f}, old {old_name} {run(old):.2f}, average {run((new + old) / 2):.2f}, '
          f'new 2/3 + old 1/3 {run(2 / 3 * new + old / 3):.2f}; median length new {np.median(L(new)):.2f} m, old {np.median(L(old)):.2f} m')


if __name__ == '__main__':
    main()
