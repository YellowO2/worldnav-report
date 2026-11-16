"""Nearest-route copying: for each val_seen route.

    python experiments/section6_limits/knn_routes.py
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from routenet import RouteNet  # noqa: E402
from score import load, score  # noqa: E402


def recording(t):
    parts = t.split('/')[-1].split('_')
    return '_'.join(parts[:-2])


def length(r):
    return np.linalg.norm(np.diff(np.concatenate([np.zeros((len(r), 1, 2)), r], 1), axis=1), axis=2).sum(1)


def embed(f, rows_img, rows_q):
    img = torch.cat([F.normalize(f['sig_pooled'][rows_img].float(), dim=-1),
                     F.normalize(f['dino_cls'][rows_img].float(), dim=-1)], 1) / 2 ** 0.5
    return img, F.normalize(f['q_text'][rows_q].float(), dim=-1)


def main():
    final.setup()
    dev, steps = 'cuda', pathnet.typical_steps()
    tr = final.load_labelled('train')
    tr_traj = {e['episode']: e['trajectory'] for e in load('data/lcvn/train.jsonl')}
    first = {}
    for i, r in enumerate(tr['q_index'].tolist()):        # one row per training route
        first.setdefault(r, i)
    ti = torch.tensor(sorted(first.values()))
    t_img, t_txt = embed(tr, tr['image_index'][ti], tr['q_index'][ti])
    t_route = tr['floor'][ti].numpy()
    ok = np.isfinite(t_route).all((1, 2))
    t_group = tr['group'][ti].numpy()
    t_rec = np.array([recording(tr_traj[tr['episodes'][i]]) for i in ti.tolist()])

    eps, va = load('data/lcvn/val_seen.jsonl'), final.load_labelled('val_seen')
    v_img, v_txt = embed(va, va['image_index'], va['q_index'])
    v_group = va['group'].numpy()
    v_rec = np.array([recording(e['trajectory']) for e in eps])

    model = RouteNet(va['q_cells'].shape[-1], va['q_last'].shape[-1]).to(dev)
    model.load_state_dict(torch.load('runs/r3/model.pt'), strict=False)
    ours = final.routes_of(model, va, dev)

    variants = {'ours (r3)': ours}
    t_img, t_txt, v_img, v_txt = t_img.to(dev), t_txt.to(dev), v_img.to(dev), v_txt.to(dev)
    for strict in (False, True):
        for w in (0.0, 0.5):
            nbr = []
            for a in range(0, len(eps), 1024):
                s = v_img[a:a + 1024] @ t_img.T + w * (v_txt[a:a + 1024] @ t_txt.T)
                bad = (v_group[a:a + 1024, None] != t_group[None]) | ~ok[None]
                if strict:
                    bad = bad | (v_rec[a:a + 1024, None] == t_rec[None])
                s[torch.from_numpy(bad).to(dev)] = -9
                nbr.append(s.topk(20, dim=1).indices.cpu())
            nbr = torch.cat(nbr).numpy()
            tag = f'{"strict" if strict else "loose "} w={w}'
            for k in (1, 5, 20):
                variants[f'copy k={k:<2d} {tag}'] = t_route[nbr[:, :k]].mean(1)
            med = np.median(length(t_route[nbr].reshape(-1, *t_route.shape[1:])).reshape(len(eps), 20), 1)
            variants[f'ours, k20 length {tag}'] = ours * (med / np.maximum(length(ours), 1e-3))[:, None, None]

    rows = {}
    for name, r in variants.items():
        rows[name] = np.array([x['score'] for x in
                               score(eps, dict(zip([e['episode'] for e in eps], final.to_moves(r, va['group'], steps))))[1]])
        print('scored', name, flush=True)
    groups = [(gn, v_group == k) for k, gn in enumerate(GROUPS) if (v_group == k).sum() >= 30]
    print(f'\n{"":34s} {"all":>6s} ' + ' '.join(f'{g[:12]:>12s}' for g, _ in groups))
    for name, s in rows.items():
        print(f'{name:34s} {s.mean():6.1f} ' + ' '.join(f'{s[m].mean():12.1f}' for _, m in groups))


if __name__ == '__main__':
    main()
