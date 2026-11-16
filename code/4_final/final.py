"""The submission: route models.

    python 4_final/final.py check --runs r1:x1 r2:x1 r3:-
"""
import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
import endnet  # noqa: E402
import pathnet  # noqa: E402
from endnet import add_qwen, batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from manual_path_score import moves_along  # noqa: E402
from routenet import RouteNet, forward_route, lift, score_loss  # noqa: E402
from score import load, score  # noqa: E402

EPOCHS = 12                       # (from the r1-r3 curves: validation is flat from epoch 8 to 16)
IMAGE_KEYS = ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls')
QWEN_KEYS = ('q_cells', 'q_last', 'q_text')


def setup():
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, 'v3'
    endnet.QWEN_TAG = 'qwen2'


def load_labelled(split):
    f = pathnet.load_split(split)
    add_qwen(f, split)
    return f


def merge(parts):
    """Several splits' feature dicts as one: per-episode tensors joined."""
    out = {'episodes': sum((p['episodes'] for p in parts), [])}
    img_off = q_off = 0
    idx, qidx = [], []
    for p in parts:
        idx.append(p['image_index'] + img_off)
        qidx.append(p['q_index'] + q_off)
        img_off += len(p['sig_pooled'])
        q_off += len(p['q_last'])
    out['image_index'], out['q_index'] = torch.cat(idx), torch.cat(qidx)
    for k in IMAGE_KEYS + QWEN_KEYS:
        out[k] = torch.cat([p[k] for p in parts])
    n = len(out['episodes'])
    for k, v in parts[0].items():
        if k in out or not torch.is_tensor(v) or len(v) != len(parts[0]['episodes']):
            continue
        out[k] = torch.cat([p[k] for p in parts])
        assert len(out[k]) == n
    return out


def make(f, idx, dev, x=None):
    b = batch(f, idx, dev)
    uv = torch.zeros(len(idx), 2, device=dev) if x is None else x[idx].to(dev).float()
    b['x_uv'] = uv
    b['x_floor'] = torch.zeros_like(uv) if x is None else lift(uv, b)
    if 'da3' in f:                                       # depth cells (log m, far share) + has-depth flag
        b['da3'] = torch.cat([f['da3'][idx].float(), f['has_da3'][idx].float()[:, None, None].expand(-1, 64, 1)], -1).to(dev)
    return b


@torch.no_grad()
def routes_of(model, f, dev, x=None):
    model.eval()
    out = [forward_route(model, make(f, torch.arange(i, min(i + 1024, len(f['episodes']))), dev, x)).float().cpu()
           for i in range(0, len(f['episodes']), 1024)]
    return torch.cat(out).numpy()


def to_moves(routes, groups, steps):
    return [moves_along(np.concatenate([[(0.0, 0.0)], r]), steps[GROUPS[int(g)]]) for r, g in zip(routes, groups)]


def check(specs):
    """Val score of trained routenet runs and of their point-by-point average."""
    setup()
    dev, steps = 'cuda', pathnet.typical_steps()
    for split in ('val_seen', 'val_unseen'):
        eps, f = load(f'data/lcvn/{split}.jsonl'), load_labelled(split)
        all_routes = []
        for spec in specs:
            name, xrun = spec.split(':')
            model = RouteNet(f['q_cells'].shape[-1], f['q_last'].shape[-1]).to(dev)
            missing, unexpected = model.load_state_dict(torch.load(f'runs/{name}/model.pt'), strict=False)
            assert not unexpected and all(k.startswith(('cell_pos16', 'mid.', 'turn.')) for k in missing), missing
            x = None if xrun == '-' else torch.load(f'runs/{xrun}/{split}_x.pt')['x']
            r = routes_of(model, f, dev, x)
            all_routes.append(r)
            s = score(eps, dict(zip([e['episode'] for e in eps], to_moves(r, f['group'], steps))))[0]['score']
            print(f'{split} {name}: {s:.2f}', flush=True)
        avg = np.mean(all_routes, 0)
        s = score(eps, dict(zip([e['episode'] for e in eps], to_moves(avg, f['group'], steps))))[0]['score']
        print(f'{split} average of {len(specs)}: {s:.2f}', flush=True)


def with_depth(f, da3=None, has=None):
    n = len(f['episodes'])
    f['da3'] = da3 if da3 is not None else torch.zeros(n, 64, 2, dtype=torch.float16)
    f['has_da3'] = has if has is not None else torch.zeros(n)
    return f


def train(name, seeds, bs=256, lr=3e-4, views=False, da3=False):
    from group_classifier import predict_groups
    setup()
    dev, steps, out = 'cuda', pathnet.typical_steps(), f'runs/{name}'
    os.makedirs(out, exist_ok=True)
    if da3:                                              # DA3 depth cells for TartanDrive (da3_tartan.py)
        from tartan import add_da3
        vu = load_labelled('val_unseen')
        add_da3(vu, load('data/lcvn/val_unseen.jsonl'))
        tr = merge([with_depth(load_labelled('train')), with_depth(load_labelled('val_seen')), vu])
    else:
        tr = merge([load_labelled(s) for s in ('train', 'val_seen', 'val_unseen')])
    keep = torch.where(tr['visible'][:, -1])[0]
    n = len(keep)
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test)
    add_qwen(test, 'test')
    test['gravity'] = test['gravity'].float()
    if da3:
        td = torch.load('features/test_da3.pt')
        assert td['episodes'] == test['episodes']
        with_depth(test, td['da3'], td['has_da3'])
        print(f'test episodes with depth: {int(td["has_da3"].sum())}', flush=True)
    print(f'training on {n} labelled episodes; test {len(test["episodes"])} episodes', flush=True)
    if views:                                            # TartanDrive views along each route (tartan.py), flagged
        ex = torch.load('features/tartan_views.pt')
        parts = [torch.load(f'features/tartan_views_qwen2_chunks/{c}') for c in ex['chunks']]
        ex.update(q_cells=torch.cat([p['cells'] for p in parts]), q_last=torch.cat([p['last'] for p in parts]),
                  q_text=torch.cat([p['text_mean'] for p in parts]))
        cam_t = pathnet.TABLE[GROUPS.index('tartan')]
        print(f'TartanDrive extra views: {len(ex["route"])}', flush=True)
    test_routes = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = RouteNet(tr['q_cells'].shape[-1], tr['q_last'].shape[-1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=EPOCHS * (n // bs), pct_start=0.05)
        for ep in range(EPOCHS):
            t0, perm, tot = time.time(), keep[torch.randperm(n)], 0.0
            for k in range(n // bs):
                idx = perm[k * bs:(k + 1) * bs]
                loss = score_loss(forward_route(model, make(tr, idx, dev)).float(), tr['floor'][idx].to(dev))
                if views:                                # end point from a later view
                    e_i = torch.randint(0, len(ex['route']), (bs // 4,))
                    down = ex['gravity'][e_i].float()
                    cam = cam_t.expand(len(e_i), -1)
                    b = dict(q_cells=ex['q_cells'][e_i].to(dev).float(), q_last=ex['q_last'][e_i].to(dev).float(),
                             q_text=ex['q_text'][e_i].to(dev).float(), cam=cam.to(dev), down=down.to(dev),
                             camera=torch.cat([cam[:, 11:13], cam[:, :1], down], 1).to(dev),
                             x_uv=torch.tensor([[1.0, 0.0]], device=dev).expand(len(e_i), -1),
                             x_floor=torch.zeros(len(e_i), 2, device=dev))
                    end_pred = forward_route(model, b).float()[:, -1]
                    et = ex['floor'][e_i].to(dev)
                    loss = loss + ((end_pred - et[:, -1]).norm(dim=-1) / et.norm(dim=-1).max(1).values.clamp(min=0.5)).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() / (n // bs)
            print(json.dumps(dict(seed=seed, epoch=ep + 1, sec=round(time.time() - t0), train_loss=round(tot, 4))), flush=True)
        torch.save(model.state_dict(), f'{out}/seed{seed}.pt')
        test_routes.append(routes_of(model, test, dev))
    avg = np.mean(test_routes, 0)
    np.save(f'{out}/test_routes.npy', avg)
    moves = to_moves(avg, test['group'], steps)
    with zipfile.ZipFile(f'{out}/submission_test.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('test.json', json.dumps({'split': 'test', 'episodes': [{'episode_id': ep, 'actions': a}
                                                                          for ep, a in zip(test['episodes'], moves)]}))
    print(f'saved {out}/submission_test.zip: {len(moves)} test episodes, {len(seeds)} seeds averaged')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['check', 'train'])
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--name')
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--views', action='store_true', help='add the flagged TartanDrive views (tartan.py views)')
    ap.add_argument('--da3', action='store_true', help='DA3 depth cells for TartanDrive photos (da3_tartan.py)')
    a = ap.parse_args()
    check(a.runs) if a.mode == 'check' else train(a.name, a.seeds, views=a.views, da3=a.da3)
