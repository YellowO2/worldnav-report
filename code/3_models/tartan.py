"""TartanDrive is about 45% of the test set.

    python 3_models/tartan.py views
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
from final import load_labelled, make, merge, routes_of, setup, to_moves  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from path_labels import resample  # noqa: E402
from routenet import RouteNet, forward_route, score_loss  # noqa: E402
from score import load, score  # noqa: E402

NO_FLAG = '--no-flag' in sys.argv   # extra views without the "later view" flag
STEP = 2                          # (arbitrary) an extra view every STEP frames
MIN_LEFT = 1.0                    # metres of route that must remain from a view


def recording(t):
    return '_'.join(t.split('/')[-1].split('_')[:2])          # 20210826_40_0001_02 -> 20210826_40


def split_routes():
    recs = sorted({recording(t) for t in json.load(open('data/lcvn/val_unseen.frames.json'))})
    random.seed(0)
    held = set(random.sample(recs, round(0.2 * len(recs))))
    return held


def subset(f, idx):
    """Some episodes of a split's feature dict (image and Qwen rows kept whole, indices still valid)."""
    t = torch.tensor(idx, dtype=torch.long)
    out = {k: (v[t] if torch.is_tensor(v) and len(v) == len(f['episodes']) and k not in ('q_cells', 'q_last', 'q_text')
               else v) for k, v in f.items()}
    out['episodes'] = [f['episodes'][i] for i in idx]
    return out


def views(held_out=False):
    """Extra views along the TartanDrive training routes: frame k."""
    import qwen_features as qf
    from gravity_all import Photos
    qf.V2 = True
    held = split_routes()
    eps = [e for e in load('data/lcvn/val_unseen.jsonl')
           if e['episode'].endswith('_2') and (recording(e['trajectory']) in held) == held_out]
    tag = 'tartan_held_views' if held_out else 'tartan_views'
    frames = json.load(open('data/lcvn/val_unseen.frames.json'))
    rows, meta = [], dict(route=[], frame=[], floor=[])
    for e in eps:
        p = np.array(e['position'], dtype=float)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        for k in range(STEP, min(len(p) - 1, len(frames[e['trajectory']])), STEP):
            if cum[-1] - cum[k] < MIN_LEFT:
                break
            c, s = np.cos(e['yaw'][k]), np.sin(e['yaw'][k])
            d = p[k:] - p[k]
            rest = np.stack([c * d[:, 0] + s * d[:, 1], -(-s * d[:, 0] + c * d[:, 1])], 1)
            rows.append((frames[e['trajectory']][k], e['instruction']))
            meta['route'].append(e['episode'][:-2])
            meta['frame'].append(k)
            meta['floor'].append(resample(rest)[0].astype(np.float32))
    print(f'{len(rows)} extra views from {len(eps)} TartanDrive training routes', flush=True)
    from geocalib import GeoCalib
    geo = GeoCalib().to('cuda')
    ph = Photos('val_unseen', [r[0] for r in rows], [GROUPS.index('tartan')] * len(rows))
    grav = []
    with torch.no_grad():
        for x, fpix in torch.utils.data.DataLoader(ph, batch_size=32, num_workers=8):
            g = geo.calibrate(x.to('cuda'), priors={'focal': fpix.float().to('cuda')})['gravity'].vec3d.reshape(len(x), 3)
            g = g * torch.where(g[:, 1:2] < 0, -1.0, 1.0)
            grav.append(g.cpu())
    print('GeoCalib done', flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    model = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda').eval()
    tar = open('data/lcvn/val_unseen.tar', 'rb')
    cdir = f'features/{tag}_qwen2_chunks'
    os.makedirs(cdir, exist_ok=True)
    CH, bs = 1024, 16
    for c0 in range(0, len(rows), CH):
        if os.path.exists(f'{cdir}/{c0:06d}.pt'):
            continue
        C, L, T = [], [], []
        for i in range(c0, min(c0 + CH, len(rows)), bs):
            b = [(qf.load_photo('val_unseen', img, tar), text) for img, text in rows[i:min(i + bs, c0 + CH, len(rows))]]
            cells, last, text_mean = qf.run(model, proc, b, torch.float16)
            C.append(torch.stack(cells).half().cpu())
            L.append(last.half().cpu())
            T.append(text_mean.half().cpu())
        torch.save(dict(cells=torch.cat(C), last=torch.cat(L), text_mean=torch.cat(T)), f'{cdir}/{c0:06d}.pt')
        print(f'{min(c0 + CH, len(rows))}/{len(rows)}', flush=True)
    torch.save(dict(route=meta['route'], frame=torch.tensor(meta['frame']), floor=torch.from_numpy(np.stack(meta['floor'])),
                    gravity=torch.cat(grav), chunks=sorted(os.listdir(cdir))), f'features/{tag}.pt')
    print(f'saved features/{tag}.pt')


@torch.no_grad()
def distance_sense(name, da3=False):
    """Does a route model judge the remaining distance from the view? On the held-out TartanDrive routes' views (all frames, every STEP), its predicted end distance against the true remaining distance, with no flag."""
    setup()
    ex = torch.load('features/tartan_held_views.pt')
    parts = [torch.load(f'features/tartan_held_views_qwen2_chunks/{c}') for c in ex['chunks']]
    q = dict(q_cells=torch.cat([p['cells'] for p in parts]), q_last=torch.cat([p['last'] for p in parts]),
             q_text=torch.cat([p['text_mean'] for p in parts]))
    model = RouteNet(q['q_cells'].shape[-1], q['q_last'].shape[-1]).to('cuda')
    model.load_state_dict(torch.load(f'runs/{name}/model.pt'), strict=False)
    model.eval()
    cam = pathnet.TABLE[GROUPS.index('tartan')].expand(len(ex['route']), -1)
    preds = []
    for i in range(0, len(ex['route']), 512):
        sl = slice(i, i + 512)
        down = ex['gravity'][sl].float()
        b = dict(q_cells=q['q_cells'][sl].cuda().float(), q_last=q['q_last'][sl].cuda().float(),
                 q_text=q['q_text'][sl].cuda().float(), cam=cam[sl].cuda(), down=down.cuda(),
                 camera=torch.cat([cam[sl][:, 11:13], cam[sl][:, :1], down], 1).cuda(),
                 x_uv=torch.zeros(len(down), 2, device='cuda'), x_floor=torch.zeros(len(down), 2, device='cuda'))
        if da3:
            dd = torch.load('features/tartan_held_views_da3.pt')['da3'][sl].float()
            b['da3'] = torch.cat([dd, torch.ones(len(dd), 64, 1)], -1).cuda()
        preds.append(forward_route(model, b).float()[:, -1].cpu())
    pred = torch.cat(preds).norm(dim=-1).numpy()
    true = ex['floor'][:, -1].norm(dim=-1).numpy()
    frame = ex['frame'].numpy()
    from scipy.stats import spearmanr
    print(f'{len(true)} views from held-out routes; rank correlation of predicted vs true remaining distance: '
          f'{spearmanr(pred, true).correlation:+.2f}')
    print('| views at frame | true remaining (m), median | predicted (m), median |')
    print('|---|---|---|')
    for lo, hi in ((2, 3), (4, 5), (6, 7), (8, 9), (10, 99)):
        m = (frame >= lo) & (frame <= hi)
        if m.any():
            print(f'| {lo}-{hi if hi < 99 else "more"} | {np.median(true[m]):.1f} | {np.median(pred[m]):.1f} |')


def add_da3(f, split_eps):
    """Per-episode DA3 cells for val_unseen (per trajectory in da3_tartan.py), has_da3 = 1."""
    d = torch.load('features/val_unseen_da3.pt')
    row = {t: i for i, t in enumerate(d['trajectories'])}
    f['da3'] = d['da3'][torch.tensor([row[e['trajectory']] for e in split_eps])]
    f['has_da3'] = torch.ones(len(split_eps))


def train(name, use_views, epochs=12, bs=256, lr=3e-4, da3=False):
    setup()
    dev, steps, out = 'cuda', pathnet.typical_steps(), f'runs/{name}'
    os.makedirs(out, exist_ok=True)
    held = split_routes()
    vu_eps, vu = load('data/lcvn/val_unseen.jsonl'), load_labelled('val_unseen')
    ti = [i for i, e in enumerate(vu_eps) if recording(e['trajectory']) not in held]
    hi = [i for i, e in enumerate(vu_eps) if recording(e['trajectory']) in held]
    if da3:
        add_da3(vu, vu_eps)
        trn = load_labelled('train')
        trn['da3'] = torch.zeros(len(trn['episodes']), 64, 2, dtype=torch.float16)
        trn['has_da3'] = torch.zeros(len(trn['episodes']))
        tr = merge([trn, subset(vu, ti)])
    else:
        tr = merge([load_labelled('train'), subset(vu, ti)])
    vals = {'tartan_held': ([vu_eps[i] for i in hi], subset(vu, hi))}
    if not da3:                                          # with depth only TartanDrive has it: score TartanDrive only
        vals['val_seen'] = (load('data/lcvn/val_seen.jsonl'), load_labelled('val_seen'))
    print(f'TartanDrive: {len(ti)} training / {len(hi)} held-out episodes', flush=True)
    if use_views:
        ex = torch.load('features/tartan_views.pt')
        parts = [torch.load(f'features/tartan_views_qwen2_chunks/{c}') for c in ex['chunks']]
        ex.update(q_cells=torch.cat([p['cells'] for p in parts]), q_last=torch.cat([p['last'] for p in parts]),
                  q_text=torch.cat([p['text_mean'] for p in parts]))
        cam_t = pathnet.TABLE[GROUPS.index('tartan')]
        ex_da3 = torch.load('features/tartan_views_da3.pt')['da3'] if da3 else None
        print(f'extra views: {len(ex["route"])}', flush=True)
    keep = torch.where(tr['visible'][:, -1])[0]
    n = len(keep)
    model = RouteNet(tr['q_cells'].shape[-1], tr['q_last'].shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * (n // bs), pct_start=0.05)
    log = []
    for ep in range(epochs):
        t0, perm, tot = time.time(), keep[torch.randperm(n)], 0.0
        for k in range(n // bs):
            idx = perm[k * bs:(k + 1) * bs]
            loss = score_loss(forward_route(model, make(tr, idx, dev)).float(), tr['floor'][idx].to(dev))
            if use_views:                                # end point of the remaining route, relative to its size
                e_i = torch.randint(0, len(ex['route']), (bs // 4,))
                down = ex['gravity'][e_i].float()
                cam = cam_t.expand(len(e_i), -1)
                b = dict(q_cells=ex['q_cells'][e_i].to(dev).float(), q_last=ex['q_last'][e_i].to(dev).float(),
                         q_text=ex['q_text'][e_i].to(dev).float(), cam=cam.to(dev), down=down.to(dev),
                         camera=torch.cat([cam[:, 11:13], cam[:, :1], down], 1).to(dev),
                         x_uv=(torch.zeros(len(e_i), 2, device=dev) if NO_FLAG else
                               torch.tensor([[1.0, 0.0]], device=dev).expand(len(e_i), -1)),   # flag: a later view
                         x_floor=torch.zeros(len(e_i), 2, device=dev))
                if da3:                                  # the view's own depth cells
                    dd = ex_da3[e_i].float()
                    b['da3'] = torch.cat([dd, torch.ones(len(dd), 64, 1)], -1).to(dev)
                end_pred = forward_route(model, b).float()[:, -1]
                et = ex['floor'][e_i].to(dev)
                loss = loss + ((end_pred - et[:, -1]).norm(dim=-1) / et.norm(dim=-1).max(1).values.clamp(min=0.5)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() / (n // bs)
        row = dict(epoch=ep + 1, sec=round(time.time() - t0), train_loss=round(tot, 4))
        if (ep + 1) % 4 == 0:
            for s, (eps, f) in vals.items():
                r = routes_of(model, f, dev)
                m = score(eps, dict(zip([e['episode'] for e in eps], to_moves(r, f['group'], steps))))[0]
                row[s] = dict(score=round(m['score'], 2), F=round(m['F'], 3), SR=round(m['SR'], 3))
            torch.save(model.state_dict(), f'{out}/model.pt')
        log.append(row)
        print(json.dumps(row), flush=True)
        json.dump(dict(epochs=log), open(f'{out}/log.json', 'w'), indent=1)


@torch.no_grad()
def distance_sense_x(name):
    """Same test for stage 1 (xnet.py): its X on each held-out TartanDrive view."""
    from endnet import EndNet, drop
    from routenet import lift
    setup()
    ex = torch.load('features/tartan_held_views.pt')
    parts = [torch.load(f'features/tartan_held_views_qwen2_chunks/{c}') for c in ex['chunks']]
    q = dict(q_cells=torch.cat([p['cells'] for p in parts]), q_last=torch.cat([p['last'] for p in parts]),
             q_text=torch.cat([p['text_mean'] for p in parts]))
    model = EndNet(False, 1, q['q_cells'].shape[-1], q['q_last'].shape[-1]).to('cuda')
    model.load_state_dict(torch.load(f'runs/{name}/model.pt'), strict=False)
    model.eval()
    n = len(ex['route'])
    cam = pathnet.TABLE[GROUPS.index('tartan')].expand(n, -1)
    S = 6
    dists = []
    for i in range(0, n, 512):
        sl = slice(i, min(i + 512, n))
        m = sl.stop - sl.start
        down = ex['gravity'][sl].float()
        b = dict(q_cells=q['q_cells'][sl].cuda().float(), q_last=q['q_last'][sl].cuda().float(),
                 q_text=q['q_text'][sl].cuda().float(), cam=cam[sl].cuda(), down=down.cuda(),
                 camera=torch.cat([cam[sl][:, 11:13], cam[sl][:, :1], down], 1).cuda(),
                 sentence=torch.zeros(m, 768, device='cuda'), sig_tokens=torch.zeros(m, 1, device='cuda'),
                 seg_move=torch.zeros(m, S, dtype=torch.long, device='cuda'), seg_side=torch.zeros(m, S, dtype=torch.long, device='cuda'),
                 seg_degree=torch.zeros(m, S, dtype=torch.long, device='cuda'), seg_phrase=torch.zeros(m, S + 1, 768, device='cuda'),
                 seg_map=torch.zeros(m, S + 1, 256, device='cuda'), has_phrase=torch.zeros(m, S + 1, dtype=torch.bool, device='cuda'))
        uv = model(b).float()
        dists.append(lift(uv, b).norm(dim=-1).cpu())
    pred = torch.cat(dists).numpy()
    true = ex['floor'][:, -1].norm(dim=-1).numpy()
    frame = ex['frame'].numpy()
    from scipy.stats import spearmanr
    print(f'{n} held-out views; rank correlation of the X\'s floor distance vs true remaining distance: '
          f'{spearmanr(pred, true).correlation:+.2f}')
    print('| views at frame | true remaining (m), median | X lifted to the floor (m), median |')
    print('|---|---|---|')
    for lo, hi in ((2, 3), (4, 5), (6, 7), (8, 9), (10, 99)):
        mk = (frame >= lo) & (frame <= hi)
        if mk.any():
            print(f'| {lo}-{hi if hi < 99 else "more"} | {np.median(true[mk]):.1f} | {np.median(pred[mk]):.1f} |')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['views', 'train', 'held_views', 'sense', 'sense_x'])
    ap.add_argument('--name')
    ap.add_argument('--views', action='store_true')
    ap.add_argument('--da3', action='store_true', help='single-photo DA3 depth cells as input (TartanDrive)')
    ap.add_argument('--no-flag', action='store_true', help='extra views carry no later-view flag')
    a = ap.parse_args()
    if a.mode == 'views':
        views()
    elif a.mode == 'held_views':
        views(held_out=True)
    elif a.mode == 'sense':
        distance_sense(a.name, a.da3)
    elif a.mode == 'sense_x':
        distance_sense_x(a.name)
    else:
        train(a.name, a.views, da3=a.da3)
