"""Path-drawing model: draws the route on the first photo.

    python 3_models/pathnet.py train --name p1 [--epochs 20]
"""
import argparse
import json
import math
import os
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, 'eval')
from floor_geometry import GROUPS, NATIVE_SIZE, axes, group_of, lift_path, load_cameras, ray, with_gravity  # noqa: E402
from score import load, score  # noqa: E402
from path_labels import K  # noqa: E402

S = 6
GRAVITY = False     # each photo's own lean (GeoCalib) for labels and lifting
CAM_INPUT = False   # the model gets the camera numbers
METRE_LOSS = 0.0    # weight of the loss on the drawn points lifted to the floor
OWN_LENGTH = False  # route length = the drawing's own lifted length


def camera_table():
    """Per group: lens (0 pinhole, 1 fisheye), fx, fy, cx, cy, k1-k4 (pinhole: 0), native W, H, height."""
    cams, rows = load_cameras(), []
    for g in GROUPS:
        c = cams[g]
        W, H = NATIVE_SIZE[g]
        lens = [0.0, c['fx'], c['fy'], c['cx'], c['cy'], 0, 0, 0, 0] if c['lens'] == 'pinhole' else [1.0] + list(c['params'])
        x, _, z = ray(W, lens[4], c)
        rows.append(lens + [W, H, c['height'], math.atan2(x, z)] + list(axes(c)[0]))
    return torch.tensor(rows, dtype=torch.float32)


def lift_torch(uv, cam, down, with_floor=False):
    """Normalised pixels (B, K, 2) -> floor (forward, right) metres (B, K, 2), differentiable."""
    lens, fx, fy, cx, cy = (cam[:, i, None] for i in range(5))
    k1, k2, k3, k4 = (cam[:, 5 + i, None] for i in range(4))
    W, H, h = cam[:, 9, None], cam[:, 10, None], cam[:, 11, None]
    mx, my = (uv[..., 0] * W - cx) / fx, (uv[..., 1] * H - cy) / fy
    rd = torch.sqrt(mx ** 2 + my ** 2 + 1e-12)
    th = rd.clamp(max=1.5)
    for _ in range(8):                              # Kannala-Brandt: solve theta (1 + k1 theta^2 + ...) = rd
        g = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8) - rd
        dg = 1 + 3 * k1 * th ** 2 + 5 * k2 * th ** 4 + 7 * k3 * th ** 6 + 9 * k4 * th ** 8
        th = (th - g / dg.clamp(min=0.1)).clamp(0, 1.8)
    s = torch.sin(th) / rd
    fish = torch.stack([mx * s, my * s, torch.cos(th)], -1)
    pin = torch.stack([mx, my, torch.ones_like(mx)], -1)
    pin = pin / pin.norm(dim=-1, keepdim=True)
    d = torch.where(lens[..., None] > 0.5, fish, pin)
    down = down / down.norm(dim=-1, keepdim=True)
    z = torch.tensor([0.0, 0.0, 1.0], device=uv.device).expand_as(down)
    fwd = z - down[:, 2:3] * down
    fwd = fwd / fwd.norm(dim=-1, keepdim=True)
    right = torch.cross(down, fwd, dim=-1)
    raw = (d * down[:, None]).sum(-1)
    dd = raw.clamp(min=h / 40)
    k = h / dd
    out = torch.stack([k * (d * fwd[:, None]).sum(-1), k * (d * right[:, None]).sum(-1)], -1)
    return (out, raw > h / 40) if with_floor else out               # with_floor: rays that meet the floor within 40 m


class PathNet(nn.Module):
    def __init__(self, d=256, layers=4, heads=8):
        super().__init__()
        self.sig, self.dino = nn.Linear(768, d), nn.Linear(1024, d)
        self.sig_pool, self.dino_cls, self.sent = nn.Linear(768, d), nn.Linear(1024, d), nn.Linear(768, d)
        self.cell_pos = nn.Parameter(torch.zeros(1, 64, d))
        self.move, self.side, self.degree = nn.Embedding(7, d), nn.Embedding(3, d), nn.Embedding(3, d)
        self.slot = nn.Parameter(torch.zeros(1, S + 1, d))
        self.phrase, self.pmap = nn.Linear(768, d), nn.Linear(256, d)
        self.group = nn.Embedding(len(GROUPS), d)
        self.camera = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.norm_in = nn.LayerNorm(d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.queries = nn.Parameter(torch.randn(1, K + 1, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, 2)
        self.point = nn.Linear(d, 3)                     # u, v (normalised), visible logit
        self.length = nn.Linear(d, 1)                    # log metres

    def forward(self, b):
        B = len(b['sig_tokens'])
        cells = torch.cat([self.sig(b['sig_tokens']) + self.cell_pos, self.dino(b['dino_tokens']) + self.cell_pos], 1)
        glob = torch.stack([self.sig_pool(b['sig_pooled']), self.dino_cls(b['dino_cls']), self.sent(b['sentence']),
                            self.camera(b['camera']) if CAM_INPUT else self.group(b['group'])], 1)
        has = b['has_phrase'].unsqueeze(-1).float()
        segs = self.slot + (self.phrase(b['seg_phrase']) + self.pmap(b['seg_map'])) * has
        moves = self.move(b['seg_move']) + self.side(b['seg_side']) + self.degree(b['seg_degree'])
        segs = segs + torch.cat([moves, torch.zeros_like(moves[:, :1])], 1)
        x = self.norm_in(torch.cat([cells, glob, segs], 1))
        pad = torch.cat([torch.zeros(B, 128 + 4, dtype=torch.bool, device=x.device),
                         b['seg_move'] == 0, torch.zeros(B, 1, dtype=torch.bool, device=x.device)], 1)
        mem = self.encoder(x, src_key_padding_mask=pad)
        q = self.decoder(self.queries.expand(B, -1, -1), mem, memory_key_padding_mask=pad)
        p = self.point(q[:, :K])
        return p[..., :2] + 0.5, p[..., 2], self.length(q[:, K]).squeeze(-1)


FEATURES = 'v2'     # features file version: v2 (features_v2.py), v3
SHIFT = False       # training on Go Stanford real's photos that match their routes' first record (files tagged _shift)


def load_split(split, with_labels=True):
    tag = '_shift' if SHIFT and split == 'train' else ''        # validation and test stay on photo 0
    path = f'features/{split}_{FEATURES}{tag}.pt'
    if os.path.exists(path):
        f = torch.load(path)
    else:                                  # stored once (dedupe_features.py)
        f = torch.load(f'features/{split}_{FEATURES}_text.pt')
        f.update(torch.load(f'features/{split}_img.pt'))
    f['gravity'] = torch.load(f'features/{split}_gravity{tag}.pt')['gravity'] if GRAVITY else None
    if with_labels:
        lab = torch.load(f'features/{split}_paths{"_grav" if GRAVITY else ""}{tag}.pt')
        assert lab['episodes'] == f['episodes']
        f.update(uv=lab['uv'], visible=lab['visible'], length=lab['length'], group=lab['group'], floor=lab['floor'])
    return f


def batch(f, idx, dev):
    rows = f['image_index'][idx]
    out = {k: f[k][rows].to(dev).float() for k in ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls')}
    for k in ('sentence', 'seg_phrase', 'seg_map'):
        out[k] = f[k][idx].to(dev).float()
    for k in ('seg_move', 'seg_side', 'seg_degree'):
        out[k] = f[k][idx].to(dev).long()
    out['has_phrase'] = f['has_phrase'][idx].to(dev)
    if 'stop_kind' in f:
        out['stop_kind'] = f['stop_kind'][idx].to(dev).long()
    out['group'] = f['group'][idx].to(dev).long()
    cam = TABLE[out['group'].cpu()]
    down = f['gravity'][idx].float() if f['gravity'] is not None else cam[:, 13:16]
    out['cam'], out['down'] = cam.to(dev), down.to(dev)
    out['camera'] = torch.cat([cam[:, 11:13], cam[:, :1], down], 1).to(dev)    # height, half-width angle, fisheye, down
    return out


def loss_fn(pred, f, idx, b, dev):
    uv, vis_logit, log_len = pred
    t_uv, t_vis = f['uv'][idx].to(dev), f['visible'][idx].to(dev).float()
    t_len = f['length'][idx].to(dev).clamp_min(0.1).log()
    l_uv = ((uv - t_uv).abs().sum(-1) * t_vis).sum() / t_vis.sum().clamp_min(1)
    l_vis = F.binary_cross_entropy_with_logits(vis_logit, t_vis)
    l_len = torch.zeros((), device=dev) if OWN_LENGTH else (log_len - t_len).abs().mean()
    loss, info = l_uv + 0.5 * l_vis + 0.5 * l_len, dict(uv=l_uv.item(), vis=l_vis.item(), len=l_len.item())
    if METRE_LOSS:
        floor = lift_torch(uv, b['cam'], b['down'])
        err = F.smooth_l1_loss(floor, f['floor'][idx].to(dev), reduction='none', beta=1.0).sum(-1)
        l_m = (err * t_vis).sum() / t_vis.sum().clamp_min(1)
        loss, info['metres'] = loss + METRE_LOSS * l_m, l_m.item()
    return loss, info


@torch.no_grad()
def predict(model, f, dev, bs=1024):
    model.eval()
    outs = []
    for i in range(0, len(f['episodes']), bs):
        idx = torch.arange(i, min(i + bs, len(f['episodes'])))
        uv, vis, ll = model(batch(f, idx, dev))
        outs.append((uv.cpu(), vis.sigmoid().cpu(), ll.exp().cpu()))
    model.train()
    return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs]), torch.cat([o[2] for o in outs])


def subset(eps, f, idx):
    """Some episodes of a split, with their features."""
    t = torch.tensor(idx, dtype=torch.long)
    sub = {k: (v[t] if torch.is_tensor(v) and len(v) == len(eps) and k != 'image_index' else v) for k, v in f.items()}
    sub['image_index'] = f['image_index'][t]
    sub['episodes'] = [f['episodes'][i] for i in idx]
    return [eps[i] for i in idx], sub


def typical_steps():
    """Median step length per camera group, from the training routes (TartanDrive: val_unseen)."""
    steps = {}
    for split in ('train', 'val_unseen'):
        for line in open(f'data/lcvn/{split}.jsonl'):
            e = json.loads(line)
            if e['episode'].endswith('_0'):
                p = np.array(e['position'])
                steps.setdefault(group_of(e['trajectory'], split), []).append(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())
    return {g: float(np.median(v)) for g, v in steps.items()}


def to_moves(uv, vis, length, grp, cams, steps, gravity=None):
    """One episode's prediction -> [dx, dy, dyaw] moves."""
    from manual_path_score import moves_along
    from rules import exact
    W, H = NATIVE_SIZE[grp]
    keep = vis > 0.5
    if keep.sum() < 2:
        keep[:2] = True
    pix = [(float(u) * W, float(v) * H) for (u, v), k in zip(uv.tolist(), keep.tolist()) if k]
    path = lift_path(pix, with_gravity(cams[grp], gravity), max_jump=1e9)
    far = np.linalg.norm(path, axis=1) > 40                       # points at the horizon: drop
    path = path[~far] if (~far).sum() >= 2 else np.array([(0.0, 0.0), (float(length), 0.0)])
    if OWN_LENGTH and length is not None and not TRUE_LENGTH:
        return moves_along(path, steps[grp])
    return moves_along(exact(path, float(length)), steps[grp])


TRUE_LENGTH = False


def evaluate(model, splits, cams, steps, dev, true_length=False):
    global TRUE_LENGTH
    TRUE_LENGTH = true_length
    rows = {}
    for split, (eps, f) in splits.items():
        uv, vis, length = predict(model, f, dev)
        if true_length:
            length = f['length']
        preds = {}
        for i, e in enumerate(eps):
            preds[e['episode']] = to_moves(uv[i], vis[i], length[i], GROUPS[int(f['group'][i])], cams, steps,
                                    None if f['gravity'] is None else f['gravity'][i])
        means, _ = score(eps, preds)
        rows[split] = {k: round(v, 3) for k, v in means.items()}
    TRUE_LENGTH = False
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'submit'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--gravity', action='store_true')
    ap.add_argument('--cam-input', action='store_true')
    ap.add_argument('--metre-loss', type=float, default=0.0)
    ap.add_argument('--own-length', action='store_true')
    ap.add_argument('--shift', action='store_true', help='train on Go Stanford real photos matching the route start')
    ap.add_argument('--exclude', nargs='*', default=[], help='camera groups left out of training')
    ap.add_argument('--only', nargs='*', default=[], help='train on, and choose the checkpoint by, these groups only')
    ap.add_argument('--init', help='start from this saved model (fine-tuning)')
    args = ap.parse_args()
    global GRAVITY, CAM_INPUT, METRE_LOSS, OWN_LENGTH, SHIFT
    GRAVITY, CAM_INPUT, METRE_LOSS, OWN_LENGTH = args.gravity, args.cam_input, args.metre_loss, args.own_length
    SHIFT = args.shift
    dev, out = 'cuda', f'runs/{args.name}'
    os.makedirs(out, exist_ok=True)
    cams, steps = load_cameras(), typical_steps()
    model = PathNet().to(dev)
    vals = {s: (load(f'data/lcvn/{s}.jsonl'), load_split(s)) for s in ('val_seen', 'val_unseen')}
    if args.only:                                  # validate on the chosen groups' episodes only
        vals = {s: subset(eps, f, [i for i in range(len(eps)) if GROUPS[int(f['group'][i])] in args.only])
                for s, (eps, f) in vals.items()}
        vals = {s: v for s, v in vals.items() if v[0]}
    if args.init:
        model.load_state_dict(torch.load(args.init))
    if args.mode == 'submit':
        model.load_state_dict(torch.load(f'{out}/model.pt'))
        return submit(model, vals, cams, steps, dev, out)
    tr = load_split('train')
    keep = torch.tensor([GROUPS[int(g)] not in args.exclude and (not args.only or GROUPS[int(g)] in args.only)
                         for g in tr['group']])
    allowed = torch.where(keep)[0]                       # training episodes used (--exclude drops camera groups)
    n = len(allowed)
    print(f'training on {n}/{len(keep)} episodes', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * (n // args.bs), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    log, best = [], -1.0
    for ep in range(args.epochs):
        t0, perm, sums = time.time(), allowed[torch.randperm(n)], {}
        for k in range(n // args.bs):
            idx = perm[k * args.bs:(k + 1) * args.bs]
            with torch.autocast('cuda', dtype=torch.float16):
                b = batch(tr, idx, dev)
                pred = model(b)
            loss, info = loss_fn([p.float() for p in pred], tr, idx, b, dev)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for a, v in info.items():
                sums[a] = sums.get(a, 0) + v / (n // args.bs)
        row = dict(epoch=ep + 1, sec=round(time.time() - t0), **{f'train_{a}': round(v, 4) for a, v in sums.items()})
        if (ep + 1) % 4 == 0 or ep + 1 == args.epochs:
            row.update(evaluate(model, vals, cams, steps, dev))
            row['phase1'] = round(float(np.mean([row[v]['score'] for v in vals])), 3)
            if row['phase1'] > best:
                best = row['phase1']
                torch.save(model.state_dict(), f'{out}/model.pt')
        log.append(row)
        print(json.dumps(row), flush=True)
        json.dump(dict(args=vars(args), epochs=log), open(f'{out}/log.json', 'w'), indent=1)
    model.load_state_dict(torch.load(f'{out}/model.pt'))
    print('true length (shape only):', json.dumps(evaluate(model, vals, cams, steps, dev, true_length=True)), flush=True)


def submit(model, vals, cams, steps, dev, out):
    from group_classifier import predict_groups
    docs = {}
    for split, (eps, f) in vals.items():
        uv, vis, length = predict(model, f, dev)
        docs[split] = {e['episode']: to_moves(uv[i], vis[i], length[i], GROUPS[int(f['group'][i])], cams, steps,
                                    None if f['gravity'] is None else f['gravity'][i])
                       for i, e in enumerate(eps)}
    test = load_split('test', with_labels=False)
    test['group'] = predict_groups(test)
    uv, vis, length = predict(model, test, dev)
    docs['test'] = {ep: to_moves(uv[i], vis[i], length[i], GROUPS[int(test['group'][i])], cams, steps,
                                 None if test['gravity'] is None else test['gravity'][i])
                    for i, ep in enumerate(test['episodes'])}
    for name, splits in (('submission_val.zip', ('val_seen', 'val_unseen')), ('submission_test.zip', ('test',))):
        with zipfile.ZipFile(f'{out}/{name}', 'w', zipfile.ZIP_DEFLATED) as z:
            for s in splits:
                z.writestr(f'{s}.json', json.dumps({'split': s, 'episodes': [{'episode_id': k, 'actions': a}
                                                                            for k, a in docs[s].items()]}))
    print('saved', out, {s: len(d) for s, d in docs.items()})


TABLE = camera_table()

if __name__ == '__main__':
    main()
