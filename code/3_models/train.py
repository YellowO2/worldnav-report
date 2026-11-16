"""First model: frozen-feature transformer that proposes K candidate routes and picks one.

    python 3_models/train.py --name v1 [--epochs 30]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, 'eval')
from score import load, score  # noqa: E402

T = 30  # longest route in the data, in steps


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def targets(episodes):
    """Waypoints in the start frame (start pose = origin, facing +x), padded by holding the last pose."""
    xy = torch.zeros(len(episodes), T, 2)
    yaw = torch.zeros(len(episodes), T)
    n = torch.zeros(len(episodes), dtype=torch.long)
    for i, e in enumerate(episodes):
        p = torch.tensor(e['position'], dtype=torch.float64)
        y = torch.tensor(e['yaw'], dtype=torch.float64)
        c, s = math.cos(y[0]), math.sin(y[0])
        d = p[1:] - p[0]
        local = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)
        k = min(len(local), T)
        n[i] = k
        xy[i, :k], yaw[i, :k] = local[:k].float(), torch.tensor([wrap(v - y[0]) for v in y[1:k + 1]])
        xy[i, k:], yaw[i, k:] = xy[i, k - 1], yaw[i, k - 1]
    return xy, yaw, n


def to_actions(xy, yaw, n):
    """Start-frame waypoints and headings -> [dx, dy, dyaw] in the agent's current frame."""
    pts = [(0.0, 0.0)] + [tuple(p) for p in xy[:n].tolist()]
    hs = [0.0] + yaw[:n].tolist()
    acts = []
    for t in range(n):
        dX, dY = pts[t + 1][0] - pts[t][0], pts[t + 1][1] - pts[t][1]
        c, s = math.cos(hs[t]), math.sin(hs[t])
        acts.append([c * dX + s * dY, -s * dX + c * dY, wrap(hs[t + 1] - hs[t])])
    return acts


class Model(nn.Module):
    """Tokens: 64 SigLIP2 image cells, 64 DA3 cells (each with its 2x2 block of depth), the SigLIP2 image."""

    def __init__(self, d=384, layers=4, k=1, ground=False):
        super().__init__()
        self.k = k
        # SigLIP2's own sentence-to-patch similarity (ground_maps.py): 3 sentences
        self.ground = nn.Linear(3, d) if ground else None
        self.img = nn.Linear(768, d)
        self.geo = nn.Linear(1024, d)
        self.depth = nn.Linear(4, d)
        self.img_pool = nn.Linear(768, d)
        self.geo_cls = nn.Linear(1024, d)
        self.txt = nn.Linear(768, d)
        self.txt_pool = nn.Linear(768, d)
        self.pos = nn.Parameter(torch.zeros(1, 64 + 64 + 3 + 48, d))
        enc = nn.TransformerEncoderLayer(d, 6, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.queries = nn.Parameter(torch.randn(1, k, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, 6, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, 2)
        self.route = nn.Linear(d, T * 4)   # per step: dx, dy (increments, so routes stay smooth), sin
        self.stop = nn.Linear(d, T)        # which step the route ends on
        self.score = nn.Linear(d, 1)

    def forward(self, img_tok, img_pool, geo_tok, geo_cls, depth, txt_tok, txt_mask, txt_pool, sims=None, valid=None):
        cells = torch.log(depth.clamp_min(1e-3)).view(-1, 8, 2, 8, 2).permute(0, 1, 3, 2, 4).reshape(-1, 64, 4)
        img = self.img(img_tok)
        if self.ground is not None:
            g = F.max_pool2d(sims.view(-1, 3, 16, 16), 2).flatten(2)                  # (B, 3, 64)
            g = (g - g.mean(-1, keepdim=True)) / (g.std(-1, keepdim=True) + 1e-4) * valid[..., None]
            img = img + self.ground(g.transpose(1, 2))
        x = torch.cat([img, self.geo(geo_tok) + self.depth(cells), self.img_pool(img_pool)[:, None],
                       self.geo_cls(geo_cls)[:, None], self.txt_pool(txt_pool)[:, None], self.txt(txt_tok)], 1)
        x = x + self.pos
        pad = torch.cat([torch.zeros(len(x), 131, dtype=torch.bool, device=x.device), ~txt_mask], 1)
        memory = self.encoder(x, src_key_padding_mask=pad)
        q = self.decoder(self.queries.expand(len(x), -1, -1), memory, memory_key_padding_mask=pad)
        r = self.route(q).view(len(x), self.k, T, 4)
        xy = torch.cumsum(r[..., :2] * 0.3, dim=2)          # 0.3 ~ a typical step
        yaw = torch.atan2(r[..., 2], r[..., 3] + 1.0)        # starts near 0 (facing ahead)
        return xy, yaw, self.stop(q), self.score(q).squeeze(-1)


def load_features(name, ground=False):
    feat = torch.load(f'features/{name}.pt')
    if ground:
        feat.update(torch.load(f'features/{name}_ground.pt'))
    return feat


def batch_inputs(feat, idx, dev):
    rows = feat['image_index'][idx]
    f = lambda key, i: feat[key][i].to(dev).float()
    out = (f('image_tokens', rows), f('image_pooled', rows), f('geo_tokens', rows), f('geo_cls', rows),
           f('depth', rows), f('text_tokens', idx), feat['text_mask'][idx].to(dev), f('text_pooled', idx))
    if 'sims' in feat:
        out = out + (f('sims', idx), feat['valid'][idx].to(dev).float())
    return out


def loss_fn(out, xy, yaw, n):
    pxy, pyaw, stop_logits, scores = out
    steps = torch.arange(T, device=xy.device)[None, :]
    valid = (steps < n[:, None]).float()                                 # (B, T)
    err = (pxy - xy[:, None]).norm(dim=-1)                               # (B, K, T)
    ade = (err * valid[:, None]).sum(-1) / n[:, None]
    best = ade.argmin(1)
    pick = lambda t: t[torch.arange(len(t)), best]
    # only the route's real steps count, as in NoMaD's action mask
    reg = (pick(err) * valid).sum() / valid.sum()
    head = (1 - torch.cos(pick(pyaw) - yaw)).mul(valid).sum(-1).div(n).mean()
    stop = F.cross_entropy(pick(stop_logits), n - 1)
    choose = F.cross_entropy(scores, best)
    return reg + head + 0.5 * stop + 0.5 * choose, dict(ade=pick(ade).mean().item())


@torch.no_grad()
def eval_loss(model, feat, xy, yaw, n, dev, batch=1024):
    """Mean training loss and best-candidate ADE on a validation split."""
    model.eval()
    total, ade, count = 0.0, 0.0, len(feat['episodes'])
    for i in range(0, count, batch):
        idx = torch.arange(i, min(i + batch, count))
        out = model(*batch_inputs(feat, idx, dev))
        loss, info = loss_fn([o.float() for o in out], xy[idx.to(dev)], yaw[idx.to(dev)], n[idx.to(dev)])
        total += loss.item() * len(idx)
        ade += info['ade'] * len(idx)
    model.train()
    return total / count, ade / count


def expected_length(xy, stop_probs):
    """Route length that maximises expected full-route fidelity."""
    pts = torch.cat([torch.zeros_like(xy[:, :1]), xy], 1)                      # (B, T+1, 2), start included
    arc = torch.cat([torch.zeros_like(xy[:, :1, 0]), (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).cumsum(1)], 1)[:, 1:]
    span = pts.norm(dim=-1).cummax(1).values[:, 1:].clamp_min(1e-3)           # (B, T): span of the route up to n
    gap = (xy[:, :, None] - xy[:, None, :]).norm(dim=-1)                       # (B, L, n) end-point distance
    over = (arc[:, None, :] / arc[:, :, None].clamp_min(1e-3)).clamp(max=1.0)  # ref length / pred length, capped at 1
    f = over * (1 - gap / span[:, None, :]).clamp_min(0)                       # (B, L, n)
    return (f * stop_probs[:, None, :]).sum(-1).argmax(-1) + 1


@torch.no_grad()
def predict(model, feat, dev, batch=1024, length='expected'):
    model.eval()
    preds = {}
    for i in range(0, len(feat['episodes']), batch):
        idx = torch.arange(i, min(i + batch, len(feat['episodes'])))
        pxy, pyaw, stop_logits, scores = model(*batch_inputs(feat, idx, dev))
        best = scores.argmax(1)
        ar = torch.arange(len(idx))
        xy, yaw, logits = pxy[ar, best], pyaw[ar, best].cpu(), stop_logits[ar, best]
        n = (expected_length(xy, logits.softmax(-1)) if length == 'expected' else logits.argmax(-1) + 1).cpu()
        xy = xy.cpu()
        for j, e in enumerate(idx.tolist()):
            preds[feat['episodes'][e]] = to_actions(xy[j], yaw[j], int(n[j]))
    model.train()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--train_split', default='train', help='val_seen for a quick smoke test')
    ap.add_argument('--flip', action='store_true', help='add the mirrored copy (features/{split}_flip.pt)')
    ap.add_argument('--ground', action='store_true', help='add SigLIP2 grounding maps (features/{split}_ground.pt)')
    args = ap.parse_args()
    dev = 'cuda'
    out_dir = f'runs/{args.name}'
    os.makedirs(out_dir, exist_ok=True)

    train_eps = load(f'data/lcvn/{args.train_split}.jsonl')
    feat = load_features(args.train_split, args.ground)
    assert feat['episodes'] == [e['episode'] for e in train_eps]
    xy, yaw, n = targets(train_eps)
    if args.flip:   # mirrored photo + left/right-swapped text
        flip = load_features(f'{args.train_split}_flip', args.ground)
        offset = len(feat['image_pooled'])
        feat = {k: v + flip[k] if k == 'episodes' else torch.cat([v, flip[k] + offset if k == 'image_index' else flip[k]])
                for k, v in feat.items()}   # (adding 0 to the bool masks would turn them into integers)
        xy, yaw, n = torch.cat([xy, xy * torch.tensor([1.0, -1.0])]), torch.cat([yaw, -yaw]), torch.cat([n, n])
        train_eps = train_eps + train_eps
    xy, yaw, n = xy.to(dev), yaw.to(dev), n.to(dev)
    vals = {s: (load(f'data/lcvn/{s}.jsonl'), load_features(s, args.ground)) for s in ('val_seen', 'val_unseen')}
    val_targets = {s: tuple(t.to(dev) for t in targets(eps)) for s, (eps, _) in vals.items()}

    model = Model(k=args.k, ground=args.ground).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(train_eps) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    log, best_score, best_split = [], -1.0, {}
    for epoch in range(args.epochs):
        t0, perm = time.time(), torch.randperm(len(train_eps))
        sums = {'loss': 0.0, 'ade': 0.0}
        for b in range(len(train_eps) // args.batch):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(*batch_inputs(feat, idx, dev))
            loss, info = loss_fn([o.float() for o in out], xy[idx.to(dev)], yaw[idx.to(dev)], n[idx.to(dev)])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums['loss'] += loss.item()
            sums['ade'] += info['ade']
        nb = len(train_eps) // args.batch
        row = dict(epoch=epoch + 1, train_loss=round(sums['loss'] / nb, 4), train_ade=round(sums['ade'] / nb, 4),
                   lr=sched.get_last_lr()[0], sec=round(time.time() - t0))
        for s, (eps, f) in vals.items():
            row[f'{s}_loss'], row[f'{s}_ade'] = (round(v, 4) for v in eval_loss(model, f, *val_targets[s], dev))
        if (epoch + 1) % 2 == 0 or epoch + 1 == args.epochs:
            for s, (eps, f) in vals.items():
                means, _ = score(eps, predict(model, f, dev), workers=8)
                row[s] = {k: round(v, 3) for k, v in means.items()}
            row['phase1'] = round((row['val_seen']['score'] + row['val_unseen']['score']) / 2, 3)  # how CodaBench ranks Phase 1
            if row['phase1'] > best_score:
                best_score = row['phase1']
                torch.save(model.state_dict(), f'{out_dir}/model.pt')
            for s in ('val_seen', 'val_unseen'):   # also the best checkpoint for each split
                if row[s]['score'] > best_split.get(s, -1.0):
                    best_split[s] = row[s]['score']
                    torch.save(model.state_dict(), f'{out_dir}/model_{s}.pt')
        log.append(row)
        print(json.dumps(row), flush=True)
        json.dump(dict(args=vars(args), epochs=log), open(f'{out_dir}/log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
