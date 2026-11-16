"""End-point predictor: the control for "how much end-point information do our inputs carry". Same inputs and body as the path model (pathnet.py), but it predicts only where the route ends, (forward, right) metres from the robot, and submits a straight line to it.

    python 3_models/endnet.py --name e1 [--drop parse|sentence|image|camera ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from manual_path_score import moves_along  # noqa: E402
from score import load, score  # noqa: E402


def metre_grid(cam, down, n=8, sub=4):
    """Per image cell (n x n), where it lies on the floor, from the camera numbers and the photo's lean."""
    B = len(cam)
    g = (torch.arange(n * sub, device=cam.device).float() + 0.5) / (n * sub)
    v, u = torch.meshgrid(g, g, indexing='ij')
    uv = torch.stack([u, v], -1).view(1, -1, 2).expand(B, -1, -1)
    fl, floor = pathnet.lift_torch(uv, cam, down, with_floor=True)          # (B, (n*sub)^2, 2)
    dist = fl.norm(dim=-1).clamp(min=0.05)
    floor = floor.float()
    dist = dist.view(B, n, sub, n, sub).permute(0, 1, 3, 2, 4).reshape(B, n * n, sub * sub)
    floor = floor.view(B, n, sub, n, sub).permute(0, 1, 3, 2, 4).reshape(B, n * n, sub * sub)
    fl = fl.view(B, n, sub, n, sub, 2).permute(0, 1, 3, 2, 4, 5).reshape(B, n * n, sub * sub, 2)
    ld = dist.log()
    mid = fl[:, :, sub * sub // 2 + sub // 2]
    bearing = torch.atan2(mid[..., 1], mid[..., 0].clamp(min=1e-3))
    return torch.stack([ld.median(-1).values, bearing, ld.min(-1).values, ld.max(-1).values, floor.mean(-1)], -1)


class EndNet(pathnet.PathNet):
    def __init__(self, grid=False, k=1, qwen_dim=0, qwen_last_dim=0):
        super().__init__()
        self.use_grid, self.k, self.qwen = grid, k, qwen_dim > 0
        self.cell_pos16 = nn.Parameter(torch.zeros(1, 256, 256))   # --sig16: positions of the 16 x 16 SigLIP2 cells
        if self.qwen:                              # Qwen3.5 features (qwen_features.py) in place of SigLIP2 + DINOv3
            self.qcell, self.qtext = nn.Linear(qwen_dim, 256), nn.Linear(qwen_dim, 256)
            self.qlast = nn.Linear(qwen_last_dim or qwen_dim, 256)
            self.qnorm = nn.LayerNorm(qwen_dim)
            self.qnorm_last = nn.LayerNorm(qwen_last_dim or qwen_dim)
        self.mid = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256))   # --oracle-mid: a known midpoint
        self.turn = nn.Sequential(nn.Linear(3, 256), nn.GELU(), nn.Linear(256, 256))  # --oracle-turn: point + has-turn
        if k > 1:                                  # k candidate end points, each with a confidence logit
            self.cands = nn.Linear(256, 3 * k)
        self.grid = nn.Sequential(nn.Linear(5, 256), nn.GELU(), nn.Linear(256, 256))
        self.end = nn.Linear(256, 2)
        self.stop_kind = nn.Embedding(5, 256)          # features_text.STOP_KINDS, added to the stop slot

    def forward(self, b):
        B = len(b['sig_tokens'])
        pos = self.cell_pos + (self.grid(metre_grid(b['cam'], b['down'])) if self.use_grid else 0)
        if self.qwen:
            cells = self.qcell(self.qnorm(b['q_cells'])) + pos
            glob = torch.stack([self.qlast(self.qnorm_last(b['q_last'])), self.qtext(self.qnorm(b['q_text'])),
                                self.sent(b['sentence']), self.camera(b['camera'])], 1)
        elif 'sig16' in b:                         # SigLIP2 at 16 x 16, DINOv3 at 8 x 8
            pos16 = self.cell_pos16 + (self.grid(metre_grid(b['cam'], b['down'], n=16, sub=2)) if self.use_grid else 0)
            cells = torch.cat([self.sig(b['sig16']) + pos16, self.dino(b['dino_tokens']) + pos], 1)
            glob = torch.stack([self.sig_pool(b['sig_pooled']), self.dino_cls(b['dino_cls']), self.sent(b['sentence']),
                                self.camera(b['camera'])], 1)
        else:
            cells = torch.cat([self.sig(b['sig_tokens']) + pos, self.dino(b['dino_tokens']) + pos], 1)
            glob = torch.stack([self.sig_pool(b['sig_pooled']), self.dino_cls(b['dino_cls']), self.sent(b['sentence']),
                                self.camera(b['camera'])], 1)
        if 'mid' in b:                             # the route's midpoint, (forward, right) metres
            glob = glob + torch.cat([torch.zeros_like(glob[:, :3]), self.mid(b['mid']).unsqueeze(1)], 1)
        if 'turn' in b:                            # the route's first turn point (or a no-turn flag), likewise
            glob = glob + torch.cat([torch.zeros_like(glob[:, :3]), self.turn(b['turn']).unsqueeze(1)], 1)
        has = b['has_phrase'].unsqueeze(-1).float()
        segs = self.slot + (self.phrase(b['seg_phrase']) + self.pmap(b['seg_map'])) * has
        moves = self.move(b['seg_move']) + self.side(b['seg_side']) + self.degree(b['seg_degree'])
        stop = self.stop_kind(b['stop_kind']).unsqueeze(1) if 'stop_kind' in b else torch.zeros_like(moves[:, :1])
        segs = segs + torch.cat([moves, stop], 1)
        x = self.norm_in(torch.cat([cells, glob, segs], 1))
        pad = torch.cat([torch.zeros(B, cells.shape[1] + 4, dtype=torch.bool, device=x.device),
                         b['seg_move'] == 0, torch.zeros(B, 1, dtype=torch.bool, device=x.device)], 1)
        mem = self.encoder(x, src_key_padding_mask=pad)
        q = self.decoder(self.queries[:, -1:].expand(B, -1, -1), mem, memory_key_padding_mask=pad)
        if self.k > 1:
            c = self.cands(q[:, 0]).view(B, self.k, 3)
            return c[..., :2], c[..., 2]
        return self.end(q[:, 0])


def drop(b, what):
    for w in what:
        if w == 'parse':
            for k in ('seg_move', 'seg_side', 'seg_degree'):
                b[k] = torch.zeros_like(b[k])
            for k in ('seg_phrase', 'seg_map'):
                b[k] = torch.zeros_like(b[k])
            b['has_phrase'] = torch.zeros_like(b['has_phrase'])
        elif w == 'sentence':
            b['sentence'] = torch.zeros_like(b['sentence'])
        elif w == 'image':
            for k in ('sig_tokens', 'sig_pooled', 'dino_tokens', 'dino_cls'):
                b[k] = torch.zeros_like(b[k])
        elif w == 'dino':
            for k in ('dino_tokens', 'dino_cls'):
                b[k] = torch.zeros_like(b[k])
        elif w == 'siglip':
            for k in ('sig_tokens', 'sig_pooled'):
                b[k] = torch.zeros_like(b[k])
        elif w == 'camera':
            b['camera'] = torch.zeros_like(b['camera'])
    return b


QWEN_TAG = 'qwen'


def add_qwen(f, split):
    q = torch.load(f'features/{split}_{QWEN_TAG}.pt')
    assert q['episodes'] == f['episodes']
    parts = [torch.load(f'features/{split}_{QWEN_TAG}_chunks/{c}') for c in q['chunks']]
    f.update(q_index=q['route_index'], q_cells=torch.cat([p['cells'] for p in parts]),
             q_last=torch.cat([p['last'] for p in parts]), q_text=torch.cat([p['text_mean'] for p in parts]))
    return f['q_cells'].shape[-1]


ORACLE_MID = False
ORACLE_TURN = False


def batch(f, idx, dev):
    b = pathnet.batch(f, idx, dev)
    if ORACLE_MID:                                 # the true point halfway along the route
        b['mid'] = f['floor'][idx][:, f['floor'].shape[1] // 2 - 1].to(dev).float()
    if 'sig16' in f:                               # --sig16: the image's 16 x 16 SigLIP2 cells
        b['sig16'] = f['sig16'][f['image_index'][idx]].to(dev).float()
    if ORACLE_TURN:                                # the true first turn point (turn_points.py)
        b['turn'] = torch.cat([f['turn'][idx], f['has_turn'][idx, None].float()], 1).to(dev).float()
    if 'q_cells' in f:
        r = f['q_index'][idx]
        b.update(q_cells=f['q_cells'][r].to(dev).float(), q_last=f['q_last'][r].to(dev).float(),
                 q_text=f['q_text'][r].to(dev).float())
    return b


def targets(f, idx, dev):
    fl = f['floor'][idx].to(dev)                                   # (n, K, 2), the last point is the route's end
    return fl[:, -1], fl.norm(dim=-1).max(1).values + 0.5


@torch.no_grad()
def evaluate(model, vals, steps, dev, what):
    model.eval()
    out = {}
    for split, (eps, f) in vals.items():
        preds, best_preds = {}, {}
        for i in range(0, len(eps), 1024):
            idx = torch.arange(i, min(i + 1024, len(eps)))
            res = model(drop(batch(f, idx, dev), what))
            if isinstance(res, tuple):             # the most confident candidate
                cands, logit = res[0].float().cpu().numpy(), res[1].float().cpu().numpy()
                true_end = f['floor'][idx][:, -1].numpy()
                end = cands[np.arange(len(idx)), logit.argmax(1)]
                oracle = cands[np.arange(len(idx)), np.abs(cands - true_end[:, None]).sum(-1).argmin(1)]
            else:
                end = oracle = res.float().cpu().numpy()
            for j, k in enumerate(idx.tolist()):
                g = GROUPS[int(f['group'][k])]
                preds[eps[k]['episode']] = moves_along(np.array([(0.0, 0.0), tuple(end[j])]), steps[g])
                best_preds[eps[k]['episode']] = moves_along(np.array([(0.0, 0.0), tuple(oracle[j])]), steps[g])
        means, rows = score(eps, preds)
        out[split] = {m: round(v, 3) for m, v in means.items()}
        out[split]['best_of_k'] = round(score(eps, best_preds)[0]['score'], 3)
        groups = [GROUPS[int(g)] for g in f['group']]
        out[split]['groups'] = {g: round(float(np.mean([r['score'] for r, gg in zip(rows, groups) if gg == g])), 1)
                                for g in GROUPS if g in groups}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--drop', nargs='*', default=[])
    ap.add_argument('--features', default='v2', help='v2, or v3 (merged parses with stop kind)')
    ap.add_argument('--grid', action='store_true', help='give each image cell its floor position in metres')
    ap.add_argument('--qwen', action='store_true', help='Qwen3.5 features (qwen_features.py) instead of SigLIP2 + DINOv3')
    ap.add_argument('--oracle-mid', action='store_true',
                    help='give the model the true midpoint (a ceiling: how much a midpoint narrows the end point)')
    ap.add_argument('--sig16', action='store_true', help='SigLIP2 image cells at 16 x 16 (sig16_features.py)')
    ap.add_argument('--frac', type=float, default=1.0,
                    help='train on this share of the training routes (a route keeps all 3 instructions); learning curve')
    ap.add_argument('--oracle-turn', action='store_true',
                    help='give the model the true first turn point (does a turn point narrow the end point?)')
    ap.add_argument('--qwen-tag', default='qwen', help='qwen (first prompt) or qwen2 (qwen_features.py --v2)')
    ap.add_argument('--k', type=int, default=1, help='candidate end points (winner-takes-all); 1 = a single answer')
    args = ap.parse_args()
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, args.features
    dev, out = 'cuda', f'runs/{args.name}'
    os.makedirs(out, exist_ok=True)
    steps = pathnet.typical_steps()
    vals = {s: (load(f'data/lcvn/{s}.jsonl'), pathnet.load_split(s)) for s in ('val_seen', 'val_unseen')}
    tr = pathnet.load_split('train')
    if args.sig16:
        for split, fs in [('train', tr)] + [(s_, v[1]) for s_, v in vals.items()]:
            fs['sig16'] = torch.load(f'features/{split}_sig16.pt')['sig16']
    if args.oracle_turn:
        for split, fs in [('train', tr)] + [(s_, v[1]) for s_, v in vals.items()]:
            t = torch.load(f'features/{split}_turns.pt')
            assert t['episodes'] == fs['episodes']
            fs.update(turn=t['turn'], has_turn=t['has_turn'])
    global QWEN_TAG, ORACLE_MID, ORACLE_TURN
    QWEN_TAG, ORACLE_MID, ORACLE_TURN = args.qwen_tag, args.oracle_mid, args.oracle_turn
    qdim = qlast = 0
    if args.qwen:
        for split, (eps, fv) in vals.items():
            qdim = add_qwen(fv, split)
        add_qwen(tr, 'train')
        qlast = tr['q_last'].shape[-1]
    model = EndNet(args.grid, args.k, qdim, qlast).to(dev)
    routes = sorted({ep[:-2] for ep in tr['episodes']})
    if args.frac < 1:                              # a random share of the routes, all 3 instructions of each
        rng = np.random.default_rng(0)
        chosen = set(rng.choice(routes, int(len(routes) * args.frac), replace=False))
        use = torch.tensor([i for i, ep in enumerate(tr['episodes']) if ep[:-2] in chosen])
    else:
        use = torch.arange(len(tr['episodes']))
    n = len(use)
    print(f'training on {n} episodes ({len(use) // 3} routes)', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * (n // args.bs), pct_start=0.05)
    log, best = [], -1.0
    for ep in range(args.epochs):
        t0, perm, tot = time.time(), use[torch.randperm(n)], 0.0
        for k in range(n // args.bs):
            idx = perm[k * args.bs:(k + 1) * args.bs]
            res = model(drop(batch(tr, idx, dev), args.drop))
            end, size = targets(tr, idx, dev)
            if args.k > 1:                         # only the closest candidate learns the end
                cands, logit = res[0].float(), res[1].float()
                err = (cands - end[:, None]).abs().sum(-1) / size[:, None]
                win = err.argmin(1)
                loss = err.gather(1, win[:, None]).mean() + 0.1 * nn.functional.cross_entropy(logit, win)
            else:
                loss = ((res.float() - end).abs().sum(-1) / size).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() / (n // args.bs)
        row = dict(epoch=ep + 1, sec=round(time.time() - t0), train_loss=round(tot, 4))
        if (ep + 1) % 4 == 0:
            row.update(evaluate(model, vals, steps, dev, args.drop))
            row['phase1'] = round((row['val_seen']['score'] + row['val_unseen']['score']) / 2, 3)
            if row['phase1'] > best:
                best = row['phase1']
                torch.save(model.state_dict(), f'{out}/model.pt')
        log.append(row)
        print(json.dumps(row), flush=True)
        json.dump(dict(args=vars(args), epochs=log), open(f'{out}/log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
