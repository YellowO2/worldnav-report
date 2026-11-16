"""Neighbour lookup.

    python 4_final/lookup_final.py --val
"""
import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, 'eval')
from floor_geometry import GROUPS, group_of  # noqa: E402
from lookup_routes import hermite, local, piece  # noqa: E402
from score import load, score  # noqa: E402
from step_oracle import at, length, moves_from  # noqa: E402

SIGLIP = 'google/siglip2-base-patch16-256'
ROBOTS = ('scand_spot', 'scand_jackal', 'stanford_real', 'stanford_sim', 'huron', 'recon', 'tartan')
SPLITS = ('train', 'val_seen', 'val_unseen')


def small(im):
    a = np.asarray(im.convert('L').resize((48, 30), Image.BILINEAR), np.float32).ravel()
    a = a - a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def z(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def rebuild(P, N):
    """Moves of the piece between labelled pieces P and N (episode records with position and yaw)."""
    pp, py = np.array(P['position']), np.unwrap(np.array(P['yaw']))
    pn, ny = np.array(N['position']), np.array(N['yaw'])
    v_prev = np.linalg.norm(np.diff(pp, axis=0), axis=1)[-3:].mean()
    v_next = np.linalg.norm(np.diff(pn, axis=0), axis=1)[:3].mean()
    turn = float(np.mean(np.diff(py)[-3:])) if len(py) > 3 else 0.0
    yaw0 = py[-1] + turn
    start = pp[-1] + v_prev * np.array([np.cos(py[-1] + 0.5 * turn), np.sin(py[-1] + 0.5 * turn)])
    end = local(pn[0] - v_next * np.array([np.cos(ny[0]), np.sin(ny[0])]), start, yaw0)
    dy = ny[0] - yaw0
    curve = hermite(end, np.array([np.cos(dy), -np.sin(dy)]))
    L = length(curve)
    n = int(np.clip(round(L / max(0.5 * (v_prev + v_next), 1e-3)), 1, 64))
    return moves_from(at(curve, np.linspace(0, L, n + 1)[1:]))


class Photos:
    """Photos as the test ones were made (squeezed to 256 x 160, JPEG)."""

    def __init__(self):
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(24)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.proc = AutoProcessor.from_pretrained(SIGLIP)
        self.model = AutoModel.from_pretrained(SIGLIP).to(self.dev).eval()
        self.frames = {s: json.load(open(f'data/lcvn/{s}.frames.json')) for s in SPLITS}
        self.tars = {s: open(f'data/lcvn/{s}.tar', 'rb') for s in SPLITS}
        self.qtables = Image.open(json.loads(open('data/test/test.jsonl').readline())['image']).quantization
        self.cache = torch.load('features/lookup_photos.pt', weights_only=False) if os.path.exists('features/lookup_photos.pt') else {}

    def image(self, key):
        if key[0] == 'test':
            return Image.open(key[1]).convert('RGB').crop((0, 48, 256, 208))
        off, size = self.frames[key[0]][key[1]][key[2]]
        self.tars[key[0]].seek(off)
        buf = io.BytesIO()
        Image.open(io.BytesIO(self.tars[key[0]].read(size))).convert('RGB').resize((256, 160), Image.BICUBIC).save(buf, 'JPEG', qtables=self.qtables)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    @torch.no_grad()
    def get(self, keys):
        """(SigLIP2 vectors, small grey copies) for photo keys ('test', path) or (split, trajectory, frame)."""
        todo = [k for k in dict.fromkeys(keys) if k not in self.cache]
        for i in range(0, len(todo), 64):
            ims = [self.image(k) for k in todo[i:i + 64]]
            px = self.proc(images=ims, return_tensors='pt')['pixel_values'].to(self.dev)
            e = F.normalize(self.model.vision_model(pixel_values=px).pooler_output.float(), dim=-1).cpu().numpy()
            for k, v, im in zip(todo[i:i + 64], e, ims):
                self.cache[k] = (v.astype(np.float16), small(im).astype(np.float16))
        if todo:
            torch.save(self.cache, 'features/lookup_photos.pt')
        return (np.stack([self.cache[k][0] for k in keys]).astype(np.float32), np.stack([self.cache[k][1] for k in keys]).astype(np.float32))


LMATCH, ROBOT, LEARNED = None, None, None
ROBOTS_ALL = ('scand_spot', 'scand_jackal', 'stanford_real', 'stanford_sim', 'huron', 'recon', 'tartan')
SIFT_W, TOP = 0.0, 15                                   # --sift: weight of the detail matcher on each hole's TOP candidates
_desc = {}


def descriptors(photos, key):
    if key not in _desc:
        import cv2
        g = cv2.resize(np.asarray(photos.image(key).convert('L')), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        kp, d = cv2.SIFT_create(nfeatures=600, contrastThreshold=0.02).detectAndCompute(g, None)
        _desc[key] = (np.float32([k.pt for k in kp]).reshape(-1, 2), d)
    return _desc[key]


def inliers(a, b):
    """Matched SIFT points between two photos that agree with one camera motion (ratio test, then RANSAC)."""
    import cv2
    (pa, da), (pb, db) = a, b
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return len(good)
    _, mask = cv2.findFundamentalMat(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]], cv2.FM_RANSAC, 2.0, 0.99)
    return int(mask.sum()) if mask is not None else len(good)


def detail(photos, holes, cand, s):
    """Extra score from local image details (SIFT): for the holes with the piece after."""
    from multiprocessing.pool import ThreadPool
    rows = [i for i, h in enumerate(holes) if h[1] is not None]
    top = {i: np.argsort(-s[i])[:TOP] for i in rows}
    for k in dict.fromkeys([holes[i][0] for i in rows] + [cand[j] for i in rows for j in top[i]]):
        descriptors(photos, k)
    def one(i):
        g = np.log1p([inliers(_desc[holes[i][0]], _desc[cand[j]]) for j in top[i]])
        return i, (g - g.mean()) / (g.std() + 1e-6)
    out = np.zeros_like(s)
    with ThreadPool(32) as pool:
        for i, g in pool.imap_unordered(one, rows, chunksize=8):
            out[i] = g.min()
            out[i, top[i]] = g
    return out


def match(photos, holes, cand):
    """holes: (previous piece key, next piece key or None)."""
    pe, pp = photos.get([h[0] for h in holes])
    ce, cp = photos.get(cand)
    s = z(pe @ ce.T) + z(pp @ cp.T)
    has = np.array([h[1] is not None for h in holes])
    if has.any():
        ne, _ = photos.get([h[1] for h in holes if h[1] is not None])
        s[has] += 0.5 * z(ne @ ce.T)
    if LMATCH is not None and has.any():                  # the trained matcher (experiments/section9_data_split/retrieval_match.py) for holes with both neighbours
        t = lambda x: torch.tensor(x)
        _, npx = photos.get([h[1] for h in holes if h[1] is not None])
        with torch.no_grad():
            s[has] = LMATCH(t(pe[has]), t(ne), t(pp[has]), t(npx), t(ce), t(cp), ROBOTS_ALL.index(ROBOT)).numpy()
    if SIFT_W:
        s = s + SIFT_W * detail(photos, holes, cand, s)
    r, c = linear_sum_assignment(-s)
    a = np.full(len(holes), -1)
    a[r] = c
    conf = np.full(len(holes), -np.inf)
    for i, j in zip(r, c):
        other = np.delete(s[i], j)
        conf[i] = s[i, j] - other.max()
    return a, conf


def test_photos():
    test = [json.loads(line) for line in open('data/test/test.jsonl')]
    labels = np.load('runs/test_group_audit.npy', allow_pickle=True).item()['cls']
    by = defaultdict(dict)                               # robot -> photo hash -> (path, episode ids)
    for e, lab in zip(test, labels):
        h = hashlib.md5(open(e['image'], 'rb').read()).hexdigest()
        by[GROUPS[int(lab)]].setdefault(h, (e['image'], []))[1].append(e['episode'])
    return test, by


CUT = dict(scand_spot=0.5, scand_jackal=0.5, stanford_real=0.0, stanford_sim=0.0, huron=1.0, recon=9.0, tartan=1.0)   # from --val: the lowest cut-off whose added pieces still beat our model


def main(val, base, name, conf_min):
    eps = {}
    for s in SPLITS:
        for e in load(f'data/lcvn/{s}.jsonl'):
            eps.setdefault(e['trajectory'], (s, []))[1].append(e)
    photos = Photos()
    test, t_by = test_photos()
    rng = np.random.default_rng(0)
    model = {}
    if val:
        val_list = load('data/lcvn/val_seen.jsonl')
        for g in ('recon', 'huron', 'stanford_real', 'scand_spot'):
            byt = defaultdict(list)
            for i, r in zip(np.load(f'runs/qsteps_{g}/val_index.npy'), np.load(f'runs/qsteps_{g}/val_routes.npy')):
                byt[val_list[i]['trajectory']].append(r)
            model.update({t: np.mean(v, 0) for t, v in byt.items()})
        import pathnet
        steps = pathnet.typical_steps()
    put = {}                                             # test episode id -> moves
    global ROBOT
    for g in ROBOTS:
        ROBOT = g
        rec, hidden = defaultdict(dict), {}
        for t, (s, _) in eps.items():
            if group_of(t, s) != g:
                continue
            key, k = piece(t)
            hide = val and (s == 'val_seen' or (s == 'val_unseen' and rng.random() < 0.4))
            if hide:
                hidden[(key, k)] = t
                rec[key]
            else:
                rec[key][k] = t
        holes, truth = [], []
        for key, d in rec.items():
            ks = set(d) | {k for (kk, k) in hidden if kk == key}
            if not d:
                continue
            for k in range(min(d), max(ks) + 2):
                if k in d or k - 1 not in d:
                    continue
                holes.append(((eps[d[k - 1]][0], d[k - 1], -1), (eps[d[k + 1]][0], d[k + 1], 0) if k + 1 in d else None, d[k - 1], d.get(k + 1)))
                truth.append(hidden.get((key, k)))
        t_keys = [('test', p) for p, _ in t_by[g].values()]
        t_ids = [ids for _, ids in t_by[g].values()]
        hid = sorted(hidden.values())
        cand = t_keys + [(eps[t][0], t, 0) for t in hid]
        a, conf = match(photos, [(h[0], h[1]) for h in holes], cand)
        nt = len(t_keys)
        both = np.array([h[1] is not None for h in holes])
        print(f'\n{g}: {nt} test photos, {len(holes)} holes ({int(both.sum())} with the piece after)' + (f', {len(hid)} hidden pieces' if val else ''), flush=True)
        if val:
            col = {t: nt + i for i, t in enumerate(hid)}
            for thr in (-np.inf, 0.0, 0.5, 1.0, 2.0):
                use = [i for i in range(len(holes)) if both[i] and a[i] >= nt and conf[i] >= thr]
                if not use:
                    continue
                right = np.mean([truth[i] is not None and col[truth[i]] == a[i] for i in use])
                ep_l, act_l, act_m = [], [], []
                for i in use:
                    t = hid[a[i] - nt]
                    mv = (LEARNED(eps[holes[i][2]][1][0], eps[holes[i][3]][1][0], g) if LEARNED else rebuild(eps[holes[i][2]][1][0], eps[holes[i][3]][1][0]))
                    for e in eps[t][1]:
                        ep_l.append(e)
                        act_l.append(mv)
                        if t in model:
                            mr = np.concatenate([[[0.0, 0.0]], model[t]])
                            n = int(np.clip(round(length(mr) / steps[g]), 1, 64))
                            act_m.append(moves_from(at(mr, np.linspace(0, length(mr), n + 1)[1:])))
                ids = [e['episode'] for e in ep_l]
                sl = score(ep_l, dict(zip(ids, act_l)), workers=16)[0]['score']
                sm = score(ep_l, dict(zip(ids, act_m)), workers=16)[0]['score'] if len(act_m) == len(ep_l) else float('nan')
                tm = [i for i in range(len(holes)) if both[i] and conf[i] >= thr and 0 <= a[i] < nt]
                print(f'  confidence >= {thr:4.1f}: {len(use):4d} hidden pieces replaced ({len(use) / max(len(hid), 1):.0%} of hidden), matched right {right:.0%}, '
                      f'rebuilt score {sl:.2f}, our model on the same {sm:.2f}; test photos that would be replaced {len(tm)} ({len(tm) / max(nt, 1):.0%})', flush=True)
        else:
            n = 0
            for i in range(len(holes)):
                if both[i] and 0 <= a[i] < nt and conf[i] >= CUT.get(g, conf_min):
                    mv = (LEARNED(eps[holes[i][2]][1][0], eps[holes[i][3]][1][0], g) if LEARNED else rebuild(eps[holes[i][2]][1][0], eps[holes[i][3]][1][0]))
                    for ep in t_ids[a[i]]:
                        put[ep] = mv
                    n += 1
            print(f'  {n} test photos replaced ({n / max(nt, 1):.0%})', flush=True)
    if not val:
        sub = json.loads(zipfile.ZipFile(f'runs/{base}/submission_test.zip').read('test.json'))
        for e in sub['episodes']:
            if e['episode_id'] in put:
                e['actions'] = put[e['episode_id']]
        os.makedirs(f'runs/{name}', exist_ok=True)
        with zipfile.ZipFile(f'runs/{name}/submission_test.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('test.json', json.dumps(sub))
        json.dump(sorted(put), open(f'runs/{name}/replaced.json', 'w'))
        print(f'saved runs/{name}/submission_test.zip: {len(put)} of {len(sub["episodes"])} episodes rebuilt from neighbours')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', action='store_true')
    ap.add_argument('--base', default='g6')
    ap.add_argument('--name', default='h1')
    ap.add_argument('--conf', type=float, default=0.0)
    ap.add_argument('--sift', type=float, default=0.0)
    ap.add_argument('--learned', action='store_true')
    ap.add_argument('--lmatch', action='store_true')
    ap.add_argument('--robots', nargs='*')
    a = ap.parse_args()
    SIFT_W = a.sift
    LEARNED = None
    if a.lmatch:
        from retrieval_match import Matcher
        LMATCH = Matcher()
        LMATCH.load_state_dict(torch.load('runs/retrieval_match/model.pt'))
        LMATCH.eval()
    if a.learned:                                         # the trained route-from-neighbours model in place of the smooth curve
        from retrieval_route import Rebuilder
        LEARNED = Rebuilder()
    if a.robots:
        ROBOTS = tuple(a.robots)
    main(a.val, a.base, a.name, a.conf)
