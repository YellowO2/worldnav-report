"""Do the test pieces sit in the holes of the labelled recordings' piece numbering.

    python experiments/section9_data_split/hole_match.py stanford_real
"""
import io
import json
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
from floor_geometry import GROUPS, group_of  # noqa: E402

SIGLIP = 'google/siglip2-base-patch16-256'


def piece(t):
    p = t.split('/')[-1].split('_')
    return '_'.join(p[:-2]) + '|' + p[-1], int(p[-2])


def main(g):
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(24)
    proc = AutoProcessor.from_pretrained(SIGLIP)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(SIGLIP).to(dev).eval()

    @torch.no_grad()
    def embed(images):
        out = []
        for i in range(0, len(images), 64):
            px = proc(images=images[i:i + 64], return_tensors='pt')['pixel_values'].to(dev)
            out.append(F.normalize(model.vision_model(pixel_values=px).pooler_output.float(), dim=-1).cpu())
        return torch.cat(out)

    frames = {s: json.load(open(f'data/lcvn/{s}.frames.json')) for s in ('train', 'val_seen')}
    tars = {s: open(f'data/lcvn/{s}.tar', 'rb') for s in frames}

    def photo(s, t, k):
        off, size = frames[s][t][k]
        tars[s].seek(off)
        return Image.open(io.BytesIO(tars[s].read(size))).convert('RGB').resize((256, 256), Image.BICUBIC)
    rec = defaultdict(dict)
    for s in frames:
        for t in frames[s]:
            if group_of(t, s) == g:
                key, k = piece(t)
                rec[key][k] = (s, t)
    test = [json.loads(line) for line in open('data/test/test.jsonl')]
    labels = np.load('runs/test_group_audit.npy', allow_pickle=True).item()['cls']
    seen, t_imgs = set(), []
    for e, lab in zip(test, labels):
        if int(lab) == GROUPS.index(g):
            h = open(e['image'], 'rb').read()
            if h not in seen:
                seen.add(h)
                t_imgs.append(e['image'])
    qtables = Image.open(test[0]['image']).quantization

    def test_like(im):
        buf = io.BytesIO()                               # squeezed to 256 x 160 and saved as the test photos are
        im.resize((256, 160), Image.BICUBIC).save(buf, 'JPEG', qtables=qtables)
        buf.seek(0)
        return Image.open(buf).convert('RGB').resize((256, 256), Image.BICUBIC)

    def test_photo(p):                                   # test photos: the picture is rows 48-208
        return Image.open(p).convert('RGB').crop((0, 48, 256, 208)).resize((256, 256), Image.BICUBIC)
    holes = []                    # 
    for key, d in rec.items():
        for k in range(min(d), max(d) + 2):
            prev, here, nxt = d.get(k - 1), d.get(k), d.get(k + 1)
            if prev is None or prev[0] != 'train':
                continue
            if here is None:
                holes.append(('test', key, k, prev[1], nxt[1] if nxt and nxt[0] == 'train' else None, None))
            elif here[0] == 'val_seen':
                holes.append(('val', key, k, prev[1], nxt[1] if nxt and nxt[0] == 'train' else None, here[1]))
    val_t = sorted(t for t in frames['val_seen'] if group_of(t, 'val_seen') == g)
    n_test_holes = sum(h[0] == 'test' for h in holes)
    print(f'{g}: {len(t_imgs)} test photos, {len(val_t)} val pieces; holes after a train piece: {n_test_holes} unlabelled '
          f'({sum(h[0] == "test" and h[4] is not None for h in holes)} with the next piece in train), {len(holes) - n_test_holes} val', flush=True)
    C = embed([test_photo(p) for p in t_imgs] + [test_like(photo('val_seen', t, 0)) for t in val_t])
    Q = embed([photo('train', h[3], -1) for h in holes])
    sims = (Q @ C.T).numpy()
    nt = len(t_imgs)
    col_of = {t: nt + i for i, t in enumerate(val_t)}
    top = np.argsort(-sims, 1)
    margin = sims[np.arange(len(holes)), top[:, 0]] - sims[np.arange(len(holes)), top[:, 1]]
    rows, cols = linear_sum_assignment(-sims)
    assigned = dict(zip(rows.tolist(), cols.tolist()))
    isval = np.array([h[0] == 'val' for h in holes])
    true = np.array([col_of[h[5]] if h[5] else -1 for h in holes])
    hit = top[:, 0] == true
    hit_a = np.array([assigned.get(i, -2) == true[i] for i in range(len(holes))])
    print(f'val holes ({isval.sum()}): true photo is the best match {hit[isval].mean():.0%}; with one-to-one assignment {hit_a[isval].mean():.0%}; '
          f'best match is a test photo {np.mean(top[isval, 0] < nt):.0%}')
    best = sims.max(1)
    inner = np.array([h[0] == 'test' and h[4] is not None for h in holes])
    print(f'unlabelled holes with the next piece in train ({inner.sum()}): best match is a test photo {np.mean(top[inner, 0] < nt):.0%}')
    for name, m in (('val holes, right match', isval & hit), ('val holes, wrong match', isval & ~hit), ('unlabelled holes', ~isval),
                    ('unlabelled, next in train', inner)):
        if m.any():
            print(f'  {name:24s} n {m.sum():4d}: best similarity {np.percentile(best[m], [10, 50, 90]).round(3)}, margin {np.percentile(margin[m], [10, 50, 90]).round(3)}')
    tm = ~isval
    print(f'unlabelled holes: best match is a test photo {np.mean(top[tm, 0] < nt):.0%} (test photos are {nt / len(C):.0%} of candidates); '
          f'assigned a test photo {np.mean([assigned.get(i, -1) < nt and assigned.get(i, -1) >= 0 for i in np.where(tm)[0]]):.0%}')
    for thr in (0.01, 0.02, 0.03):
        print(f'  margin >= {thr}: val holes {np.mean(margin[isval] >= thr):.0%} kept, right {hit[isval & (margin >= thr)].mean():.0%}; '
              f'unlabelled holes {np.mean(margin[tm] >= thr):.0%} kept')
    for thr in (0.01, 0.02):
        print(f'  margin >= {thr}, unlabelled with next in train: {np.mean(margin[inner] >= thr):.0%} kept, of those matched to a test photo {np.mean(top[inner & (margin >= thr), 0] < nt):.0%}')
    torch.save(dict(holes=holes, t_imgs=t_imgs, val_t=val_t, sims=sims, assigned=assigned), f'features/hole_match_{g}.pt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pick = [i for i in np.argsort(-margin) if tm[i] and holes[i][4] is not None][:4] + \
           [i for i in np.where(tm)[0] if holes[i][4] is not None][10:14]
    fig, ax = plt.subplots(len(pick), 5, figsize=(13, 2.7 * len(pick)))
    for r, i in enumerate(pick):
        h = holes[i]
        ims = [photo('train', h[3], -1)] + [test_photo(t_imgs[c]) if c < nt else photo('val_seen', val_t[c - nt], 0) for c in top[i, :3]] + [photo('train', h[4], 0)]
        ttl = ['piece before: last photo'] + [f'match {j + 1}: {sims[i, c]:.3f} ({"test" if c < nt else "val"})' for j, c in enumerate(top[i, :3])] + ['piece after: first photo']
        for a, im, t in zip(ax[r], ims, ttl):
            a.imshow(im)
            a.set_title(t, fontsize=8)
            a.axis('off')
    plt.tight_layout()
    plt.savefig(f'runs/hole_match_{g}.png', dpi=70)
    print(f'saved runs/hole_match_{g}.png')


if __name__ == '__main__':
    main(sys.argv[1])
