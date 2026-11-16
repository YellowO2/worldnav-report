"""LoRA pilot with the standard noisy-data tricks.

    python experiments/section7_per_robot/lora_pilot2.py recon [--epochs 2]
"""
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageOps

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
import qwen_features as qf  # noqa: E402
import siglip_steps as ss  # noqa: E402
from curves import log_curve  # noqa: E402
from endnet import batch as ep_batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import recording, scored  # noqa: E402
from lora_pilot import BS, TOP, features, routes  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load  # noqa: E402

LEFT, RIGHT = 1, 2                                       # features_v2.SIDES
QTABLES = None                                           # the test photos' JPEG tables (set in main)


def test_like(im):
    """As the test photos were made: 256 x 256 (load_photo)."""
    import io as _io
    buf = _io.BytesIO()
    im.save(buf, 'JPEG', qtables=QTABLES)
    buf.seek(0)
    from PIL import Image as _Image
    return _Image.open(buf).convert('RGB')


def swap_words(t):
    return re.sub(r'\b(left|right|Left|Right|LEFT|RIGHT)\b',
                  lambda m: {'left': 'right', 'right': 'left', 'Left': 'Right', 'Right': 'Left', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}[m.group(1)], t)


def main(group, epochs=2, rank=16):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, 'v3'
    steps = pathnet.typical_steps()
    g = GROUPS.index(group)
    tr, va = pathnet.load_split('train'), pathnet.load_split('val_seen')
    tr_eps, va_eps = load('data/lcvn/train.jsonl'), load('data/lcvn/val_seen.jsonl')
    sizes = {k: int(max(tr[f'seg_{k}'].max(), va[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), va['stop_kind'].max())) + 1
    tr_rows, va_rows = routes('train', tr, tr_eps, g), routes('val_seen', va, va_eps, g)
    tr_rows = [r for r in tr_rows if torch.isfinite(tr['floor'][r[0]]).all()]
    recs = sorted({recording(tr_eps[r[0]]['trajectory']) for r in tr_rows})
    random.seed(0)
    held = set(random.sample(recs, len(recs) // 10))
    fit = [r for r in tr_rows if recording(tr_eps[r[0]]['trajectory']) not in held]
    hold = [r for r in tr_rows if recording(tr_eps[r[0]]['trajectory']) in held]
    samples = [(r, i) for r in fit for i in r[3]]       # (route row, episode index): one per instruction
    print(f'{group}: fit {len(fit)} routes = {len(samples)} instruction samples, held-out {len(hold)}, val {len(va_rows)}', flush=True)
    global QTABLES
    import json as _json
    from PIL import Image as _Image
    QTABLES = _Image.open(_json.loads(open('data/test/test.jsonl').readline())['image']).quantization
    trar, vatar = open('data/lcvn/train.tar', 'rb'), open('data/lcvn/val_seen.tar', 'rb')
    photo = {}
    for rows, tar, key in ((fit + hold, trar, 'tr'), (va_rows, vatar, 'va')):
        for r in rows:
            im = test_like(qf.load_photo(None, r[1], tar))      # every photo processed like the test ones
            photo[(key, r[0])] = im
            photo[(key, r[0], 'm')] = ImageOps.mirror(im)

    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    base = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda')
    layers = '|'.join(str(i) for i in range(TOP, 32))
    target = rf'.*language_model\.layers\.({layers})\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))'
    torch.manual_seed(0)
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=target))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = 3 * 2560, 2560
    pdim = ss.parse_vec(tr, torch.tensor([fit[0][0]]), sizes).shape[1]
    head = ss.GridRoute(['qwen'], ['q_last', 'q_text'], pdim).cuda()
    opt = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-4},
                             {'params': head.parameters(), 'lr': 3e-4}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 3e-4], total_steps=epochs * (len(samples) // BS), pct_start=0.05)
    scaler = torch.amp.GradScaler()

    def head_out(f, key, rows_eps, mirror):
        """rows_eps: [(route row, episode index)]."""
        idx = torch.tensor([i for _, i in rows_eps])
        eps = tr_eps if key == 'tr' else va_eps
        imgs = [photo[(key, r[0], 'm')] if m else photo[(key, r[0])] for (r, _), m in zip(rows_eps, mirror)]
        texts = [swap_words(eps[i]['instruction']) if m else eps[i]['instruction'] for (_, i), m in zip(rows_eps, mirror)]
        with torch.autocast('cuda', dtype=torch.float16):
            cells, last, text = features(model, proc, imgs, texts)
        b = ep_batch(f, idx, 'cuda')
        parse = ss.parse_vec(f, idx, sizes)
        camera = b['camera'].clone()
        mm = torch.tensor(mirror, device='cuda')
        if mm.any():                                     # mirrored: swap the parse's left/right
            side = f['seg_side'][idx].long()
            swapped = torch.where(side == LEFT, RIGHT, torch.where(side == RIGHT, LEFT, side))
            f2 = dict(seg_move=f['seg_move'], seg_side=f['seg_side'].clone(), seg_degree=f['seg_degree'], stop_kind=f['stop_kind'])
            f2['seg_side'][idx] = swapped.to(f2['seg_side'].dtype)
            parse = torch.where(torch.tensor(mirror)[:, None], ss.parse_vec(f2, idx, sizes), parse)
            camera[mm, 3] = -camera[mm, 3]
        out = head.run(dict(camera=camera, parse=parse.cuda(), qwen=cells, q_last=last, q_text=text)).float()
        out[mm, :, 1] = -out[mm, :, 1]                    # back to the original frame
        return out

    @torch.no_grad()
    def predict(rows, f, key, tta):
        model.eval()
        head.eval()
        outs = []
        for i in range(0, len(rows), 8):
            part = rows[i:i + 8]
            if tta:
                combos = [(r, e, m) for r in part for e in r[3] for m in (False, True)]
                o = head_out(f, key, [(r, e) for r, e, _ in combos], [m for _, _, m in combos]).cpu().numpy()
                o = o.reshape(len(part), -1, *o.shape[1:]).mean(1) if all(len(r[3]) == len(part[0][3]) for r in part) else \
                    np.stack([o[sum(2 * len(q[3]) for q in part[:j]):sum(2 * len(q[3]) for q in part[:j + 1])].mean(0) for j in range(len(part))])
            else:
                o = head_out(f, key, [(r, r[0]) for r in part], [False] * len(part)).cpu().numpy()
            outs.append(o)
        model.train()
        head.train()
        return np.concatenate(outs)

    def episode_scores(rows, pred, f, eps):
        idx = torch.tensor([i for r in rows for i in r[3]])
        return scored(eps, f, idx, np.concatenate([np.repeat(p[None], len(r[3]), 0) for r, p in zip(rows, pred)]), steps)

    run = f'lora_pilot2_{group}'
    for ep in range(1, epochs + 1):
        random.seed(ep)
        random.shuffle(samples)
        t = time.time()
        for k in range(len(samples) // BS):
            part = samples[k * BS:(k + 1) * BS]
            mirror = [random.random() < 0.5 for _ in part]
            opt.zero_grad()
            loss = 0.0
            for h in (slice(0, BS // 2), slice(BS // 2, BS)):
                pr = part[h]
                pred = head_out(tr, 'tr', pr, mirror[h])
                idx = torch.tensor([i for _, i in pr])
                l = score_loss(pred, tr['floor'][idx].cuda()) / 2
                scaler.scale(l).backward()
                loss = loss + l.detach()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp['params']], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if k % 200 == 0:
                print(f'  epoch {ep} batch {k}/{len(samples) // BS}: loss {loss.item():.3f}, {time.time() - t:.0f} s', flush=True)
                log_curve(run, epoch=ep - 1 + k / (len(samples) // BS), train_loss=float(loss))
        res = {}
        for tta in (False, True):
            res[f'held_{"tta" if tta else "plain"}'] = episode_scores(hold, predict(hold, tr, 'tr', tta), tr, tr_eps)
            res[f'val_{"tta" if tta else "plain"}'] = episode_scores(va_rows, predict(va_rows, va, 'va', tta), va, va_eps)
        log_curve(run, epoch=ep, **res)
        print(f'epoch {ep}: ' + ', '.join(f'{k} {v:.2f}' for k, v in res.items()) +
              f'   (plain LoRA pilot best: held-out 43.85, val 43.84-43.95; frozen 1 seed 43.58 / 42.93)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], epochs=int(a[a.index('--epochs') + 1]) if '--epochs' in a else 2)
