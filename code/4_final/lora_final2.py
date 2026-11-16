"""lora_final.py with the noisy-data tricks of experiments/section7_per_robot/lora_pilot2.py.

    python 4_final/lora_final2.py
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
import qwen_features as qf  # noqa: E402
import siglip_steps as ss  # noqa: E402
import lora_pilot2 as lp2  # noqa: E402
from curves import log_curve  # noqa: E402
from endnet import batch as ep_batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from group_classifier import predict_groups  # noqa: E402
from lora_pilot import BS, TOP, features, routes  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load  # noqa: E402


def main(epochs=1, rank=16, tartan_mode=False, seed=0):
    import json
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, 'v3'
    out_dir = ('runs/lora_tartan2' if tartan_mode else 'runs/lora_final2') + (f'_s{seed}' if seed else '')
    lp2.QTABLES = Image.open(json.loads(open('data/test/test.jsonl').readline())['image']).quantization
    tartan = GROUPS.index('tartan')
    data = {}
    for split in ('train', 'val_seen') + (('val_unseen',) if tartan_mode else ()):
        f, eps = pathnet.load_split(split), load(f'data/lcvn/{split}.jsonl')
        groups = [tartan] if split == 'val_unseen' else [g for g in range(len(GROUPS)) if g != tartan]
        rows = [r for g in groups for r in routes(split, f, eps, g)]
        rows = [r for r in rows if torch.isfinite(f['floor'][r[0]]).all()]
        data[split] = (f, eps, rows, open(f'data/lcvn/{split}.tar', 'rb'))
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test).cpu()
    t_eps, t_rows, t_index = qf.routes_of('test')
    eps_of_route = {}
    for k, r in enumerate(t_index):
        eps_of_route.setdefault(r, []).append(k)
    t_keep = [r for r in range(len(t_rows)) if (int(test['group'][eps_of_route[r][0]]) == tartan) == tartan_mode]
    sizes = {k: int(max(data['train'][0][f'seg_{k}'].max(), test[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(data['train'][0]['stop_kind'].max(), test['stop_kind'].max())) + 1
    t0 = time.time()
    photo = {}
    for split, (f, eps, rows, tar) in data.items():
        for r in rows:
            im = lp2.test_like(qf.load_photo(None, r[1], tar))
            photo[(split, r[0])], photo[(split, r[0], 'm')] = im, ImageOps.mirror(im)
    for r in t_keep:                                     # test photos are already 256 x 256 JPEGs
        im = qf.load_photo('test', t_rows[r][0], None)
        photo[('test', r)], photo[('test', r, 'm')] = im, ImageOps.mirror(im)
    samples = [(split, r, i) for split, (f, eps, rows, tar) in data.items() for r in rows for i in r[3]]
    print(f'{len(samples)} instruction samples from {sum(len(v[2]) for v in data.values())} routes; {len(t_keep)} test routes; '
          f'photos in {time.time() - t0:.0f} s', flush=True)

    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    base = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda')
    layers = '|'.join(str(i) for i in range(TOP, 32))
    target = rf'.*language_model\.layers\.({layers})\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))'
    torch.manual_seed(seed)
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=target))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = 3 * 2560, 2560
    pdim = ss.parse_vec(test, torch.tensor([0]), sizes).shape[1]
    head = ss.GridRoute(['qwen'], ['q_last', 'q_text'], pdim).cuda()
    opt = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-4},
                             {'params': head.parameters(), 'lr': 3e-4}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 3e-4], total_steps=epochs * (len(samples) // BS), pct_start=0.05)
    scaler = torch.amp.GradScaler()
    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    ema = [p.detach().clone() for p in trainable]       # exponential moving average of the weights (decay EMA_DECAY)
    EMA_DECAY = 0.999

    def head_out(f, eps, key, row_ep, mirror):
        idx = torch.tensor([i for _, i in row_ep])
        rid = [r if key == 'test' else r[0] for r, _ in row_ep]
        imgs = [photo[(key, q, 'm')] if m else photo[(key, q)] for q, m in zip(rid, mirror)]
        texts = [lp2.swap_words(eps[i]['instruction']) if m else eps[i]['instruction'] for (_, i), m in zip(row_ep, mirror)]
        with torch.autocast('cuda', dtype=torch.float16):
            cells, last, text = features(model, proc, imgs, texts)
        b = ep_batch(f, idx, 'cuda')
        parse, camera = ss.parse_vec(f, idx, sizes), b['camera'].clone()
        mm = torch.tensor(mirror, device='cuda')
        if mm.any():
            side = f['seg_side'][idx].long()
            f2 = dict(seg_move=f['seg_move'], seg_side=f['seg_side'].clone(), seg_degree=f['seg_degree'], stop_kind=f['stop_kind'])
            f2['seg_side'][idx] = torch.where(side == lp2.LEFT, lp2.RIGHT, torch.where(side == lp2.RIGHT, lp2.LEFT, side)).to(f2['seg_side'].dtype)
            parse = torch.where(torch.tensor(mirror)[:, None], ss.parse_vec(f2, idx, sizes), parse)
            camera[mm, 3] = -camera[mm, 3]
        out = head.run(dict(camera=camera, parse=parse.cuda(), qwen=cells, q_last=last, q_text=text)).float()
        out[mm, :, 1] = -out[mm, :, 1]
        return out

    model.train()
    head.train()
    for ep in range(1, epochs + 1):
        random.seed(ep + 1000 * seed)
        batches = []
        for split in data:
            ss_ = [s for s in samples if s[0] == split]
            random.shuffle(ss_)
            batches += [ss_[i:i + BS] for i in range(0, len(ss_) - BS + 1, BS)]
        random.shuffle(batches)
        t = time.time()
        for k, part in enumerate(batches):
            split = part[0][0]
            f, eps = data[split][0], data[split][1]
            mirror = [random.random() < 0.5 for _ in part]
            opt.zero_grad()
            loss = 0.0
            for h in (slice(0, BS // 2), slice(BS // 2, BS)):
                row_ep = [(r, i) for _, r, i in part[h]]
                pred = head_out(f, eps, split, row_ep, mirror[h])
                idx = torch.tensor([i for _, i in row_ep])
                l = score_loss(pred, f['floor'][idx].cuda()) / 2
                scaler.scale(l).backward()
                loss = loss + l.detach()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp['params']], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                for e_, p_ in zip(ema, trainable):
                    e_.mul_(EMA_DECAY).add_(p_.detach(), alpha=1 - EMA_DECAY)
            if k % 400 == 0:
                print(f'  epoch {ep} batch {k}/{len(batches)}: loss {loss.item():.3f}, {time.time() - t:.0f} s', flush=True)
                log_curve(os.path.basename(out_dir), epoch=ep - 1 + k / len(batches), train_loss=float(loss))
        print(f'epoch {ep} done ({time.time() - t:.0f} s)', flush=True)
    model.eval()
    head.eval()

    def predict_all():
        preds = {}
        with torch.no_grad():
            for i in range(0, len(t_keep), 4):
                rs = t_keep[i:i + 4]
                combos = [(r, e, m) for r in rs for e in eps_of_route[r] for m in (False, True)]
                o = head_out(test, t_eps, 'test', [(r, e) for r, e, _ in combos], [m for _, _, m in combos]).cpu().numpy()
                j = 0
                for r in rs:
                    n = 2 * len(eps_of_route[r])
                    preds[r] = o[j:j + n].mean(0)
                    j += n
        return preds
    final_preds = predict_all()
    with torch.no_grad():                                # the EMA weights, then the average of both predictions
        for e_, p_ in zip(ema, trainable):
            p_.copy_(e_)
    ema_preds = predict_all()
    preds = {r: (final_preds[r] + ema_preds[r]) / 2 for r in final_preds}
    if False:
        for i in range(0, len(t_keep), 4):
            rs = t_keep[i:i + 4]
            combos = [(r, e, m) for r in rs for e in eps_of_route[r] for m in (False, True)]
            o = head_out(test, t_eps, 'test', [(r, e) for r, e, _ in combos], [m for _, _, m in combos]).cpu().numpy()
            j = 0
            for r in rs:
                n = 2 * len(eps_of_route[r])
                preds[r] = o[j:j + n].mean(0)
                j += n
    covered = [k for k, r in enumerate(t_index) if r in preds]
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/test_routes.npy', dict(index=np.array(covered), routes=np.stack([preds[t_index[k]] for k in covered]),
                                              episodes=[t_eps[k]['episode'] for k in covered]), allow_pickle=True)
    print(f'saved {out_dir}/test_routes.npy: {len(covered)} test episodes (6-way test-time average per route)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--tartan', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    main(a.epochs, tartan_mode=a.tartan, seed=a.seed)
