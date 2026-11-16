"""Final LoRA model for the robots other than TartanDrive: Qwen3.5-4B with LoRA adapters.

    python 4_final/lora_final.py --epochs 3
"""
import argparse
import json
import os
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
import qwen_features as qf  # noqa: E402
import siglip_steps as ss  # noqa: E402
from endnet import batch as ep_batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from group_classifier import predict_groups  # noqa: E402
from lora_pilot import BS, TOP, features, routes  # noqa: E402
from routenet import score_loss  # noqa: E402
from curves import log_curve  # noqa: E402
from score import load  # noqa: E402

OUT = 'runs/lora_final'


def train_and_predict(epochs, rank=16, tartan_mode=False, seed=0):
    """tartan_mode: train on every robot's labelled routes including all TartanDrive ones."""
    out_dir = ('runs/lora_tartan' if tartan_mode else OUT) + (f'_s{seed}' if seed else '')
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, 'v3'
    tartan = GROUPS.index('tartan')
    data = {}
    for split in ('train', 'val_seen') + (('val_unseen',) if tartan_mode else ()):
        f, eps = pathnet.load_split(split), load(f'data/lcvn/{split}.jsonl')
        groups = [tartan] if split == 'val_unseen' else [g for g in range(len(GROUPS)) if g != tartan]
        rows = [r for g in groups for r in routes(split, f, eps, g)]
        rows = [r for r in rows if torch.isfinite(f['floor'][r[0]]).all()]
        data[split] = (f, rows, open(f'data/lcvn/{split}.tar', 'rb'))
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test).cpu()
    t_eps, t_rows, t_index = qf.routes_of('test')                # one row per test route (3 episodes share a photo)
    first = {}
    for k, r in enumerate(t_index):
        first.setdefault(r, k)
    t_keep = [r for r in range(len(t_rows)) if (int(test['group'][first[r]]) == tartan) == tartan_mode]
    sizes = {k: int(max(data['train'][0][f'seg_{k}'].max(), test[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(data['train'][0]['stop_kind'].max(), test['stop_kind'].max())) + 1
    t0 = time.time()
    photo = {}
    for split, (f, rows, tar) in data.items():
        for r in rows:
            photo[(split, r[0])] = qf.load_photo(None, r[1], tar)
    for r in t_keep:
        photo[('test', r)] = qf.load_photo('test', t_rows[r][0], None)
    print(f'{sum(len(v[1]) for v in data.values())} labelled routes, {len(t_keep)} test routes; photos in {time.time() - t0:.0f} s', flush=True)

    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    base = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda')
    layers = '|'.join(str(i) for i in range(TOP, 32))
    target = rf'.*language_model\.layers\.({layers})\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))'
    torch.manual_seed(seed)                              # the adapters' random start
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=target))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = 3 * 2560, 2560
    pdim = ss.parse_vec(test, torch.tensor([0]), sizes).shape[1]
    torch.manual_seed(seed)
    head = ss.GridRoute(['qwen'], ['q_last', 'q_text'], pdim).cuda()
    n_items = sum(len(rows) for f, rows, tar in data.values())
    opt = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-4},
                             {'params': head.parameters(), 'lr': 3e-4}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 3e-4], total_steps=epochs * (n_items // BS), pct_start=0.05)
    scaler = torch.amp.GradScaler()

    def head_out(f, idx, imgs, texts):
        with torch.autocast('cuda', dtype=torch.float16):
            cells, last, text = features(model, proc, imgs, texts)
        b = ep_batch(f, idx, 'cuda')
        return head.run(dict(camera=b['camera'], parse=ss.parse_vec(f, idx, sizes).cuda(), qwen=cells, q_last=last, q_text=text)).float()

    model.train()
    head.train()
    for ep in range(1, epochs + 1):
        random.seed(ep + 1000 * seed)
        batches = []                                     # each batch from one split (separate feature tables)
        for split, (f, rows, tar) in data.items():
            rows = rows[:]
            random.shuffle(rows)
            batches += [(split, rows[i:i + BS]) for i in range(0, len(rows) - BS + 1, BS)]
        random.shuffle(batches)
        t = time.time()
        for k, (split, part) in enumerate(batches):
            opt.zero_grad()
            f = data[split][0]
            loss = 0.0
            for half in (part[:BS // 2], part[BS // 2:]):   # two micro-batches of 8: longer instructions ran out of memory
                idx = torch.tensor([r[0] for r in half])
                pred = head_out(f, idx, [photo[(split, r[0])] for r in half], [r[2] for r in half])
                l = score_loss(pred, f['floor'][idx].cuda()) / 2
                scaler.scale(l).backward()
                loss = loss + l.detach()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp['params']], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if k % 200 == 0:
                print(f'  epoch {ep} batch {k}/{len(batches)}: loss {loss.item():.3f}, {time.time() - t:.0f} s', flush=True)
                log_curve(os.path.basename(out_dir), epoch=ep + k / len(batches), train_loss=float(loss))
        print(f'epoch {ep} done ({time.time() - t:.0f} s)', flush=True)
    model.eval()
    head.eval()
    preds = {}
    with torch.no_grad():
        for i in range(0, len(t_keep), 16):
            rs = t_keep[i:i + 16]
            idx = torch.tensor([first[r] for r in rs])
            out = head_out(test, idx, [photo[('test', r)] for r in rs], [t_rows[r][1] for r in rs]).cpu().numpy()
            for r, o in zip(rs, out):
                preds[r] = o
    covered = [k for k, r in enumerate(t_index) if r in preds]
    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/test_routes.npy', dict(index=np.array(covered), routes=np.stack([preds[t_index[k]] for k in covered]),
                                              episodes=[t_eps[k]['episode'] for k in covered]), allow_pickle=True)
    print(f'saved {out_dir}/test_routes.npy: {len(covered)} test episodes', flush=True)


def mix(src, name, lora_dir=OUT, weight=0.5):
    final.setup()
    steps = pathnet.typical_steps()
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test).cpu()
    base = np.load(f'runs/{src}/test_routes.npy')
    d = np.load(f'{lora_dir}/test_routes.npy', allow_pickle=True).item()
    assert [test['episodes'][k] for k in d['index']] == d['episodes']
    out = base.copy()
    out[d['index']] = (1 - weight) * base[d['index']] + weight * d['routes']
    os.makedirs(f'runs/{name}', exist_ok=True)
    np.save(f'runs/{name}/test_routes.npy', out)
    moves = final.to_moves(out, test['group'], steps)
    with zipfile.ZipFile(f'runs/{name}/submission_test.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('test.json', json.dumps({'split': 'test', 'episodes': [{'episode_id': ep, 'actions': a}
                                                                          for ep, a in zip(test['episodes'], moves)]}))
    print(f'saved runs/{name}/submission_test.zip: {len(d["index"])} episodes averaged with the LoRA routes')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--mix')
    ap.add_argument('--name')
    ap.add_argument('--tartan', action='store_true', help='TartanDrive model and test routes (runs/lora_tartan)')
    ap.add_argument('--weight', type=float, default=0.5, help='with --mix: the LoRA routes weight')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lora-dir', help='with --mix: the LoRA routes folder (default runs/lora_tartan or runs/lora_final)')
    a = ap.parse_args()
    if a.mix:
        mix(a.mix, a.name, a.lora_dir or ('runs/lora_tartan' if a.tartan else OUT), a.weight)
    else:
        train_and_predict(a.epochs, tartan_mode=a.tartan, seed=a.seed)
