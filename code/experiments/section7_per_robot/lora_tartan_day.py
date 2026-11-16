"""Does LoRA fine-tuning help TartanDrive in places it has not seen? Same LoRA model as lora_final.py.

    python experiments/section7_per_robot/lora_tartan_day.py [--epochs 3]
"""
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import pathnet  # noqa: E402
import qwen_features as qf  # noqa: E402
import siglip_steps as ss  # noqa: E402
from endnet import batch as ep_batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from ladder import scored  # noqa: E402
from lora_pilot import BS, TOP, features, routes  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load  # noqa: E402

FIT_DAYS, REPORT_DAYS = {'20210902', '20210903'}, {'20210826', '20210828', '20210910'}


def day(e):
    return e['trajectory'].split('/')[-1].split('_')[0]


def main(epochs=3, rank=16):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    pathnet.GRAVITY, pathnet.CAM_INPUT, pathnet.FEATURES = True, True, 'v3'
    steps = pathnet.typical_steps()
    tartan = GROUPS.index('tartan')
    data, eps_of = {}, {}
    for split in ('train', 'val_seen', 'val_unseen'):
        f, eps = pathnet.load_split(split), load(f'data/lcvn/{split}.jsonl')
        eps_of[split] = eps
        if split == 'val_unseen':
            rows = routes(split, f, eps, tartan)
        else:
            rows = [r for g in range(len(GROUPS)) if g != tartan for r in routes(split, f, eps, g)]
        rows = [r for r in rows if torch.isfinite(f['floor'][r[0]]).all()]
        data[split] = (f, rows, open(f'data/lcvn/{split}.tar', 'rb'))
    vu_f, vu_rows, vu_tar = data['val_unseen']
    report = [r for r in vu_rows if day(eps_of['val_unseen'][r[0]]) in REPORT_DAYS]
    data['val_unseen'] = (vu_f, [r for r in vu_rows if day(eps_of['val_unseen'][r[0]]) in FIT_DAYS], vu_tar)
    sizes = {k: int(max(data['train'][0][f'seg_{k}'].max(), vu_f[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(data['train'][0]['stop_kind'].max(), vu_f['stop_kind'].max())) + 1
    photo = {}
    for split, (f, rows, tar) in data.items():
        for r in rows:
            photo[(split, r[0])] = qf.load_photo(None, r[1], tar)
    for r in report:
        photo[('report', r[0])] = qf.load_photo(None, r[1], vu_tar)
    print({s: len(v[1]) for s, v in data.items()}, f'report routes {len(report)}', flush=True)

    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    base = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda')
    layers = '|'.join(str(i) for i in range(TOP, 32))
    target = rf'.*language_model\.layers\.({layers})\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))'
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=target))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = 3 * 2560, 2560
    pdim = ss.parse_vec(vu_f, torch.tensor([0]), sizes).shape[1]
    torch.manual_seed(0)
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

    log = []
    for ep in range(1, epochs + 1):
        model.train()
        head.train()
        random.seed(ep)
        batches = []
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
            for half in (part[:BS // 2], part[BS // 2:]):
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
            if k % 400 == 0:
                print(f'  epoch {ep} batch {k}/{len(batches)}: loss {loss.item():.3f}, {time.time() - t:.0f} s', flush=True)
        model.eval()
        head.eval()
        with torch.no_grad():
            pr = np.concatenate([head_out(vu_f, torch.tensor([r[0] for r in report[i:i + 16]]),
                                          [photo[('report', r[0])] for r in report[i:i + 16]], [r[2] for r in report[i:i + 16]]).cpu().numpy()
                                 for i in range(0, len(report), 16)])
        idx = torch.tensor([i for r in report for i in r[3]])
        ep_routes = np.concatenate([np.repeat(p[None], len(r[3]), 0) for r, p in zip(report, pr)])
        s = scored(eps_of['val_unseen'], vu_f, idx, ep_routes, steps)
        print(f'epoch {ep}: report days {s:.2f} ({time.time() - t:.0f} s)   frozen: f5-like 62.31, new 62.87 (3 seeds each)', flush=True)
        log.append(dict(epoch=ep, report=s))
        json.dump(log, open('runs/lora_tartan_day.json', 'w'), indent=1)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(epochs=int(a[a.index('--epochs') + 1]) if '--epochs' in a else 3)
