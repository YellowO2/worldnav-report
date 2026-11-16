"""Pilot: does fine-tuning Qwen3.5-4B help? LoRA adapters in the top half of the language layers.

    python experiments/section7_per_robot/lora_pilot.py recon --check
"""
import io
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
from ladder import recording, scored  # noqa: E402
from routenet import score_loss  # noqa: E402
from score import load  # noqa: E402

BS = 16
TOP = 16                         # adapters in layers TOP..31 (of 32)


def routes(split, f, eps, g):
    """One row per route of robot g: (landmark-style episode index, photo item, instruction)."""
    frames = json.load(open(f'data/lcvn/{split}.frames.json'))
    by = {}
    for i, e in enumerate(eps):
        if int(f['group'][i]) == g:
            by.setdefault(e['episode'][:-2], {})[e['episode'][-1]] = i
    rows = []
    for key, d in by.items():
        i = d.get('2', d.get('0'))
        rows.append((i, frames[eps[i]['trajectory']][0], eps[i]['instruction'], [d[s] for s in sorted(d)]))
    return rows


def features(model, proc, imgs, texts):
    msgs = [[{'role': 'user', 'content': [{'type': 'text', 'text': qf.V2_BEFORE.format(t)}, {'type': 'image', 'image': im},
                                          {'type': 'text', 'text': qf.V2_AFTER}]}] for im, t in zip(imgs, texts)]
    enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors='pt', padding=True, enable_thinking=False).to('cuda')
    enc['pixel_values'] = enc['pixel_values'].half()
    hs = model(**enc, output_hidden_states=True).hidden_states
    n = len(hs) - 1
    ids = enc['input_ids']
    img_id = model.config.image_token_id
    cells = torch.stack([hs[round(qf.LAYER_FRACTION * n)][b][ids[b] == img_id] for b in range(len(imgs))])
    last = torch.cat([hs[round(fr * n)][:, -1] for fr in qf.LAYERS], -1)
    mask = enc['attention_mask'].bool() & (ids != img_id)
    text_mean = (hs[-1] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    return cells.float(), last.float(), text_mean.float()


def main(group, check=False, epochs=3, rank=16, top=TOP, all_modules=False):
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
    print(f'{group}: fit {len(fit)} routes, held-out {len(hold)}, val {len(va_rows)}', flush=True)
    t0 = time.time()
    trar, vatar = open('data/lcvn/train.tar', 'rb'), open('data/lcvn/val_seen.tar', 'rb')
    photo = {}
    for rows, tar in ((fit + hold, trar), (va_rows, vatar)):
        for r in rows:
            photo[(id(tar), r[0])] = qf.load_photo(None, r[1], tar)
    tr_key, va_key = id(trar), id(vatar)
    print(f'photos loaded in {time.time() - t0:.0f} s', flush=True)

    proc = AutoProcessor.from_pretrained(qf.MODEL)
    proc.tokenizer.padding_side = 'left'
    base = AutoModelForImageTextToText.from_pretrained(qf.MODEL, dtype=torch.float16).to('cuda')
    layers = '|'.join(str(i) for i in range(top, 32))
    extra = r'|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)' if all_modules else ''
    target = rf'.*language_model\.layers\.({layers})\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj){extra})'
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=target))
    if top < 8:                                          # memory: gradients through all layers
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()                      # adapters in fp32, the frozen model in fp16
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = 3 * 2560, 2560
    pdim = ss.parse_vec(tr, torch.tensor([fit[0][0]]), sizes).shape[1]
    torch.manual_seed(0)
    head = ss.GridRoute(['qwen'], ['q_last', 'q_text'], pdim).cuda()
    print(f'adapters: {n_lora / 1e6:.1f} M trainable numbers in layers {top}-31 (rank {rank}, linear attention {all_modules}); head {sum(p.numel() for p in head.parameters()) / 1e6:.1f} M', flush=True)
    opt = torch.optim.AdamW([{'params': [p for p in model.parameters() if p.requires_grad], 'lr': 1e-4},
                             {'params': head.parameters(), 'lr': 3e-4}], weight_decay=0.05)
    total = epochs * (len(fit) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-4, 3e-4], total_steps=max(total, 1), pct_start=0.05)
    scaler = torch.amp.GradScaler()

    def run_batch(rows, f, key, train):
        idx = torch.tensor([r[0] for r in rows])
        with torch.autocast('cuda', dtype=torch.float16):
            cells, last, text = features(model, proc, [photo[(key, r[0])] for r in rows], [r[2] for r in rows])
        b = ep_batch(f, idx, 'cuda')
        o = dict(camera=b['camera'], parse=ss.parse_vec(f, idx, sizes).cuda(), qwen=cells, q_last=last, q_text=text)
        return head.run(o).float(), idx

    @torch.no_grad()
    def predict(rows, f, key):
        model.eval()
        head.eval()
        out = np.concatenate([run_batch(rows[i:i + 32], f, key, False)[0].cpu().numpy() for i in range(0, len(rows), 32)])
        model.train()
        head.train()
        return out

    def episode_routes(rows, pred, n):
        """Route predictions -> every episode of those routes (the 3 instructions share a route)."""
        idx = [i for r in rows for i in r[3]]
        return torch.tensor(idx), np.concatenate([np.repeat(p[None], len(r[3]), 0) for r, p in zip(rows, pred)])

    model.train()
    head.train()
    if check:
        for k in range(3):
            torch.cuda.synchronize()
            t = time.time()
            pred, idx = run_batch(fit[k * BS:(k + 1) * BS], tr, tr_key, True)
            loss = score_loss(pred, tr['floor'][idx].cuda())
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            print(f'batch {k}: {time.time() - t:.2f} s, loss {loss.item():.3f}, peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB', flush=True)
            opt.zero_grad()
        print(f'one pass over {len(fit)} routes at that speed: about {(time.time() - t) * len(fit) / BS / 60:.0f} min')
        return
    log = []
    for ep in range(1, epochs + 1):
        random.shuffle(fit)
        t = time.time()
        for k in range(len(fit) // BS):
            opt.zero_grad()
            loss = 0.0
            for part in (fit[k * BS:k * BS + BS // 2], fit[k * BS + BS // 2:(k + 1) * BS]):   # two micro-batches of 8
                pred, idx = run_batch(part, tr, tr_key, True)
                l = score_loss(pred, tr['floor'][idx].cuda()) / 2
                scaler.scale(l).backward()
                loss = loss + l.detach()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp['params']], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if k % 100 == 0:
                print(f'  epoch {ep} batch {k}/{len(fit) // BS}: loss {loss.item():.3f}, {time.time() - t:.0f} s', flush=True)
        hi, hp = episode_routes(hold, predict(hold, tr, tr_key), None)
        vi, vp = episode_routes(va_rows, predict(va_rows, va, va_key), None)
        h, v = scored(tr_eps, tr, hi, hp, steps), scored(va_eps, va, vi, vp, steps)
        print(f'epoch {ep}: held-out {h:.2f}, val {v:.2f} ({time.time() - t:.0f} s)   light LoRA pilot: 43.29 / 42.76 at epoch 4, '
              f'best 43.85 / 43.84-43.95 at 6-7', flush=True)
        log.append(dict(epoch=ep, held=h, val=v))
        tag = f'lora_pilot_{group}_top{top}_r{rank}' + ('_all' if all_modules else '')
        json.dump(log, open(f'runs/{tag}.json', 'w'), indent=1)
        from curves import log_curve
        log_curve(tag, epoch=ep, held=h, val=v)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], check='--check' in a, epochs=int(a[a.index('--epochs') + 1]) if '--epochs' in a else 3,
         rank=int(a[a.index('--rank') + 1]) if '--rank' in a else 16,
         top=int(a[a.index('--top') + 1]) if '--top' in a else TOP, all_modules='--all-modules' in a)
