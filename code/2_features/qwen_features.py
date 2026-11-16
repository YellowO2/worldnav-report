"""Frozen Qwen3.5-4B features: the model reads each route's first photo.

    python 2_features/qwen_features.py val_seen|val_unseen|test|train [--check] [--v2]
"""
import collections
import hashlib
import json
import sys

import torch
from PIL import Image

MODEL = 'Qwen/Qwen3.5-4B'
LAYER_FRACTION = 2 / 3          # image tokens from two thirds of the way up
PROMPT = 'A robot is about to follow this route instruction from where this photo was taken: "{}" Where will it go?'
# v2: the instruction first, then the photo
V2 = '--v2' in sys.argv
V2_BEFORE = 'A robot will walk following this instruction: "{}". This is the view from its front camera:'
V2_AFTER = 'Where on the floor in this image will the robot stop?'
LAYERS = (1 / 2, 2 / 3, 1)      # v2: the final position's state at these depths


def routes_of(split):
    path = 'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    eps = [json.loads(line) for line in open(path)]
    frames = None if split == 'test' else json.load(open(f'data/lcvn/{split}.frames.json'))
    groups = collections.OrderedDict()
    for e in eps:
        key = hashlib.md5(open(e['image'], 'rb').read()).hexdigest() if split == 'test' else e['episode'][:-2]
        groups.setdefault(key, []).append(e)
    rows, index = [], {}
    for key, group in groups.items():
        if split == 'test':
            order = sorted(group, key=lambda e: len(e['instruction']))
            text = order[len(order) // 2]['instruction']                  # middle length = landmark style
            image = group[0]['image']
        else:
            by = {e['episode'][-1]: e for e in group}
            text = (by.get('2') or by.get('0') or group[0])['instruction']
            image = frames[group[0]['trajectory']][0]
        for e in group:
            index[e['episode']] = len(rows)
        rows.append((image, text))
    return eps, rows, [index[e['episode']] for e in eps]


def load_photo(split, item, tar):
    import io
    if isinstance(item, str):
        img = Image.open(item)
    else:
        tar.seek(item[0])
        img = Image.open(io.BytesIO(tar.read(item[1])))
    return img.convert('RGB').resize((256, 256), Image.BICUBIC)


@torch.no_grad()
def run(model, proc, batch, dtype):
    if V2:
        msgs = [[{'role': 'user', 'content': [{'type': 'text', 'text': V2_BEFORE.format(t)}, {'type': 'image', 'image': img},
                                              {'type': 'text', 'text': V2_AFTER}]}] for img, t in batch]
    else:
        msgs = [[{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': PROMPT.format(t)}]}]
                for img, t in batch]
    enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                   return_tensors='pt', padding=True, enable_thinking=False).to('cuda')
    if 'pixel_values' in enc:
        enc['pixel_values'] = enc['pixel_values'].to(dtype)
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    layer = round(LAYER_FRACTION * (len(hs) - 1))
    image_id = model.config.image_token_id
    ids = enc['input_ids']
    cells = [hs[layer][b][ids[b] == image_id] for b in range(len(batch))]
    last = torch.cat([hs[round(fr * (len(hs) - 1))][:, -1] for fr in LAYERS], -1) if V2 else hs[-1][:, -1]
    mask = enc['attention_mask'].bool() & (ids != image_id)
    text_mean = (hs[-1] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
    return cells, last, text_mean


def main(split, check=False, bs=16):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    eps, rows, route_index = routes_of(split)
    if check:
        rows = rows[:8]
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = 'left'
    model16 = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float16).to('cuda').eval()
    model32 = None
    tar = None if split == 'test' else open(f'data/lcvn/{split}.tar', 'rb')
    import os
    tag = 'qwen2' if V2 else 'qwen'
    chunk_dir = f'features/{split}_{tag}_chunks'          # saved every CHUNK routes: a cut-off run continues
    os.makedirs(chunk_dir, exist_ok=True)
    CHUNK = 1024
    C, L, T, redo = [], [], [], 0
    for c0 in range(0, len(rows), CHUNK):
        cpath = f'{chunk_dir}/{c0:06d}.pt'
        if not check and os.path.exists(cpath):
            d = torch.load(cpath)
            C.append(d['cells'])
            L.append(d['last'])
            T.append(d['text_mean'])
            continue
        n0 = len(C)
        for i in range(c0, min(c0 + CHUNK, len(rows)), bs):
            batch = [(load_photo(split, img, tar), text) for img, text in rows[i:min(i + bs, c0 + CHUNK, len(rows))]]
            cells, last, text_mean = run(model16, proc, batch, torch.float16)
            bad = any(not torch.isfinite(c).all() for c in cells) or not torch.isfinite(last).all()
            if bad:                                          # redo this batch in fp32
                redo += 1
                if model32 is None:
                    model32 = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).to('cuda').eval()
                cells, last, text_mean = run(model32, proc, batch, torch.float32)
            assert all(c.shape[0] == 64 for c in cells), f'image tokens per photo: {[c.shape[0] for c in cells]}'
            C.append(torch.stack(cells).half().cpu())
            L.append(last.half().cpu())
            T.append(text_mean.half().cpu())
            if check:
                print('cells', C[-1].shape, 'nan in fp16 batch:', bad, 'value range', float(C[-1].float().abs().max()))
        if check:
            return
        part = dict(cells=torch.cat(C[n0:]), last=torch.cat(L[n0:]), text_mean=torch.cat(T[n0:]))
        torch.save(part, cpath)
        C[n0:], L[n0:], T[n0:] = [part['cells']], [part['last']], [part['text_mean']]
        print(f'{split}: {min(c0 + CHUNK, len(rows))}/{len(rows)} routes, batches redone in fp32: {redo}', flush=True)
    torch.save(dict(episodes=[e['episode'] for e in eps], route_index=torch.tensor(route_index),
                    chunks=sorted(os.listdir(chunk_dir))), f'features/{split}_{tag}.pt')   # the rows stay in the chunks
    print(f'{split}: saved {len(rows)} routes; batches redone in fp32: {redo}')


if __name__ == '__main__':
    main(sys.argv[1], check='--check' in sys.argv)
