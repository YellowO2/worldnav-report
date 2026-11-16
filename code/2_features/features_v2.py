"""Frozen features for the path-drawing model: SigLIP2 (meaning, text) and DINOv3-L (layout).

    python 2_features/features_v2.py train|val_seen|val_unseen|test [--shift]
"""
import io
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

sys.path.insert(0, 'eval')
from floor_geometry import group_of, start_photo  # noqa: E402

SIGLIP = 'google/siglip2-base-patch16-256'
DINO = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
S = 6
MOVES = ['pad', 'straight', 'turn', 'curve', 'turn_in_place', 'stairs_up', 'stairs_down']
SIDES = ['none', 'left', 'right']
DEGREES = ['normal', 'slight', 'sharp']


class Photos(torch.utils.data.Dataset):
    def __init__(self, split, episodes, sig_proc, dino_proc, shift=False):
        self.sig_proc, self.dino_proc, self.tar_path, self.tar = sig_proc, dino_proc, f'data/lcvn/{split}.tar', None
        if split == 'test':
            self.items, self.index = [e['image'] for e in episodes], list(range(len(episodes)))
            return
        frames = json.load(open(f'data/lcvn/{split}.frames.json'))
        rows, self.items = {}, []
        for e in episodes:
            if e['trajectory'] not in rows:
                rows[e['trajectory']] = len(self.items)
                self.items.append(start_photo(frames[e['trajectory']], group_of(e['trajectory'], split), shift))
        self.index = [rows[e['trajectory']] for e in episodes]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        if isinstance(item, str):
            img = Image.open(item)
        else:
            if self.tar is None:
                self.tar = open(self.tar_path, 'rb')
            self.tar.seek(item[0])
            img = Image.open(io.BytesIO(self.tar.read(item[1])))
        img = img.convert('RGB').resize((256, 256), Image.BICUBIC)     # every split as the test photos are: 256 x 256
        return (self.sig_proc(images=img, return_tensors='pt')['pixel_values'][0],
                self.dino_proc(images=img, return_tensors='pt')['pixel_values'][0])


def pool8(tokens):
    side = int(tokens.shape[1] ** 0.5)
    g = tokens.transpose(1, 2).reshape(len(tokens), -1, side, side).float()
    return F.adaptive_avg_pool2d(g, 8).flatten(2).transpose(1, 2).half().cpu()


def slots(parsed):
    """Per-episode move ids and landmark phrases (S segments + the stop)."""
    move, side, deg, phr = [0] * S, [0] * S, [0] * S, [None] * (S + 1)
    if parsed:
        for k, s in enumerate(parsed['segments'][:S]):
            move[k] = MOVES.index(s['move']) if s['move'] in MOVES else 0
            side[k] = SIDES.index(s.get('side', 'none')) if s.get('side', 'none') in SIDES else 0
            deg[k] = DEGREES.index(s.get('degree', 'normal')) if s.get('degree', 'normal') in DEGREES else 0
            phr[k] = s.get('until') or s.get('at') or None
        phr[S] = parsed.get('stop_at') or None
    return move, side, deg, phr


@torch.no_grad()
def main(split, batch=128, shift=False):
    dev = 'cuda'
    path = 'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    eps = [json.loads(line) for line in open(path)]
    parsed = {}
    for line in open(f'features/{split}_parsed.jsonl'):
        r = json.loads(line)
        parsed[r['episode']] = r['parsed']
    sig_proc = AutoProcessor.from_pretrained(SIGLIP)
    sig = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(dev).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO, size={'height': 256, 'width': 256})
    dino = AutoModel.from_pretrained(DINO, dtype=torch.float32).to(dev).eval()   # fp16 overflows to NaN in DINOv3
    n_reg = getattr(dino.config, 'num_register_tokens', 0)

    # text: whole sentences, and every distinct landmark phrase once
    def encode(texts):
        out = []
        for i in range(0, len(texts), 512):
            t = sig_proc(text=texts[i:i + 512], padding='max_length', max_length=64, truncation=True,
                         return_tensors='pt').to(dev)
            out.append(F.normalize(sig.get_text_features(**t).pooler_output.float(), dim=-1).half())
        return torch.cat(out)

    sentence = encode([e['instruction'] for e in eps]).cpu()
    sl = [slots(parsed.get(e['episode'])) for e in eps]
    phrases = sorted({p for s in sl for p in s[3] if p})
    phrase_emb = encode([f'a photo of {p}' for p in phrases]) if phrases else torch.zeros(0, 768, device=dev)
    pid = {p: i for i, p in enumerate(phrases)}

    photos = Photos(split, eps, sig_proc, dino_proc, shift)
    by_image = [[] for _ in range(len(photos))]
    for e, row in enumerate(photos.index):
        by_image[row].append(e)
    n_ep = len(eps)
    seg_phrase = torch.zeros(n_ep, S + 1, 768, dtype=torch.float16)
    seg_map = torch.zeros(n_ep, S + 1, 256, dtype=torch.float16)
    has_phrase = torch.zeros(n_ep, S + 1, dtype=torch.bool)
    sig_tok, sig_pool, dino_tok, dino_cls = [], [], [], []
    for b, (sp, dp) in enumerate(torch.utils.data.DataLoader(photos, batch_size=batch, num_workers=8)):
        out = sig.vision_model(pixel_values=sp.to(dev, torch.float16))
        sig_tok.append(pool8(out.last_hidden_state))
        sig_pool.append(out.pooler_output.cpu())
        per_patch = F.normalize(sig.vision_model.head(out.last_hidden_state.reshape(-1, 1, 768)).view(len(sp), 256, -1).float(), dim=-1)
        d = dino(pixel_values=dp.to(dev, torch.float32)).last_hidden_state.half()
        dino_cls.append(d[:, 0].cpu())
        dino_tok.append(pool8(d[:, 1 + n_reg:]))
        for j in range(len(sp)):
            for e in by_image[b * batch + j]:
                for k, p in enumerate(sl[e][3]):
                    if p:
                        m = per_patch[j] @ phrase_emb[pid[p]].float()
                        seg_map[e, k] = ((m - m.mean()) / (m.std() + 1e-4)).half().cpu()
                        seg_phrase[e, k] = phrase_emb[pid[p]].cpu()
                        has_phrase[e, k] = True
        if b % 20 == 0:
            print(f'images {b * batch + len(sp)}/{len(photos)}', flush=True)
    torch.save(dict(episodes=[e['episode'] for e in eps], image_index=torch.tensor(photos.index),
                    sig_tokens=torch.cat(sig_tok), sig_pooled=torch.cat(sig_pool), dino_tokens=torch.cat(dino_tok),
                    dino_cls=torch.cat(dino_cls), sentence=sentence,
                    seg_move=torch.tensor([s[0] for s in sl], dtype=torch.int8),
                    seg_side=torch.tensor([s[1] for s in sl], dtype=torch.int8),
                    seg_degree=torch.tensor([s[2] for s in sl], dtype=torch.int8),
                    seg_phrase=seg_phrase, seg_map=seg_map, has_phrase=has_phrase,
                    parsed=torch.tensor([parsed.get(e['episode']) is not None for e in eps])),
               f'features/{split}_v2{"_shift" if shift else ""}.pt')
    print(f'{split}: {n_ep} episodes, {len(photos)} images, {len(phrases)} distinct phrases, '
          f'parsed {sum(parsed.get(e["episode"]) is not None for e in eps)}/{n_ep}')


if __name__ == '__main__':
    main(sys.argv[1], shift='--shift' in sys.argv)
