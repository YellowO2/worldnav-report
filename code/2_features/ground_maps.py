"""Caches where each sentence of the instruction is in the photo, as SigLIP2 itself sees it.

    python 2_features/ground_maps.py train|val_seen|val_unseen|test [--flip]
"""
import json
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from features import SIGLIP, FirstFrames, mirror_text

SENTENCES = 3


def sentences(text):
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', text) if p.strip()]
    parts = parts[-SENTENCES:]
    return [''] * (SENTENCES - len(parts)) + parts


@torch.no_grad()
def main(split, batch=128, flip=False):
    dev = 'cuda'
    path = 'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    episodes = [json.loads(line) for line in open(path)]
    processor = AutoProcessor.from_pretrained(SIGLIP)
    model = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(dev).eval()

    texts = [s for e in episodes for s in sentences(mirror_text(e['instruction']) if flip else e['instruction'])]
    valid = torch.tensor([s != '' for s in texts]).view(-1, SENTENCES)
    emb = []
    for i in range(0, len(texts), 512):
        t = processor(text=[s or '.' for s in texts[i:i + 512]], padding='max_length', max_length=64,
                      truncation=True, return_tensors='pt').to(dev)
        emb.append(F.normalize(model.get_text_features(**t).pooler_output.float(), dim=-1))
    emb = torch.cat(emb).view(len(episodes), SENTENCES, -1)                     # (n_episodes, 3, 768) on GPU

    frames = FirstFrames(split, episodes, processor, flip=flip)
    by_image = [[] for _ in range(len(frames))]
    for e, row in enumerate(frames.index):
        by_image[row].append(e)
    sims = torch.zeros(len(episodes), SENTENCES, 256, dtype=torch.float16)
    vision = model.vision_model
    for b, (px, _) in enumerate(torch.utils.data.DataLoader(frames, batch_size=batch, num_workers=8)):
        patches = vision(pixel_values=px.to(dev, torch.float16)).last_hidden_state      # (B, 256, 768)
        per_patch = vision.head(patches.reshape(-1, 1, patches.shape[-1])).view(len(px), 256, -1)
        per_patch = F.normalize(per_patch.float(), dim=-1)
        for j in range(len(px)):
            eps = by_image[b * batch + j]
            sims[eps] = torch.einsum('ekd,pd->ekp', emb[eps], per_patch[j]).half().cpu()
        if b % 20 == 0:
            print(f'images {b * batch + len(px)}/{len(frames)}', flush=True)
    torch.save(dict(sims=sims, valid=valid), f'features/{split}{"_flip" if flip else ""}_ground.pt')
    print(f'{split}: {len(episodes)} episodes')


if __name__ == '__main__':
    main(sys.argv[1], flip='--flip' in sys.argv)
