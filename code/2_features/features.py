"""Caches frozen encoder features once per split, so training only runs a small head.

    python 2_features/features.py train|val_seen|val_unseen|test [--flip] (--flip: mirrored photo, left/right swapped in the text)
"""
import io
import re
import json
import os
import sys
import tarfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

SIGLIP = 'google/siglip2-base-patch16-256'
DA3 = 'depth-anything/DA3MONO-LARGE'
DA3_SIZE = 336   # multiple of its 14-pixel patch
TEXT_LEN = 48   # SigLIP2 pads to 64
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class FirstFrames(torch.utils.data.Dataset):
    """One first frame per trajectory (per episode for test), read straight from the packed tar."""

    def __init__(self, split, episodes, processor, flip=False):
        self.processor, self.tar_path, self.tar, self.flip = processor, f'data/lcvn/{split}.tar', None, flip
        if split == 'test':
            self.items, self.index = [e['image'] for e in episodes], list(range(len(episodes)))
            return
        frames = json.load(open(f'data/lcvn/{split}.frames.json'))
        rows, self.items = {}, []
        for e in episodes:
            if e['trajectory'] not in rows:
                rows[e['trajectory']] = len(self.items)
                self.items.append(frames[e['trajectory']][0])
        self.index = [rows[e['trajectory']] for e in episodes]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        if isinstance(item, str):
            image = Image.open(item)
        else:
            if self.tar is None:
                self.tar = open(self.tar_path, 'rb')
            self.tar.seek(item[0])
            image = Image.open(io.BytesIO(self.tar.read(item[1])))
        image = image.convert('RGB')
        if self.flip:
            image = ImageOps.mirror(image)
        da3 = image.resize((DA3_SIZE, DA3_SIZE), Image.BICUBIC)
        da3 = (torch.from_numpy(np.asarray(da3)).permute(2, 0, 1).float() / 255 - IMAGENET_MEAN) / IMAGENET_STD
        return self.processor(images=image, return_tensors='pt')['pixel_values'][0], da3


def load_da3():
    """Builds the Depth Anything 3 network from its config without the package's export/IO extras."""
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    net = create_object(load_config(MODEL_REGISTRY['da3mono-large']))
    sd = load_file(snapshot_download(DA3) + '/model.safetensors')
    net.load_state_dict({k.removeprefix('model.'): v for k, v in sd.items()})
    return net.eval()


def pool_grid(tokens, side_out):
    side = int(tokens.shape[1] ** 0.5)
    grid = tokens.transpose(1, 2).reshape(len(tokens), -1, side, side).float()
    return F.adaptive_avg_pool2d(grid, side_out).flatten(2).transpose(1, 2).half().cpu()


def mirror_text(text):
    """Left and right swapped, for the mirrored copy of an episode."""
    swap = {'left': 'right', 'right': 'left', 'leftward': 'rightward', 'rightward': 'leftward',
            'leftmost': 'rightmost', 'rightmost': 'leftmost', 'clockwise': 'counterclockwise',
            'counterclockwise': 'clockwise', 'anticlockwise': 'clockwise', 'counter-clockwise': 'clockwise'}

    def sub(m):
        w = swap[m.group(0).lower()]
        return w.capitalize() if m.group(0)[0].isupper() else w
    return re.sub(r'\b(' + '|'.join(sorted(swap, key=len, reverse=True)) + r')\b', sub, text, flags=re.IGNORECASE)


@torch.no_grad()
def main(split, batch=128, flip=False):
    path = f'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    episodes = [json.loads(line) for line in open(path)]
    dev = 'cuda'
    processor = AutoProcessor.from_pretrained(SIGLIP)
    siglip = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(dev).eval()
    da3 = load_da3().to(dev)
    frames = FirstFrames(split, episodes, processor, flip=flip)

    tokens, pooled, geo_tokens, geo_cls, depth = [], [], [], [], []
    for i, (px, dx) in enumerate(torch.utils.data.DataLoader(frames, batch_size=batch, num_workers=8)):
        out = siglip.vision_model(pixel_values=px.to(dev, torch.float16))
        tokens.append(pool_grid(out.last_hidden_state, 8))                  # (B, 256, 768) -> (B, 64, 768)
        pooled.append(out.pooler_output.cpu())
        x = dx.to(dev)[:, None]                                              # (B, 1, 3, 336, 336)
        with torch.autocast('cuda', dtype=torch.float16):
            feats, _ = da3.backbone(x, cam_token=None, export_feat_layers=[])
        with torch.autocast('cuda', enabled=False):                          # the depth head runs in float32, as in DA3
            d = da3._process_depth_head(feats, DA3_SIZE, DA3_SIZE).depth[:, 0].float()   # (B, 336, 336)
        tok, cls = feats[-1]
        geo_tokens.append(pool_grid(tok[:, 0], 8))                           # (B, 576, 1024) -> (B, 64, 1024)
        geo_cls.append(cls[:, 0].half().cpu())
        d = F.adaptive_avg_pool2d(d[:, None], 16)[:, 0]
        depth.append((d / d.flatten(1).median(1).values[:, None, None].clamp_min(1e-6)).half().cpu())
        if i % 20 == 0:
            print(f'images {i * batch + len(px)}/{len(frames)}', flush=True)
    del da3

    texts = [mirror_text(e['instruction']) if flip else e['instruction'] for e in episodes]
    text_pooled, text_tokens, text_mask = [], [], []
    for i in range(0, len(texts), batch):
        t = processor(text=texts[i:i + batch], padding='max_length', max_length=64, truncation=True,
                      return_tensors='pt').to(dev)
        out = siglip.text_model(input_ids=t['input_ids'])
        text_pooled.append(siglip.get_text_features(input_ids=t['input_ids']).pooler_output.cpu())
        text_tokens.append(out.last_hidden_state[:, :TEXT_LEN].cpu())
        text_mask.append((t['input_ids'][:, :TEXT_LEN] != processor.tokenizer.pad_token_id).cpu())

    os.makedirs('features', exist_ok=True)
    torch.save(dict(episodes=[e['episode'] for e in episodes], image_index=torch.tensor(frames.index),
                    image_tokens=torch.cat(tokens), image_pooled=torch.cat(pooled),
                    geo_tokens=torch.cat(geo_tokens), geo_cls=torch.cat(geo_cls), depth=torch.cat(depth),
                    text_pooled=torch.cat(text_pooled).half(), text_tokens=torch.cat(text_tokens).half(),
                    text_mask=torch.cat(text_mask)), f'features/{split}{"_flip" if flip else ""}.pt')
    print(f'{split}: {len(episodes)} episodes, {len(frames)} images')


if __name__ == '__main__':
    main(sys.argv[1], flip='--flip' in sys.argv)
