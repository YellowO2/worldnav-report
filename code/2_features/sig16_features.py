"""SigLIP2 image cells at full 16 x 16 resolution (features_v2.py pools them to 8 x 8).

    python 2_features/sig16_features.py train|val_seen|val_unseen|test
"""
import json
import sys

import torch
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from features_v2 import DINO, SIGLIP, Photos


@torch.no_grad()
def main(split, batch=256):
    path = 'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    eps = [json.loads(line) for line in open(path)]
    sig_proc = AutoProcessor.from_pretrained(SIGLIP)
    sig = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to('cuda').eval()
    photos = Photos(split, eps, sig_proc, AutoImageProcessor.from_pretrained(DINO, size={'height': 256, 'width': 256}))
    out = []
    for b, (sp, _) in enumerate(torch.utils.data.DataLoader(photos, batch_size=batch, num_workers=8)):
        out.append(sig.vision_model(pixel_values=sp.to('cuda', torch.float16)).last_hidden_state.half().cpu())
        if b % 20 == 0:
            print(f'{split}: {b * batch + len(sp)}/{len(photos)}', flush=True)
    t = torch.cat(out)
    assert t.shape[1] == 256 and torch.isfinite(t).all()
    torch.save(dict(sig16=t), f'features/{split}_sig16.pt')
    print(f'{split}: saved {tuple(t.shape)}')


if __name__ == '__main__':
    main(sys.argv[1])
