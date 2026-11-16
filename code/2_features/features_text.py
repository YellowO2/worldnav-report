"""The parse part of the features rebuilt from the merged per-route parses (build_parses.py).

    python 2_features/features_text.py train|val_seen|val_unseen|test
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoProcessor

from features_v2 import DINO, S, SIGLIP, Photos, slots

STOP_KINDS = ['none', 'object', 'end', 'after turn', 'past landmark']


@torch.no_grad()
def main(split, batch=128):
    dev = 'cuda'
    path = 'data/test/test.jsonl' if split == 'test' else f'data/lcvn/{split}.jsonl'
    eps = [json.loads(line) for line in open(path)]
    rows = {json.loads(l)['episode']: json.loads(l) for l in open(f'features/{split}_parsed2.jsonl')}
    f = torch.load(f'features/{split}_v2.pt')
    assert f['episodes'] == [e['episode'] for e in eps]
    sig_proc = AutoProcessor.from_pretrained(SIGLIP)
    sig = AutoModel.from_pretrained(SIGLIP, dtype=torch.float16).to(dev).eval()
    dino_proc = AutoImageProcessor.from_pretrained(DINO, size={'height': 256, 'width': 256})

    def encode(texts):
        out = []
        for i in range(0, len(texts), 512):
            t = sig_proc(text=texts[i:i + 512], padding='max_length', max_length=64, truncation=True,
                         return_tensors='pt').to(dev)
            out.append(F.normalize(sig.get_text_features(**t).pooler_output.float(), dim=-1).half())
        return torch.cat(out)

    sl = [slots(rows[e['episode']]['parsed']) for e in eps]
    phrases = sorted({p for s in sl for p in s[3] if p})
    phrase_emb = encode([f'a photo of {p}' for p in phrases]) if phrases else torch.zeros(0, 768, device=dev)
    pid = {p: i for i, p in enumerate(phrases)}
    photos = Photos(split, eps, sig_proc, dino_proc)
    assert photos.index == f['image_index'].tolist()
    by_image = [[] for _ in range(len(photos))]
    for e, row in enumerate(photos.index):
        by_image[row].append(e)
    n_ep = len(eps)
    seg_phrase = torch.zeros(n_ep, S + 1, 768, dtype=torch.float16)
    seg_map = torch.zeros(n_ep, S + 1, 256, dtype=torch.float16)
    has_phrase = torch.zeros(n_ep, S + 1, dtype=torch.bool)
    for b, (sp, _) in enumerate(torch.utils.data.DataLoader(photos, batch_size=batch, num_workers=8)):
        out = sig.vision_model(pixel_values=sp.to(dev, torch.float16))
        per_patch = F.normalize(sig.vision_model.head(out.last_hidden_state.reshape(-1, 1, 768)).view(len(sp), 256, -1).float(), dim=-1)
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
    f.update(seg_move=torch.tensor([s[0] for s in sl], dtype=torch.int8),
             seg_side=torch.tensor([s[1] for s in sl], dtype=torch.int8),
             seg_degree=torch.tensor([s[2] for s in sl], dtype=torch.int8),
             seg_phrase=seg_phrase, seg_map=seg_map, has_phrase=has_phrase,
             parsed=torch.tensor([rows[e['episode']]['parsed'] is not None for e in eps]),
             stop_kind=torch.tensor([STOP_KINDS.index(rows[e['episode']]['stop_kind']) for e in eps], dtype=torch.int8))
    torch.save(f, f'features/{split}_v3.pt')
    print(f'{split}: {n_ep} episodes, {len(phrases)} distinct phrases, stop kinds',
          {k: int((f['stop_kind'] == i).sum()) for i, k in enumerate(STOP_KINDS)})


if __name__ == '__main__':
    main(sys.argv[1])
