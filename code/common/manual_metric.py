"""Are the dataset units metres? Metric depth at hand-marked stop landmarks vs the true route's end distance.

    python common/manual_metric.py
"""
import io
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import source  # noqa: E402

# case number in config/cases.json ->
GRID = 20
MARKS = {2: (7, 10), 4: (10, 14), 8: (12, 11), 9: (9, 11), 10: (9, 10), 12: (11, 12),
         14: (4, 12), 20: (10, 12), 21: (6, 12), 22: (9, 12), 23: (13, 11), 24: (12, 11),
         25: (12, 11), 27: (7, 12), 29: (9, 11), 30: (13, 14), 31: (8, 12), 34: (14, 12),
         36: (15, 11), 37: (10, 10), 38: (10, 9), 42: (8, 10), 43: (10, 11), 44: (3, 11), 45: (10, 10)}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def load_da3(preset, repo):
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    net = create_object(load_config(MODEL_REGISTRY[preset]))
    sd = load_file(snapshot_download(repo) + '/model.safetensors')
    print(repo, net.load_state_dict({k.removeprefix('model.'): v for k, v in sd.items()}, strict=False))
    return net.eval()


def load_nested():
    # 1.1: retrained after a training bug
    return load_da3('da3nested-giant-large', 'depth-anything/DA3NESTED-GIANT-LARGE-1.1')


def cell_median(depth, col, row):
    h, w = depth.shape
    return float(np.median(depth[row * h // GRID:(row + 1) * h // GRID, col * w // GRID:(col + 1) * w // GRID]))


@torch.no_grad()
def main():
    dev = 'cuda'
    cases = json.load(open('config/cases.json'))
    frames = json.load(open('data/lcvn/train.frames.json'))
    tar = open('data/lcvn/train.tar', 'rb')
    nested = load_nested().to(dev)
    metric = load_da3('da3metric-large', 'depth-anything/DA3METRIC-LARGE').to(dev)
    dp_proc = DepthProImageProcessorFast.from_pretrained('apple/DepthPro-hf')
    depth_pro = DepthProForDepthEstimation.from_pretrained('apple/DepthPro-hf', dtype=torch.float16).to(dev).eval()

    rows = []
    for k, (col, row) in MARKS.items():
        c = cases[k]
        o, s = frames[c['trajectory']][0]
        tar.seek(o)
        photo = Image.open(io.BytesIO(tar.read(s))).convert('RGB')
        w, h = photo.size
        side = 504 / max(w, h)
        W, H = round(w * side / 14) * 14, round(h * side / 14) * 14
        x = torch.from_numpy(np.asarray(photo.resize((W, H), Image.BICUBIC))).permute(2, 0, 1).float() / 255
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD)[None, None].to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            out = nested(x)
        d_nested = out.depth[0, 0].float().cpu().numpy()
        focal_in = float(out.intrinsics[0, 0, :2, :2].diagonal().mean())              # at the network's input size
        focal = focal_in / side                                                         # in original pixels
        with torch.autocast('cuda', dtype=torch.float16):
            raw = metric(x).depth[0, 0].float().cpu().numpy()
        d_metric = focal_in * raw / 300      # DA3 README: metric_depth = focal * net_output / 300
        inputs = dp_proc(images=photo, return_tensors='pt').to(dev)
        post = dp_proc.post_process_depth_estimation(depth_pro(pixel_values=inputs['pixel_values'].half()),
                                                     target_sizes=[(h, w)])[0]
        d_pro = post['predicted_depth'].float().cpu().numpy()
        p = np.array(c['position'])
        end = float(np.linalg.norm(p[-1] - p[0]))
        dn, dm, dp = (cell_median(d, col, row) for d in (d_nested, d_metric, d_pro))
        rows.append(dict(case=k, source=source(c), phrase=c['phrase'], end=end, da3=dn, da3_metric=dm, depth_pro=dp,
                         focal_da3=focal, focal_pro=float(post['focal_length'])))
        print(f'#{k:2d} {rows[-1]["source"]:8s} {c["phrase"][:28]:28s} stopped {end:5.2f}  nested-1.1 {dn:6.2f}m  '
              f'metric-large {dm:6.2f}m  DepthPro {dp:6.2f}m  focal nested {focal:4.0f} Pro {rows[-1]["focal_pro"]:4.0f}',
              flush=True)
    json.dump(rows, open('config/metric.json', 'w'), indent=1)
    print()
    keys = ('da3', 'da3_metric', 'depth_pro')
    for src in ('scand', 'recon', 'stanford', 'huron', None):
        R = [r for r in rows if src is None or r['source'] == src]
        for key in keys:
            ratios = np.array([r['end'] / r[key] for r in R])
            corr = np.corrcoef([r[key] for r in R], [r['end'] for r in R])[0, 1]
            print(f'{src or "all":8s} {key:10s} n={len(R):2d} stopped/depth median {np.median(ratios):.2f} '
                  f'(min {ratios.min():.2f}, max {ratios.max():.2f}) corr {corr:.2f}')


if __name__ == '__main__':
    main()
