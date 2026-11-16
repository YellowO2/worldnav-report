"""Each fisheye robot's lens from AnyCalib (ICCV 2025, github.com/javrtg/AnyCalib).

    python common/anycalib_lens.py
"""
import io
import json
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import source  # noqa: E402

SOURCES, PER_SOURCE, OUT_HFOV = ('stanford', 'huron', 'recon'), 50, 100.0


def undistort_kb(img, lens, out_w):
    """PIL fisheye image -> 100 deg pinhole PIL image through a Kannala-Brandt lens."""
    import cv2
    fx, fy, cx, cy, k1, k2, k3, k4 = lens['params']
    W, H = img.size
    out_h = int(round(out_w * H / W))
    f_pin = (out_w / 2) / math.tan(math.radians(OUT_HFOV / 2))
    u, v = np.meshgrid(np.arange(out_w) + 0.5, np.arange(out_h) + 0.5)
    x, y = (u - out_w / 2) / f_pin, (v - out_h / 2) / f_pin
    r = np.hypot(x, y)
    th = np.arctan(r)
    thd = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8)
    s = np.where(r > 1e-9, thd / np.maximum(r, 1e-9), 1.0)
    mx, my = (cx + fx * x * s).astype(np.float32), (cy + fy * y * s).astype(np.float32)
    return Image.fromarray(cv2.remap(np.asarray(img), mx, my, cv2.INTER_LINEAR)), f_pin


@torch.no_grad()
def main():
    from anycalib import AnyCalib
    dev = 'cuda'
    model = AnyCalib(model_id='anycalib_gen').to(dev)
    frames = json.load(open('data/lcvn/train.frames.json'))
    tar = open('data/lcvn/train.tar', 'rb')
    random.seed(2)
    trajs = sorted(frames)
    random.shuffle(trajs)
    params, sample = {s: [] for s in SOURCES}, {}
    for t in trajs:
        s = source({'trajectory': t})
        if s not in params or len(params[s]) >= PER_SOURCE:
            continue
        tar.seek(frames[t][0][0])
        img = Image.open(io.BytesIO(tar.read(frames[t][0][1]))).convert('RGB')
        x = torch.tensor(np.array(img), dtype=torch.float32, device=dev).permute(2, 0, 1) / 255
        out = model.predict(x, cam_id='kb:4')
        params[s].append(out['intrinsics'].flatten().cpu().numpy().tolist())
        sample.setdefault(s, []).append(img)
        if all(len(v) >= PER_SOURCE for v in params.values()):
            break
    lenses, rows = {}, []
    for s, P in params.items():
        P = np.array(P)
        med = np.median(P, 0)
        lo, hi = np.percentile(P, 25, 0), np.percentile(P, 75, 0)
        lenses[s] = dict(model='kb:4', params=med.tolist(), size=list(sample[s][0].size),
                         out_width=sample[s][0].size[0] * (2 if sample[s][0].size[0] <= 160 else 1))
        th_edge = None
        print(f'{s:9s} fx {med[0]:.1f} ({lo[0]:.1f}-{hi[0]:.1f})  fy {med[1]:.1f}  centre ({med[2]:.1f}, {med[3]:.1f})  '
              f'k {np.round(med[4:], 3).tolist()}', flush=True)
        for img in sample[s][:3]:
            b, _ = undistort_kb(img, lenses[s], lenses[s]['out_width'])
            h = 260
            A, B = img.resize((int(img.size[0] * h / img.size[1]), h)), b.resize((int(b.size[0] * h / b.size[1]), h))
            row = Image.new('RGB', (A.size[0] + B.size[0] + 20, h + 22), 'white')
            ImageDraw.Draw(row).text((2, 4), f'{s}: original | corrected with AnyCalib lens', fill='black')
            row.paste(A, (0, 22))
            row.paste(B, (A.size[0] + 20, 22))
            rows.append(row)
    json.dump(lenses, open('config/fisheye_anycalib.json', 'w'), indent=1)
    W = max(r.size[0] for r in rows)
    sheet = Image.new('RGB', (W, sum(r.size[1] + 8 for r in rows)), 'white')
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.size[1] + 8
    sheet.save('runs/undistort_anycalib.png')
    print('saved runs/undistort_anycalib.png')


if __name__ == '__main__':
    main()
