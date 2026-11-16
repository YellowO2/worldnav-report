"""Does the distance to the stop phrase in the photo carry length information our model misses? For one robot's val routes: where the stop phrase's SigLIP2 map (16 x 16) peaks, that patch's floor point (the photo's own lean), its distance in metres.

    python experiments/section6_limits/stop_distance_check.py recon qsteps_recon
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
from floor_geometry import NATIVE_SIZE, ground_point, group_of, load_cameras, with_gravity  # noqa: E402
from path_labels import start_frame  # noqa: E402
from score import load  # noqa: E402


def length(r):
    return float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum())


def main(group, run):
    eps = load('data/lcvn/val_seen.jsonl')
    text = torch.load('features/val_seen_v3_text.pt')
    grav = torch.load('features/val_seen_gravity.pt')['gravity'].numpy()
    idx = np.load(f'runs/{run}/val_index.npy')
    ours = np.load(f'runs/{run}/val_routes.npy')
    cam0, (W, H) = load_cameras()[group], NATIVE_SIZE[group]
    rk = lambda x: np.argsort(np.argsort(x))  # noqa: E731
    rows = []
    for k, i in enumerate(idx):
        e = eps[i]
        if not e['episode'].endswith('_2') or not bool(text['has_phrase'][i, -1]):
            continue
        m = text['seg_map'][i, -1].float().view(16, 16).numpy()
        r, c = np.unravel_index(m.argmax(), m.shape)
        g = ground_point((c + 0.5) / 16 * W, (r + 0.5) / 16 * H, with_gravity(cam0, grav[i]))
        tl = length(start_frame(e['position'], e['yaw'][0]))
        ol = length(np.concatenate([[[0, 0]], ours[k]]))
        rows.append((np.hypot(*g) if g is not None else np.nan, m.max(), tl, ol, r))
    a = np.array(rows, float)
    ok = np.isfinite(a[:, 0]) & (a[:, 0] < 30)
    d, peak, tl, ol, row = a[ok].T
    err = np.log(tl / ol)
    print(f'{group}: {len(a)} val routes with a stop phrase; peak on the floor within 30 m: {ok.sum()} ({ok.mean():.0%})')
    print(f'  distance to the stop phrase: median {np.median(d):.1f} m')
    print(f'  rank corr with true length {np.corrcoef(rk(d), rk(tl))[0, 1]:+.2f}; with our length {np.corrcoef(rk(d), rk(ol))[0, 1]:+.2f}; '
          f'with our error log(true / ours) {np.corrcoef(rk(d), rk(err))[0, 1]:+.2f}')
    strong = peak > np.median(peak)
    print(f'  strong peaks only ({strong.sum()}): corr with true length {np.corrcoef(rk(d[strong]), rk(tl[strong]))[0, 1]:+.2f}, '
          f'with our error {np.corrcoef(rk(d[strong]), rk(err[strong]))[0, 1]:+.2f}')
    print(f'  peak image row (higher = farther) rank corr with true length {np.corrcoef(rk(-row), rk(tl))[0, 1]:+.2f}, '
          f'with our error {np.corrcoef(rk(-row), rk(err))[0, 1]:+.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
