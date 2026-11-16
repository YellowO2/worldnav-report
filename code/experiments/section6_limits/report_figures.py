"""Report figures sized for an A4 page (17 cm wide).

Figure 6.1: routes whose labels turn the other way from the instruction: the start photo with the recorded route
(green) and our per-robot model's route (magenta, never trained on these photos).
Figure 7.1: typical routes of the per-robot models (saved val routes), ours against the truth.

    python eval/report_figures.py   ->  runs/manual/fig6_1_labels.png, runs/manual/fig7_1_examples.png
"""
import io
import json
import random
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS, load_cameras, project, with_gravity  # noqa: E402
from score import load, score  # noqa: E402

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7})
LABELS = ['jackal_2019-12-23-14-56-39_6_r01_3_0_0',                   # says turn right, goes left (candidates W5)
          'Feb-16-2023-bww1-intloss_00000032_6_3_0_0',               # says turn right, goes left (W13)
          'Feb-23-2023-soda3-intloss_00000003_4_12_0_0']             # says turn left, goes right (W16)
STOPS = ['jackal_2019-10-20-15-57-42_1_r00_4_0_2',                   # stop in front of the white building: 1.7 m
         'random_mdps_A_Spot_Jester_Jester_Wed_Nov_10_63_0_0_0_2',  # to the glass doors at the end: 2.6 m
         'jackal_2019-08-02-18-16-17_5_r02_3_0_2']                  # near the white house at the end of the path: 1.3 m
EXAMPLES = [('recon', 4), ('huron', 2), ('stanford_real', 6), ('recon', 12)]   # panels of samples_view.py's sheets

final.setup()
va, eps = final.load_labelled('val_seen'), load('data/lcvn/val_seen.jsonl')
frames = json.load(open('data/lcvn/val_seen.frames.json'))
tar = open('data/lcvn/val_seen.tar', 'rb')
cams = load_cameras()
where = {e['episode']: i for i, e in enumerate(eps)}


def photo(i, which):
    o, z = frames[eps[i]['trajectory']][which]
    tar.seek(o)
    return Image.open(io.BytesIO(tar.read(z))).convert('RGB')


def on_photo(ax, i, routes):
    img = photo(i, 0)
    cam = with_gravity(cams[GROUPS[int(va['group'][i])]], va['gravity'][i].numpy())
    ax.imshow(img)
    for pts, col in routes:
        pix = np.array([q if q else (np.nan, np.nan) for q in
                        (project(*p, cam) if p[0] > 0.05 else None for p in np.concatenate([[[0.3, 0]], pts]))], float)
        ax.plot(pix[:, 0], pix[:, 1], '-', color=col, lw=1.8)
        ax.plot(pix[-1, 0], pix[-1, 1], 'o', color=col, mec='k', mew=0.6, ms=4)
    ax.set(xlim=(0, img.size[0]), ylim=(img.size[1], 0))
    ax.axis('off')


def length(r):
    return float(np.linalg.norm(np.diff(np.concatenate([[[0, 0]], r]), axis=0), axis=1).sum())


def from_above(ax, routes, title):
    lim = 1.0
    for pts, col, lab in routes:
        t = np.concatenate([[[0, 0]], pts])
        ax.plot(t[:, 1], t[:, 0], '-', color=col, lw=1.5, label=lab)
        ax.plot(t[-1, 1], t[-1, 0], 'o', color=col, ms=3.5)
        lim = max(lim, np.abs(t).max())
    lim *= 1.15
    ax.plot(0, 0, 'k^', ms=5)
    ax.set(xlim=(-lim, lim), ylim=(-0.15 * lim, lim), aspect='equal')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_title(title, fontsize=7)


def wrap(s, n):
    return '\n'.join(textwrap.wrap(s, n))


def sample(group, j):
    """Index of panel j of samples_view.py's sheet for this robot (same random draw)."""
    g = GROUPS.index(group)
    vi = [i for i, e in enumerate(eps) if int(va['group'][i]) == g and e['episode'].endswith('_2')]
    random.seed(0)
    return sorted(random.sample(vi, 16))[j]


def labels_figure():
    """Top row: wrong turns, with our route. Bottom row: stops long before the named landmark (recorded route only)."""
    fig, ax = plt.subplots(2, 3, figsize=(6.9, 4.7))
    for a, ep in zip(ax[0], LABELS):
        i = where[ep]
        g = GROUPS[int(va['group'][i])]
        idx = list(np.load(f'runs/qsteps_{g}/val_index.npy'))
        ours = np.load(f'runs/qsteps_{g}/val_routes.npy')[idx.index(i)]
        on_photo(a, i, [(va['floor'][i].numpy(), 'lime'), (ours, 'magenta')])
        a.set_title(wrap(f'{g.replace("_", " ")}: "{eps[i]["instruction"]}"', 40), loc='left', fontsize=6.5, x=0, ha='left')
    for a, ep in zip(ax[1], STOPS):
        i = where[ep]
        g = GROUPS[int(va['group'][i])]
        true = va['floor'][i].numpy()
        on_photo(a, i, [(true, 'lime')])
        text = eps[i]['instruction']
        a.set_title(wrap(f'{g.replace("_", " ")}, route {length(true):.1f} m: "{text}"', 46), loc='left', fontsize=6,
                    x=0, ha='left')
    plt.tight_layout(h_pad=0.4, w_pad=0.6, rect=(0.03, 0, 1, 1))
    for row, label in zip(ax, ('Wrong turns', 'Unclear stops')):
        box = row[0].get_position()
        fig.text(0.008, (box.y0 + box.y1) / 2, label, rotation=90, va='center', ha='left', fontsize=8, fontweight='bold')
    plt.savefig('runs/manual/fig6_1_labels.png', dpi=220, bbox_inches='tight', pad_inches=0.02)


def examples_figure():
    steps = pathnet.typical_steps()
    fig, ax = plt.subplots(2, 4, figsize=(6.9, 3.9), gridspec_kw=dict(width_ratios=[1.3, 1, 1.3, 1]))
    for k, (group, j) in enumerate(EXAMPLES):
        i = sample(group, j)
        idx = list(np.load(f'runs/qsteps_{group}/val_index.npy'))
        ours = np.load(f'runs/qsteps_{group}/val_routes.npy')[idx.index(i)]
        true = va['floor'][i].numpy()
        sc = score([eps[i]], {eps[i]['episode']: final.to_moves(ours[None], va['group'][i:i + 1], steps)[0]})[1][0]['score']
        a_img, a_top = ax[k // 2, 2 * (k % 2)], ax[k // 2, 2 * (k % 2) + 1]
        on_photo(a_img, i, [(true, 'lime'), (ours, 'magenta')])
        a_img.set_title(wrap(f'{group.replace("_", " ")}: "{eps[i]["instruction"]}"', 52), loc='left', fontsize=6, x=0, ha='left')
        from_above(a_top, [(true, 'green', 'true'), (ours, 'magenta', 'ours')],
                   f'score {sc:.0f}\ntrue {length(true):.1f} m, ours {length(ours):.1f} m')
        if k == 0:
            a_top.legend(fontsize=6, loc='upper right', handlelength=1.2)
    plt.tight_layout(h_pad=0.8, w_pad=0.4)
    plt.savefig('runs/manual/fig7_1_examples.png', dpi=220, bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    labels_figure()
    examples_figure()
    print('saved runs/manual/fig6_1_labels.png, runs/manual/fig7_1_examples.png')
