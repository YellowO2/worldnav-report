"""Training losses and validation scores per epoch for path-model runs.

    python common/plot_curves.py p1 p2 ...
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for n, c in zip(sys.argv[1:], ['tab:gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']):
    p = f'runs/{n}/log.json'
    if not os.path.exists(p):
        continue
    L = json.load(open(p))['epochs']
    ep = [r['epoch'] for r in L]
    ax[0].plot(ep, [r['train_uv'] for r in L], '-o', ms=3, color=c, label=n)
    if 'train_metres' in L[0]:
        ax[1].plot(ep, [r['train_metres'] for r in L], '-o', ms=3, color=c, label=n)
    v = [r for r in L if 'phase1' in r]
    ax[2].plot([r['epoch'] for r in v], [r['val_seen']['score'] for r in v], '-o', color=c, label=f'{n} val_seen')
    ax[2].plot([r['epoch'] for r in v], [r['val_unseen']['score'] for r in v], '--s', color=c, label=f'{n} val_unseen')
ax[0].set(title='training pixel loss (L1, image widths)', xlabel='epoch')
ax[1].set(title='training metre loss (smooth L1, metres)', xlabel='epoch')
ax[2].set(title='validation score (out of 100)', xlabel='epoch')
for a in ax:
    a.grid(alpha=.3)
    a.legend(fontsize=7)
plt.tight_layout()
plt.savefig('runs/curves.png', dpi=80)
