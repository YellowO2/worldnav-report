"""The per-robot submission: each robot's winning design from the step-by-step study.

    python 4_final/final_groups.py --name g1 --only tartan
"""
import argparse
import copy
import json
import os
import sys
import zipfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'eval')
import final  # noqa: E402
import pathnet  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from group_classifier import predict_groups  # noqa: E402
import siglip_steps as ss  # noqa: E402
from routenet import score_loss  # noqa: E402
from curves import log_curve  # noqa: E402

# robot -> (cells, globals, trained on all robots?, epochs, picked epoch)
DESIGNS = {
    'recon': (['qwen'], ['q_last', 'q_text'], True, 16, 6),
    'huron': (['qwen', 'dino', 'sig'], ['q_last', 'q_text', 'sentence'], False, 16, 16),
    'stanford_real': (['qwen'], ['q_last', 'q_text', 'sentence'], True, 16, 8),
    'scand_spot': (['qwen'], ['q_last', 'q_text', 'sentence'], True, 16, 12),
}
DESIGNS['stanford_sim'] = DESIGNS['stanford_real']      # experiments/section7_per_robot/final_compare.py
DESIGNS['scand_jackal'] = DESIGNS['scand_spot']         # best for SCAND Jackal (58.6)
MIX_WITH_OLD = {'recon', 'huron', 'stanford_real', 'scand_spot', 'scand_jackal'}   # average with f6's routes there
DESIGNS['tartan'] = (['qwen', 'dino'], ['q_last', 'q_text'], True, 16, 16)   # experiments/section7_per_robot/tartan_steps.py
DEFAULT = DESIGNS['recon']
SMOKE = None                             # --smoke: this many batches per epoch
TARTAN_FROM = None                       # an earlier run's TartanDrive routes instead


LATER_HURON = (['qwen'], ['q_last', 'q_text', 'sentence'], False, 16, 16)   # --later: no DINOv3 / SigLIP grids of later photos


def main(name, only=None, seeds=(0, 1, 2), later=False):
    final.setup()
    steps = pathnet.typical_steps()
    designs = dict(DESIGNS, huron=LATER_HURON) if later else DESIGNS
    tr = final.merge([final.load_labelled(s) for s in ('train', 'val_seen') + (() if later else ('val_unseen',))])
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test).cpu()
    final.add_qwen(test, 'test')
    test['gravity'] = test['gravity'].float()
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = tr['q_last'].shape[-1], tr['q_cells'].shape[-1]
    if later:                                            # later-start examples of every labelled route
        from later_data import add_later, drop_images
        assert only and 'tartan' not in only
        drop_images(tr)
        drop_images(test)
        n_first = add_later(tr, ('train', 'val_seen'), list(only))      # rows from n_first on are later-start examples
    sizes = {k: int(max(tr[f'seg_{k}'].max(), test[f'seg_{k}'].max())) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(max(tr['stop_kind'].max(), test['stop_kind'].max())) + 1
    ok = torch.isfinite(tr['floor']).flatten(1).all(1)
    tartan = GROUPS.index('tartan')
    pdim = ss.parse_vec(tr, torch.arange(2), sizes).shape[1]
    out = np.load(f'runs/{TARTAN_FROM or "f5"}/test_routes.npy')    # filled in below, robot by robot
    assert len(out) == len(test['episodes'])
    dev = ss.DEV

    def make(f, idx, cells, globs):
        from endnet import batch
        b = batch(f, idx, dev)
        im = f['image_index'][idx]
        o = dict(camera=b['camera'], parse=ss.parse_vec(f, idx, sizes).to(dev))
        if 'sig' in cells:
            o['sig'] = f['sig_tokens'][im].to(dev).float()
        if 'dino' in cells:
            o['dino'] = f['dino_tokens'][im].to(dev).float()
        if 'sentence' in globs:
            o['sentence'] = f['sentence'][idx].to(dev).float()
        o['qwen'], o['q_last'], o['q_text'] = b['q_cells'], b['q_last'], b['q_text']
        return o

    trained = {}
    for g, gn in enumerate(GROUPS):
        vi = torch.where(test['group'] == g)[0]
        if (gn == 'tartan' and TARTAN_FROM) or len(vi) == 0 or (only and gn not in only):
            continue
        cells, globs, every, epochs, pick = designs.get(gn, DEFAULT)
        with_tartan = gn == 'tartan'             # only TartanDrive's design trained on TartanDrive routes too
        key = (tuple(cells), tuple(globs), gn if later or not every else (every, with_tartan), epochs)
        if key not in trained:                   # one run serves every robot sharing the design
            picks = {designs.get(x, DEFAULT)[4] for x in GROUPS if (not only or x in only) and (test['group'] == GROUPS.index(x)).any()
                     and (tuple(designs.get(x, DEFAULT)[0]), tuple(designs.get(x, DEFAULT)[1]),
                          x if later or not designs.get(x, DEFAULT)[2] else (designs.get(x, DEFAULT)[2], x == 'tartan'),
                          designs.get(x, DEFAULT)[3]) == key}
            tag = '+'.join(cells + globs) + ('_all' if every and not later else f'_{gn}')
            robots = (torch.ones_like(ok) if with_tartan else tr['group'] != tartan) if every else tr['group'] == g
            if later and every:                  # the recipe of experiments/section7_per_robot/later_starts.py arm 5: this robot's first photos and
                first = torch.arange(len(ok)) < n_first      # later starts + the other robots' first photos
                stanford = torch.isin(tr['group'], torch.tensor([GROUPS.index('stanford_real'), GROUPS.index('stanford_sim')]))
                robots = (tr['group'] == g) | (first & (tr['group'] != tartan) & ~stanford)
            ti = torch.where(robots & ok)[0]
            models = {p: [] for p in picks}
            for seed in seeds:
                torch.manual_seed(seed)
                m = ss.GridRoute(cells, globs, pdim).to(dev)
                opt = torch.optim.AdamW(m.parameters(), lr=ss.LR, weight_decay=0.05)
                n = len(ti)
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, ss.LR, total_steps=epochs * (n // ss.BS), pct_start=0.05)
                for ep in range(1, max(picks) + 1):      # the schedule of a full run, stopped at the last picked epoch
                    perm = ti[torch.randperm(n)]
                    tot = 0.0
                    for k in range(min(n // ss.BS, SMOKE or 10 ** 9)):
                        idx = perm[k * ss.BS:(k + 1) * ss.BS]
                        loss = score_loss(m.run(make(tr, idx, cells, globs)).float(), tr['floor'][idx].to(dev))
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()
                        sched.step()
                        tot += float(loss)
                    log_curve(f'{name}_{tag}_seed{seed}', epoch=ep, train_loss=tot / min(n // ss.BS, SMOKE or 10 ** 9))
                    if ep in picks:
                        models[ep].append(copy.deepcopy(m).eval())
                        if later:
                            os.makedirs(f'runs/{name}', exist_ok=True)
                            torch.save(m.state_dict(), f'runs/{name}/model_{tag}_seed{seed}_epoch{ep}.pt')
                print(f'{gn}: design {key[:2]}, seed {seed} trained on {n} episodes, epochs {sorted(picks)}', flush=True)
            trained[key] = models
        with torch.no_grad():
            preds = [np.concatenate([m.run(make(test, vi[i:i + 1024], cells, globs)).float().cpu().numpy()
                                     for i in range(0, len(vi), 1024)]) for m in trained[key][pick]]
        out[vi.numpy()] = np.mean(preds, 0)
        os.makedirs(f'runs/{name}', exist_ok=True)
        np.save(f'runs/{name}/routes_{gn}.npy', dict(index=vi.numpy(), routes=np.mean(preds, 0)), allow_pickle=True)
        print(f'{gn}: {len(vi)} test episodes, saved runs/{name}/routes_{gn}.npy', flush=True)
    if only:
        return
    write(name, out, test, steps)


def write(name, out, test, steps):
    os.makedirs(f'runs/{name}', exist_ok=True)
    np.save(f'runs/{name}/test_routes.npy', out)
    moves = final.to_moves(out, test['group'], steps)
    with zipfile.ZipFile(f'runs/{name}/submission_test.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('test.json', json.dumps({'split': 'test', 'episodes': [{'episode_id': ep, 'actions': a}
                                                                          for ep, a in zip(test['episodes'], moves)]}))
    print(f'saved runs/{name}/submission_test.zip: {len(moves)} test episodes')


def combine(name, old='f6', more=(), mix=None, src=None):
    """The per-robot routes saved by --only runs."""
    final.setup()
    steps = pathnet.typical_steps()
    test = pathnet.load_split('test', with_labels=False)
    test['group'] = predict_groups(test).cpu()
    base = np.load(f'runs/{old}/test_routes.npy')
    out = base.copy()
    for gn in GROUPS:
        p = f'runs/{src or name}/routes_{gn}.npy'
        if not os.path.exists(p):
            print(f'{gn}: no new routes, keeping {old}', flush=True)
            continue
        d = np.load(p, allow_pickle=True).item()
        assert (test['group'][d['index']] == GROUPS.index(gn)).all()
        extra = [np.load(f'runs/{m}/routes_{gn}.npy', allow_pickle=True).item() for m in more
                 if os.path.exists(f'runs/{m}/routes_{gn}.npy')]
        assert all((x['index'] == d['index']).all() for x in extra)
        d['routes'] = np.mean([d['routes']] + [x['routes'] for x in extra], 0)    # equal seed counts per run
        mixed = mix if mix is not None else MIX_WITH_OLD
        out[d['index']] = (d['routes'] + base[d['index']]) / 2 if gn in mixed else d['routes']
        print(f'{gn}: {len(d["index"])} episodes, {"averaged with " + old if gn in mixed else "new only"}', flush=True)
    write(name, out, test, steps)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--only', nargs='*', help='train and save these robots only (no zip)')
    ap.add_argument('--combine', action='store_true', help='assemble the saved robots into the zip, averaging with f6')
    ap.add_argument('--seeds', nargs='*', type=int, default=[0, 1, 2])
    ap.add_argument('--more', nargs='*', default=[], help='with --combine: other runs whose per-robot routes are averaged in')
    ap.add_argument('--from-run', default='g1', help='with --combine: the run whose per-robot routes are the base')
    ap.add_argument('--mix', nargs='*', help='with --combine: robots averaged with f6 (default MIX_WITH_OLD)')
    ap.add_argument('--later', action='store_true', help='with --only: add later-start examples (2_features/later_data.py); models saved')
    ap.add_argument('--smoke', type=int, help='batches per epoch, to check the code path end to end')
    a = ap.parse_args()
    SMOKE = a.smoke
    if a.combine:
        combine(a.name, more=a.more, mix=set(a.mix) if a.mix is not None else None, src=a.from_run)
    else:
        main(a.name, a.only, tuple(a.seeds), a.later)
