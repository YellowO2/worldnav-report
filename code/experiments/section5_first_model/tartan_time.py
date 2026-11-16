"""TartanDrive: the per-move output (the points ARE the steps).

    python experiments/section5_first_model/tartan_time.py
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
import final  # noqa: E402
import siglip_steps as ss  # noqa: E402
import time_routes as T  # noqa: E402
from curves import log_curve  # noqa: E402
from endnet import batch  # noqa: E402
from floor_geometry import GROUPS  # noqa: E402
from score import load, score  # noqa: E402


def main():
    final.setup()
    vu, vu_eps = final.load_labelled('val_unseen'), load('data/lcvn/val_unseen.jsonl')
    tr = final.load_labelled('train')
    data = final.merge([tr, vu])
    eps = load('data/lcvn/train.jsonl') + vu_eps
    off = len(tr['episodes'])
    del tr
    ss.GLOB_DIMS['q_last'], ss.CELL_DIMS['qwen'] = data['q_last'].shape[-1], data['q_cells'].shape[-1]
    sizes = {k: int(data[f'seg_{k}'].max()) + 1 for k in ('move', 'side', 'degree')}
    sizes['stop'] = int(data['stop_kind'].max()) + 1
    Tt, nt = T.time_targets(eps)
    day = [e['trajectory'].split('/')[-1].split('_')[0] for e in vu_eps]
    val = torch.tensor([off + i for i, r in enumerate(day) if r in {'20210826', '20210828', '20210910'}])
    hold = torch.tensor([off + i for i, r in enumerate(day) if r == '20210829'])
    fit_t = torch.tensor([off + i for i, r in enumerate(day) if r in {'20210902', '20210903'}])
    ok = torch.isfinite(data['floor']).flatten(1).all(1)
    every = torch.cat([torch.where(ok[:off] & (data['group'][:off] != GROUPS.index('tartan')))[0], fit_t])
    print(f'tartan: fit {len(fit_t)} episodes, pick {len(hold)}, report {len(val)}; all robots {len(every)}; '
          f'true moves on the report days 10/50/90%: {np.percentile(nt[val].numpy(), [10, 50, 90])}', flush=True)
    pdim = ss.parse_vec(data, fit_t[:2], sizes).shape[1]
    dev = ss.DEV

    def make(idx):
        b = batch(data, idx, dev)
        return dict(camera=b['camera'], parse=ss.parse_vec(data, idx, sizes).to(dev), qwen=b['q_cells'], q_last=b['q_last'], q_text=b['q_text'])

    @torch.no_grad()
    def predict(m, idx):
        m.eval()
        out = [m.run2(make(idx[i:i + 512])) for i in range(0, len(idx), 512)]
        m.train()
        return torch.cat([o[0].float().cpu() for o in out]), torch.cat([o[1].float().log_softmax(-1).cpu() for o in out])

    def official(idx, P, counts):
        e = [eps[i] for i in idx.tolist()]
        acts = [T.moves_from(p, int(c)) for p, c in zip(P.numpy(), counts.tolist())]
        return score(e, dict(zip([x['episode'] for x in e], acts)), workers=8)[0]

    def outputs(idx, P, logp):
        ne, nm = [], []
        for i in range(0, len(idx), 256):
            a, b = T.choose_n(P[i:i + 256].to(dev), logp[i:i + 256].to(dev))
            ne.append(a.cpu()), nm.append(b.cpu())
        ne, nm = torch.cat(ne), torch.cat(nm)
        return {'count by expected score': official(idx, P, ne), 'most likely count': official(idx, P, nm),
                'true count (ceiling for the count)': official(idx, P, nt[idx])}, ne

    ph, pv = {e: [] for e in ss.CHECKPOINTS}, {e: [] for e in ss.CHECKPOINTS}
    for seed in range(int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else 3):
        torch.manual_seed(seed)
        m = T.TimeRoute(pdim).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=ss.LR, weight_decay=0.05)
        n = len(every)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, ss.LR, total_steps=ss.EPOCHS * (n // ss.BS), pct_start=0.05)
        for ep in range(1, ss.EPOCHS + 1):
            perm = every[torch.randperm(n)]
            tot = 0.0
            for k in range(n // ss.BS):
                idx = perm[k * ss.BS:(k + 1) * ss.BS]
                P, lg = m.run2(make(idx))
                tt, nn_ = Tt[idx].to(dev), nt[idx].to(dev)
                s, gap, span, mk = T.seq_score(P.float(), tt, nn_)
                loss = -s.mean() + 0.2 * ((gap * mk).sum(1) / mk.sum(1) / span).mean() + 0.5 * nn.functional.cross_entropy(lg.float(), nn_ - 1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss)
            if ep in ss.CHECKPOINTS:
                ph[ep].append(predict(m, hold))
                pv[ep].append(predict(m, val))
                P, lp = ph[ep][-1]
                hs = official(hold, P, lp.argmax(-1) + 1)['score']
                acc = float((lp.argmax(-1) + 1 == nt[hold]).float().mean())
                print(f'  seed {seed} epoch {ep}: train loss {tot / (n // ss.BS):.3f}, pick day (most likely count) {hs:.2f}, count exactly right {acc:.0%}', flush=True)
                log_curve('tartan_time', seed=seed, epoch=ep, train_loss=tot / (n // ss.BS), held=hs, held_count_acc=acc)
    avg = lambda d, e: (torch.stack([x[0] for x in d[e]]).mean(0), torch.stack([x[1] for x in d[e]]).exp().mean(0).log())
    hs = {e: official(hold, avg(ph, e)[0], avg(ph, e)[1].argmax(-1) + 1)['score'] for e in ss.CHECKPOINTS}
    best = max(hs, key=hs.get)
    (Ph, lh), (Pv, lv) = avg(ph, best), avg(pv, best)
    rh, _ = outputs(hold, Ph, lh)
    rv, ne = outputs(val, Pv, lv)
    step = lambda P, c: np.array([np.linalg.norm(np.diff(np.concatenate([[[0, 0]], p[:int(k)]]), axis=0), axis=1).mean() for p, k in zip(P.numpy(), c.tolist())])
    true_step = np.array([np.linalg.norm(np.diff(np.array(eps[i]['position']), axis=0), axis=1).mean() for i in val.tolist()])
    err = np.log(step(Pv, ne) / true_step)
    print(f'\nper-move output: epoch {best} (pick day by checkpoint ' + ' '.join(f'{hs[e]:.2f}' for e in ss.CHECKPOINTS) + f'); report days: mean step error sd {err.std():.3f}, '
          f'bias {err.mean():+.3f}, corr {np.corrcoef(np.log(step(Pv, ne)), np.log(true_step))[0, 1]:.2f}; count exactly right {float((lv.argmax(-1) + 1 == nt[val]).float().mean()):.0%}; '
          f'predicted moves 10/50/90%: {np.percentile(ne.numpy(), [10, 50, 90])}', flush=True)
    for name in rv:
        a, b = rh[name], rv[name]
        print(f'  {name:36s} pick day {a["score"]:6.2f} | report days {b["score"]:6.2f}  SR {b["SR"]:.3f}  ATE {b["ATE"]:.3f}  RPE {b["RPE"]:.3f}  F {b["F"]:.3f}  H {b["H"]:.3f}', flush=True)
    print('  reference: 16-point path + step size   pick day  61.51 | report days  62.49  SR 0.423  ATE 1.527  RPE 0.220  F 0.734  H 0.931')
    json.dump(dict(epoch=best, held=hs, pick={k: v['score'] for k, v in rh.items()}, report=rv), open('runs/tartan_time.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
