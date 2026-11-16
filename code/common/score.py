"""Scores predicted action sequences against a converted split with the official metric functions.

    python common/score.py data/lcvn/val_seen.jsonl predictions.json
"""
import json
import math
import multiprocessing
import sys

import numpy as np

from official_metric import (absolute_trajectory_error, full_route_fidelity, heading_and_turn_fidelity,
                             relative_pose_error, success_rate)

MAX_ACTIONS, MAX_STEP = 64, 2.46


def rollout(actions, x, y, yaw):
    xy, yaws = [(x, y)], [yaw]
    for dx, dy, dyaw in actions:
        x += math.cos(yaw) * dx - math.sin(yaw) * dy
        y += math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw += dyaw
        xy.append((x, y))
        yaws.append(yaw)
    return xy, yaws


def clean(actions):
    out = []
    for dx, dy, dyaw in actions[:MAX_ACTIONS]:
        n = math.hypot(dx, dy)
        if n > MAX_STEP:
            dx, dy = dx * MAX_STEP / n, dy * MAX_STEP / n
        out.append((dx, dy, max(-6.29, min(6.29, dyaw))))
    return out


def episode_metrics(actions, ref_xy, ref_yaw):
    ref_xy = [tuple(p) for p in ref_xy]
    pred_xy, pred_yaw = rollout(clean(actions), *ref_xy[0], ref_yaw[0])
    step = float(np.mean([math.dist(a, b) for a, b in zip(ref_xy, ref_xy[1:])]))
    m = dict(SR=success_rate(pred_xy, ref_xy[-1], step),
             ATE=absolute_trajectory_error(pred_xy, ref_xy),
             RPE=relative_pose_error(pred_xy, ref_xy),
             F=full_route_fidelity(pred_xy, ref_xy),
             H=heading_and_turn_fidelity(pred_xy, pred_yaw, ref_xy, list(ref_yaw)))
    m['score'] = 100 * m['F'] * (1 + m['H']) / 2 * (0.7 + 0.1 * m['SR'] + 0.1 / (1 + m['ATE']) + 0.1 / (1 + m['RPE']))
    return m


def _one(args):
    return episode_metrics(*args)


def score(episodes, predictions, workers=8):
    """Mean metrics over the episodes, and each episode's row."""
    jobs = [(predictions[e['episode']], e['position'], e['yaw']) for e in episodes]
    with multiprocessing.Pool(workers) as pool:
        rows = pool.map(_one, jobs, chunksize=64)
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


def load(path):
    """Episodes of a converted split."""
    with open(path) as f:
        episodes = [json.loads(line) for line in f]
    for e in episodes:
        e['yaw'] = np.asarray(e['yaw'], dtype=float).reshape(-1).tolist()
    return episodes


if __name__ == '__main__':
    with open(sys.argv[2]) as f:
        preds = json.load(f)
    means, _ = score(load(sys.argv[1]), preds)
    print(json.dumps({k: round(v, 4) for k, v in means.items()}))
