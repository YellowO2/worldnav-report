"""Census of the missing pieces per robot (train + val_seen + val_unseen labelled).

    python experiments/section9_data_split/hole_census.py
"""
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, 'eval')
from floor_geometry import group_of  # noqa: E402
from lookup_final import ROBOTS, SPLITS, test_photos  # noqa: E402
from lookup_routes import piece  # noqa: E402
from score import load  # noqa: E402


def main():
    rec = defaultdict(lambda: defaultdict(set))
    for s in SPLITS:
        for t in {e['trajectory'] for e in load(f'data/lcvn/{s}.jsonl')}:
            key, k = piece(t)
            rec[group_of(t, s)][key].add(k)
    _, t_by = test_photos()
    done = Counter()
    try:
        rep = set(json.load(open('runs/h1/replaced.json')))
        for g in ROBOTS:
            done[g] = sum(any(i in rep for i in ids) for _, ids in t_by[g].values())
    except FileNotFoundError:
        pass
    print('robot           test photos  replaced in h1 | closed runs of missing pieces by length (pieces in them): 1, 2, 3, 4-6, 7+ | missing after the last piece is unknown')
    for g in ROBOTS:
        runs = Counter()
        for key, ks in rec[g].items():
            ks = sorted(ks)
            for a, b in zip(ks, ks[1:]):
                if b - a > 1:
                    runs[b - a - 1] += 1
        n = lambda lo, hi: sum(v * k for k, v in runs.items() if lo <= k <= hi)
        print(f'{g:15s} {len(t_by[g]):6d} {done[g]:12d}       | {n(1, 1):5d} {n(2, 2):5d} {n(3, 3):5d} {n(4, 6):5d} {n(7, 10 ** 6):6d}   ({len(rec[g])} recordings)')


if __name__ == '__main__':
    main()
