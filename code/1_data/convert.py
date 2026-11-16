"""Reads every traj_data.pkl in a packed split (see pack.py) and writes one JSON line per episode.

    python 1_data/convert.py data/lcvn/val_seen.tar
"""
import os
import collections
import io
import json
import pickle
import sys
import tarfile

import numpy as np

ALLOWED = {
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
    ('collections', 'OrderedDict'),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f'refused {module}.{name}')
        return super().find_class(module, name)


def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist()) if x.dtype == object else x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def main(tar_path):
    base = tar_path[:-len('.tar')]
    frames = collections.defaultdict(dict)   # trajectory -> frame index -> [offset, size] in the tar
    episodes = []
    with tarfile.open(tar_path) as tar:
        for m in tar:
            traj, name = m.name.rsplit('/', 1)
            if name.endswith('.jpg'):
                frames[traj][int(name[:-4])] = [m.offset_data, m.size]
            elif name.endswith('.pkl'):
                d = plain(SafeUnpickler(io.BytesIO(tar.extractfile(m).read())).load())
                episodes.append({'episode': f'{traj.split("/", 1)[1]}_{name[:-4]}', 'trajectory': traj, **d})
    with open(base + '.jsonl', 'w') as f:
        for e in episodes:
            e['n_frames'] = len(frames[e['trajectory']])
            f.write(json.dumps(e) + '\n')
    with open(base + '.frames.json', 'w') as f:
        json.dump({t: [v[i] for i in sorted(v)] for t, v in frames.items()}, f)
    print(f'{tar_path}: {len(episodes)} episodes, {len(frames)} trajectories, keys {sorted(episodes[0])}')


def main_test(folder):
    out = os.path.join(os.path.dirname(os.path.dirname(folder.rstrip('/'))), 'test.jsonl')
    names = sorted(os.listdir(folder))
    with open(out, 'w') as f:
        for name in names:
            with open(os.path.join(folder, name, 'traj_data.pkl'), 'rb') as p:
                d = plain(SafeUnpickler(p).load())
            f.write(json.dumps({'episode': name, 'image': os.path.join(folder, name, '0.jpg'), **d}) + '\n')
    print(f'{out}: {len(names)} episodes, keys {sorted(d)}')


if __name__ == '__main__':
    main(sys.argv[1]) if sys.argv[1].endswith('.tar') else main_test(sys.argv[1])
