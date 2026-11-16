"""Packs one LCVN split, streamed as a tar on stdin, into a smaller tar."""
import hashlib
import io
import sys
import tarfile


def main(out_path):
    seen = {}  # frame path -> sha1, for the current trajectory only
    current = None
    kept = dropped = differ = 0
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as src, tarfile.open(out_path, 'w') as out:
        for m in src:
            if not m.isfile():
                continue
            folder, name = m.name.rsplit('/', 1)
            traj, style = folder.rsplit('_', 1)
            if traj != current:
                current, seen = traj, {}
            data = src.extractfile(m).read()
            if name.endswith('.pkl'):
                path = f'{traj}/{style}.pkl'
            else:
                digest = hashlib.sha1(data).hexdigest()
                path = f'{traj}/{name}'
                if path in seen:
                    if seen[path] == digest:
                        dropped += 1
                        continue
                    path, differ = f'{traj}/{style}/{name}', differ + 1
                seen.setdefault(f'{traj}/{name}', digest)
            info = tarfile.TarInfo(path)
            info.size = len(data)
            out.addfile(info, io.BytesIO(data))
            kept += 1
    print(f'{out_path}: kept {kept}, dropped duplicate frames {dropped}, differing frames {differ}')


if __name__ == '__main__':
    main(sys.argv[1])
