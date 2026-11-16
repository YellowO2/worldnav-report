#!/usr/bin/env bash
# Streams each LCVN split from Hugging Face straight into pack.py, so the 85 GB of tars never touch the disk.
# Run on the TC1 login node:  nohup bash 1_data/download.sh > data/download.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."      # the code folder; data goes in data/lcvn/
mkdir -p data/lcvn
for split in val_seen val_unseen train; do
    [ -f "data/lcvn/$split.done" ] && continue
    echo "[$(date)] $split"
    curl -L --fail --retry 5 -s "https://huggingface.co/datasets/fly1113/LCVN/resolve/main/$split.tar" \
        | python3 1_data/pack.py "data/lcvn/$split.tar"
    touch "data/lcvn/$split.done"
done
echo "[$(date)] done"; du -sh data/lcvn
