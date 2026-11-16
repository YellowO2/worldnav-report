#!/usr/bin/env bash
# Builds ~/worldnav/venv on TC1. The V100 is Volta (sm_70): the cu126 wheels still include it, cu128 and later do not.
# The V100 has no fast bfloat16, so train in float16 autocast.
set -euo pipefail
cd "$(dirname "$0")"
# module load fails in non-login shells, so use TC1's Python 3.11 directly (it lacks sqlite3).
rm -rf venv
/tc1apps/python/3.11.14/bin/python3.11 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -U pip
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install --no-cache-dir -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
