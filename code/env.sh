# Puts every code folder on the import path. Run scripts from this folder:  source env.sh; python 4_final/final.py ...
# (Scripts import each other by name, e.g. "import pathnet", so all folders must be on PYTHONPATH.)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$ROOT/common:$ROOT/2_features:$ROOT/3_models:$ROOT/4_final:$(find "$ROOT/experiments" -mindepth 1 -maxdepth 1 -type d | tr '\n' ':')${PYTHONPATH:-}"
