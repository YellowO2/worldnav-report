#!/bin/bash
# Runs any command on a TC1 GPU node:  sbatch --time=60 --job-name=NAME job.sh python 3_models/train.py ...
# Always pass --time (minutes, at most 360).
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -e
cd "$SLURM_SUBMIT_DIR"           # submit from the code folder
source venv/bin/activate
source env.sh
export HF_HOME=~/.cache/huggingface PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$@"
