#!/bin/bash
# RECON recipe test (arm 6: all robots but Stanford, no later starts; then arm 4 or 5: own + own later starts + other robots' first photos), the later-photo Qwen features for every other
# non-TartanDrive robot (train + val_seen), and the Stanford real confirmation (arms 0 and 1), side by side on one GPU.
#   sbatch --time=300 --mem=128G --cpus-per-task=16 --job-name=laterall job.sh bash run_later.sh
python experiments/section7_per_robot/later_starts.py recon --arms 6 45 > logs/later_recon_recipe.out 2>&1 &
python 2_features/later_feats.py stanford_real
python experiments/section7_per_robot/later_starts.py stanford_real --arms 0 1 > logs/later_confirm_stanford_real.out 2>&1 &
for g in scand_spot huron stanford_sim scand_jackal; do python 2_features/later_feats.py $g; done
for g in recon huron stanford_real stanford_sim scand_spot scand_jackal; do python 2_features/later_feats.py $g val_seen; done
wait
