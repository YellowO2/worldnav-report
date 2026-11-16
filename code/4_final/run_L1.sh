#!/bin/bash
# The later-start final models (final_groups.py --later), one per robot, side by side on one GPU. Stanford real and
# Stanford sim are left out (later starts hurt on Stanford real); drop any robot whose confirmation fails.
#   sbatch --time=360 --mem=180G --cpus-per-task=16 --job-name=L1 job.sh bash run_L1.sh
for g in recon scand_spot scand_jackal huron; do
  python 4_final/final_groups.py --name L1 --later --only $g > logs/L1_$g.out 2>&1 &
done
wait
