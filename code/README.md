# WorldNav code (SC4001 project)

Language-conditioned open-loop navigation, RoboWorld 2026 Track 1 (LCVN). One start photo and an instruction in,
a list of moves out. This folder holds the code for the final model and the scripts behind every table in the
report. The full log of runs, one row per experiment with its result, is notes/runs.md.

## Layout

The folders follow the pipeline, then the report:

| Folder | What is in it |
|---|---|
| 1_data/ | download and unpack the LCVN data |
| 2_features/ | instruction parsing and frozen features (Qwen3.5-4B, SigLIP2, DINOv3, depth, camera lean), cached once |
| 3_models/ | the route models and the camera (robot) classifier |
| 4_final/ | training the final models and building the submission zip |
| common/ | scoring (the organisers' metric), camera geometry, training curves |
| experiments/sectionN_*/ | one script per experiment, grouped by report section |
| config/ | measured camera calibration constants (small JSON files the code reads) |

Every script starts with a docstring: what it does, the protocol, and the command to run it.

## Setup and data

Run everything from this folder, after `source env.sh` (it puts every folder on the import path; the scripts
import each other by name).

1. Environment (Python 3.11, CUDA 12.6 wheels for the V100): `bash setup_tc1.sh` makes venv/ and installs torch
   2.7.1 and requirements.txt. The pretrained models (Qwen3.5-4B, SigLIP2, DINOv3, Depth Anything 3, Granite 4.2
   3B) download from Hugging Face on first use.
2. Labelled data: `bash 1_data/download.sh` streams the LCVN splits (train, val_seen, val_unseen) from Hugging Face
   (fly1113/LCVN) into data/lcvn/ (about 85 GB of tars, packed on the fly), then
   `python 1_data/convert.py data/lcvn/{split}.tar` writes the episode lists.
3. Test data: the competition's test set from CodaBench, unpacked into data/test/, then
   `python 1_data/convert.py data/test/WorldNav_test/test`.
4. GPU jobs on the school cluster: `sbatch --time=60 --job-name=NAME job.sh python <script> ...` from this folder.

Cached features go in features/, trained models and outputs in runs/.

## Pipeline (final model)

1. Parse the instructions: 2_features/build_parses.py (rule parser, rule_parse.py; LLM parser, parse_instructions.py)
2. Cache features per split: 2_features/qwen_features.py --v2, features_v2.py (SigLIP2, DINOv3), gravity_all.py
   (camera lean), da3_tartan.py (depth, TartanDrive), path_labels.py (training targets)
3. Robot of each test photo: 3_models/group_classifier.py
4. Final models, in the order they were built and averaged:
   - f1 to f6: 4_final/final.py, final_mix.py
   - g1, g2 (per-robot designs): 4_final/final_groups.py
   - g3 to g6 (LoRA fine-tuned Qwen averaged in): 4_final/lora_final.py, lora_final2.py, final_g6.py
   - final (later starts): 4_final/run_L1.sh (final_groups.py --later), then final_g7.py
5. Data-split lookup, reported separately: 4_final/lookup_final.py
6. Scoring: common/score.py with the organisers' common/official_metric.py

## Report tables to scripts

| Report | Script(s) |
|---|---|
| 4.1 simple baseline | experiments/section4_baseline/baselines.py |
| Table 5.1 what each part adds | 3_models/train.py, 2_features/ground_maps.py; section5_first_model/tune_base.py, tune_qwen.py, time_routes.py, k_ceiling.py; 3_models/siglip_steps.py |
| Table 10.1 output formats | section5_first_model/time_routes.py, later_speed.py, anchor_routes.py, tartan_time.py, tartan_shape.py, tartan_speed_head.py |
| 6.2 wrong turns | section6_limits/turn_check.py, side_check_saved.py, post_turn.py |
| 6.3 and Table 6.1 where to stop | section6_limits/step_oracle.py, stop_distance_check.py, residual_hunt.py |
| Table 7.1 ladders | section7_per_robot/ladder.py; section5_first_model/tune_base.py |
| Table 7.2 best design per robot | 3_models/siglip_steps.py; section7_per_robot/tartan_steps.py, final_compare.py, group_mix.py |
| Table 7.3 later starts | 2_features/later_feats.py, later_data.py; section7_per_robot/later_starts.py, tartan_later.py |
| 7.4 LoRA fine-tuning | section7_per_robot/lora_pilot.py, lora_pilot2.py, lora_tartan_day.py |
| Table 8.1 test submissions | 4_final/ (see the pipeline above); section8_results/tartan_steps_count.py |
| Tables 9.1, 9.2 data-split lookup | section9_data_split/hole_census.py, hole_match.py, lookup_routes.py, retrieval_match.py, retrieval_route.py; 4_final/lookup_final.py |

The section 9 scripts use the continuity of the recordings (the data split), not the photo and the instruction.
Their score is reported separately from the model's.

## Curves

Every run appends its per-epoch training loss and held-out scores to runs/curves/{run}.json (common/curves.py);
common/plot_curves.py draws them.
