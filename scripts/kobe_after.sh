#!/usr/bin/env bash
set -u
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
L=/tmp/overnight.log
while ps -eo cmd | grep -q "[e]val_push.py"; do sleep 120; done
echo "=== Kobe 재시작 $(date +%F\ %H:%M:%S)" >> $L
cd "$WBT"
"$PY" scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name hooneyskywalker-humanoid-org/wandb-registry-motions/kobe_level1 \
  --adaptive_uniform_ratio 1.0 --headless --logger wandb \
  --log_project_name g1-motion-tracking --run_name kobe_level1_u1 \
  > /home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/train_logs/kobe_level1.log 2>&1
echo "=== Kobe 종료 $? $(date +%F\ %H:%M:%S)" >> $L
