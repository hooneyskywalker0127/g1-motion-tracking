#!/usr/bin/env bash
# 컴퓨터 이동으로 27,000에서 끊긴 walk4_subject1 학습을 이어서 30,000까지.
# rsl_rl은 tot_iter = 불러온 iter + max_iterations 라서 남은 3,000만 준다.
set -u
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
ORG=hooneyskywalker-humanoid-org
SEQ=walk4_subject1
RUN=2026-09-04_13-15-59_walk4_subject1_resume
LOGDIR=/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/train_logs

cd "$WBT" || exit 1
"$PY" scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name "$ORG/wandb-registry-motions/$SEQ" \
  --headless --logger wandb \
  --log_project_name g1-motion-tracking \
  --run_name "${SEQ}_resume2" \
  --resume True \
  --load_run "$RUN" \
  --checkpoint model_27000.pt \
  --max_iterations 3000 \
  > "$LOGDIR/${SEQ}_resume2.log" 2>&1
echo "walk4 재개 종료코드 $? $(date +%F\ %H:%M:%S)"
