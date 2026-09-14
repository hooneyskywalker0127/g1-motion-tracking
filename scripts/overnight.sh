#!/usr/bin/env bash
# 렌더가 끝나면 이어서 돌린다.
#   1) 쌍 영상 자동 검사
#   2) Isaac 외란 평가 (MuJoCo 쪽 sim2sim_push 와 조건 맞춤)
#   3) Isaac PolySim 기준 평가 (전역 0.5 m)
#   4) Kobe 재학습 (adaptive_uniform_ratio 1.0)
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
L=/tmp/overnight.log

echo "=== 렌더 종료 확인 $(date +%F\ %H:%M:%S)" >> $L
bash "$R/scripts/check_pair_videos.sh" >> $L 2>&1

echo "=== Isaac 외란 평가 시작 $(date +%H:%M:%S)" >> $L
mkdir -p "$R/outputs/eval_push"
for M in /home/sehoon/motions_fixed/*.npz; do
  S=$(basename "$M" .npz); [ "$S" = kobe_level1 ] && continue
  [ -f "$R/outputs/eval_push/$S.json" ] && continue
  RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$S"/ 2>/dev/null | sort | tail -1)
  [ -n "$RUN" ] || continue
  RUN=$(basename "${RUN%/}")
  (cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval_push.py \
     --task=Tracking-Flat-G1-v0 --num_envs 30 --load_run="$RUN" \
     --checkpoint=model_29999.pt --motion_file="$M" --force_push \
     --out "$R/outputs/eval_push/$S.json") > /tmp/ep_push_$S.log 2>&1
  echo "  $S 종료 $? $(date +%H:%M:%S)" >> $L
done

echo "=== Isaac PolySim 기준 평가 시작 $(date +%H:%M:%S)" >> $L
mkdir -p "$R/outputs/eval_polysim"
for M in /home/sehoon/motions_fixed/*.npz; do
  S=$(basename "$M" .npz); [ "$S" = kobe_level1 ] && continue
  [ -f "$R/outputs/eval_polysim/$S.json" ] && continue
  RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$S"/ 2>/dev/null | sort | tail -1)
  [ -n "$RUN" ] || continue
  RUN=$(basename "${RUN%/}")
  (cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval_dump.py \
     --task=Tracking-Flat-G1-v0 --num_envs 30 --load_run="$RUN" \
     --checkpoint=model_29999.pt --motion_file="$M" \
     --out "$R/outputs/eval_polysim/$S.json") > /tmp/ep_poly_$S.log 2>&1
  echo "  $S 종료 $? $(date +%H:%M:%S)" >> $L
done

echo "=== Kobe 학습 시작 $(date +%H:%M:%S)" >> $L
(cd "$WBT" && "$PY" scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
   --registry_name hooneyskywalker-humanoid-org/wandb-registry-motions/kobe_level1 \
   --adaptive_uniform_ratio 1.0 --headless --logger wandb \
   --log_project_name g1-motion-tracking --run_name kobe_level1_u1) \
   > "$R/outputs/train_logs/kobe_level1.log" 2>&1
echo "=== Kobe 종료 $? $(date +%F\ %H:%M:%S)" >> $L
