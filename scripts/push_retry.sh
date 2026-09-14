#!/usr/bin/env bash
# PolySim 평가가 끝난 뒤 외란 평가를 다시 돌린다 (순서 버그 수정본).
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
L=/tmp/overnight.log
while ps -eo cmd | grep -q "[e]val_dump.py"; do sleep 120; done
echo "=== Isaac 외란 평가 재시도 $(date +%H:%M:%S)" >> $L
rm -f "$R"/outputs/eval_push/*.json
for M in /home/sehoon/motions_fixed/*.npz; do
  S=$(basename "$M" .npz); [ "$S" = kobe_level1 ] && continue
  RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$S"/ 2>/dev/null | sort | tail -1)
  [ -n "$RUN" ] || continue
  RUN=$(basename "${RUN%/}")
  (cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval_push.py \
     --task=Tracking-Flat-G1-v0 --num_envs 30 --load_run="$RUN" \
     --checkpoint=model_29999.pt --motion_file="$M" --force_push \
     --out "$R/outputs/eval_push/$S.json") > /tmp/ep_push_$S.log 2>&1
  echo "  $S 종료 $? $(date +%H:%M:%S)" >> $L
done
echo "=== 외란 평가 완료 $(date +%F\ %H:%M:%S)" >> $L
