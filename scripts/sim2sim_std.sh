#!/usr/bin/env bash
# 표준 구조 sim2sim 을 시퀀스마다 돌린다 (src/sim2sim.py).
# ROS 런치가 필요 없다 — MuJoCo 를 직접 돌리므로 CPU 만 쓴다.
#
#   bash scripts/sim2sim_std.sh [초] [시퀀스 ...]
set -u
SECS=${1:-60}; shift || true
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
OUT=$REPO/outputs/sim2sim_std
mkdir -p "$OUT"
if [ $# -gt 0 ]; then SEQS="$*"; else
  SEQS=$(ls -1 /home/sehoon/colcon_ws/policies/*.onnx | xargs -n1 basename | sed 's/\.onnx$//'); fi
for SEQ in $SEQS; do
  [ -f "$OUT/$SEQ.npz" ] && { echo "건너뜀 $SEQ"; continue; }
  "$PY" "$REPO/src/sim2sim.py" "$SEQ" --start_step 0 --seconds "$SECS" \
        --out "$OUT/$SEQ.npz" 2>&1 | tail -1
done
