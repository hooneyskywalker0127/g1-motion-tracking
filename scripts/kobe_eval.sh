#!/usr/bin/env bash
# Kobe 학습이 끝난 뒤 한 번에 돌리는 평가. PolySim Table III 와 맞추기 위한 것이다.
#   PolySim  IsaacSim_DR → MuJoCo  성공 0.100 · E_g-mpjpe 272.6 mm (Kobe, 10회 시행)
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
M=/home/sehoon/motions_fixed/kobe_level1.npz
SEQ=kobe_level1

RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"/ | sort | tail -1); RUN=$(basename "${RUN%/}")
CKPT=$(ls -1 "$WBT/logs/rsl_rl/g1_flat/$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
echo "=== 런 $RUN · 체크포인트 model_$CKPT.pt"

# 1) ONNX 내보내기 (play.py 가 policy.onnx 를 만든다)
(cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
   --num_envs 1 --load_run="$RUN" --checkpoint="model_$CKPT.pt" \
   --motion_file="$M" --headless) > /tmp/kobe_play.log 2>&1
ONNX=$(ls -t "$WBT"/logs/rsl_rl/g1_flat/"$RUN"/exported/policy.onnx 2>/dev/null | head -1)
[ -n "$ONNX" ] && cp -v "$ONNX" /home/sehoon/colcon_ws/policies/$SEQ.onnx

# 2) Isaac 평가 (자체 기준 + PolySim 0.5 m 기준)
(cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval_dump.py --task=Tracking-Flat-G1-v0 \
   --num_envs 100 --load_run="$RUN" --checkpoint="model_$CKPT.pt" \
   --motion_file="$M" --out "$R/outputs/eval_polysim/$SEQ.json") > /tmp/kobe_eval.log 2>&1
echo "--- Isaac"; cat "$R/outputs/eval_polysim/$SEQ.json" 2>/dev/null

# 3) MuJoCo 10회 시행, 모션 전체 길이
(cd "$R" && OMP_NUM_THREADS=1 "$PY" src/sim2sim_trials.py "$SEQ" --n 10 --seconds 0 \
   --out outputs/lr/kobe_trials.npz) 2>&1 | grep -v Warn
echo "완료 $(date +%F\ %H:%M:%S)"
