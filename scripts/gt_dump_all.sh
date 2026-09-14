#!/usr/bin/env bash
# 아이작 쪽 프레임별 전역오차(g_t)를 17개 시퀀스 전부 남긴다.
# 목적: 평가 지평 H 를 바꿔가며 성공률을 다시 재는 것을 MuJoCo 뿐 아니라
#       아이작에서도 해서, 지평 효과가 시뮬레이터 특성이 아님을 보인다.
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
L=$R/outputs/train_logs/gt_dump.log
mkdir -p "$R/outputs/eval_gt"

# 코베 학습·평가 체인이 끝날 때까지 기다린다.
while pgrep -f "[r]sl_rl/train.py" >/dev/null || pgrep -f "[k]obe_chain.sh" >/dev/null; do
  sleep 300
done
echo "=== g_t 덤프 시작 $(date +%F\ %H:%M:%S)" >> "$L"

for SEQ in aiming1_subject1 dance2_subject3 jumps1_subject1 obstacles2_subject1 \
           obstacles3_subject3 run2_subject4 walk1_subject1 walk1_subject2 \
           walk1_subject5 walk2_subject1 walk2_subject3 walk2_subject4 \
           walk3_subject1 walk3_subject2 walk3_subject4 walk3_subject5 walk4_subject1; do
  OUT=$R/outputs/eval_gt/$SEQ.json
  [ -f "${OUT%.json}_dump.npz" ] && { echo "건너뜀 $SEQ" >> "$L"; continue; }
  RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"/ 2>/dev/null | sort | tail -1)
  [ -z "$RUN" ] && { echo "! 런 없음 $SEQ" >> "$L"; continue; }
  RUN=$(basename "${RUN%/}")
  CKPT=$(ls -1 "$WBT/logs/rsl_rl/g1_flat/$RUN"/model_*.pt 2>/dev/null \
         | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  [ -z "$CKPT" ] && { echo "! 체크포인트 없음 $SEQ" >> "$L"; continue; }
  echo "--- $SEQ · $RUN · model_$CKPT.pt · $(date +%H:%M:%S)" >> "$L"
  (cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval_dump.py \
     --task=Tracking-Flat-G1-v0 --num_envs 100 --load_run="$RUN" \
     --checkpoint="model_$CKPT.pt" --motion_file=/home/sehoon/motions_fixed/$SEQ.npz \
     --out "$OUT") >> "$L" 2>&1
  echo "    종료코드 $?" >> "$L"
done
echo "=== g_t 덤프 종료 $(date +%F\ %H:%M:%S)" >> "$L"
