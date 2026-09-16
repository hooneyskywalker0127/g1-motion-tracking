#!/usr/bin/env bash
# 대칭 프로토콜로 17 시퀀스를 다시 잰다.
#   sim      Isaac,  교란 끔,  100 환경 (교란이 없으므로 사실상 결정론 1 표본)
#   sim-dr   Isaac,  초기 교란 켬, 100 환경
#   sim2sim  MuJoCo, 같은 초기 교란, 100 시행
# 교란 분포는 양쪽 다 tracking_env_cfg.py 의 MotionCommandCfg 값이다.
# 채점 정의는 양쪽 다 src/score_standard.py 와 eval_sym.py 로 같게 맞췄다.
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
OUT=$R/outputs/sym
L=$OUT/run.log
mkdir -p "$OUT"

SEQS="aiming1_subject1 dance2_subject3 jumps1_subject1 obstacles2_subject1
      obstacles3_subject3 run2_subject4 walk1_subject1 walk1_subject2
      walk1_subject5 walk2_subject1 walk2_subject3 walk2_subject4
      walk3_subject1 walk3_subject2 walk3_subject4 walk3_subject5 walk4_subject1"

echo "=== 시작 $(date +%F\ %H:%M:%S)" >> "$L"

# 헤드라인은 sim-dr 과 sim2sim 의 짝이다. sim-dr 을 먼저 다 받고 sim 을 받는다.
for COND in simdr sim; do
 for SEQ in $SEQS; do
  RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"/ 2>/dev/null | sort | tail -1)
  [ -z "$RUN" ] && { echo "! 런 없음 $SEQ" >> "$L"; continue; }
  RUN=$(basename "${RUN%/}")
  CKPT=$(ls -1 "$WBT/logs/rsl_rl/g1_flat/$RUN"/model_*.pt 2>/dev/null \
         | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  [ -z "$CKPT" ] && { echo "! 체크포인트 없음 $SEQ" >> "$L"; continue; }

  {
    J=$OUT/${SEQ}_${COND}.json
    [ -f "$J" ] && { echo "건너뜀 $SEQ $COND" >> "$L"; continue; }
    FLAG=""; [ "$COND" = simdr ] && FLAG="--init_randomize"
    echo "--- $SEQ $COND · $RUN · model_$CKPT.pt · $(date +%H:%M:%S)" >> "$L"
    (cd "$WBT" && OMP_NUM_THREADS=1 timeout 3600 "$PY" scripts/rsl_rl/eval_sym.py \
       --task=Tracking-Flat-G1-v0 --num_envs 100 --load_run="$RUN" \
       --checkpoint="model_$CKPT.pt" \
       --motion_file=/home/sehoon/motions_fixed/$SEQ.npz \
       --out "$J" $FLAG) >> "$L" 2>&1
    echo "    종료코드 $?" >> "$L"
  }
 done
done

echo "=== Isaac 끝, MuJoCo 시작 $(date +%F\ %H:%M:%S)" >> "$L"

# MuJoCo 는 CPU 만 쓰므로 시퀀스를 병렬로 돌린다.
cd "$R"
printf '%s\n' $SEQS | xargs -P 12 -I{} sh -c \
  "OMP_NUM_THREADS=1 $PY src/sim2sim_trials.py {} --n 100 \
     --out $OUT/{}_sim2sim.npz >> $L 2>&1"

echo "=== 종료 $(date +%F\ %H:%M:%S)" >> "$L"
