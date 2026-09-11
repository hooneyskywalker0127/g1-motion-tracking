#!/usr/bin/env bash
# 레퍼런스 시작 스텝을 바꿔 가며 같은 시퀀스를 기록한다.
#
# 리타게팅 첫 프레임이 IK 미수렴이라 레퍼런스 앞 2프레임에 가짜 속도가 들어간다.
# start_step 을 올리면 그 구간을 건너뛴다. Retargeting Matters(2510.02252) V-D 가
# 시작 프레임이 sim2sim 성공률을 크게 좌우한다고 보고한 것과 같은 축이다.
#
#   bash scripts/start_step_probe.sh <시퀀스> <초> <스텝 ...>
set -u
SEQ=$1; SECS=$2; shift 2
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
OUT=$REPO/outputs/start_step
mkdir -p "$OUT"

for STEP in "$@"; do
  pkill -9 -f "mujoco_sim|run_mtc_sim" 2>/dev/null; sleep 5
  LOG=/tmp/ss_${SEQ}_$STEP.log
  setsid bash /home/sehoon/run_mtc_sim_step.sh "$SEQ" "$STEP" > "$LOG" 2>&1 &
  ok=0
  for i in $(seq 1 90); do
    grep -qa "activated walking_controller" "$LOG" && { ok=1; break; }
    sleep 1
  done
  [ "$ok" -ne 1 ] && { echo "  !! start_step=$STEP 활성 실패"; continue; }
  sleep 3
  ( set +u; source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
    python3 "$REPO/scripts/record_policy_io.py" "$OUT/${SEQ}_s${STEP}.npz" \
      --seconds "$SECS" ) 2>&1 | tail -1
done
pkill -9 -f "mujoco_sim|run_mtc_sim" 2>/dev/null
