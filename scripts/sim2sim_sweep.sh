#!/usr/bin/env bash
# 공식 sim-to-sim 경로를 시퀀스마다 돌리고, 컨트롤러가 발행하는 값만 기록한다.
# 물리·관측은 건드리지 않는다. 기록기는 scripts/record_policy_io.py.
#
#   bash scripts/sim2sim_sweep.sh [초] [시퀀스 ...]
set -u
SECS=${1:-260}; shift || true
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
OUT=$REPO/outputs/policy_io
POL=/home/sehoon/colcon_ws/policies
mkdir -p "$OUT"

if [ $# -gt 0 ]; then SEQS="$*"; else SEQS=$(ls -1 "$POL"/*.onnx | xargs -n1 basename | sed 's/\.onnx$//'); fi

cleanup() {
  for pat in "mujoco_sim" "ros2 launch motion_tracking" "controller_manager/spawner" \
             "robot_state_publisher" "joy_linux_node" "joy_teleop"; do
    ps -eo pid,cmd | grep -F "$pat" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  done
  sleep 4
}
trap cleanup EXIT

for SEQ in $SEQS; do
  [ -f "$POL/$SEQ.onnx" ] || { echo "정책 없음 $SEQ"; continue; }
  [ -f "$OUT/$SEQ.npz" ] && { echo "건너뜀 $SEQ (이미 있음)"; continue; }
  echo "=== $SEQ $(date +%H:%M:%S)"
  cleanup
  LOG=/tmp/s2s_$SEQ.log
  setsid bash /home/sehoon/run_mtc_sim.sh "$SEQ" > "$LOG" 2>&1 &

  # 컨트롤러가 활성화될 때까지 기다린다. 실패도 같이 본다.
  ok=0
  for i in $(seq 1 60); do
    grep -qa "Configured and activated walking_controller" "$LOG" && { ok=1; break; }
    grep -qaE "FATAL|process has died" "$LOG" && break
    sleep 1
  done
  if [ "$ok" -ne 1 ]; then echo "  !! 컨트롤러 활성 실패"; continue; fi
  sleep 3

  ( source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
    python3 "$REPO/scripts/record_policy_io.py" "$OUT/$SEQ.npz" --seconds "$SECS" ) 2>&1 | tail -1
done
echo "전체 완료 $(date +%H:%M:%S)"
