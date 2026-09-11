#!/usr/bin/env bash
# 시퀀스 하나를 상류가 문서화한 기동 순서로 돌린다.
#
#   README:116  "The robot should enter standby controller in the beginning."
#   real.launch.py:54  active_list = ["state_estimator", "standby_controller"]
#
# mujoco.launch.py 는 walking_controller 를 바로 활성화해서, 스포너가 뜨는
# 수 초 동안 아무도 로봇을 잡지 않는다. MJCF 에 keyframe 이 없어 관절 0 으로
# 스폰되므로 그 사이에 주저앉고, 정책은 무너진 로봇을 인계받는다.
# 그래서 활성화 직후 standby 로 넘겨 기준자세를 잡게 한 뒤 정책으로 되돌린다.
# 되돌릴 때 MotionOnnxPolicy::reset() 이 timeStep_ = startStep_ 로 되돌린다.
#
#   bash scripts/run_seq_standby.sh <시퀀스> <출력.npz> [초]
set -u
SEQ=$1; OUT=$2; SECS=${3:-260}
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
LOG=/tmp/s2s_$SEQ.log

pkill -9 -f "mujoco_sim|run_mtc_sim" 2>/dev/null; sleep 5
setsid bash /home/sehoon/run_mtc_sim.sh "$SEQ" > "$LOG" 2>&1 &

for i in $(seq 1 90); do
  grep -qa "activated walking_controller" "$LOG" && break
  sleep 1
done
grep -qa "activated walking_controller" "$LOG" || { echo "  !! 활성 실패"; exit 1; }

sw() { ( set +u; source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
         ros2 control switch_controllers --activate "$1" --deactivate "$2" ) ; }

sw standby_controller walking_controller >/dev/null 2>&1
sleep 8                       # 기준자세로 일어설 시간
sw walking_controller standby_controller >/dev/null 2>&1

( set +u; source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
  python3 "$REPO/scripts/record_policy_io.py" "$OUT" --seconds "$SECS" ) 2>&1 | tail -1
pkill -9 -f "mujoco_sim|run_mtc_sim" 2>/dev/null
