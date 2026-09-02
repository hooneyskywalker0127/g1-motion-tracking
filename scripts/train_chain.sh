#!/usr/bin/env bash
# 시퀀스를 하나씩 순서대로 학습시킨다. BeyondMimic은 모션당 정책 하나라
# 병렬로 돌릴 수 없고, 앞 학습이 끝나야 다음이 시작된다.
#
# 순서는 configs/train_order.txt (발 오차가 작은 것부터).
# 멈추려면 /tmp/stop_train_chain 파일을 만들면 된다. 진행 중인 학습은
# 끝까지 가고 그다음 것을 시작하지 않는다.
set -u
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
ORG=hooneyskywalker-humanoid-org
LOGDIR=/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/train_logs
mkdir -p "$LOGDIR"

# 이미 도는 학습이 있으면 끝날 때까지 기다린다.
while pgrep -f "rsl_rl/train.py" > /dev/null; do sleep 60; done

for seq in $(cat "$REPO/configs/train_order.txt"); do
  [ -f /tmp/stop_train_chain ] && { echo "중지 요청 확인, 종료 $(date +%F\ %H:%M:%S)"; break; }
  echo "=== $seq 학습 시작 $(date +%F\ %H:%M:%S) ==="
  start=$SECONDS
  (cd "$WBT" && "$PY" scripts/rsl_rl/train.py \
      --task=Tracking-Flat-G1-v0 \
      --registry_name "$ORG/wandb-registry-motions/$seq" \
      --headless --logger wandb \
      --log_project_name g1-motion-tracking \
      --run_name "$seq") > "$LOGDIR/$seq.log" 2>&1
  echo "=== $seq 종료코드 $? · $(( (SECONDS-start)/60 ))분 · $(date +%F\ %H:%M:%S) ==="
done
echo "체인 완료 $(date +%F\ %H:%M:%S)"
