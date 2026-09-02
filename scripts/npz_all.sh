#!/usr/bin/env bash
# 선별한 시퀀스의 csv를 BeyondMimic npz로 바꿔 wandb registry에 올린다.
# csv_to_npz.py는 시퀀스마다 Isaac Sim을 새로 띄우므로 한 번에 하나씩 돈다.
# 인자를 주면 그 시퀀스만, 없으면 selected.txt 전부.
set -u
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python

if [ $# -gt 0 ]; then SEQS="$*"; else SEQS=$(cat "$REPO/outputs/metrics/selected.txt"); fi

for seq in $SEQS; do
  csv="$REPO/outputs/csv/$seq.csv"
  [ -f "$csv" ] || { echo "SKIP $seq (csv 없음)"; continue; }
  echo "=== $seq 시작 $(date +%H:%M:%S) ==="
  start=$SECONDS
  (cd "$WBT" && timeout 900 "$PY" scripts/csv_to_npz.py \
      --input_file "$csv" --input_fps 30 --output_name "$seq" --headless) \
      > "/tmp/npz_$seq.log" 2>&1
  code=$?
  ok=$(grep -ac "Motion saved to wandb registry" "/tmp/npz_$seq.log")
  echo "=== $seq 종료코드 $code · 업로드 $ok · $((SECONDS-start))초 ==="
done
echo "배치 완료 $(date +%H:%M:%S)"
