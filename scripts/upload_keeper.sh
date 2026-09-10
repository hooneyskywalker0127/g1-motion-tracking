#!/usr/bin/env bash
# 업로더가 죽으면 다시 띄운다.
# 자리를 비운 사이 업로더가 죽어 드라이브에 아무것도 안 올라가는 일을 막는다.
#
# 끝나는 조건은 개수가 아니라 실제 상태다. 시퀀스 (3)개 각각에 대해
# 병합 영상 (3)개와 설명 txt (3)개가 드라이브 기록에 있으면 종료한다.
# 개수로 세면 목표가 바뀔 때마다 틀린다.
set -u
U=/home/sehoon/Documents/GitHub/g1-motion-tracking/scripts/upload_gdrive.sh
LOG=/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/train_logs/upload.log
DONE=/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/.uploaded
SEQS="walk2_subject4 aiming1_subject1 walk1_subject1"

all_done() {
  local seq tier f
  for seq in $SEQS; do
    for tier in compare progression randomization; do
      for f in "${seq}_${tier}.mp4" "${seq}_${tier}_youtube.txt"; do
        grep -q "/$f\$" "$DONE" 2>/dev/null || return 1
      done
    done
  done
  return 0
}

while true; do
  if all_done; then
    echo "$(date +%F\ %H:%M:%S) 영상 (9)개와 설명 (9)개 모두 업로드됨, keeper 종료"
    break
  fi
  if ! pgrep -f "upload_gdrive.sh" > /dev/null; then
    echo "$(date +%F\ %H:%M:%S) 업로더가 없어 다시 띄움 (기록 $(wc -l < "$DONE") 건)"
    setsid "$U" >> "$LOG" 2>&1 < /dev/null &
  fi
  sleep 120
done
