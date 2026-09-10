#!/usr/bin/env bash
# 시퀀스별 병합 영상 세 개(비교/발전과정/랜덤화)와 설명문을 완성되는 대로 구글드라이브에 올린다.
# 드라이브 구조는 이미 올라간 것과 같게 gdrive:g1-motion-tracking/<시퀀스>/<파일> 평면이다.
# clips/ 안의 중간 렌더는 올리지 않는다.
#
# 완성 판정: ffprobe로 열리고, 60초 동안 크기가 변하지 않은 파일.
# 한 번 올린 파일은 .uploaded 목록에 적어 두 번 올리지 않는다.
set -u
SRC=/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking
DST=gdrive:g1-motion-tracking
DONE=/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/.uploaded
SEQS="walk2_subject4 aiming1_subject1 walk1_subject1"
touch "$DONE"

ready() {  # 다 쓰인 영상인지, 길이가 정상인지 본다
  local f=$1 d
  [ -f "$f" ] || return 1
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null) || return 1
  [ -n "$d" ] || return 1
  # 가장 긴 클립이 (261)초다. 400초를 넘으면 폭주한 결과이므로 올리지 않는다.
  if [ "$(printf '%.0f' "$d")" -gt 400 ]; then
    echo "길이 비정상 ${d}초, 올리지 않음: $(basename "$f")"; return 1
  fi
  local a b
  a=$(stat -c %s "$f"); sleep 60; b=$(stat -c %s "$f")
  [ "$a" = "$b" ]
}

while true; do
  left=0
  for seq in $SEQS; do
    for tier in compare progression randomization; do
      for f in "$SRC/$seq/$tier/${seq}_${tier}.mp4" \
               "$SRC/$seq/$tier/${seq}_${tier}_youtube.txt"; do
        grep -qxF "$f" "$DONE" && continue
        [ -f "$f" ] || { left=$((left+1)); continue; }
        case "$f" in
          *.mp4) ready "$f" || { left=$((left+1)); continue; } ;;
        esac
        echo "올림 $(date +%H:%M:%S): $seq/$(basename "$f")"
        if rclone copyto "$f" "$DST/$seq/$(basename "$f")" --stats-one-line; then
          echo "$f" >> "$DONE"
          # auto_pipeline 과 기록을 공유해 같은 파일을 두 번 올리지 않는다.
          echo "$seq/$(basename "$f")" >> /home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/uploaded.txt
          echo "완료: $seq/$(basename "$f")"
        else
          echo "실패, 다음 회차에 재시도: $seq/$(basename "$f")"
          left=$((left+1))
        fi
      done
    done
  done
  [ "$left" -eq 0 ] && { echo "전부 업로드 완료 $(date +%F\ %H:%M:%S)"; break; }
  sleep 120
done
