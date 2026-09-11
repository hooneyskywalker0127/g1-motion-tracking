#!/usr/bin/env bash
# sim-to-sim 사이드바이사이드 영상과 설명문을 구글드라이브에 올린다.
# 이 영상들만 따로 받을 수 있게 **한 폴더**에 평평하게 올린다: gdrive:g1-sim2sim/
#
# 물리적으로 무효한 롤아웃(src/sim2sim_validity.py 판정)은 기본으로 건너뛴다.
# 전부 올리려면 ALL=1 을 준다.
set -u
SRC=/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking
DST=gdrive:g1-sim2sim
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
ALL=${ALL:-0}

VALID=$("$PY" "$REPO/src/sim2sim_validity.py" 2>/dev/null \
        | awk '/정상 완주|넘어짐\(정상\)/{print $1}')
[ "$ALL" = "1" ] && VALID=$(ls -1 "$SRC" | grep -v '\.')

echo "올릴 시퀀스:"; echo "$VALID" | tr '\n' ' '; echo; echo

for seq in $VALID; do
  d="$SRC/$seq/sim2sim"
  vid="$d/${seq}_sidebyside.mp4"
  txt="$d/${seq}_sim2sim_youtube.txt"
  memo="$d/${seq}_메모.md"
  # 메모는 sim-to-sim 절만 뽑아 시퀀스 이름을 붙여 따로 만든다
  awk '/^## Sim-to-Sim/{f=1} f' "$SRC/$seq/유튜브_메모.md" > "$memo" 2>/dev/null
  [ -f "$vid" ] || { echo "건너뜀 $seq (영상 없음)"; continue; }
  # 쓰는 중인 파일은 올리지 않는다
  if pgrep -f "ffmpeg.*${seq}_sidebyside" >/dev/null; then
    echo "건너뜀 $seq (아직 쓰는 중)"; continue
  fi
  ffprobe -v error -show_entries format=duration -of csv=p=0 "$vid" >/dev/null 2>&1 \
    || { echo "건너뜀 $seq (영상 손상)"; continue; }
  echo "=== $seq $(date +%H:%M:%S)"
  for f in "$vid" "$txt" "$memo"; do
    [ -f "$f" ] && rclone copy "$f" "$DST/" --transfers 1 \
      --drive-chunk-size 32M --stats-one-line --stats 30s 2>&1 | grep -v NOTICE
  done
done
echo "완료 $(date +%H:%M:%S)"
