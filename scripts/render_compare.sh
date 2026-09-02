#!/usr/bin/env bash
# 사람 bvh 원본과 리타게팅된 G1을 한 영상에 담는다.
# 인자: 시퀀스 이름, [raw|scaled]
set -e

SEQ=${1:-walk1_subject1}
HUMAN=${2:-raw}
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
DEST=/home/sehoon/Desktop/참고/할일/26/09/260902

source /home/sehoon/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd "$REPO"

OUT=outputs/videos/${SEQ}_${HUMAN}.mp4
MUJOCO_GL=egl python src/render_compare.py \
  --bvh_file /home/sehoon/data/lafan1/"$SEQ".bvh \
  --human "$HUMAN" \
  --overlay \
  --out "$OUT"

mkdir -p "$DEST"
cp "$OUT" "$DEST"/
echo "복사 완료: $DEST/$(basename $OUT)"
