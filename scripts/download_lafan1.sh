#!/usr/bin/env bash
# LAFAN1 원본 BVH + G1 리타게팅 정답지 다운로드
set -e

DST=/home/sehoon/data/lafan1
REF=/home/sehoon/data/lafan1_g1_ref
HF=/home/sehoon/miniconda3/envs/env_isaaclab/bin/hf

mkdir -p "$DST" "$REF"

echo "[1/3] downloading lafan1.zip (144 MB)"
URL=https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip
curl -L -C - -o "$DST/lafan1.zip" "$URL"

echo "[2/3] extracting bvh files"
unzip -o -q "$DST/lafan1.zip" -d "$DST"
find "$DST" -name "*.bvh" | wc -l | xargs echo "  bvh files:"

echo "[3/3] downloading G1 reference retarget"
"$HF" download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset --local-dir "$REF"

echo
echo "done."
echo "  source bvh : $DST"
echo "  g1 ref csv : $REF/g1"
