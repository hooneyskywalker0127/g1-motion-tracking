#!/usr/bin/env bash
# 학습이 끝난 정책들을 하나씩 평가해 outputs/eval/<시퀀스>.json에 남긴다.
# 지표와 성공 판정은 GMR 논문(arXiv:2510.02252)을 따른다.
#
# 인자: [환경 수] [--randomize]
set -u
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
MOTIONS=/home/sehoon/motions
NUM_ENVS=${1:-100}
EXTRA=${2:-}
OUTDIR="$REPO/outputs/eval"
mkdir -p "$OUTDIR"

# 학습 로그 폴더 이름은 <날짜>_<시퀀스> 형식이다. 같은 시퀀스가 여러 번 학습됐으면
# 가장 최근 것을 쓴다.
for run in $(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*/ | sort); do
  name=$(basename "$run")
  seq=${name#*_*_}
  ckpt=$(ls -1 "$run"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  [ -z "$ckpt" ] && { echo "건너뜀: $name 체크포인트 없음"; continue; }
  [ -f "$MOTIONS/$seq.npz" ] || { echo "건너뜀: $seq npz 없음"; continue; }

  out="$OUTDIR/$seq.json"
  [ -f "$out" ] && { echo "건너뜀: $seq 이미 평가됨"; continue; }

  echo "=== $seq · model_$ckpt.pt · 환경 $NUM_ENVS $(date +%F\ %H:%M:%S) ==="
  (cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval.py \
      --task=Tracking-Flat-G1-v0 --num_envs "$NUM_ENVS" \
      --load_run="$name" --checkpoint="model_$ckpt.pt" \
      --motion_file="$MOTIONS/$seq.npz" \
      --out "$out" $EXTRA) 2>&1 | grep -E "\[RESULT\]|\[INFO\]: Wrote|Error"
done
echo "평가 완료 $(date +%F\ %H:%M:%S)"
