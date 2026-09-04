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

# 학습 로그 폴더는 <날짜>_<시각>_<시퀀스> 형식이다. 같은 시퀀스를 여러 번 돌렸으면
# 이름순 뒤쪽이 최신이므로 나중 것이 앞의 것을 덮어쓰게 둔다.
declare -A RUN_OF
for run in $(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*/ 2>/dev/null | sort); do
  name=$(basename "$run")
  seq=${name#*_*_}; seq=${seq%_resume}
  RUN_OF[$seq]=$name
done

for seq in "${!RUN_OF[@]}"; do
  name=${RUN_OF[$seq]}
  run="$WBT/logs/rsl_rl/g1_flat/$name"

  out="$OUTDIR/$seq.json"
  [ -f "$out" ] && { echo "건너뜀: $seq 이미 평가됨"; continue; }

  ckpt=$(ls -1 "$run"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  # 30000회를 다 돌지 못한 학습은 아직 평가 대상이 아니다.
  if [ -z "$ckpt" ] || [ "$ckpt" -lt 29999 ]; then
    echo "건너뜀: $seq 학습 미완료 (최종 체크포인트 ${ckpt:-없음})"
    continue
  fi

  # npz는 손으로 복사해 둔 것이 있으면 그것을, 없으면 학습 때 받아 둔 artifact를 쓴다.
  motion="$MOTIONS/$seq.npz"
  [ -f "$motion" ] || motion="$WBT/artifacts/$seq:v0/motion.npz"
  [ -f "$motion" ] || { echo "건너뜀: $seq npz 없음"; continue; }

  echo "=== $seq · model_$ckpt.pt · 환경 $NUM_ENVS $(date +%F\ %H:%M:%S) ==="
  (cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/eval.py \
      --task=Tracking-Flat-G1-v0 --num_envs "$NUM_ENVS" \
      --load_run="$name" --checkpoint="model_$ckpt.pt" \
      --motion_file="$motion" \
      --out "$out" $EXTRA) 2>&1 | grep -E "\[RESULT\]|\[INFO\]: Wrote|Error"
done
echo "평가 완료 $(date +%F\ %H:%M:%S)"
