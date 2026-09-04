#!/usr/bin/env bash
# 한 시퀀스의 체크포인트별 정책을 렌더해 레퍼런스와 함께 한 화면에 붙인다.
# 아래에는 학습 곡선 그림을 같이 넣는다.
#
# 인자: 시퀀스 이름 [출력 폴더]
set -e
SEQ=$1
BASE=${2:-/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking}/$SEQ
DEST="$BASE/progression"
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
WORK="$DEST/clips"
CURVE="$BASE/${SEQ}_error_curve.png"

# 오차 곡선 그림은 학습 로그에서 바로 그린다. wandb 화면을 캡처할 필요가 없다.
mkdir -p "$BASE"
[ -f "$CURVE" ] || "$PY" "$(dirname "$0")/make_error_curve.py" "$SEQ" "$CURVE"

RUN=$(basename "$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"*/ | sort | tail -1)")
MOTION=/home/sehoon/motions/$SEQ.npz
[ -f "$MOTION" ] || MOTION="$WBT/artifacts/$SEQ:v0/motion.npz"
FRAMES=$("$PY" -c "import numpy as np;print(len(np.load('$MOTION')['joint_pos']))")
FPS=$("$PY" -c "import numpy as np;print(int(np.asarray(np.load('$MOTION')['fps']).reshape(-1)[0]))")
EPLEN=$("$PY" -c "print($FRAMES/$FPS + 5)")

mkdir -p "$WORK" "$DEST"
cd "$WBT"
export OMP_NUM_THREADS=1

# 30000회와 레퍼런스는 좌우 비교용으로 이미 뽑아둔 것을 재사용한다.
[ -f "$WORK/iter_30000.mp4" ] || cp "$BASE/compare/${SEQ}_policy.mp4" "$WORK/iter_30000.mp4"
[ -f "$WORK/reference.mp4" ] || cp "$BASE/${SEQ}_reference.mp4" "$WORK/reference.mp4"

for ck in 1000 5000 10000 20000; do
  out="$WORK/iter_$(printf %05d $ck).mp4"
  [ -f "$out" ] && { echo "건너뜀: $ck 이미 렌더됨"; continue; }
  echo "=== $SEQ · model_$ck.pt $(date +%H:%M:%S) ==="
  "$PY" scripts/rsl_rl/play.py \
    --task=Tracking-Flat-G1-v0 --num_envs=1 \
    --headless --video --video_length "$FRAMES" \
    --video_out "$out" \
    --load_run="$RUN" --checkpoint="model_$ck.pt" \
    --motion_file="$MOTION" --start_frame=0 --no_randomization \
    env.episode_length_s="$EPLEN" env.scene.contact_forces.debug_vis=false \
    env.commands.motion.debug_vis=false
done

lab() {  # $1 입력 인덱스, $2 라벨, $3 출력 이름
  echo "[$1:v]scale=960:540,pad=960:600:0:0:color=0x101010,drawtext=fontfile=$FONT:expansion=none:text='$2':fontcolor=white:fontsize=30:x=(w-text_w)/2:y=558[$3];"
}

TITLE="Training progression - $SEQ"
NOTE="same motion, same start frame, same camera in every panel  |  one training run, sampled at five checkpoints"

FILTER="$(lab 0 '1,000 iterations' a)$(lab 1 '5,000 iterations' b)$(lab 2 '10,000 iterations' c)$(lab 3 '20,000 iterations' d)$(lab 4 '30,000 iterations - final policy' e)$(lab 5 'retargeting reference - no physics' f)"
FILTER="$FILTER[a][b][c]hstack=inputs=3[r1];[d][e][f]hstack=inputs=3[r2];[r1][r2]vstack=inputs=2[g];"
# 학습 곡선 그림을 아래에 붙인다. 폭을 맞추고 남는 자리는 배경색으로 채운다.
FILTER="$FILTER[6:v]scale=-1:560,pad=2880:600:(2880-iw)/2:20:color=0x101010[cv];"
FILTER="$FILTER[g][cv]vstack=inputs=2:shortest=1[gc];"
FILTER="$FILTER[gc]pad=iw:ih+170:0:100:color=0x101010[p];"
FILTER="$FILTER[p]drawtext=fontfile=$FONT:expansion=none:text='$TITLE':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=30,"
FILTER="$FILTER drawtext=fontfile=$FONT:expansion=none:text='$NOTE':fontcolor=0xd0d0d0:fontsize=28:x=(w-text_w)/2:y=h-46[out]"

ffmpeg -y \
  -i "$WORK/iter_01000.mp4" -i "$WORK/iter_05000.mp4" -i "$WORK/iter_10000.mp4" \
  -i "$WORK/iter_20000.mp4" -i "$WORK/iter_30000.mp4" -i "$WORK/reference.mp4" \
  -loop 1 -i "$CURVE" \
  -filter_complex "$FILTER" -map "[out]" -shortest \
  -c:v libx264 -pix_fmt yuv420p -crf 21 "$DEST/${SEQ}_progression.mp4"

echo "완료: $DEST/${SEQ}_progression.mp4"
