#!/usr/bin/env bash
# 도메인 랜덤화를 켠 조건에서 정책을 굴려 영상으로 남긴다.
# 밀치기가 들어간 순간에는 몸통에 화살표가 그려진다.
#
# 인자: 시퀀스 이름 [출력 폴더]
set -e
SEQ=$1
BASE=${2:-/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking}/$SEQ
DEST="$BASE/randomization"
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

RUN=$(basename "$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"*/ | sort | tail -1)")
MOTION=/home/sehoon/motions/$SEQ.npz
[ -f "$MOTION" ] || MOTION="$WBT/artifacts/$SEQ:v0/motion.npz"
FRAMES=$("$PY" -c "import numpy as np;print(len(np.load('$MOTION')['joint_pos']))")
FPS=$("$PY" -c "import numpy as np;print(int(np.asarray(np.load('$MOTION')['fps']).reshape(-1)[0]))")
EPLEN=$("$PY" -c "print($FRAMES/$FPS + 5)")

mkdir -p "$DEST"
cd "$WBT"
export OMP_NUM_THREADS=1

if [ -f "$DEST/${SEQ}_randomized.mp4" ]; then
  echo "건너뜀: 랜덤화 영상 이미 있음"
else
  "$PY" scripts/rsl_rl/play.py \
    --task=Tracking-Flat-G1-v0 --num_envs=1 \
    --headless --video --video_length "$FRAMES" \
    --video_out "$DEST/${SEQ}_randomized.mp4" \
    --load_run="$RUN" --checkpoint=model_29999.pt \
    --motion_file="$MOTION" --start_frame=0 --show_pushes \
    env.episode_length_s="$EPLEN" env.scene.contact_forces.debug_vis=false \
    env.commands.motion.debug_vis=false
fi

T1="$SEQ  -  Trained policy under domain randomization"
T2="$SEQ  -  Retargeting reference (no physics)"
M1="randomization on: random push every 1-3 s, randomized friction, torso CoM, joint offsets and reset state"
M2="the red arrow marks each push - direction and length follow the applied velocity"

# 랜덤화를 켠 조건이라 정책이 레퍼런스에서 벗어나거나 에피소드가 끊길 수 있다.
# 그때는 0프레임부터 다시 시작하므로 좌우 위상이 어긋난다. 그것도 결과의 일부다.
ffmpeg -y -i "$DEST/${SEQ}_randomized.mp4" -i "$BASE/${SEQ}_reference.mp4" \
 -filter_complex "\
[0:v]setpts=PTS-STARTPTS[l];[1:v]setpts=PTS-STARTPTS[r];\
[l][r]hstack=inputs=2[v];\
[v]pad=iw:ih+210:0:100:color=0x101010[p];\
[p]drawtext=fontfile=$FONT:expansion=none:text='$T1':fontcolor=white:fontsize=36:x=(1280-text_w)/2:y=32,\
drawtext=fontfile=$FONT:expansion=none:text='$T2':fontcolor=white:fontsize=36:x=1280+(1280-text_w)/2:y=32,\
drawtext=fontfile=$FONT:expansion=none:text='$M1':fontcolor=0xd0d0d0:fontsize=28:x=(w-text_w)/2:y=h-88,\
drawtext=fontfile=$FONT:expansion=none:text='$M2':fontcolor=0xd0d0d0:fontsize=28:x=(w-text_w)/2:y=h-44[out]" \
 -map "[out]" -shortest -c:v libx264 -pix_fmt yuv420p -crf 20 "$DEST/${SEQ}_randomization.mp4"

echo "완료: $DEST/${SEQ}_randomization.mp4"
