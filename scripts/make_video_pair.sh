#!/usr/bin/env bash
# Isaac 과 MuJoCo 를 같은 레퍼런스 프레임에서 시작해 같은 길이로 찍고 좌우로 붙인다.
#
# 정렬은 구성으로 성립한다.
#   MuJoCo  src/sim2sim.py 가 제어 스텝마다 한 프레임을 50 fps 로 쓴다
#   Isaac   play.py 가 1/step_dt = 50 fps 로 쓴다 (play.py:355)
#   둘 다 프레임 0 에서 시작하고 같은 스텝 수를 찍으므로
#   같은 프레임 번호가 같은 레퍼런스 스텝이다 (260912 레퍼런스 차이 0.000000 로 확인)
#
# 자막은 render_tracking.sh 와 같은 규약을 쓴다.
# 결과는 26/09/g1_motion_tracking/<시퀀스>/compare/ 에 둔다.
#
#   bash scripts/make_video_pair.sh <시퀀스> [초]   (초를 안 주면 모션 전체)
set -e
SEQ=$1
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
MOTION=/home/sehoon/motions_fixed/$SEQ.npz
OUT=$R/outputs/videos_pair
DEST="/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking/$SEQ/isaac_mujoco"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
mkdir -p "$OUT" "$DEST"

SECS=${2:-$("$PY" -c "import numpy as np;print(int(len(np.load('$MOTION')['joint_pos'])/50))")}
STEPS=$(python3 -c "print(int($SECS*50))")
# 기본 episode_length_s 는 10초다. 안 넘기면 500프레임마다 타임아웃으로
# 리셋돼 로봇이 처음으로 돌아간다 (260913 확인). 전체 길이보다 넉넉히 준다.
EPLEN=$(python3 -c "print(float($SECS)+5.0)")

"$PY" "$R/src/sim2sim.py" "$SEQ" --start_step 0 --seconds "$SECS" \
      --motion "$MOTION" --video "$OUT/${SEQ}_mujoco.mp4" | tail -1

RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"/ | sort | tail -1); RUN=$(basename "${RUN%/}")
CKPT=$(ls -1 "$WBT/logs/rsl_rl/g1_flat/$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
(cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 \
   --num_envs 1 --load_run="$RUN" --checkpoint="model_$CKPT.pt" \
   --motion_file="$MOTION" --start_frame 0 --no_randomization \
   --headless --video --video_length "$STEPS" --video_out "$OUT/${SEQ}_isaac.mp4" \
   env.episode_length_s="$EPLEN" \
   env.commands.motion.debug_vis=false env.scene.contact_forces.debug_vis=false) 2>&1 \
   | grep -E "start at motion frame|Wrote" || true

# 자막. 양쪽을 같은 정의로 내는 일은 src/caption_pair.py 가 한다.
IFS='|' read -r L1 L2 L3 L4 L5 <<< "$("$PY" "$R/src/caption_pair.py" "$SEQ")"

T1="Isaac Lab   trained policy (30,000 iterations)"
T2="MuJoCo   same policy, zero-shot"
HD="                                     Isaac Lab       MuJoCo"

# 표 다섯 줄은 등폭 글꼴이라야 열이 맞는다.
MONO=/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf
[ -f "$MONO" ] || MONO=$FONT
ROW() { echo "drawtext=fontfile=$MONO:expansion=none:text='$1':fontcolor=$2:fontsize=22:x=(w-text_w)/2:y=h-$3"; }

ffmpeg -y -hide_banner -loglevel error \
  -i "$OUT/${SEQ}_isaac.mp4" -i "$OUT/${SEQ}_mujoco.mp4" \
  -filter_complex "\
[0:v]setpts=PTS-STARTPTS,scale=1280:720[l];[1:v]setpts=PTS-STARTPTS,scale=1280:720[r];\
[l][r]hstack=inputs=2[v];\
[v]pad=iw:ih+300:0:100:color=0x101010[p];\
[p]drawtext=fontfile=$FONT:expansion=none:text='$T1':fontcolor=white:fontsize=34:x=(1280-text_w)/2:y=30,\
drawtext=fontfile=$FONT:expansion=none:text='$T2':fontcolor=white:fontsize=34:x=1280+(1280-text_w)/2:y=30,\
drawtext=fontfile=$FONT:expansion=none:text='$SEQ':fontcolor=0x9fd0ff:fontsize=26:x=(w-text_w)/2:y=h-196,\
$(ROW "$HD" 0x808080 162),\
$(ROW "$L1" 0xd0d0d0 132),\
$(ROW "$L2" 0xd0d0d0 104),\
$(ROW "$L3" 0xd0d0d0 76),\
$(ROW "$L4" 0xd0d0d0 48),\
$(ROW "$L5" 0xd0d0d0 20)[out]" \
  -map "[out]" -c:v libx264 -preset veryfast -pix_fmt yuv420p -crf 20 \
  "$OUT/${SEQ}_compare.mp4"

cp -f "$OUT/${SEQ}_compare.mp4" "$DEST/${SEQ}_isaac_mujoco.mp4"
printf "%-24s " "$SEQ"
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,r_frame_rate,width,height \
  -of csv=p=0 "$OUT/${SEQ}_compare.mp4"
