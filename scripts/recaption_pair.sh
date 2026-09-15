#!/usr/bin/env bash
# 좌우 원본은 그대로 두고 자막만 다시 얹는다.
# make_video_pair.sh 는 Isaac 렌더와 MuJoCo 롤아웃을 처음부터 다시 돌리는데,
# 자막만 바꿀 때는 그게 낭비다. outputs/videos_pair 에 남은 *_isaac.mp4 와
# *_mujoco.mp4 를 다시 합치기만 한다.
set -e
SEQ=$1
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
OUT=$R/outputs/videos_pair
DEST="/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking/$SEQ/isaac_mujoco"
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
mkdir -p "$DEST"
for f in "$OUT/${SEQ}_isaac.mp4" "$OUT/${SEQ}_mujoco.mp4"; do
  [ -f "$f" ] || { echo "! 원본 없음 $f"; exit 1; }
done

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
ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,width,height \
  -of csv=p=0 "$OUT/${SEQ}_compare.mp4"
