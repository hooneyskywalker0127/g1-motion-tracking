#!/usr/bin/env bash
# Isaac(정책) ↔ MuJoCo(sim-to-sim) 사이드바이사이드를 처음부터 다시 만든다.
#
#  1. state.npz 에서 MuJoCo 를 추적 카메라로 오프스크린 렌더 (로봇을 따라간다)
#  2. 두 패널을 붙이고, 패널마다 자기 지표를 자막으로 넣는다
#  3. 기존 *_mujoco.mp4 / *_sidebyside.mp4 는 지운다
#
# 인자를 주면 그 시퀀스만, 없으면 전부.
set -u
ROOT=/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
SYSPY=/home/sehoon/miniconda3/bin/python
FONT=/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf
export MUJOCO_GL=egl

if [ $# -gt 0 ]; then SEQS="$*"; else
  SEQS=$(for d in "$ROOT"/*/; do s=$(basename "$d")
          [ -f "$d/sim2sim/${s}_state.npz" ] && echo "$s"; done)
fi

for SEQ in $SEQS; do
  S2S="$ROOT/$SEQ/sim2sim"
  POL="$ROOT/$SEQ/compare/${SEQ}_policy.mp4"
  MJ="$S2S/${SEQ}_mujoco.mp4"
  OUT="$S2S/${SEQ}_sidebyside.mp4"
  [ -f "$POL" ] || { echo "SKIP $SEQ (policy 영상 없음)"; continue; }

  echo "=== $SEQ $(date +%H:%M:%S) ==="
  rm -f "$OUT" "$S2S/${SEQ}_mujoco_track.mp4"

  # 1. MuJoCo 추적 카메라 렌더. 이미 추적본이 있으면 건너뛴다.
  if [ -f "$MJ.track" ]; then
    echo "  추적 렌더 있음, 건너뜀"
  else
    rm -f "$MJ"
    "$PY" "$REPO/src/render_mujoco_state.py" "$SEQ" \
          --distance 2.598 --elevation -35.26 --azimuth 45 --out "$MJ" || { echo "  !! 렌더 실패"; continue; }
    touch "$MJ.track"
  fi

  # 2. Isaac 영상 앞을 잘라 두 패널의 시점을 맞춘다.
  #    레퍼런스 앞에 T자세 보정 구간이 있어 Isaac 은 그것까지 그리는데
  #    MuJoCo 기록은 그 뒤에서 시작했다. 그 차이만큼 Isaac 을 건너뛴다.
  OFF=$("$PY" "$REPO/src/sim2sim_offset.py" "$SEQ" | awk '{print $2}')
  case "$OFF" in ""|-1) OFF=0 ;; esac
  echo "  Isaac 앞 ${OFF}초 건너뜀"

  # 3. 자막 문구를 두 json 에서 뽑는다
  read -r L R <<<"$("$SYSPY" "$REPO/src/sim2sim_caption.py" "$SEQ")"
  [ -n "$L" ] || { echo "  !! 자막 생성 실패"; continue; }
  LEFT=$(echo "$L" | tr '~' ' '); RIGHT=$(echo "$R" | tr '~' ' ')

  # 4. 합성. 위 제목띠 70px, 아래 자막띠 110px.
  ffmpeg -v error -y \
    -ss "$OFF" -i "$POL" -i "$MJ" \
    -filter_complex "\
[0:v]fps=30,scale=960:540,setsar=1[a];\
[1:v]fps=30,scale=960:540,setsar=1[b];\
[a][b]hstack=inputs=2:shortest=1[v];\
[v]pad=1920:720:0:70:color=black[p];\
[p]drawtext=fontfile=$FONT:text='$SEQ  -  Trained policy (30,000 iterations)':\
fontcolor=white:fontsize=32:x=28:y=18:expansion=none,\
drawtext=fontfile=$FONT:text='Isaac Sim (정책)':fontcolor=white:fontsize=28:x=28:y=622:expansion=none,\
drawtext=fontfile=$FONT:text='$LEFT':fontcolor=0x8CE99A:fontsize=24:x=28:y=664:expansion=none,\
drawtext=fontfile=$FONT:text='MuJoCo (sim-to-sim)':fontcolor=white:fontsize=28:x=988:y=622:expansion=none,\
drawtext=fontfile=$FONT:text='$RIGHT':fontcolor=0xFFC078:fontsize=24:x=988:y=664:expansion=none[o]" \
    -map "[o]" -c:v libx264 -preset veryfast -crf 24 -pix_fmt yuv420p "$OUT" 2>&1 | head -3

  [ -f "$OUT" ] && echo "  완료 $(du -h "$OUT"|cut -f1)" || echo "  !! 합성 실패"
done
echo "전체 완료 $(date +%H:%M:%S)"
