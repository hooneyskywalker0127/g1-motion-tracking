#!/usr/bin/env bash
# 한 시퀀스에 대해 정책 영상, 레퍼런스 영상, 좌우 합성본을 만든다.
# 자막의 수치는 outputs/eval/<시퀀스>.json에서 읽는다.
#
# 인자: 시퀀스 이름 [출력 폴더]
set -e
SEQ=$1
BASE=${2:-/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking}/$SEQ
DEST="$BASE/compare"
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf

RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"*/ | sort | tail -1)
RUN=$(basename "$RUN")
MOTION=/home/sehoon/motions/$SEQ.npz
[ -f "$MOTION" ] || MOTION="$WBT/artifacts/$SEQ:v0/motion.npz"
FRAMES=$("$PY" -c "import numpy as np;print(len(np.load('$MOTION')['joint_pos']))")
FPS=$("$PY" -c "import numpy as np;print(int(np.asarray(np.load('$MOTION')['fps']).reshape(-1)[0]))")
EPLEN=$("$PY" -c "print($FRAMES/$FPS + 5)")

mkdir -p "$DEST"
cd "$WBT"
export OMP_NUM_THREADS=1

echo "=== $SEQ · $RUN · $FRAMES 프레임 ==="

# 이미 뽑아 둔 영상이 있으면 다시 그리지 않는다.
if [ -f "$DEST/${SEQ}_policy.mp4" ]; then
  echo "건너뜀: 정책 영상 이미 있음"
else
# 정책: 물리 위에서 실행. 리셋 때 위상이 다시 뽑히지 않도록 0프레임 고정하고,
# 에피소드 길이를 클립 전체보다 길게 잡아 중간에 끊기지 않게 한다.
"$PY" scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 --num_envs=1 \
  --headless --video --video_length "$FRAMES" \
  --video_out "$DEST/${SEQ}_policy.mp4" \
  --load_run="$RUN" --checkpoint=model_29999.pt \
  --motion_file="$MOTION" --start_frame=0 --no_randomization \
  env.episode_length_s="$EPLEN" env.scene.contact_forces.debug_vis=false \
  env.commands.motion.debug_vis=false

fi

# 레퍼런스: 물리 없이 프레임마다 자세를 써넣고 렌더. 랜덤화와 무관하다.
if [ -f "$BASE/${SEQ}_reference.mp4" ]; then
  echo "건너뜀: 레퍼런스 영상 이미 있음"
else
"$PY" scripts/render_motion_video.py \
  --headless --motion_file "$MOTION" \
  --video_length "$FRAMES" --output "$BASE/${SEQ}_reference.mp4"
fi

# 자막 수치는 평가 결과에서 읽는다. 아직 평가 전이면 지표 줄을 비운다.
IFS='|' read -r M1 M2 <<< "$("$PY" -c "
import json, pathlib
p = pathlib.Path('$REPO/outputs/eval/$SEQ.json')
if p.exists():
    d = json.load(open(p))
    print(f\"completion {d['success_rate']*100:.0f}% over {d['num_envs']} rollouts, no domain randomization\"
          f\"|E_g-mpbpe {d['e_g_mpbpe_mm']:.0f} mm   E_mpbpe {d['e_mpbpe_mm']:.0f} mm   E_mpjpe {d['e_mpjpe_rad']:.3f} rad\")
else:
    print('not evaluated yet| ')
")"

T1="Trained policy (30,000 iterations)"
T2="Retargeting reference (mocap to G1)"

# drawtext는 콜론을 옵션 구분자로, %를 확장 문법으로 읽으므로 expansion=none을 준다.
ffmpeg -y -i "$DEST/${SEQ}_policy.mp4" -i "$BASE/${SEQ}_reference.mp4" \
 -filter_complex "\
[0:v]setpts=PTS-STARTPTS[l];[1:v]setpts=PTS-STARTPTS[r];\
[l][r]hstack=inputs=2[v];\
[v]pad=iw:ih+210:0:100:color=0x101010[p];\
[p]drawtext=fontfile=$FONT:expansion=none:text='$SEQ  -  $T1':fontcolor=white:fontsize=40:x=(1280-text_w)/2:y=30,\
drawtext=fontfile=$FONT:expansion=none:text='$SEQ  -  $T2':fontcolor=white:fontsize=40:x=1280+(1280-text_w)/2:y=30,\
drawtext=fontfile=$FONT:expansion=none:text='$M1':fontcolor=0xd0d0d0:fontsize=30:x=(w-text_w)/2:y=h-88,\
drawtext=fontfile=$FONT:expansion=none:text='$M2':fontcolor=0xd0d0d0:fontsize=30:x=(w-text_w)/2:y=h-44[out]" \
 -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$DEST/${SEQ}_compare.mp4"

echo "완료: $DEST"
