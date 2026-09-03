#!/usr/bin/env bash
# 학습된 정책과 레퍼런스 모션을 같은 카메라로 렌더해 좌우로 붙인다.
# 인자: 시퀀스 이름, 학습 로그 폴더 이름
set -e

SEQ=${1:-walk2_subject4}
RUN=${2:-2026-09-02_16-30-24_walk2_subject4}
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
MOTION=/home/sehoon/motions/${SEQ}.npz
DEST=/home/sehoon/Desktop/참고/할일/26/09/260903
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
FRAMES=$(${PY} -c "import numpy as np;print(len(np.load('${MOTION}')['joint_pos']))")

mkdir -p "$DEST"
cd "$WBT"
export OMP_NUM_THREADS=1

# 정책: 물리 위에서 실행. 리셋마다 위상이 다시 뽑히지 않도록 0프레임 고정,
# 에피소드 길이는 클립 전체를 덮도록 늘린다.
${PY} scripts/rsl_rl/play.py \
  --task=Tracking-Flat-G1-v0 --num_envs=1 \
  --headless --video --video_length "$FRAMES" \
  --video_out "$DEST/${SEQ}_policy.mp4" \
  --load_run="$RUN" --checkpoint=model_29999.pt \
  --motion_file="$MOTION" --start_frame=0 \
  env.episode_length_s=240.0 env.scene.contact_forces.debug_vis=false \
  env.commands.motion.debug_vis=false

# 레퍼런스: 물리 없이 프레임마다 자세를 써넣고 렌더.
${PY} scripts/render_motion_video.py \
  --headless --motion_file "$MOTION" \
  --video_length "$FRAMES" --output "$DEST/${SEQ}_retarget.mp4"

T1="Trained policy (30,000 iterations)"
T2="Retargeting reference (mocap to G1)"
M1="tracking error vs reference   -   body position 4.8 cm  |  joint position 0.66 rad  |  anchor position 16.8 cm"
M2="mean reward 36.80  |  98.8% of episodes end by timeout, not tracking failure"

# 자막은 로봇을 가리지 않도록 위아래 띠 안에만 넣는다.
# drawtext는 콜론을 옵션 구분자로, %를 확장 문법으로 해석하므로 주의.
ffmpeg -y -i "$DEST/${SEQ}_policy.mp4" -i "$DEST/${SEQ}_retarget.mp4" \
 -filter_complex "\
[0:v]setpts=PTS-STARTPTS[l];[1:v]setpts=PTS-STARTPTS[r];\
[l][r]hstack=inputs=2[v];\
[v]pad=iw:ih+210:0:100:color=0x101010[p];\
[p]drawtext=fontfile=$FONT:expansion=none:text='$T1':fontcolor=white:fontsize=44:x=(1280-text_w)/2:y=28,\
drawtext=fontfile=$FONT:expansion=none:text='$T2':fontcolor=white:fontsize=44:x=1280+(1280-text_w)/2:y=28,\
drawtext=fontfile=$FONT:expansion=none:text='$M1':fontcolor=0xd0d0d0:fontsize=30:x=(w-text_w)/2:y=h-88,\
drawtext=fontfile=$FONT:expansion=none:text='$M2':fontcolor=0xd0d0d0:fontsize=30:x=(w-text_w)/2:y=h-44[out]" \
 -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$DEST/${SEQ}_compare.mp4"

echo "완료: $DEST/${SEQ}_compare.mp4"
