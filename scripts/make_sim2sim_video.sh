#!/usr/bin/env bash
# Isaac 과 MuJoCo 를 같은 레퍼런스 step 에서 시작해 같은 길이로 찍고 좌우로 붙인다.
#
# 정렬은 추정이 아니라 구성으로 맞춘다.
#   MuJoCo  기록 n번째 메시지 = 레퍼런스 step k0 + n
#           k0 는 관측 앞 29개를 ONNX joint_pos 와 대조해 확정한다 (오차 0.0e+00)
#   Isaac   play.py --start_frame k0 → n번째 스텝 = 레퍼런스 step k0 + n
#           (play.py:198-206 이 term.time_steps 를 그 값으로 고정한다)
# 둘 다 50 Hz 이므로 같은 프레임 번호가 같은 순간이고, 프레임 수를 맞추면 같이 끝난다.
#
# 카메라는 양쪽 다 로봇을 따라간다.
#   Isaac   viewer.origin_type = "asset_root"  (tracking_env_cfg.py:319-322)
#   MuJoCo  오버레이 MJCF 의 <camera name="track" mode="trackcom">
#
#   bash scripts/make_sim2sim_video.sh <시퀀스> [초]
set -e
SEQ=$1
SECS=${2:-30}
REPO=/home/sehoon/Documents/GitHub/g1-motion-tracking
WBT=/home/sehoon/Projects/whole_body_tracking
PY=/home/sehoon/miniconda3/envs/env_isaaclab/bin/python
OUT=$REPO/outputs/videos_sim2sim
mkdir -p "$OUT"

FPS=30
STEPS=$(python3 -c "print(int($SECS*50))")

K0=$($PY - "$SEQ" <<'EOP'
import json, sys, pathlib
p = pathlib.Path("/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/metrics/sim2sim.json")
d = {r["motion"]: r for r in json.loads(p.read_text())}
print(d[sys.argv[1]]["start_time_step"])
EOP
)
echo "=== $SEQ · 레퍼런스 시작 step $K0 · ${SECS}초 (${STEPS} 스텝)"

# MuJoCo — 기록한 상태를 MuJoCo 렌더러로 그린다. 물리를 다시 돌리지 않는다.
$PY "$REPO/src/render_mujoco_recording.py" "$SEQ" \
    --seconds "$SECS" --fps $FPS --out "$OUT/${SEQ}_mujoco.mp4"

# Isaac — 공식 play.py 로 같은 시작 step 에서 같은 길이를 찍는다.
RUN=$(ls -1d "$WBT"/logs/rsl_rl/g1_flat/*_"$SEQ"/ | sort | tail -1)
RUN=$(basename "${RUN%/}")
CKPT=$(ls -1 "$WBT/logs/rsl_rl/g1_flat/$RUN"/model_*.pt | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
MOTION=/home/sehoon/motions/$SEQ.npz
[ -f "$MOTION" ] || MOTION="$WBT/artifacts/$SEQ:v0/motion.npz"

(cd "$WBT" && OMP_NUM_THREADS=1 "$PY" scripts/rsl_rl/play.py \
    --task=Tracking-Flat-G1-v0 --num_envs 1 \
    --load_run="$RUN" --checkpoint="model_$CKPT.pt" \
    --motion_file="$MOTION" --start_frame "$K0" --no_randomization \
    --video --video_length "$STEPS" --video_out "$OUT/${SEQ}_isaac.mp4") \
    2>&1 | grep -E "start at motion frame|Wrote|Error" || true

# 좌우 합성. 높이를 맞추고 같은 프레임 번호가 같은 순간이 되게 둔다.
ffmpeg -y -hide_banner -loglevel error \
  -i "$OUT/${SEQ}_isaac.mp4" -i "$OUT/${SEQ}_mujoco.mp4" \
  -filter_complex "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack=inputs=2" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/${SEQ}_compare.mp4"

for f in isaac mujoco compare; do
  printf "%-10s " "$f"
  ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,width,height \
    -of csv=p=0 "$OUT/${SEQ}_$f.mp4" 2>/dev/null || echo "없음"
done
