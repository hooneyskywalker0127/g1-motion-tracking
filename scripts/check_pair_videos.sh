#!/usr/bin/env bash
# 쌍 영상이 멀쩡한지 자동 검사한다.
#   1) Isaac 패널이 얼어 있지 않은가 (멀리 떨어진 프레임 간 차이)
#   2) Isaac 패널에 로봇이 남아 있는가 (패널 표준편차가 배경 수준이면 비어 있음)
# 260913: episode_length_s 를 안 넘겨 500프레임마다 리셋되던 것을 놓쳤다. 그 재발 방지.
set -u
R=/home/sehoon/Documents/GitHub/g1-motion-tracking
cd "$R"
printf "%-24s %8s %8s %8s  %s\n" 시퀀스 프레임 움직임 로봇존재 판정
for f in outputs/videos_pair/*_compare.mp4; do
  s=$(basename "$f" _compare.mp4)
  n=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$f")
  read -r mv pres <<< "$(/home/sehoon/miniconda3/envs/env_isaaclab/bin/python - "$f" "$n" <<'PY'
import subprocess, sys, numpy as np
f, n = sys.argv[1], int(sys.argv[2])
W, H = 2560, 970
def grab(k):
    p = subprocess.run(["ffmpeg","-v","error","-i",f,"-vf",f"select=eq(n\\,{k})",
                        "-vsync","0","-frames:v","1","-f","rawvideo","-pix_fmt","gray","-"],
                       capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8)
    return a[:W*H].reshape(H, W)[100:820, :1280] if a.size >= W*H else None
ks = [int(n*r) for r in (0.05, 0.35, 0.65, 0.92)]
fr = [g for g in (grab(k) for k in ks) if g is not None]
if len(fr) < 2:
    print("nan nan"); sys.exit()
mv = float(np.mean([np.abs(fr[i].astype(float)-fr[i+1].astype(float)).mean() for i in range(len(fr)-1)]))
pres = float(np.mean([x.std() for x in fr]))
print(f"{mv:.2f} {pres:.2f}")
PY
)"
  v=$(python3 -c "print('X' if float('$mv')<3.0 or float('$pres')<12.0 else 'O')" 2>/dev/null || echo "?")
  printf "%-24s %8s %8s %8s  %s\n" "$s" "$n" "$mv" "$pres" "$v"
done
