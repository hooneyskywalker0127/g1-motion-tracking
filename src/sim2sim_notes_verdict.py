#!/usr/bin/env python3
"""각 시퀀스의 sim2sim 유튜브 설명문 끝에 롤아웃 유효성 판정을 덧붙인다.

무효 롤아웃을 "정책이 실패했다"로 읽지 않도록, 실패의 종류를 영상 설명에 남긴다.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim2sim_validity import scan

BASE = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")

TEXT = {
 "정상 완주": None,
 "넘어짐(정상)":
  "Rollout check: the policy loses the reference and falls. The pelvis stays\n"
  "within a physically sensible range throughout, so this is a genuine\n"
  "sim-to-sim failure of the policy and is shown as such.",
 "수치 발산":
  "Rollout check: NOT a policy failure. The run tracks normally for most of a\n"
  "minute or more and then the solver diverges -- the pelvis leaves the range\n"
  "-1.0 to 1.5 m, in some runs reaching tens of metres. That is a simulator\n"
  "blow-up, not a fall. The error numbers above are meaningless for this clip\n"
  "and it is published only to document the failure mode.",
 "초기상태 이상":
  "Rollout check: NOT a policy failure. The robot starts the recording already\n"
  "collapsed -- pelvis height outside 0.60-0.95 m, where a standing G1 sits near\n"
  "0.79 m -- while the reference motion begins standing at about 0.81 m. The\n"
  "rollout was wrong before the policy did anything. The error numbers above are\n"
  "meaningless for this clip and it is published only to document the failure mode.",
}

for name, verdict, z0, zmin, zmax in scan():
    f = BASE / name / "sim2sim" / f"{name}_sim2sim_youtube.txt"
    if not f.exists():
        continue
    body = f.read_text().split("\n\nRollout check:")[0].rstrip()
    note = TEXT.get(verdict)
    if note:
        body += ("\n\n" + note +
                 f"\n(start pelvis {z0:.3f} m, min {zmin:.3f} m, max {zmax:.3f} m)")
    f.write_text(body + "\n")
    print(f"{name:22s} {verdict}")
