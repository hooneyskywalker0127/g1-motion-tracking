"""IGRIS-C 공개 URDF 에서 IsaacLab 학습용 URDF 를 만든다.

원 모델: github.com/robrosinc/igris_c_description_public (LICENSE 없음 — 여기 복사하지
않는다. 사용자의 로컬 클론에서 읽어 로컬에 쓴다.)

먼저 그쪽 저장소에서 펼친다. (MJCF 도 prepare_mjcf.py 용으로 같이 펼친다)
    python scripts/export_model.py --format urdf --base-type pelvis --parallel false --end-effector hand
    python scripts/export_model.py --format xml  --base-type pelvis --parallel false --end-effector hand --fixed false
그 결과를 받아서
  - 손가락 관절 22개를 fixed 로 바꾼다. 손은 편 채로 굳는다(prepare_mjcf.py 와 같다).
  - 관절 토크 한계를 채운다. URDF 에는 effort="1000" 자리 값만 있다. MJCF 의 액추에이터
    클래스(actuator_150/120/90/60/8/7 → ±150/120/90/60/8/7 Nm)에서 관절마다 읽는다.
  - package:// 메시 경로를 절대 경로로 바꾼다.
URDF 에는 MJCF 의 *_backlash 관절이 없어 따로 뺄 것이 없다. 31 자유도가 남는다.

    python scripts/igris/prepare_urdf.py <igris 클론> <출력 urdf>
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

src_root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])

# MJCF 에서 관절별 액추에이터 클래스 → 토크 한계
mj = ET.parse(src_root / "mujoco" / "igris_c_v2_pelvis_hand_free.xml").getroot()
effort = {}
for j in mj.iter("joint"):
    m = re.fullmatch(r"actuator_(\d+)", j.get("class", ""))
    if m and j.get("name"):
        effort[j.get("name")] = float(m.group(1))

tree = ET.parse(src_root / "urdf" / "igris_c_v2_pelvis_hand.urdf")
root = tree.getroot()
FINGERS = ("thumb", "index", "middle", "ring", "little")

n_fixed, n_actuated = 0, 0
for j in root.findall("joint"):
    name = j.get("name")
    if j.get("type") != "revolute":
        continue
    if any(f in name for f in FINGERS):
        j.set("type", "fixed")
        for tag in ("limit", "axis", "dynamics"):
            for el in j.findall(tag):
                j.remove(el)
        n_fixed += 1
        continue
    j.find("limit").set("effort", str(effort[name]))
    n_actuated += 1

for mesh in root.iter("mesh"):
    mesh.set("filename", mesh.get("filename").replace("package://igris_c_description", str(src_root)))

out.parent.mkdir(parents=True, exist_ok=True)
tree.write(out)
print(f"{out}: 구동 관절 {n_actuated}개, 손가락 {n_fixed}개 fixed")
