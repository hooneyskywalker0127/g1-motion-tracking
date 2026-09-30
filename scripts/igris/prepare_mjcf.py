"""IGRIS-C 공개 MJCF 에서 리타게팅용 운동학 모델을 만든다.

원 모델: github.com/robrosinc/igris_c_description_public (LICENSE 없음 — 파일을
여기 복사하지 않는다. 이 스크립트가 사용자의 로컬 클론에서 읽어 로컬에 쓴다.)

먼저 그쪽 저장소에서 펼친다.
    python scripts/export_model.py --format xml --base-type pelvis \
        --parallel false --end-effector none --fixed false
그 결과(mujoco/igris_c_v2_pelvis_none_free.xml)를 받아서
  - 관절마다 붙은 *_backlash 관절(±0.01 rad, 29개. 목에는 없다)을 뺀다. 31 자유도가 남는다.
  - actuator / sensor / keyframe 을 뺀다. IK 에는 필요 없고, keyframe 은 관절을 빼면
    qpos 길이가 안 맞는다. (그쪽 actuator 는 kv 와 dampratio 를 같이 적어
    MuJoCo 3.12 가 읽지도 못한다.)
  - meshdir 을 절대 경로로 바꿔 어디에 둬도 열리게 한다.

    python scripts/igris/prepare_mjcf.py <igris 클론> <출력 xml>
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

src_root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
src = src_root / "mujoco" / "igris_c_v2_pelvis_none_free.xml"

tree = ET.parse(src)
root = tree.getroot()

root.find("compiler").set("meshdir", str(src_root / "meshes" / "igris_c_v2") + "/")

for tag in ("actuator", "sensor", "keyframe"):
    for el in root.findall(tag):
        root.remove(el)

removed = 0
for parent in root.iter():
    for j in list(parent.findall("joint")):
        if j.get("name", "").endswith("_backlash"):
            parent.remove(j)
            removed += 1

out.parent.mkdir(parents=True, exist_ok=True)
tree.write(out)
print(f"{src.name} -> {out}  (backlash 관절 {removed}개 뺌)")
