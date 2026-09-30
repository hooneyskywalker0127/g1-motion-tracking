"""GMR 의 bvh_lafan1_to_g1.json 에서 IGRIS-C 용 설정을 만든다.

두 로봇 모두 영자세에서 몸체 좌표계가 월드와 나란하다(G1 은 고관절 두 링크만 5도
기울어 있다). 그래서 회전 오프셋·가중치는 G1 것을 그대로 쓰고 몸체 이름과
사람→로봇 길이 비율만 바꾼다. 예외는 팔꿈치 아래 두 몸체다. G1 은 영자세에서
팔뚝이 앞(+x)을 향하고 IGRIS 는 아래(-z)로 곧게 뻗는다. 같은 오프셋을 쓰면 IK 가
팔꿈치를 한계(0, 곧게 편 쪽)에 붙인다(첫 시도에서 80~88% 프레임). 그래서 두 몸체의
오프셋 뒤에 Ry(-90도)를 곱해 IGRIS 의 -z 를 G1 의 +x 자리에 둔다.
GMR 은 목표 = 사람 회전 * 오프셋 으로 쓴다(motion_retarget.py offset_human_data).

비율은 두 모델의 영자세 관절 위치에서 쟀다.

  다리 (골반→발바닥) G1 0.792 m   IGRIS 0.890 m   x1.12  → 0.9  -> 1.01
    (발목 원점으로 재면 x1.08 인데, IGRIS 는 발바닥이 발목 아래 7.1cm 로 G1(3.5cm)의
     두 배라 골반이 낮게 잡혀 14클립 전부에서 발이 바닥에 박혔다.)
  몸통 (골반→어깨)   G1 0.292 m   IGRIS 0.332 m   x1.14  → 0.9  -> 1.0
  팔 (어깨→손목)     G1 ~0.43 m   IGRIS 0.486 m   x1.1   → 0.75 -> 0.82

    python scripts/igris/make_ik_config.py <GMR 루트>

GMR 쪽 general_motion_retargeting/params.py 에는 손으로 네 줄을 더한다.
    ROBOT_XML_DICT            "igris_c": ASSET_ROOT / "igris_c" / "igris_c_v2_31dof.xml"
    IK_CONFIG_DICT bvh_lafan1 "igris_c": IK_CONFIG_ROOT / "bvh_lafan1_to_igris.json"
    ROBOT_BASE_DICT           "igris_c": "pelvis"
    VIEWER_CAM_DISTANCE_DICT  "igris_c": 2.5
xml 은 prepare_mjcf.py 로 만든다.
"""

import json
import sys
from pathlib import Path

from scipy.spatial.transform import Rotation as R

gmr = Path(sys.argv[1])
cfg_dir = gmr / "general_motion_retargeting" / "ik_configs"
cfg = json.loads((cfg_dir / "bvh_lafan1_to_g1.json").read_text())

BODY = {
    "pelvis": "pelvis",
    "left_hip_yaw_link": "l_upper_leg",
    "left_knee_link": "l_lower_leg",
    "left_ankle_roll_link": "l_foot_original",
    "right_hip_yaw_link": "r_upper_leg",
    "right_knee_link": "r_lower_leg",
    "right_ankle_roll_link": "r_foot_original",
    "torso_link": "torso",
    "left_shoulder_yaw_link": "l_upper_arm",
    "left_elbow_link": "l_elbow_bracket",
    "left_wrist_yaw_link": "l_wrist_connector",
    "right_shoulder_yaw_link": "r_upper_arm",
    "right_elbow_link": "r_elbow_bracket",
    "right_wrist_yaw_link": "r_wrist_connector",
}
SCALE = {"Hips": 1.01, "Spine2": 1.0,
         "LeftUpLeg": 1.01, "RightUpLeg": 1.01, "LeftLeg": 1.01, "RightLeg": 1.01,
         "LeftFootMod": 1.01, "RightFootMod": 1.01,
         "LeftArm": 0.82, "RightArm": 0.82, "LeftForeArm": 0.82, "RightForeArm": 0.82,
         "LeftHand": 0.82, "RightHand": 0.82}

FOREARM = {"l_elbow_bracket", "l_wrist_connector", "r_elbow_bracket", "r_wrist_connector"}
RY = R.from_euler("y", -90, degrees=True)

cfg["robot_root_name"] = BODY[cfg["robot_root_name"]]
for t in ("ik_match_table1", "ik_match_table2"):
    cfg[t] = {BODY[k]: v for k, v in cfg[t].items()}
    for body in FOREARM:
        q = R.from_quat(cfg[t][body][4], scalar_first=True) * RY
        cfg[t][body][4] = [round(float(x), 8) for x in q.as_quat(scalar_first=True)]
assert set(cfg["human_scale_table"]) == set(SCALE)
cfg["human_scale_table"] = SCALE

out = cfg_dir / "bvh_lafan1_to_igris.json"
out.write_text(json.dumps(cfg, indent=4))
print(out)
