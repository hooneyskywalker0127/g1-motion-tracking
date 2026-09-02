"""77개 리타게팅 결과의 IK 목표 추적 오차를 잰다.

GMR이 목표로 쓴 scaled_human_data는 pkl에 저장되지 않으므로 리타게팅을
다시 돌리면서 잰다. 결과는 outputs/metrics/quality.csv.

위치 가중치가 0~5인 몸통/어깨/고관절은 MuJoCo 바디 원점이 사람 관절 중심과
달라 상수 오프셋이 섞인다. 오차로 쓸 수 없어 제외하고, GMR이 실제로 위치를
구속하는 골반(100)/발목(50)/손목(10)만 본다.
"""
import argparse
import glob
import json
import os
import time

import mujoco as mj
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file

GROUPS = {
    "pelvis": [("pelvis", "Hips")],
    "feet": [("left_ankle_roll_link", "LeftFootMod"),
             ("right_ankle_roll_link", "RightFootMod")],
    "hands": [("left_wrist_yaw_link", "LeftHand"),
              ("right_wrist_yaw_link", "RightHand")],
}

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="/home/sehoon/data/lafan1")
parser.add_argument("--out", default="outputs/metrics/quality.csv")
parser.add_argument("--robot", default="unitree_g1")
args = parser.parse_args()

model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
data = mj.MjData(model)
ids = {g: [(model.body(rb).id, hj) for rb, hj in pairs]
       for g, pairs in GROUPS.items()}

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
files = sorted(glob.glob(os.path.join(args.src, "*.bvh")))
rows = []

for k, path in enumerate(files, 1):
    seq = os.path.basename(path).replace(".bvh", "")
    t0 = time.time()
    frames, height = load_bvh_file(path, format="lafan1")
    rt = GMR(src_human="bvh_lafan1", tgt_robot=args.robot,
             actual_human_height=height)

    per = {g: [] for g in GROUPS}
    for f in frames:
        data.qpos[:] = rt.retarget(f)
        mj.mj_forward(model, data)
        tgt = rt.scaled_human_data
        for g, pairs in ids.items():
            per[g].append(np.mean([
                np.linalg.norm(data.xpos[bid] - np.asarray(tgt[hj][0]))
                for bid, hj in pairs]))

    row = {"seq": seq, "frames": len(frames)}
    for g in GROUPS:
        v = np.array(per[g]) * 100.0  # cm
        row[f"{g}_mean"] = v.mean()
        row[f"{g}_p95"] = np.percentile(v, 95)
        row[f"{g}_max"] = v.max()
    rows.append(row)
    print(f"[{k:2d}/{len(files)}] {seq:28s} "
          f"feet {row['feet_mean']:5.2f}  hands {row['hands_mean']:5.2f} cm  "
          f"{time.time() - t0:5.1f}s", flush=True)

cols = ["seq", "frames"] + [f"{g}_{s}" for g in GROUPS
                            for s in ("mean", "p95", "max")]
with open(args.out, "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) if c in ("seq", "frames")
                         else f"{r[c]:.3f}" for c in cols) + "\n")
print(f"\nsaved: {args.out}")
