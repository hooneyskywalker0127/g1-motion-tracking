"""LAFAN1 bvh 전체를 G1으로 리타게팅해 pkl로 저장한다.

GMR의 bvh_to_robot_dataset.py는 이 버전에서 없는 함수(load_lafan1_file)와
없는 키(src_human="bvh")를 참조해 동작하지 않는다. 검증된 단일 경로
(bvh_to_robot.py와 같은 호출)를 그대로 반복한다.
"""
import argparse
import glob
import os
import pickle
import time

import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="/home/sehoon/data/lafan1")
parser.add_argument("--dst", default="/home/sehoon/Documents/GitHub/"
                                     "g1-motion-tracking/outputs/retarget")
parser.add_argument("--robot", default="unitree_g1")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--override", action="store_true")
args = parser.parse_args()

os.makedirs(args.dst, exist_ok=True)
files = sorted(glob.glob(os.path.join(args.src, "*.bvh")))
print(f"{len(files)} bvh files")

failed = []
for path in files:
    seq = os.path.basename(path).replace(".bvh", "")
    out = os.path.join(args.dst, f"{seq}.pkl")
    if os.path.exists(out) and not args.override:
        print(f"skip {seq}")
        continue

    t0 = time.time()
    try:
        frames, human_height = load_bvh_file(path, format="lafan1")
        retargeter = GMR(src_human="bvh_lafan1", tgt_robot=args.robot,
                         actual_human_height=human_height)
        qpos = np.array([retargeter.retarget(f)
                         for f in tqdm(frames, desc=seq, leave=False)])
    except Exception as e:
        print(f"FAILED {seq}: {e}")
        failed.append((seq, str(e)))
        continue

    with open(out, "wb") as f:
        pickle.dump({
            "fps": args.fps,
            "root_pos": qpos[:, :3],
            # wxyz -> xyzw, bvh_to_robot.py의 저장 형식과 맞춘다
            "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
            "dof_pos": qpos[:, 7:],
            "local_body_pos": None,
            "link_body_list": None,
        }, f)
    print(f"{seq}  {len(frames)} frames  {time.time() - t0:.1f}s")

print()
print(f"완료 {len(files) - len(failed)} / {len(files)}")
for seq, err in failed:
    print(f"  실패 {seq}: {err}")
