"""사람 골격과 리타게팅된 G1을 한 영상에 나란히 렌더한다.

GMR의 RobotMotionViewer는 녹화 시 로봇만 다시 그려서 사람 쪽이 영상에 안 남는다.
여기서는 오프스크린 렌더러의 scene에 사람 뼈대를 직접 얹어 함께 담는다.

--human raw    bvh 원본 그대로. 관절 24개, 사람 실제 치수. 기본값.
--human scaled GMR이 IK 목표로 쓰는 값. 팔다리를 로봇 비율로 줄이고 보정을
               더한 뒤라 사람의 임바디먼트가 아니다. IK 수렴만 볼 때 쓴다.
"""
import argparse
import os

import imageio
import mujoco as mj
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from tqdm import tqdm

from overlay import Overlay

# GMR이 IK로 쫓는 사람 관절 14개의 연결. scaled_human_data 키 기준.
# 목/머리/발끝은 추적 대상이 아니라 없다.
# bvh 원본 24관절의 연결.
BONES_RAW = [
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToe"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToe"),
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Neck"), ("Neck", "Head"),
    ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
]

# GMR이 IK로 쫓는 14관절의 연결. 목/머리/발끝은 추적 대상이 아니라 없다.
BONES_SCALED = [
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFootMod"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFootMod"),
    ("Hips", "Spine2"),
    ("Spine2", "LeftArm"), ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightArm"), ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--bvh_file", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--robot", default="unitree_g1")
parser.add_argument("--format", default="lafan1", choices=["lafan1", "nokov"])
parser.add_argument("--side", type=float, default=1.2, help="사람을 옆으로 밀 거리(m)")
parser.add_argument("--human", default="raw", choices=["raw", "scaled"],
                    help="raw=bvh 원본, scaled=GMR의 IK 목표")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--max_frames", type=int, default=0, help="0이면 전체")
parser.add_argument("--overlay", action="store_true", help="영어 자막 얹기")
args = parser.parse_args()

frames, human_height = load_bvh_file(args.bvh_file, format=args.format)
if args.max_frames:
    frames = frames[:args.max_frames]

retargeter = GMR(src_human=f"bvh_{args.format}", tgt_robot=args.robot,
                 actual_human_height=human_height)

model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
data = mj.MjData(model)
base_id = model.body(ROBOT_BASE_DICT[args.robot]).id
renderer = mj.Renderer(model, height=args.height, width=args.width)

cam = mj.MjvCamera()
cam.distance = 3.5
cam.elevation = -12
cam.azimuth = 135

offset = np.array([0.0, args.side, 0.0])
BONE_RGBA = np.array([1.0, 0.55, 0.0, 1.0])
JOINT_RGBA = np.array([1.0, 0.85, 0.3, 1.0])

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)


def add_geom(scene, gtype, size, pos, rgba):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, type=gtype, size=size, pos=pos,
                    mat=np.eye(3).flatten(), rgba=rgba)
    scene.ngeom += 1
    return g


bones = BONES_RAW if args.human == "raw" else BONES_SCALED

ov = None
if args.overlay:
    seq = os.path.basename(args.bvh_file).replace(".bvh", "")
    hp = {k: np.asarray(frames[0][k][0]) for k in
          ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftArm", "LeftForeArm"]}
    human_seg = {
        "thigh": np.linalg.norm(hp["LeftUpLeg"] - hp["LeftLeg"]),
        "shank": np.linalg.norm(hp["LeftLeg"] - hp["LeftFoot"]),
        "upper_arm": np.linalg.norm(hp["LeftArm"] - hp["LeftForeArm"]),
    }
    mj.mj_forward(model, data)
    rp = lambda n: data.xpos[model.body(n).id].copy()
    robot_seg = {
        "thigh": np.linalg.norm(rp("left_hip_pitch_link") - rp("left_knee_link")),
        "shank": np.linalg.norm(rp("left_knee_link") - rp("left_ankle_pitch_link")),
        "upper_arm": np.linalg.norm(
            rp("left_shoulder_pitch_link") - rp("left_elbow_link")),
    }
    ov = Overlay(seq, args.fps, human_seg, robot_seg)

# GMR이 위치까지 구속하는 바디만 본다. 위치 가중치가 0~5인 몸통/어깨/고관절은
# MuJoCo 바디 원점이 사람 관절 중심과 달라 상수 오프셋이 섞이므로 오차로 못 쓴다.
ERR_GROUPS = {
    "feet": [("left_ankle_roll_link", "LeftFootMod"),
             ("right_ankle_roll_link", "RightFootMod")],
    "hands": [("left_wrist_yaw_link", "LeftHand"),
              ("right_wrist_yaw_link", "RightHand")],
}
err_ids = {g: [(model.body(rb).id, hj) for rb, hj in pairs]
           for g, pairs in ERR_GROUPS.items()}
err_sum = {g: 0.0 for g in ERR_GROUPS}
err_n = 0

for i, frame in enumerate(tqdm(frames, desc="rendering")):
    qpos = retargeter.retarget(frame)
    data.qpos[:3] = qpos[:3]
    data.qpos[3:7] = qpos[3:7]
    data.qpos[7:] = qpos[7:]
    mj.mj_forward(model, data)

    cam.lookat = data.xpos[base_id] + offset / 2.0
    renderer.update_scene(data, camera=cam)
    scene = renderer.scene

    human = frame if args.human == "raw" else retargeter.scaled_human_data
    if args.human == "raw":
        # FootMod 두 개는 라판에 없고 GMR이 IK용으로 만든 것이라 뺀다.
        human = {k: v for k, v in human.items() if not k.endswith("FootMod")}
    for name, (pos, _rot) in human.items():
        add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]),
                 np.asarray(pos) + offset, JOINT_RGBA)
    for a, b in bones:
        if a not in human or b not in human:
            continue
        pa = np.asarray(human[a][0]) + offset
        pb = np.asarray(human[b][0]) + offset
        g = add_geom(scene, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                     np.zeros(3), BONE_RGBA)
        if g is not None:
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.018, pa, pb)

    img = renderer.render()
    if ov is not None:
        tgt = retargeter.scaled_human_data
        err = {}
        for g, pairs in err_ids.items():
            err[g] = 100.0 * float(np.mean([
                np.linalg.norm(data.xpos[bid] - np.asarray(tgt[hj][0]))
                for bid, hj in pairs]))
            err_sum[g] += err[g]
        err_n += 1
        err["mean"] = {g: err_sum[g] / err_n for g in err_sum}
        img = ov.draw(img, i, err)
    writer.append_data(img)

writer.close()
print(f"saved: {args.out}")
