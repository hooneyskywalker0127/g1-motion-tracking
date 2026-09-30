"""같은 LAFAN1 클립을 G1 과 IGRIS-C 로 리타게팅한 결과를 사람 골격과 한 장면에 담는다.

두 로봇을 MjSpec 으로 한 모델에 붙여 같은 바닥, 같은 카메라로 그린다. 그래서 키와
보폭 차이가 화면에 그대로 나온다. 왼쪽부터 G1, 사람(bvh 원본), IGRIS.
리타게팅은 다시 풀지 않고 retarget_all.py 가 저장한 pkl 을 읽는다.

GMR 은 골반 궤적도 월드 원점 기준으로 비율만큼 줄인다(G1 0.9, IGRIS 1.01). 그래서
G1 은 사람 경로의 0.88배를 걸어 방 안에서 사람과 최대 0.65m 벌어진다. 나란히 보이려고
매 프레임 각 로봇의 골반 수평 위치를 사람 골반에 맞추고 옆으로만 민다. 자세는 그대로다.

    MUJOCO_GL=egl python src/render_retarget_two.py --seq walk1_subject2 --out x.mp4 \
        --start 0 --seconds 60
"""
import argparse
import os
import pickle

import imageio
import mujoco as mj
import numpy as np
from general_motion_retargeting import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# bvh 원본 관절 연결. render_compare.py 의 BONES_RAW 와 같다(그 파일은 import 하면
# 인자 파싱이 돌아서 못 가져온다).
BONES_RAW = [
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToe"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToe"),
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Head"),
    ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--seq", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--start", type=float, default=0.0, help="초")
parser.add_argument("--seconds", type=float, default=0.0, help="0이면 끝까지")
parser.add_argument("--gap", type=float, default=2.0, help="셋 사이 간격(m). 1920 폭, 화각 20도·거리 9.5 에서 화면 3등분 자리")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--still", type=int, default=-1, help="이 프레임 하나만 png 로")
args = parser.parse_args()

FPS = 30
ROBOTS = [  # (접두사, GMR 이름, pkl 폴더, 옆으로 민 거리의 부호, 라벨)
    ("g1_", "unitree_g1", "outputs/retarget", -1, "Unitree G1  1.32 m / 35 kg"),
    ("igris_", "igris_c", "outputs/retarget_igris", +1, "IGRIS-C  1.5 m / 58 kg"),
]

spec = mj.MjSpec()
spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], group=1,
                        rgba=[0.82, 0.84, 0.87, 1])
spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
spec.visual.global_.offwidth = args.width
spec.visual.global_.offheight = args.height
# 셋이 좌우로 2m 씩 떨어져 있어 화각이 넓으면 양끝을 보는 방향이 ±25도 갈라진다.
# 같은 방향을 봐도 한쪽은 정면, 한쪽은 옆모습으로 보였다. 망원으로 ±12도까지 줄인다.
spec.visual.global_.fovy = 20
for prefix, name, *_ in ROBOTS:
    child = mj.MjSpec.from_file(str(ROBOT_XML_DICT[name]))
    frame = spec.worldbody.add_frame()
    frame.attach_body(child.worldbody.first_body(), prefix, "")
model = spec.compile()
data = mj.MjData(model)

motions = []
for prefix, _name, folder, sign, _ in ROBOTS:
    r = pickle.load(open(os.path.join(REPO, folder, f"{args.seq}.pkl"), "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    jid = [j for j in range(model.njnt)
           if mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j).startswith(prefix)]
    adr = model.jnt_qposadr[jid[0]]
    motions.append((q, adr, np.array([0.0, sign * args.gap, 0.0])))

human_frames, _ = load_bvh_file(f"/home/sehoon/data/lafan1/{args.seq}.bvh", format="lafan1")
n = min(len(human_frames), *(len(q) for q, *_ in motions))
t0 = int(args.start * FPS)
t1 = n if args.seconds <= 0 else min(n, t0 + int(args.seconds * FPS))
if args.still >= 0:
    t0, t1 = args.still, args.still + 1

renderer = mj.Renderer(model, height=args.height, width=args.width)
cam = mj.MjvCamera()
cam.distance = 9.5
cam.elevation = -10
cam.azimuth = 180  # -x 를 보므로 화면 왼쪽이 -y 다
# 시각 형상(그룹 1)만 그린다. 충돌 형상은 G1 이 그룹 0, IGRIS 가 그룹 2(주황)에 있다.
opt = mj.MjvOption()
opt.geomgroup[:] = [0, 1, 0, 0, 0, 0]

BONE = np.array([1.0, 0.55, 0.0, 1.0])
JOINT = np.array([1.0, 0.85, 0.3, 1.0])


def add_geom(scene, gtype, size, pos, rgba):
    g = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(g, type=gtype, size=size, pos=pos, mat=np.eye(3).flatten(), rgba=rgba)
    scene.ngeom += 1
    return g


font_b = ImageFont.truetype(FONT_B, 34)
font = ImageFont.truetype(FONT, 26)
labels = [ROBOTS[0][4], "LAFAN1 human (bvh)", ROBOTS[1][4]]


def draw_text(img, t):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, args.width, 70], fill=(14, 16, 20))
    d.text((30, 16), f"Same LAFAN1 clip, two robots   ·   {args.seq}   ·   GMR retargeting",
           font=font_b, fill=(255, 255, 255))
    d.text((args.width - 150, 20), f"{t / FPS:6.1f} s", font=font, fill=(170, 176, 186))
    d.rectangle([0, args.height - 60, args.width, args.height], fill=(14, 16, 20))
    for k, s in enumerate(labels):
        w = d.textlength(s, font=font)
        d.text(((k + 0.5) * args.width / 3 - w / 2, args.height - 46), s, font=font,
               fill=(255, 150, 40) if k == 1 else (230, 232, 236))
    return np.asarray(im)


writer = None
if args.still < 0:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = imageio.get_writer(args.out, fps=FPS, macro_block_size=1, quality=8)

for t in tqdm(range(t0, t1)):
    hips = np.asarray(human_frames[t]["Hips"][0])
    for q, adr, off in motions:
        data.qpos[adr:adr + q.shape[1]] = q[t]
        data.qpos[adr:adr + 2] = hips[:2] + off[:2]
    mj.mj_kinematics(model, data)
    cam.lookat = np.array([hips[0], hips[1], 0.75])
    renderer.update_scene(data, camera=cam, scene_option=opt)
    scene = renderer.scene
    human = {k: v for k, v in human_frames[t].items() if not k.endswith("FootMod")}
    for _, (pos, _rot) in human.items():
        add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]), np.asarray(pos), JOINT)
    for a, b in BONES_RAW:
        if a in human and b in human:
            g = add_geom(scene, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), BONE)
            mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.018,
                             np.asarray(human[a][0]), np.asarray(human[b][0]))
    img = draw_text(renderer.render(), t)
    if writer is None:
        Image.fromarray(img).save(args.out)
    else:
        writer.append_data(img)

if writer is not None:
    writer.close()
print(f"saved: {args.out}")
