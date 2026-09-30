"""IGRIS-C 관절을 G1 정책의 29 슬롯에 대응시키는 표를 만든다: q_G1 = s * q_IGRIS + c.

G1 정책을 IGRIS 에서 출발점으로 쓰려면(Any2Any, arXiv 2605.23733 의 kinematic alignment)
관절 입력·출력을 G1 좌표로 바꿔야 한다. 이름으로 짝지으면 틀린다.
  - 허리 셋은 IGRIS 축이 음수(0 -1 0 등)라 부호가 반대다.
  - 팔꿈치는 G1 이 영자세에서 90도 굽어 있어 영점이 90도 다르다.
  - 손목은 이름이 엇갈린다. G1 팔뚝은 +x, IGRIS 팔뚝은 -z 를 따라 뻗어서, 팔뚝 축으로
    도는 관절이 G1 은 roll(x), IGRIS 는 yaw(z) 다.
그래서 짝은 기하로 정하고, 같은 LAFAN1 14클립을 두 로봇으로 리타게팅한 결과에서
부호(s = ±1)는 상관의 부호로, 영점(c)은 중앙값으로 잡는다. 기울기를 자유롭게 맞추면
관절 한계에 붙은 프레임 때문에 1 보다 작게 나와 쓰지 않는다.
목 두 관절(neck_yaw, neck_pitch)은 G1 에 없어 0 으로 고정한다.

    python scripts/igris/fit_slot_map.py <GMR 루트> <시퀀스 목록> <출력 json>
"""
import json
import pickle
import sys

import mujoco
import numpy as np

gmr, seq_file, out = sys.argv[1:]
jn = lambda m: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
G = jn(mujoco.MjModel.from_xml_path(f"{gmr}/assets/unitree_g1/g1_mocap_29dof.xml"))
I = jn(mujoco.MjModel.from_xml_path(f"{gmr}/assets/igris_c/igris_c_v2_31dof.xml"))

PAIR = {}
for g in G:
    side = "l_" if g.startswith("left") else "r_" if g.startswith("right") else ""
    core = g.replace("left_", "").replace("right_", "").replace("_joint", "")
    core = {"knee": "knee_pitch", "elbow": "elbow_pitch",
            "wrist_roll": "wrist_yaw", "wrist_yaw": "wrist_roll"}.get(core, core)
    PAIR[g] = side + core

seqs = [s.strip() for s in open(seq_file) if s.strip() and not s.startswith("#")]
qg = np.concatenate([pickle.load(open(f"outputs/retarget/{s}.pkl", "rb"))["dof_pos"] for s in seqs])
qi = np.concatenate([pickle.load(open(f"outputs/retarget_igris/{s}.pkl", "rb"))["dof_pos"] for s in seqs])

table = {}
print(f"{len(qg)} 프레임")
print(f"{'G1':28s} {'IGRIS':16s} {'s':>3s} {'c(도)':>7s} {'상관':>6s} {'잔차(도)':>8s}")
for k, g in enumerate(G):
    a, b = qi[:, I.index(PAIR[g])], qg[:, k]
    r = np.corrcoef(a, b)[0, 1]
    s = 1.0 if r > 0 else -1.0
    c = float(np.median(b - s * a))
    res = np.degrees(np.sqrt(np.mean((s * a + c - b) ** 2)))
    table[g] = {"igris": PAIR[g], "s": s, "c": c}
    print(f"{g:28s} {PAIR[g]:16s} {s:+3.0f} {np.degrees(c):7.1f} {r:+6.2f} {res:8.1f}")

json.dump({"g1_joints": G, "map": table, "igris_fixed": ["neck_yaw", "neck_pitch"]},
          open(out, "w"), indent=2)
print(out)
