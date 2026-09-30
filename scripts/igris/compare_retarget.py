"""같은 LAFAN1 클립을 G1 과 IGRIS-C 로 리타게팅한 결과를 나란히 점검한다.

  발바닥최저  발 몸체 원점 - (원점에서 발바닥까지 거리). 음수면 바닥에 박힌다
  <-3cm       발바닥이 3cm 넘게 박힌 프레임 수
  한계        관절 한계에 붙은 프레임 비율, 관절 중 최댓값
  >15r/s      어느 관절이든 15 rad/s 를 넘는 프레임 수(튐)

    python scripts/igris/compare_retarget.py <GMR 루트> <시퀀스 목록 파일>
"""
import pickle
import sys

import mujoco
import numpy as np

gmr, seq_file = sys.argv[1:]
ROBOTS = [
    ("IGRIS", f"{gmr}/assets/igris_c/igris_c_v2_31dof.xml", "outputs/retarget_igris",
     ["l_foot_original", "r_foot_original"], 0.071),
    ("G1", f"{gmr}/assets/unitree_g1/g1_mocap_29dof.xml", "outputs/retarget",
     ["left_ankle_roll_link", "right_ankle_roll_link"], 0.035),
]


def stats(xml, pkl, feet, sole):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    r = pickle.load(open(pkl, "rb"))
    q = np.concatenate([r["root_pos"], r["root_rot"][:, [3, 0, 1, 2]], r["dof_pos"]], 1)
    fz = []
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        fz.append(min(d.xpos[m.body(f).id][2] for f in feet))
    sz = np.array(fz) - sole
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    dp = r["dof_pos"]
    lim = ((dp <= lo + 1e-3) | (dp >= hi - 1e-3)).mean(0).max()
    vel = np.abs(np.diff(dp, axis=0)) * r["fps"]
    return sz.min(), (sz < -0.03).sum(), lim, (vel > 15).any(1).sum()


seqs = [s.strip() for s in open(seq_file) if s.strip() and not s.startswith("#")]
head = " | ".join(f"{n:>5s} 발바닥최저  <-3cm   한계 >15r/s" for n, *_ in ROBOTS)
print(f"{'클립':22s} {head}")
for s in seqs:
    cols = []
    for _, xml, out, feet, sole in ROBOTS:
        a = stats(xml, f"{out}/{s}.pkl", feet, sole)
        cols.append(f"{a[0]:16.3f} {a[1]:6d} {a[2]:6.1%} {a[3]:6d}")
    print(f"{s:22s} " + " | ".join(cols))
