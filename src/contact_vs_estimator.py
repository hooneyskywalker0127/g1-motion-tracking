#!/usr/bin/env python3
"""추정기 오차가 양발 부양 구간과 겹치는지 본다.

`legged_estimation/LinearKalmanFilter.h` 의 상태추정기는 IMU 적분을 발 접촉으로
보정한다. 접촉 판정 문턱은 `contactForceThreshold = 150.` 이다. 양발이 모두 그
아래로 내려가면 보정이 끊기고 IMU 적분만 남는다.

관절각은 측정값이니 낮은 발을 지면에 놓고 순기구학으로 몸통 높이를 따로 구해
추정기 값과 비교한다. 그 차이가 부양 구간에 몰리는지 본다.

    python3 src/contact_vs_estimator.py [시퀀스 ...]
"""
import pathlib
import sys

import mujoco as mj
import numpy as np
import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
MJCF = "/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
CONTACT_TH = 150.0     # N, LinearKalmanFilter.h
ANKLE_TO_SOLE = 0.02


def analyse(seq, stride=5):
    d = np.load(ROOT / "outputs" / "policy_io_contact" / f"{seq}.npz", allow_pickle=True)
    io, io_t, od, od_t = d["io"], d["io_t"], d["od"], d["od_t"]
    wl = np.interp(io_t, d["wl_t"], d["wl"])
    wr = np.interp(io_t, d["wr_t"], d["wr_"])
    air = (wl < CONTACT_TH) & (wr < CONTACT_TH)

    md = {e.key: e.value for e in
          onnx.load(f"/home/sehoon/colcon_ws/policies/{seq}.onnx").metadata_props}
    qdef = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    jn = md["joint_names"].split(",")

    m = mj.MjModel.from_xml_path(MJCF)
    data = mj.MjData(m)
    qadr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in jn]
    tid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "torso_link")
    aid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)
           for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    base = np.stack([np.interp(io_t, od_t, od[:, i]) for i in range(7)], axis=1)
    idx = np.arange(0, len(io), stride)
    gap = np.empty(len(idx))
    for k, i in enumerate(idx):
        q = base[i, 3:7]
        q = q / (np.linalg.norm(q) + 1e-12)
        data.qpos[:3] = base[i, :3]
        data.qpos[3:7] = q
        data.qpos[qadr] = io[i, 73:102] + qdef
        mj.mj_kinematics(m, data)
        est = data.xpos[tid][2]
        kin = est - min(data.xpos[a][2] for a in aid) + ANKLE_TO_SOLE
        gap[k] = est - kin
    return gap, air[idx]


if __name__ == "__main__":
    seqs = sys.argv[1:] or sorted(
        p.stem for p in (ROOT / "outputs" / "policy_io_contact").glob("*.npz"))
    print(f"{'시퀀스':<22}{'부양비율':>9}{'접지중 불일치':>14}{'부양중 불일치':>14}"
          f"{'0.1m초과(접지)':>15}{'0.1m초과(부양)':>15}")
    for s in seqs:
        try:
            gap, air = analyse(s)
        except Exception as e:
            print(f"{s:<22} 건너뜀 ({type(e).__name__})")
            continue
        g_on, g_air = gap[~air], gap[air]
        f = lambda a: f"{np.median(a):.3f}" if a.size else "-"
        p = lambda a: f"{(np.abs(a) > 0.1).mean()*100:.1f}%" if a.size else "-"
        print(f"{s:<22}{air.mean()*100:>8.1f}%{f(g_on):>14}{f(g_air):>14}"
              f"{p(g_on):>15}{p(g_air):>15}")
