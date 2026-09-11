#!/usr/bin/env python3
"""실패 판정이 로봇 탓인지 상태추정기 탓인지 가른다.

Isaac 종료 조건은 월드 높이를 쓰는데, MuJoCo 배포에서 그 높이는 상태추정기가 준다.
관절각은 측정값이라 믿을 수 있으니, 발을 지면에 놓고 순기구학으로 몸통 높이를
따로 구해 추정기 값과 비교한다. 둘이 어긋나면 로봇이 아니라 추정기가 튄 것이다.

BeyondMimic 논문이 한계로 적은 대목이다.

    failures caused by state estimation drift, particularly when the end-effector
    contact assumption is violated

    python3 src/check_estimator.py [시퀀스 ...]
"""
import pathlib
import sys

import mujoco as mj
import numpy as np
import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
MJCF = "/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
ANKLE_TO_SOLE = 0.02   # 발목 롤 링크 원점에서 발바닥까지


def check(seq, stride=10):
    d = np.load(ROOT / "outputs" / "policy_io" / f"{seq}.npz", allow_pickle=True)
    io, io_t, od, od_t = d["io"], d["io_t"], d["od"], d["od_t"]
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
    gap = []
    for i in range(0, len(io), stride):
        q = base[i, 3:7]
        q = q / (np.linalg.norm(q) + 1e-12)
        data.qpos[:3] = base[i, :3]
        data.qpos[3:7] = q
        data.qpos[qadr] = io[i, 73:102] + qdef
        mj.mj_kinematics(m, data)
        est = data.xpos[tid][2]
        kin = est - min(data.xpos[a][2] for a in aid) + ANKLE_TO_SOLE
        gap.append(est - kin)
    return np.array(gap)


if __name__ == "__main__":
    seqs = sys.argv[1:] or sorted(p.stem for p in (ROOT / "outputs" / "policy_io").glob("*.npz"))
    print(f"{'시퀀스':<22}{'중앙 불일치':>12}{'최대 음의 불일치':>17}{'0.1m 초과 비율':>16}")
    for s in seqs:
        try:
            g = check(s)
        except Exception as e:
            print(f"{s:<22} 건너뜀 ({type(e).__name__})")
            continue
        print(f"{s:<22}{np.median(g):>12.3f}{g.min():>17.3f}"
              f"{(np.abs(g) > 0.1).mean()*100:>15.1f}%")
