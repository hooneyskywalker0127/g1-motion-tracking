#!/usr/bin/env python3
"""기록한 policy_io 에서 링크 위치 오차(E-mpbpe, mm)를 낸다.

Isaac 쪽 scripts/rsl_rl/eval.py 의 `e_mpbpe_mm` 과 같은 양이다.

    rel = torch.norm(command.body_pos_relative_w - robot_body_pos, dim=-1).mean(dim=-1)

골반 상대 좌표로 계산하므로 두 시뮬레이터의 월드 프레임이 달라도 상관없다.
odom 도 필요 없다. 쓰는 값은 둘뿐이다.

  측정  io[:, 73:102] + default_joint_pos  → MuJoCo 모델에 넣고 순기구학
  기준  ONNX 의 body_pos_w, body_quat_w    → 레퍼런스 골반 기준으로 환산

MJCF 는 시뮬레이터가 실제로 쓰는 것과 같은 파일을 쓴다.
"""
import argparse
import json
import pathlib

import mujoco as mj
import numpy as np
import onnx
import onnxruntime as ort

POLICY_DIR = pathlib.Path("/home/sehoon/colcon_ws/policies")
MJCF = "/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
EVAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "eval"
CTRL_HZ = 50.0


def quat_to_mat(q):
    """(w, x, y, z) → 3x3"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--dir", default=str(pathlib.Path(__file__).resolve().parent.parent
                                        / "outputs" / "policy_io"))
    ap.add_argument("--stride", type=int, default=10, help="몇 스텝마다 계산할지")
    a = ap.parse_args()

    d = np.load(pathlib.Path(a.dir) / f"{a.seq}.npz", allow_pickle=True)
    io = d["io"]

    path = POLICY_DIR / f"{a.seq}.onnx"
    md = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    qdef = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    jnames = md["joint_names"].split(",")
    bnames = md["body_names"].split(",")
    pelvis = bnames.index("pelvis")

    m = mj.MjModel.from_xml_path(MJCF)
    data = mj.MjData(m)
    # ONNX 관절 순서 → MuJoCo qpos 주소
    qadr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in jnames]
    bid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in bnames]

    def ref(step):
        return sess.run(["joint_pos", "body_pos_w", "body_quat_w"],
                        {"obs": np.zeros((1, 160), np.float32),
                         "time_step": np.array([[step]], np.float32)})

    # 시작 time_step 을 찾는다 (관측 앞 29개 = 그 스텝의 레퍼런스 관절 위치)
    k0 = None
    for k in range(20000):
        if float(np.abs(ref(k)[0][0] - io[0, :29]).max()) < 1e-6:
            k0 = k
            break
    if k0 is None:
        raise SystemExit("시작 time_step 을 못 찾았다")

    errs = []
    for i in range(0, len(io), a.stride):
        _, bpos, bquat = ref(k0 + i)
        bpos, bquat = bpos[0], bquat[0]
        R = quat_to_mat(bquat[pelvis])
        ref_rel = (bpos - bpos[pelvis]) @ R          # 레퍼런스 골반 프레임

        data.qpos[:] = 0.0
        data.qpos[3] = 1.0                            # 베이스 쿼터니언 단위
        data.qpos[qadr] = io[i, 73:102] + qdef
        mj.mj_kinematics(m, data)
        rob = data.xpos[bid]
        Rr = data.xmat[bid[pelvis]].reshape(3, 3)
        rob_rel = (rob - rob[pelvis]) @ Rr

        errs.append(np.linalg.norm(ref_rel - rob_rel, axis=-1).mean())

    errs = np.array(errs) * 1000.0                    # m → mm
    ev = EVAL_DIR / f"{a.seq}.json"
    isaac = json.loads(ev.read_text()).get("e_mpbpe_mm") if ev.exists() else None
    print(f"{a.seq:<22} MuJoCo E-mpbpe {errs.mean():7.2f} mm "
          f"(중앙 {np.median(errs):.2f}, 최대 {errs.max():.2f}) "
          f"· Isaac {isaac if isaac is None else f'{isaac:.2f}'} mm"
          + ("" if isaac is None or not np.isfinite(isaac)
             else f" · 차이 {errs.mean() - isaac:+.2f}"))


if __name__ == "__main__":
    main()
