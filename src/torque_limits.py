#!/usr/bin/env python3
"""기록한 관측에서 PD 토크를 복원해 Isaac 의 effort_limit_sim 과 비교한다.

MuJoCo 배포는 URDF 의 토크 제한을 강제하지 않는다. 실행 로그가 그렇게 말한다.

    [controller_manager]: Enforcing command limits is disabled.
                          Command limits from URDF will be ignored.

Isaac 은 `whole_body_tracking/robots/g1.py` 의 `effort_limit_sim` 으로 강제한다.
그래서 같은 정책이 두 시뮬에서 다른 토크를 쓴다. 얼마나 다른지 센다.

토크는 컨트롤러가 하드웨어에 넣는 PD 식을 그대로 복원한다.
    tau = kp * (default + action_scale * action - q) - kd * dq
게인·스케일·기본자세는 ONNX 메타데이터, q·dq·action 은 기록한 관측이다.

    python3 src/torque_limits.py [--dir policy_io]
"""
import argparse
import glob
import pathlib

import numpy as np
import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent

# whole_body_tracking/robots/g1.py 의 effort_limit_sim
LIMITS = {"hip_yaw": 88.0, "hip_roll": 139.0, "hip_pitch": 88.0, "knee": 139.0,
          "ankle": 50.0, "waist": 50.0, "shoulder": 25.0, "elbow": 25.0,
          "wrist": 25.0}


def limit_of(name):
    for key, value in LIMITS.items():
        if key in name:
            return value
    return None


def analyse(seq, path):
    d = np.load(path, allow_pickle=True)
    io = d["io"]
    md = {e.key: e.value for e in
          onnx.load(f"/home/sehoon/colcon_ws/policies/{seq}.onnx").metadata_props}
    names = md["joint_names"].split(",")
    qdef = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    kp = np.array([float(x) for x in md["joint_stiffness"].split(",")])
    kd = np.array([float(x) for x in md["joint_damping"].split(",")])
    scale = np.array([float(x) for x in md["action_scale"].split(",")])

    target = qdef + scale * io[:, 160:189]
    tau = kp * (target - (io[:, 73:102] + qdef)) - kd * io[:, 102:131]

    over = np.zeros(len(io), dtype=bool)
    worst = (None, 0.0, 0.0)
    for i, n in enumerate(names):
        lim = limit_of(n)
        if lim is None:
            continue
        peak = float(np.abs(tau[:, i]).max())
        over |= np.abs(tau[:, i]) > lim
        if peak > lim and peak / lim > (worst[2] / worst[1] if worst[1] else 0):
            worst = (n, lim, peak)
    return over.mean(), worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="policy_io")
    a = ap.parse_args()

    print(f"{'시퀀스':<22}{'초과 스텝':>10}{'최대 초과관절':>24}{'제한':>7}{'최대':>8}")
    for f in sorted(glob.glob(str(ROOT / "outputs" / a.dir / "*.npz"))):
        seq = pathlib.Path(f).stem
        try:
            ratio, (name, lim, peak) = analyse(seq, f)
        except Exception as exc:
            print(f"{seq:<22} 건너뜀 ({type(exc).__name__})")
            continue
        if name is None:
            print(f"{seq:<22}{ratio*100:>9.1f}%{'-':>24}")
        else:
            print(f"{seq:<22}{ratio*100:>9.1f}%{name:>24}{lim:>7.0f}{peak:>8.1f}")
