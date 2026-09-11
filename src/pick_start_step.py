#!/usr/bin/env python3
"""레퍼런스에서 안전한 시작 프레임을 고른다.

Retargeting Matters (arXiv:2510.02252, V-D) 가 sim2sim 성공률이 시작 프레임에
크게 좌우된다고 보고하지만 고르는 기준은 주지 않는다. 권고만 한다.

    We recommend ensuring that the start pose of the reference motion is such
    that the robot can safely reach it once policy inference starts.

그 선행 연구인 HuB (arXiv:2505.07294) 가 원리를 준다 — 양발 지지(double support)
상태에서 안정되어 있어야 한다는 것.

    Challenging balance motions are often sensitive to the initial pose, and even
    slight instability in the double-support stance prior to execution can
    adversely affect performance.

그래서 여기서는 두 조건을 건다.
  1. 양 발목이 모두 낮다      → 양발 지지
  2. 관절 속도가 작다          → 정지에 가깝다
조건을 만족하는 가장 이른 프레임을 고른다. 값을 지어내지 않고 조건만 건다.

    python3 src/pick_start_step.py [시퀀스 ...]
"""
import argparse
import pathlib

import numpy as np
import onnx
import onnxruntime as ort

POLICY_DIR = pathlib.Path("/home/sehoon/colcon_ws/policies")
FOOT_Z = 0.09       # m, 발목 링크 높이 상한 (접지로 볼 값)
VEL = 1.0           # rad/s, 관절 속도 상한
SCAN = 600          # 프레임 (12초)


def scan(seq):
    path = POLICY_DIR / f"{seq}.onnx"
    md = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    bn = md["body_names"].split(",")
    la, ra = bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")
    pel = bn.index("pelvis")

    rows = []
    for k in range(SCAN):
        jp, jv, bp = sess.run(["joint_pos", "joint_vel", "body_pos_w"],
                              {"obs": np.zeros((1, 160), np.float32),
                               "time_step": np.array([[k]], np.float32)})
        bp = bp[0]
        rows.append((float(bp[la, 2]), float(bp[ra, 2]), float(bp[pel, 2]),
                     float(np.abs(jv[0]).max())))
    a = np.array(rows)
    ok = (a[:, 0] < FOOT_Z) & (a[:, 1] < FOOT_Z) & (a[:, 3] < VEL)
    first = int(np.argmax(ok)) if ok.any() else -1
    return a, ok, first


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*")
    args = ap.parse_args()
    seqs = args.seqs or sorted(p.stem for p in POLICY_DIR.glob("*.onnx"))

    print(f"{'시퀀스':<22}{'프레임0 발z':>13}{'프레임0 속도':>13}"
          f"{'조건만족 최초':>14}{'12초내 만족비율':>16}")
    for s in seqs:
        a, ok, first = scan(s)
        print(f"{s:<22}{max(a[0,0],a[0,1]):>13.3f}{a[0,3]:>13.2f}"
              f"{(first if first >= 0 else '없음'):>14}{ok.mean()*100:>15.0f}%")
