"""ASAP 의 리타게팅된 G1 모션 pkl 을 BeyondMimic csv 로 바꾼다.

PolySim 이 IsaacSim_DR → MuJoCo 에서 성공률 0.100 을 보고한 모션(Kobe)을
우리 파이프라인으로 그대로 돌려보려고 만든다. 모션을 맞춰야 비교가 된다.

ASAP 는 g1_29dof_anneal_23dof 로 손목 6축을 빼고 23축만 쓴다.
빠진 손목은 0 으로 채운다.
csv 열: root xyz(3) + root quat xyzw(4) + dof(29, Isaac 관절순서)
"""
import argparse, joblib, numpy as np

ASAP_DOF = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

ap = argparse.ArgumentParser()
ap.add_argument("pkl")
ap.add_argument("out_csv")
ap.add_argument("--isaac_order", default="/home/sehoon/colcon_ws/policies/walk1_subject1.onnx")
ap.add_argument("--z_offset", type=float, default=0.0,
                help="루트 z 에서 뺄 값. ASAP 모션은 지면보다 약 5 cm 떠 있다")
a = ap.parse_args()

import onnx
md = {e.key: e.value for e in onnx.load(a.isaac_order).metadata_props}
ISAAC_DOF = md["joint_names"].split(",")
assert len(ISAAC_DOF) == 29
# csv 는 URDF 순서를 기대한다 (csv_to_npz 가 그렇게 읽고 Isaac 이 내부에서 재정렬한다).
# ISAAC 순서로 넣으면 레퍼런스 관절이 뒤섞인다 (260914 확인, error_joint_pos 2.44 rad).
# 정상 시퀀스의 csv 와 npz 를 대조해 URDF 순서 이름표를 되찾는다.
_c = np.loadtxt("outputs/csv_fixed/walk1_subject1.csv", delimiter=",", max_rows=1)[7:]
_n = np.load("/home/sehoon/motions_fixed/walk1_subject1.npz")["joint_pos"][0]
_perm = [int(np.argmin(np.abs(_c - x))) for x in _n]
URDF_DOF = [None] * 29
for _i, _k in enumerate(_perm):
    URDF_DOF[_k] = ISAAC_DOF[_i]
assert None not in URDF_DOF, "URDF 순서 복원 실패"

d = joblib.load(a.pkl)
m = list(d.values())[0]
n = len(m["dof"])
dof29 = np.zeros((n, 29), np.float64)          # 손목 6축은 0 으로 남는다
for j, name in enumerate(ASAP_DOF):
    dof29[:, URDF_DOF.index(name)] = m["dof"][:, j]

root = np.asarray(m["root_trans_offset"], np.float64).copy()
root[:, 2] -= a.z_offset
rows = np.concatenate([root,
                       np.asarray(m["root_rot"], np.float64),   # xyzw, csv 규약과 같다
                       dof29], axis=1)
np.savetxt(a.out_csv, rows, delimiter=",", fmt="%.6f")
missing = [n_ for n_ in URDF_DOF if n_ not in ASAP_DOF]
print(f"{a.out_csv}  {n} 프레임  fps {float(m['fps']):.0f}  "
      f"0 으로 채운 관절 {len(missing)}개 {missing}")
