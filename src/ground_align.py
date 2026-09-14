"""리타게팅된 모션의 부양을 프레임별로 고친다.

상수 오프셋으로는 안 된다 — 부양량이 클립 안에서 변한다.
Kobe 에 0.087 을 일괄로 빼니 가라앉은 국면 중앙은 맞았지만(+0.016)
어떤 프레임은 10 cm 파고들었다.

접지해야 할 국면(루트 높이 하위 30%)에서만 발바닥 높이를 재고,
그 값을 시간축으로 부드럽게 이어 보정 곡선을 만든다.
체공 구간은 양옆 접지 구간을 잇는 보간으로 처리해 점프 동역학을 안 망친다.
"""
import argparse, numpy as np, mujoco, onnx

ap = argparse.ArgumentParser()
ap.add_argument("csv_in")
ap.add_argument("csv_out")
ap.add_argument("--stance_pct", type=float, default=30.0,
                help="루트 높이 하위 몇 %% 를 접지 국면으로 볼지")
ap.add_argument("--target", type=float, default=0.016,
                help="접지 국면에서 목표로 하는 발바닥 높이 (walk1_subject1 기준)")
ap.add_argument("--smooth", type=int, default=9, help="보정 곡선 이동평균 창(프레임)")
ap.add_argument("--mjcf", default="/home/sehoon/mtc_overlay/opt/ros/jazzy/share/"
                                  "unitree_description/mjcf/g1.xml")
ap.add_argument("--order_onnx", default="/home/sehoon/colcon_ws/policies/walk1_subject1.onnx")
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(a.mjcf)
d = mujoco.MjData(m)
jn = {e.key: e.value for e in
      onnx.load(a.order_onnx).metadata_props}["joint_names"].split(",")
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]

verts = {}
for i in range(m.ngeom):
    if not (m.geom_contype[i] or m.geom_conaffinity[i]):
        continue
    t, s = m.geom_type[i], m.geom_size[i]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[i]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].reshape(-1, 3)
        verts[i] = v[:: max(1, len(v) // 300)]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        verts[i] = np.array([[0, 0, -s[0]]])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        verts[i] = np.array([[0, 0, s[1]], [0, 0, -s[1]], [s[0], 0, s[1]], [-s[0], 0, s[1]]])

def lowest():
    return min(((v @ d.geom_xmat[i].reshape(3, 3).T)[:, 2].min() + d.geom_xpos[i][2])
               for i, v in verts.items())

rows = np.loadtxt(a.csv_in, delimiter=",")
sole = np.empty(len(rows))
for t, r in enumerate(rows):
    d.qpos[:3] = r[:3]
    d.qpos[3:7] = r[[6, 3, 4, 5]]          # csv 는 xyzw, MuJoCo 는 wxyz
    d.qpos[qadr] = r[7:]
    mujoco.mj_forward(m, d)
    sole[t] = lowest()

rootz = rows[:, 2]
stance = rootz <= np.percentile(rootz, a.stance_pct)
if stance.sum() < 3:
    raise SystemExit("접지 국면 프레임이 너무 적다")

# 접지 프레임에서의 필요 보정량을 전체 시간축으로 보간한다
idx = np.arange(len(rows))
need = sole[stance] - a.target
off = np.interp(idx, idx[stance], need)
if a.smooth > 1:
    k = np.ones(a.smooth) / a.smooth
    off = np.convolve(np.pad(off, a.smooth // 2, mode="edge"), k, mode="valid")[:len(off)]

out = rows.copy()
out[:, 2] -= off
np.savetxt(a.csv_out, out, delimiter=",", fmt="%.6f")
print(f"{a.csv_out}  보정량 중앙 {np.median(off):+.3f} m  "
      f"최소 {off.min():+.3f}  최대 {off.max():+.3f}")
print(f"  보정 전 접지국면 발바닥 중앙 {np.median(sole[stance]):+.3f} → 목표 {a.target:+.3f}")
