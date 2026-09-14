"""우리 BeyondMimic 모션 npz 의 접지 상태를 메시 정점 기준으로 전수 확인한다.

측정법은 src/asap_sole_check.py 와 같다. 정상 모션에서 0 근처가 나오는 것을
확인하고 쓰는 것이 전제다 (walk1_subject1 최저 -0.014).
"""
import glob, os, numpy as np, mujoco, onnx

X = "/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
m = mujoco.MjModel.from_xml_path(X)
d = mujoco.MjData(m)
jn = {e.key: e.value for e in
      onnx.load("/home/sehoon/colcon_ws/policies/walk1_subject1.onnx").metadata_props
      }["joint_names"].split(",")
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]

verts = {}
for i in range(m.ngeom):
    if not (m.geom_contype[i] or m.geom_conaffinity[i]):
        continue
    t, s = m.geom_type[i], m.geom_size[i]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[i]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].reshape(-1, 3)
        verts[i] = v[:: max(1, len(v) // 400)]
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        verts[i] = np.array([[0, 0, -s[0]]])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        verts[i] = np.array([[0, 0, s[1]], [0, 0, -s[1]], [s[0], 0, s[1]],
                             [-s[0], 0, s[1]], [s[0], 0, -s[1]], [-s[0], 0, -s[1]]])
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        verts[i] = np.array([[a * s[0], b * s[1], c * s[2]]
                             for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])

def lowest():
    return min(((v @ d.geom_xmat[i].reshape(3, 3).T)[:, 2].min() + d.geom_xpos[i][2])
               for i, v in verts.items())

rows = []
for f in sorted(glob.glob("/home/sehoon/motions_fixed/*.npz")):
    z = np.load(f)
    lo = []
    for t in range(0, len(z["joint_pos"]), 10):    # 10프레임(0.2초) 간격
        d.qpos[:3] = z["body_pos_w"][t, 0]
        d.qpos[3:7] = z["body_quat_w"][t, 0]
        d.qpos[qadr] = z["joint_pos"][t]
        mujoco.mj_forward(m, d)
        lo.append(lowest())
    lo = np.array(lo)
    rows.append((os.path.basename(f)[:-4], lo.min(), lo.mean(),
                 float((lo < -0.02).mean())))

rows.sort(key=lambda r: r[1])
print(f"{'모션':<22}{'최저 z':>9}{'평균':>9}{'2cm 이상 관통 비율':>18}")
for n, mn, mu, fr in rows:
    print(f"{n:<22}{mn:>+9.3f}{mu:>+9.3f}{fr:>18.1%}")
