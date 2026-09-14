"""리타게팅한 클립 전체의 물리 결함을 네 가지로 전수 측정한다.

지표는 ULTRA(arXiv:2603.03279)와 PHUMA(arXiv:2510.26236)가 쓰는 것을 따른다.
  관통      발바닥 최저 z 가 음수인 프레임 비율과 최대 깊이
  부양      접지해야 할 국면에 발이 떠 있는 정도 (최저 z 가 임계 위)
  미끄러짐  발이 접지한 동안의 접선 속도
  한계위반  관절 속도가 로봇 한계를 넘는 프레임 비율
발바닥은 메시 정점을 월드로 옮겨 잰다. 바디 원점이나 bounding sphere 로는 틀린다.
"""
import argparse, glob, os, pickle, numpy as np, mujoco, onnx

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="outputs/retarget_fixed")
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--out", default="outputs/metrics/defect_census.csv")
a = ap.parse_args()

X = "/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
m = mujoco.MjModel.from_xml_path(X)
d = mujoco.MjData(m)
jn = {e.key: e.value for e in
      onnx.load("/home/sehoon/colcon_ws/policies/walk1_subject1.onnx").metadata_props
      }["joint_names"].split(",")
# pkl/csv 의 dof 순서는 URDF 순서이고 ONNX(Isaac) 순서와 다르다.
# npz 와 csv 를 대조해 얻은 순열로 이름을 되찾는다.
_c = np.loadtxt("outputs/csv_fixed/walk1_subject1.csv", delimiter=",", max_rows=1)[7:]
_n = np.load("/home/sehoon/motions_fixed/walk1_subject1.npz")["joint_pos"][0]
_perm = [int(np.argmin(np.abs(_c - x))) for x in _n]      # npz[i] = csv[_perm[i]]
urdf_names = [None] * len(jn)
for i, k in enumerate(_perm):
    urdf_names[k] = jn[i]
assert None not in urdf_names, "관절 순서 복원 실패"
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in urdf_names]
FOOT = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

verts, gbody = {}, {}
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
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        verts[i] = np.array([[p * s[0], q * s[1], r * s[2]]
                             for p in (-1, 1) for q in (-1, 1) for r in (-1, 1)])
    gbody[i] = m.geom_bodyid[i]

FOOT_GEOM = [i for i in verts if gbody[i] in FOOT]

def foot_low():
    """좌우 발 각각의 최저 표면 z"""
    out = []
    for b in FOOT:
        z = [((verts[i] @ d.geom_xmat[i].reshape(3, 3).T)[:, 2].min() + d.geom_xpos[i][2])
             for i in FOOT_GEOM if gbody[i] == b]
        out.append(min(z) if z else np.nan)
    return out

rows = []
for f in sorted(glob.glob(os.path.join(a.src, "*.pkl"))):
    seq = os.path.basename(f)[:-4]
    mo = pickle.load(open(f, "rb"))
    root, rot, dof = (np.asarray(mo["root_pos"]), np.asarray(mo["root_rot"]),
                      np.asarray(mo["dof_pos"]))
    fps = float(mo["fps"])
    idx = np.arange(0, len(dof), a.stride)
    lo, fx = [], []
    for t in idx:
        d.qpos[:3] = root[t]
        q = rot[t]                       # pkl 은 xyzw 로 저장돼 있다
        d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[qadr] = dof[t]
        mujoco.mj_forward(m, d)
        lo.append(foot_low())
        fx.append([d.xpos[b][:2].copy() for b in FOOT])
    lo = np.array(lo); fx = np.array(fx)
    dt = a.stride / fps

    pen = (lo < -0.005)
    pen_frac = float(pen.any(axis=1).mean())
    pen_max = float(max(0.0, -lo.min()))

    stance = lo < 0.01                                   # 접지로 보는 프레임
    vel = np.linalg.norm(np.diff(fx, axis=0), axis=-1) / dt
    both = stance[:-1] & stance[1:]
    slip = float(vel[both].mean()) if both.any() else 0.0
    slip_max = float(vel[both].max()) if both.any() else 0.0

    # 두 발 모두 임계 위 = 체공. 체공 비율이 과하면 부양 의심
    air = float((lo.min(axis=1) > 0.05).mean())

    jv = np.abs(np.diff(dof, axis=0)) * fps
    viol = float((jv > 23.0).any(axis=1).mean())         # 23 rad/s 초과 프레임 비율

    rows.append((seq, pen_frac, pen_max, slip, slip_max, air, viol, len(dof) / fps))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as fp:
    fp.write("seq,pen_frac,pen_max_m,slip_mean_mps,slip_max_mps,air_frac,jvel_viol_frac,dur_s\n")
    for r in rows:
        fp.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f\n" % r)

rows.sort(key=lambda r: -r[1])
print(f"{'클립':<26}{'관통비율':>9}{'최대깊이':>9}{'미끄럼':>8}{'체공':>7}{'속도위반':>9}")
for r in rows[:20]:
    print(f"{r[0]:<26}{r[1]:>9.1%}{r[2]:>9.3f}{r[3]:>8.3f}{r[5]:>7.1%}{r[6]:>9.2%}")
print(f"\n{len(rows)}개 · 저장 {a.out}")
