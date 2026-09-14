"""ASAP 공개 모션 51개의 부양을 접지 국면 기준으로 전수 측정한다.

체공이 많은 점프 모션을 부양으로 오인하지 않으려면
루트 높이가 하위 30% 인 프레임(=서 있거나 착지)만 봐야 한다.
기준: BeyondMimic 정상 모션의 접지 국면 발바닥 중앙 -0.038 (csv 좌표계)
"""
import glob, os, numpy as np, mujoco, onnx, joblib

X = "/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
D = ("/home/sehoon/Projects/ASAP/humanoidverse/data/motions/"
     "g1_29dof_anneal_23dof/TairanTestbed/singles")
ASAP = ['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint',
 'left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint',
 'right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
 'waist_yaw_joint','waist_roll_joint','waist_pitch_joint','left_shoulder_pitch_joint',
 'left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint',
 'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint',
 'right_elbow_joint']

m = mujoco.MjModel.from_xml_path(X)
d = mujoco.MjData(m)
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ASAP]
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

def low():
    return min(((v @ d.geom_xmat[i].reshape(3, 3).T)[:, 2].min() + d.geom_xpos[i][2])
               for i, v in verts.items())

REF = -0.038
rows = []
for f in sorted(glob.glob(D + "/*.pkl")):
    mo = list(joblib.load(f).values())[0]
    name = os.path.basename(f).split("video_")[-1].replace("_amass.pkl", "")
    root, rot, dof = (np.asarray(mo["root_trans_offset"]), np.asarray(mo["root_rot"]),
                      np.asarray(mo["dof"]))
    lo = []
    for t in range(len(dof)):
        d.qpos[:3] = root[t]
        q = rot[t]; d.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        d.qpos[qadr] = dof[t]
        mujoco.mj_forward(m, d)
        lo.append(low())
    lo = np.array(lo)
    st = lo[root[:, 2] <= np.percentile(root[:, 2], 30)]
    rows.append((name, float(np.median(st)), float(st.min()),
                 float((st > REF + 0.03).mean()), len(dof)))

rows.sort(key=lambda r: -r[1])
print(f"{'모션':<36}{'접지중앙':>9}{'접지최소':>9}{'기준+3cm 초과':>14}{'프레임':>7}")
for n, md, mn, fr, k in rows:
    print(f"{n:<36}{md:>+9.3f}{mn:>+9.3f}{fr:>14.0%}{k:>7}")
md = np.array([r[1] for r in rows]); fr = np.array([r[3] for r in rows])
print(f"\n{len(rows)}개 · 기준 {REF:+.3f} (BeyondMimic 정상 모션의 접지국면 중앙)")
print(f"  접지국면 중앙의 중앙값 {np.median(md):+.3f}  최소 {md.min():+.3f}  최대 {md.max():+.3f}")
print(f"  기준보다 3 cm 이상 뜬 클립 {(md > REF + 0.03).sum()}개 / {len(rows)}")
print(f"  기준보다 6 cm 이상 뜬 클립 {(md > REF + 0.06).sum()}개")
