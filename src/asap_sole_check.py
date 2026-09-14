"""ASAP 공개 G1 모션의 실제 발바닥 접지 높이를 잰다.

충돌 geom 이 대부분 메시라서 bounding sphere 로는 못 잰다.
메시 정점을 직접 월드로 옮겨 최저 z 를 구한다. 물리는 안 쓴다.
0 이면 정확히 접지, 양수면 떠 있고, 음수면 지면을 파고든다.
"""
import glob, os, numpy as np, mujoco, joblib

X = "/home/sehoon/Projects/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.xml"
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

verts = {}          # geom id -> 로컬 정점 (N,3)
for i in range(m.ngeom):
    if not (m.geom_contype[i] or m.geom_conaffinity[i]):
        continue
    t = m.geom_type[i]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[i]
        a, n = m.mesh_vertadr[k], m.mesh_vertnum[k]
        v = m.mesh_vert[a:a + n].reshape(-1, 3)
        verts[i] = v[:: max(1, len(v) // 400)]          # 400점이면 최저점 찾기에 충분하다
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        verts[i] = np.array([[0, 0, -m.geom_size[i][0]]])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, h = m.geom_size[i][0], m.geom_size[i][1]
        verts[i] = np.array([[0, 0, h], [0, 0, -h], [r, 0, h], [-r, 0, h],
                             [r, 0, -h], [-r, 0, -h], [0, r, h], [0, -r, h]])

def lowest():
    z = []
    for i, v in verts.items():
        R = d.geom_xmat[i].reshape(3, 3)
        z.append((v @ R.T)[:, 2].min() + d.geom_xpos[i][2])
    return min(z)

rows = []
for f in sorted(glob.glob(D + "/*.pkl")):
    mo = list(joblib.load(f).values())[0]
    name = os.path.basename(f).split("video_")[-1].replace("_amass.pkl", "")
    lo = []
    for t in range(len(mo["dof"])):
        d.qpos[:3] = mo["root_trans_offset"][t]
        r = mo["root_rot"][t]; d.qpos[3:7] = [r[3], r[0], r[1], r[2]]
        d.qpos[qadr] = mo["dof"][t]
        mujoco.mj_forward(m, d)
        lo.append(lowest())
    lo = np.array(lo)
    rows.append((name, float(lo.min()), float(np.median(lo)), float((lo > 0.05).mean())))

rows.sort(key=lambda r: -r[3])
print(f"{'모션':<36}{'최저 z':>9}{'중앙 z':>9}{'0.05 초과':>10}")
for n, mn, md, fr in rows:
    print(f"{n:<36}{mn:>+9.3f}{md:>+9.3f}{fr:>10.0%}")
mn = np.array([r[1] for r in rows]); md = np.array([r[2] for r in rows])
fr = np.array([r[3] for r in rows])
print(f"\n{len(rows)}개")
print(f"  최저 z   중앙값 {np.median(mn):+.3f}")
print(f"  중앙 z   중앙값 {np.median(md):+.3f}  (참고: walk1_subject1 은 +0.018)")
print(f"  0.05 m 초과 프레임 비율의 중앙값 {np.median(fr):.0%}")
print(f"  그 비율이 50% 넘는 클립 {(fr > 0.5).sum()}개 / {len(rows)}")
