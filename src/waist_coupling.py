"""허리를 돌려 시선을 바꿀 때 손이 얼마나 따라 도는지, 그리고 팔이 얼마나 보상해야 하는지."""
import numpy as np, mujoco

X = "/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
m = mujoco.MjModel.from_xml_path(X); d = mujoco.MjData(m)

def q(name): return m.joint(name).qposadr[0]
HAND = "right_wrist_yaw_link"
ARM = [f"right_{j}" for j in ("shoulder_pitch_joint","shoulder_roll_joint",
       "shoulder_yaw_joint","elbow_joint","wrist_roll_joint","wrist_pitch_joint",
       "wrist_yaw_joint")]

def fk(waist_yaw=0.0, arm=None):
    d.qpos[:] = 0.0; d.qpos[3] = 1.0
    d.qpos[q("right_shoulder_pitch_joint")] = 0.3
    d.qpos[q("right_elbow_joint")] = 0.8
    if arm is not None:
        for n, v in zip(ARM, arm): d.qpos[q(n)] = v
    d.qpos[q("waist_yaw_joint")] = waist_yaw
    mujoco.mj_forward(m, d)
    return d.body(HAND).xpos.copy(), d.body("torso_link").xmat.copy().reshape(3,3)

p0, R0 = fk(0.0)
print(f"기준 자세 손 위치 {np.round(p0,3)}  (어깨피치 0.3, 팔꿈치 0.8 rad)")
print(f"\n{'허리yaw':>8} {'시선회전':>9} {'손 이동(cm)':>12} {'팔 보상 필요각(도,합)':>21}")
for deg in (15, 30, 45, 60, 90, 120, 150):
    w = np.deg2rad(deg)
    p1, R1 = fk(w)
    move = np.linalg.norm(p1 - p0) * 100
    # 손을 제자리에 두려면 팔이 얼마나 움직여야 하는가 (수치 IK, 위치만)
    arm = np.array([0.3,0,0,0.8,0,0,0])
    for _ in range(300):
        pa, _ = fk(w, arm)
        err = p0 - pa
        if np.linalg.norm(err) < 1e-4: break
        J = np.zeros((3, len(ARM)))
        for k in range(len(ARM)):
            a2 = arm.copy(); a2[k] += 1e-4
            J[:, k] = (fk(w, a2)[0] - pa) / 1e-4
        arm += np.linalg.pinv(J) @ err * 0.5
    comp = np.rad2deg(np.abs(arm - np.array([0.3,0,0,0.8,0,0,0]))).sum()
    resid = np.linalg.norm(p0 - fk(w, arm)[0]) * 100
    tag = "" if resid < 0.5 else f"  (잔차 {resid:.1f}cm — 도달 불가)"
    print(f"{deg:7d}도 {deg:8d}도 {move:12.1f} {comp:21.1f}{tag}")
