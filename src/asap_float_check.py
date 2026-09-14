"""ASAP 모션 전체가 지면에서 얼마나 떠 있는지 MuJoCo 순운동학으로 잰다.

csv 의 루트 위치·자세와 관절각을 그대로 꽂고 mj_forward 만 돌려
모든 바디의 최저 z 를 본다. 물리는 안 쓴다.
기준: BeyondMimic 정상 모션은 접지 시 최저 바디 z 가 약 0.033~0.061 m.
"""
import glob, os, numpy as np, mujoco, onnx

MJCF = ("/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml")
jn = {e.key: e.value for e in
      onnx.load("/home/sehoon/colcon_ws/policies/walk1_subject1.onnx").metadata_props
      }["joint_names"].split(",")
m = mujoco.MjModel.from_xml_path(MJCF)
d = mujoco.MjData(m)
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]

rows = []
for f in sorted(glob.glob("/tmp/asap_csv/*.csv")):
    a = np.loadtxt(f, delimiter=",")
    lo = []
    for r in a:
        d.qpos[:3] = r[:3]
        d.qpos[3:7] = r[[6, 3, 4, 5]]     # csv 는 xyzw, MuJoCo 는 wxyz
        d.qpos[qadr] = r[7:]
        mujoco.mj_forward(m, d)
        lo.append(d.xpos[1:, 2].min())   # 0 번은 world 라 항상 z=0 이다
    lo = np.array(lo)
    rows.append((os.path.basename(f)[:-4], lo.min(), lo.mean(), len(a)))

rows.sort(key=lambda r: r[1])
print(f"{'모션':<30}{'최저 z':>9}{'평균 최저 z':>12}{'프레임':>7}")
for n, mn, mu, k in rows:
    print(f"{n:<30}{mn:>9.3f}{mu:>12.3f}{k:>7}")
mns = np.array([r[1] for r in rows])
print(f"\n{len(rows)}개 · 최저 z 의 중앙값 {np.median(mns):.3f} m")
print(f"  0.09 m 이상 떠 있는 것 {(mns > 0.09).sum()}개 / 0.02 m 이하로 파고드는 것 {(mns < 0.02).sum()}개")
print("  참고 · BeyondMimic 정상 모션 walk1_subject1 은 최저 0.033, 선 자세에서 0.061")
