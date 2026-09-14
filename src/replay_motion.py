"""레퍼런스 npz 를 MuJoCo 에 그대로 꽂아 운동학 재생만 한다 (물리 없음).

학습을 걸기 전에 모션이 멀쩡한지 눈으로 확인하려고 만든다.
정책이 없어도 되고 CPU 만 쓴다.
"""
import argparse, numpy as np, mujoco, imageio, onnx

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("out_mp4")
ap.add_argument("--order_onnx", default="/home/sehoon/colcon_ws/policies/walk1_subject1.onnx")
ap.add_argument("--mjcf", default="/home/sehoon/mtc_overlay/opt/ros/jazzy/share/"
                                  "unitree_description/mjcf/g1.xml")
ap.add_argument("--fps", type=int, default=50)
a = ap.parse_args()

d = np.load(a.npz)
jp, bp, bq = d["joint_pos"], d["body_pos_w"], d["body_quat_w"]
jn = {e.key: e.value for e in onnx.load(a.order_onnx).metadata_props}["joint_names"].split(",")

model = mujoco.MjModel.from_xml_path(a.mjcf)
data = mujoco.MjData(model)
qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]

r = mujoco.Renderer(model, height=720, width=960)
cam = mujoco.MjvCamera(); cam.distance = 3.0; cam.elevation = -15; cam.azimuth = 135
w = imageio.get_writer(a.out_mp4, fps=a.fps, macro_block_size=1)
for t in range(len(jp)):
    data.qpos[:3] = bp[t, 0]
    data.qpos[3:7] = bq[t, 0]
    data.qpos[qadr] = jp[t]
    mujoco.mj_forward(model, data)
    cam.lookat = data.qpos[:3]
    r.update_scene(data, cam); w.append_data(r.render())
w.close()
lo = np.array([data.xpos[i][2] for i in range(model.nbody)])
print(f"{a.out_mp4}  {len(jp)} 프레임  {len(jp)/a.fps:.1f}초")
