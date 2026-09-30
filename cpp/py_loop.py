#!/usr/bin/env python3
"""rt_loop.cpp 와 **같은 루프**를 파이썬으로 돈다.

두 가지를 본다.
  1. 포팅 검증 — 같은 입력에 같은 궤적이 나오는가. 지연만 재고 동작이 틀리면 뜻이 없다.
  2. 지연 비교 — 같은 일을 하는 두 구현의 꼬리가 어떻게 다른가.

읽는 파일도 C++ 과 같다(meta.json, ref_*.bin). src/sim2sim_student.py 를
그대로 쓰지 않은 이유는, 그쪽은 로깅과 렌더까지 하므로 언어 차이가 아니라
하는 일의 차이가 섞이기 때문이다.
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np
import onnxruntime as ort

SIM_DT = 0.001
DECIMATION = 20
NUM_OBS = 260
VEL_SCALE = 0.05

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=".")
ap.add_argument("--onnx", required=True)
ap.add_argument("--mjcf", required=True)
ap.add_argument("--steps", type=int, default=1000)
a = ap.parse_args()

D = pathlib.Path(a.dir)
md = json.loads((D / "meta.json").read_text())
nq, nb = md["num_dofs"], md["num_bodies"]
qdef = np.array(md["default_joint_pos"])
kp = np.array(md["stiffness"])
kd = np.array(md["damping"])
ascale = np.array(md["action_scale"])
tlim = np.array(md["torque_limit"])
anchor = md["anchor"]

rd = lambda n, *s: np.fromfile(D / n, dtype=np.float64).reshape(-1, *s)
ref_jp, ref_jv = rd("ref_jp.bin", nq), rd("ref_jv.bin", nq)
ref_bp, ref_bq = rd("ref_bp.bin", nb, 3), rd("ref_bq.bin", nb, 4)

m = mujoco.MjModel.from_xml_path(a.mjcf)
m.opt.timestep = SIM_DT
d = mujoco.MjData(m)
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in md["joint_names"]]
vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in md["joint_names"]]
bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in md["body_names"]]

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])

vscale = np.ones(NUM_OBS, dtype=np.float32)
vscale[29:58] = VEL_SCALE
vscale[67:73] = VEL_SCALE
vscale[102:131] = VEL_SCALE
vscale[189:218] = VEL_SCALE

# sim2sim.py reset_to_reference 와 같다. 관절 속도까지 레퍼런스에서 받는다 —
# 이걸 빼면 움직이는 레퍼런스를 정지 상태에서 쫓게 되어 첫 걸음에 밀린다.
d.qpos[0:3] = ref_bp[0, 0]        # body_names[0] 는 pelvis
d.qpos[3:7] = ref_bq[0, 0]
d.qpos[qadr] = ref_jp[0]
d.qvel[vadr] = ref_jv[0]
mujoco.mj_forward(m, d)

last_action = np.zeros(nq)
pd = qdef.copy()
t_infer, t_phys, t_cycle = [], [], []
steps = min(a.steps, md["frames"])

for k in range(steps):
    c0 = time.perf_counter()

    jp, jv, bp, bq = ref_jp[k], ref_jv[k], ref_bp[k], ref_bq[k]
    rob_pos = d.xpos[bid[anchor]]
    R = d.xmat[bid[anchor]].reshape(3, 3)
    w, x, y, z = bq[anchor]
    Rr = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    anchor_pos_b = R.T @ (bp[anchor] - rob_pos)
    rel = R.T @ Rr
    Rb = d.xmat[bid[0]].reshape(3, 3)
    blin = Rb.T @ d.qvel[0:3]

    q = d.qpos[qadr]
    dq = d.qvel[vadr]
    body_diff_b = (bp - d.xpos[bid]) @ R      # (R.T @ e) 를 행마다

    obs = np.concatenate([
        jp, jv, anchor_pos_b, rel[:, :2].reshape(-1),
        blin, d.qvel[3:6], q - qdef, dq, last_action,
        jp - q, jv - dq, body_diff_b.reshape(-1),
    ]).astype(np.float32) * vscale

    i0 = time.perf_counter()
    action = sess.run(["actions"], {"obs": obs[None]})[0][0]
    i1 = time.perf_counter()
    t_infer.append((i1 - i0) * 1e3)

    last_action = action.astype(np.float64)
    pd = qdef + ascale * last_action

    p0 = time.perf_counter()
    for _ in range(DECIMATION):
        tq = (pd - d.qpos[qadr]) * kp - d.qvel[vadr] * kd
        d.qfrc_applied[vadr] = np.clip(tq, -tlim, tlim)
        mujoco.mj_step(m, d)
    p1 = time.perf_counter()
    t_phys.append((p1 - p0) * 1e3)
    t_cycle.append((p1 - c0) * 1e3)


def report(name, v, budget=0.0):
    v = np.asarray(v)
    line = (f"{name:<22} p50 {np.percentile(v,50):6.3f}  p99 {np.percentile(v,99):6.3f}  "
            f"p99.9 {np.percentile(v,99.9):6.3f}  max {v.max():6.3f} ms")
    if budget:
        line += f"   예산 {budget:.0f}ms 초과 {(v > budget).sum()}/{len(v)}"
    print(line)


print(f"스텝 {steps}, 제어 50Hz, 물리 1kHz")
report("추론", t_infer)
report("물리 20스텝", t_phys)
report("제어 주기 전체", t_cycle, 20.0)

np.savetxt(D / "lat_py.csv", np.c_[t_infer, t_phys, t_cycle],
           delimiter=",", header="infer_ms,phys_ms,cycle_ms", comments="")
with open(D / "state_py.txt", "w") as f:
    for v in list(d.qpos[0:7]) + list(d.qpos[qadr]):
        f.write(f"{v:.12g}\n")
