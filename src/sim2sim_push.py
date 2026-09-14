"""외란을 주고 두 시뮬을 비교한다.

Policy-Aware Simulator Learning (2605.29032) 의 지적 —
정상 궤적에서 두 시뮬이 일치해도, 정책이 평소 안 가는 상태에서
엔진이 크게 다르면 전이가 깨진다. 그 영역을 보려면 밀어넣어야 한다.

BeyondMimic 학습의 push_robot 이벤트와 같은 규격으로 민다.
  interval 1~3초, 몸통 속도에 더한다
  x,y +-0.5 m/s · z +-0.2 · roll,pitch +-0.52 rad/s · yaw +-0.78
"""
import argparse, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim import Sim2Sim
from sim2sim_polysim import score

VEL = dict(x=(-0.5, 0.5), y=(-0.5, 0.5), z=(-0.2, 0.2),
           roll=(-0.52, 0.52), pitch=(-0.52, 0.52), yaw=(-0.78, 0.78))

ap = argparse.ArgumentParser()
ap.add_argument("seqs", nargs="*")
ap.add_argument("--n", type=int, default=5, help="시드 수")
ap.add_argument("--seconds", type=float, default=0.0, help="0 이면 모션 전체 길이")
ap.add_argument("--interval", type=float, default=2.0, help="평균 밀치기 간격(초)")
ap.add_argument("--scale", type=float, default=1.0, help="밀치기 세기 배율")
ap.add_argument("--motion_dir", default="/home/sehoon/motions_fixed")
ap.add_argument("--out", default="outputs/lr/push.npz")
a = ap.parse_args()

import glob
seqs = a.seqs or sorted(os.path.basename(f)[:-4]
                        for f in glob.glob(f"{a.motion_dir}/*.npz"))
res = {}
for seq in seqs:
    m = f"{a.motion_dir}/{seq}.npz"
    if not os.path.exists(m) or not os.path.exists(
            f"/home/sehoon/colcon_ws/policies/{seq}.onnx"):
        continue
    sim = Sim2Sim(seq, motion_npz=m)
    nsteps = (int(a.seconds / 0.02) if a.seconds
              else len(np.load(m)['joint_pos']) - 1)
    orig_run = sim.run
    rows = []
    for s in range(a.n):
        rng = np.random.default_rng(1000 + s)
        sim.last_action = np.zeros_like(sim.last_action)
        # 밀치기를 넣기 위해 스텝 루프를 직접 돈다
        sim.reset_to_reference(0)
        import mujoco
        log = {"ref_body_pos_w": [], "rob_body_pos_w": [],
               "ref_joint_pos": [], "rob_joint_pos": [], "step": []}
        pd = sim.default_dof_pos.copy()
        next_push = rng.uniform(1.0, 3.0) if s else 1e9   # seed 0 은 무외란
        t_sec = 0.0
        for i in range(nsteps * 20):
            if i % 20 == 0:
                t = i // 20
                obs = sim.observation(t)
                act = sim.policy(obs, t)
                sim.last_action = act.copy()
                pd = sim.default_dof_pos + sim.action_scale * act
                jp, _, bpos, _ = sim.reference(t)
                log["step"].append(t)
                log["ref_body_pos_w"].append(bpos.copy())
                log["rob_body_pos_w"].append(sim.data.xpos[sim.bid].copy())
                log["ref_joint_pos"].append(jp.copy())
                log["rob_joint_pos"].append(sim.data.qpos[sim.qadr].copy())
                t_sec = t * 0.02
                if t_sec >= next_push:
                    v = np.array([rng.uniform(*VEL[k]) * a.scale
                                  for k in ("x", "y", "z", "roll", "pitch", "yaw")])
                    # Isaac 은 root_vel_w(월드 프레임 6D)에 더한다.
                    # MuJoCo 자유관절 qvel 은 선속도가 월드, 각속도가 바디 로컬이라
                    # 각속도만 월드→바디로 돌려서 더해야 조건이 같아진다.
                    R = sim.data.xmat[sim.bid[0]].reshape(3, 3)
                    sim.data.qvel[:3] += v[:3]
                    sim.data.qvel[3:6] += R.T @ v[3:]
                    next_push = t_sec + rng.uniform(1.0, 3.0)   # Isaac 과 같은 균등 1~3초
            tq = (pd - sim.data.qpos[sim.qadr]) * sim.stiffness \
                 - sim.data.qvel[sim.vadr] * sim.damping
            sim.data.qfrc_applied[sim.vadr] = np.clip(
                tq, -sim.torque_limits, sim.torque_limits)
            mujoco.mj_step(sim.model, sim.data)
        d = {k: np.asarray(v) for k, v in log.items()}
        r = score(d)
        rows.append((s, r["success"], r["e_g_mpjpe"], r["e_mpjpe"], r["over_frac"]))
    res[seq] = rows
    ok = sum(1 for r in rows[1:] if r[1])
    base = rows[0]
    print(f"{seq:<22} 무외란 {'O' if base[1] else 'X'} "
          f"E_mpjpe {base[3]:.1f} mm  |  외란 성공 {ok}/{len(rows)-1}  "
          f"E_mpjpe {np.mean([r[3] for r in rows[1:] if r[1]]) if ok else float('nan'):.1f} mm",
          flush=True)

np.savez(a.out, **{k: np.array(v, dtype=object) for k, v in res.items()})
