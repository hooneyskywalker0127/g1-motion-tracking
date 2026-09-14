"""시퀀스마다 여러 번 시행하며 프레임별 전역오차를 통째로 남긴다.

sim2sim_trials.py 는 성공 여부만 남겨서 평가 지평 H 를 바꿔 다시 잴 수 없다.
지평 효과를 보려면 시행마다 오차 시계열이 필요하다.
"""
import argparse, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim import Sim2Sim

ap = argparse.ArgumentParser()
ap.add_argument("seqs", nargs="*")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--noise", type=float, default=0.02)
ap.add_argument("--motion_dir", default="/home/sehoon/motions_fixed")
ap.add_argument("--out", default="outputs/metrics/horizon_trials.npz")
a = ap.parse_args()

seqs = a.seqs or sorted(os.path.basename(f)[:-4] for f in
                        glob.glob(f"{a.motion_dir}/*.npz"))
out = {}
for seq in seqs:
    m = f"{a.motion_dir}/{seq}.npz"
    if not os.path.exists(m):
        print(f"{seq} 모션 없음", flush=True); continue
    sim = Sim2Sim(seq, motion_npz=m)
    nsteps = len(np.load(m)["joint_pos"]) - 1
    orig = sim.reset_to_reference
    curves = []
    for s in range(a.n):
        rng = np.random.default_rng(s)
        def reset(step, _o=orig, _r=rng, _s=s):
            _o(step)
            if _s:
                sim.data.qpos[sim.qadr] += _r.normal(0, a.noise, len(sim.qadr))
                sim.data.qpos[3:7] += _r.normal(0, a.noise, 4)
                sim.data.qpos[3:7] /= np.linalg.norm(sim.data.qpos[3:7])
        sim.reset_to_reference = reset
        sim.last_action = np.zeros_like(sim.last_action)
        d = sim.run(0, nsteps, None)
        e = np.linalg.norm(d["ref_body_pos_w"] - d["rob_body_pos_w"],
                           axis=-1).mean(axis=1)
        curves.append(e.astype(np.float16))
    out[seq] = np.array(curves)
    ok = (out[seq] > 0.5).any(axis=1)
    print(f"{seq:<22} {len(curves)}회 · 전장 성공 {(~ok).sum()}/{a.n} · "
          f"{out[seq].shape[1]/50:.0f}초", flush=True)

if out:
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, **out)
    print(f"저장 {a.out}")
