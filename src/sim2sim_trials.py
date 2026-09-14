"""PolySim 방식으로 시퀀스마다 여러 번 시행해 성공률을 낸다.

지금까지 우리 성공률은 시퀀스당 결정론적 1회였다.
PolySim 의 0.100 은 10회 시행 중 1회 성공이라는 뜻이다.
표본 수가 다르면 나란히 놓을 수 없으므로 시행 횟수를 맞춘다.
초기 관절각·루트 자세에만 잡음을 준다 (학습 쪽 DR 과는 무관).
"""
import argparse, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim import Sim2Sim
from sim2sim_polysim import score

ap = argparse.ArgumentParser()
ap.add_argument("seqs", nargs="*")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seconds", type=float, default=0.0,
                help="0 이면 모션 전체 길이")
ap.add_argument("--noise", type=float, default=0.02)
ap.add_argument("--motion_dir", default="/home/sehoon/motions_fixed")
ap.add_argument("--out", default="outputs/lr/trials.npz")
a = ap.parse_args()

seqs = a.seqs or sorted(os.path.basename(f)[:-4] for f in
                        glob.glob(f"{a.motion_dir}/*.npz"))
allr = {}
for seq in seqs:
    m = f"{a.motion_dir}/{seq}.npz"
    if not os.path.exists(m):
        print(f"{seq} 모션 없음"); continue
    sim = Sim2Sim(seq, motion_npz=m)
    nsteps = (int(a.seconds / 0.02) if a.seconds
              else len(np.load(m)["joint_pos"]) - 1)
    orig = sim.reset_to_reference
    res = []
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
        r = score(sim.run(0, nsteps, None))
        res.append(r)
    ok = [r for r in res if r["success"]]
    allr[seq] = [float(r["success"]) for r in res]
    m_ = lambda k: np.mean([r[k] for r in ok]) if ok else float("nan")
    print(f"{seq:<22} 성공 {len(ok)}/{a.n}={len(ok)/a.n:.3f}  "
          f"E_g-mpjpe {m_('e_g_mpjpe'):.1f}  E_mpjpe {m_('e_mpjpe'):.1f}",
          flush=True)

if allr:
    v = np.array([np.mean(x) for x in allr.values()])
    print(f"\n{len(allr)}개 시퀀스 · 시행 {a.n}회 · 평균 성공률 {v.mean():.3f}")
    np.savez(a.out, **{k: np.array(x) for k, x in allr.items()})
