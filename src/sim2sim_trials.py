"""시퀀스마다 여러 번 시행해 성공률과 오차를 낸다.

초기 교란을 Isaac 쪽 MotionCommandCfg 와 같은 분포로 준다. 그래야 Isaac 의
sim-dr 열과 MuJoCo 의 sim2sim 열이 같은 조건이 되고, 둘의 비를 전이 유지율로
읽을 수 있다. 예전에는 여기가 가우시안 0.02 였는데 Isaac 쪽과 분포도 다르고
근거도 없는 값이었다.

교란을 거는 방식도 Isaac 을 그대로 따른다 (commands.py 250~275행 확인).
  루트 위치   += U(pose_range)
  루트 자세   = quat_from_euler_xyz(U(roll), U(pitch), U(yaw)) * 원래 자세   (좌곱)
  루트 속도   += U(velocity_range).  선속도는 월드, 각속도는 월드로 더한 뒤
                MuJoCo free joint 규약(body-local)에 맞춰 돌려 넣는다
  관절 각도   += U(joint_position_range), 소프트 한계로 클립
  관절 속도   교란 없음

채점은 src/score_standard.py 하나만 쓴다. Isaac 쪽 eval_sym.py 가 같은 정의를
쓴다.

    python src/sim2sim_trials.py [시퀀스...] --n 100
"""
import argparse, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim import Sim2Sim
from score_standard import score

# Isaac tracking_env_cfg.py 의 MotionCommandCfg 값 그대로 (확인, 88~98행)
POSE_RANGE = dict(x=(-0.05, 0.05), y=(-0.05, 0.05), z=(-0.01, 0.01),
                  roll=(-0.1, 0.1), pitch=(-0.1, 0.1), yaw=(-0.2, 0.2))
VEL_RANGE = dict(x=(-0.5, 0.5), y=(-0.5, 0.5), z=(-0.2, 0.2),
                 roll=(-0.52, 0.52), pitch=(-0.52, 0.52), yaw=(-0.78, 0.78))
JOINT_RANGE = (-0.1, 0.1)
KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]


def quat_from_euler_xyz(r, p, y):
    """isaaclab.utils.math.quat_from_euler_xyz 와 같은 식. 반환은 wxyz."""
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def perturb(sim, rng):
    """reset_to_reference 직후의 상태에 Isaac 과 같은 초기 교란을 건다."""
    d = sim.data
    lo, hi = np.array([POSE_RANGE[k][0] for k in KEYS]), np.array([POSE_RANGE[k][1] for k in KEYS])
    s = rng.uniform(lo, hi)
    d.qpos[:3] += s[:3]
    d.qpos[3:7] = qmul(quat_from_euler_xyz(*s[3:]), d.qpos[3:7])

    lo, hi = np.array([VEL_RANGE[k][0] for k in KEYS]), np.array([VEL_RANGE[k][1] for k in KEYS])
    s = rng.uniform(lo, hi)
    d.qvel[:3] += s[:3]
    # qvel[3:6] 은 body-local 규약이다. 월드 델타를 현재 자세로 돌려 넣는다.
    w, x, y, z = d.qpos[3:7]
    R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                  [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                  [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])
    d.qvel[3:6] += R.T @ s[3:]

    q = d.qpos[sim.qadr] + rng.uniform(*JOINT_RANGE, len(sim.qadr))
    lim = sim.model.jnt_range[sim.model.dof_jntid[sim.vadr]]
    d.qpos[sim.qadr] = np.clip(q, lim[:, 0], lim[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 이면 모션 전체 길이")
    ap.add_argument("--no_perturb", action="store_true", help="교란 없이 결정론 1회 (sim 열)")
    ap.add_argument("--motion_dir", default="/home/sehoon/motions_fixed")
    ap.add_argument("--out", default="outputs/sym/mujoco_trials.npz")
    a = ap.parse_args()

    seqs = a.seqs or sorted(os.path.basename(f)[:-4] for f in
                            glob.glob(f"{a.motion_dir}/*.npz"))
    store = {}
    for seq in seqs:
        m = f"{a.motion_dir}/{seq}.npz"
        if not os.path.exists(m):
            print(f"{seq} 모션 없음", flush=True); continue
        sim = Sim2Sim(seq, motion_npz=m)
        nsteps = (int(a.seconds / 0.02) if a.seconds
                  else len(np.load(m)["joint_pos"]) - 1)
        orig = sim.reset_to_reference
        rows = []
        n = 1 if a.no_perturb else a.n
        for s in range(n):
            rng = np.random.default_rng(s)

            def reset(step, _o=orig, _r=rng):
                _o(step)
                if not a.no_perturb:
                    perturb(sim, _r)

            sim.reset_to_reference = reset
            sim.last_action = np.zeros_like(sim.last_action)
            r = score(sim.run(0, nsteps, None))
            rows.append([r["bm_success"], r["poly_success"], r["mpkpe"], r["r_mpkpe"],
                         r["jl2"], r["jvel"], r["alive_frames"],
                         r["mpkpe_full"], r["r_mpkpe_full"], r["jl2_full"]])
        A = np.array(rows, float)
        store[seq] = A
        okm = A[:, 0].astype(bool)
        f = lambda c: A[okm, c].mean() if okm.any() else float("nan")
        print(f"{seq:<22} S_bm {A[:,0].mean():.3f}  S_poly {A[:,1].mean():.3f}  "
              f"MPKPE {f(2):7.1f}  R-MPKPE {f(3):6.1f}  E_joint {f(4):.3f}", flush=True)

    if store:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez(a.out, **store)
        A = np.concatenate(list(store.values()))
        print(f"\n{len(store)}개 시퀀스 · 시행 {n}회 · S_bm {A[:,0].mean():.3f}"
              f" · S_poly {A[:,1].mean():.3f} · {a.out}")


if __name__ == "__main__":
    main()
