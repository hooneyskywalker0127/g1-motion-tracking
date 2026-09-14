"""jumps1_subject1 의 10회 롤아웃 중 실패 회차가 어디서 어떻게 죽는지 본다.

접촉 순간에 죽는지 체공 중에 죽는지부터 가른다.
실패 판정은 PolySim 기준 — 전역 바디 위치 오차 평균이 0.5 m 를 넘는 시점.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim import Sim2Sim
import mujoco

SEQ = sys.argv[1] if len(sys.argv) > 1 else "jumps1_subject1"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
sim = Sim2Sim(SEQ, motion_npz=f"/home/sehoon/motions_fixed/{SEQ}.npz")
feet = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
orig = sim.reset_to_reference

print(f"{'시행':>4}{'실패 스텝':>10}{'실패 시각':>10}{'그때 발높이 왼/오른':>22}"
      f"{'직전 30스텝 접촉비':>20}")
for s in range(N):
    rng = np.random.default_rng(s)
    def reset(step, _o=orig, _r=rng, _s=s):
        _o(step)
        if _s:
            sim.data.qpos[sim.qadr] += _r.normal(0, 0.02, len(sim.qadr))
            sim.data.qpos[3:7] += _r.normal(0, 0.02, 4)
            sim.data.qpos[3:7] /= np.linalg.norm(sim.data.qpos[3:7])
    sim.reset_to_reference = reset
    sim.last_action = np.zeros_like(sim.last_action)
    log = sim.run(0, 3000, None)
    ref, rob = log["ref_body_pos_w"], log["rob_body_pos_w"]
    per_t = np.linalg.norm(ref - rob, axis=-1).mean(axis=1)
    bad = np.argmax(per_t > 0.5) if (per_t > 0.5).any() else -1
    if bad < 0:
        print(f"{s:>4}{'성공':>10}"); continue
    onset = int(np.argmax(per_t > 0.15))
    li, ri = (sim.body_names.index("left_ankle_roll_link"),
              sim.body_names.index("right_ankle_roll_link"))
    zl = np.array([rob[t][li][2] for t in range(max(0, onset - 50), onset + 1)])
    zr = np.array([rob[t][ri][2] for t in range(max(0, onset - 50), onset + 1)])
    air = float((np.minimum(zl, zr) > 0.10).mean())
    rzl = np.array([ref[t][li][2] for t in range(max(0, onset - 50), onset + 1)])
    rzr = np.array([ref[t][ri][2] for t in range(max(0, onset - 50), onset + 1)])
    rair = float((np.minimum(rzl, rzr) > 0.10).mean())
    print(f"{s:>4}  발산시작 {onset:>5} ({onset/50:>5.1f}초)  "
          f"직전 1초 체공비율 로봇 {air:.2f} 레퍼런스 {rair:.2f}  "
          f"0.5m 도달 {bad} ({bad/50:.1f}초)")
    continue
    # 발 높이는 로그에 없으니 바디 인덱스로 다시 찾는다
    fz = [rob[bad][sim.body_names.index(n)][2] if n in sim.body_names else np.nan
          for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    w = slice(max(0, bad - 30), bad)
    zs = np.array([[rob[t][sim.body_names.index(n)][2]
                    for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
                   for t in range(w.start, w.stop)])
    contact = float((zs.min(axis=1) < 0.08).mean()) if len(zs) else float("nan")
    print(f"{s:>4}{bad:>10}{bad/50:>9.1f}초"
          f"{fz[0]:>11.3f}{fz[1]:>11.3f}{contact:>20.2f}")
