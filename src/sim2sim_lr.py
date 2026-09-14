"""MuJoCo 좌우 비대칭이 계통오차인지 단일 롤아웃의 우연인지 가린다.

Isaac 평가는 num_envs=100 의 평균이고 MuJoCo 는 결정론적 1회다.
비교가 성립하려면 MuJoCo 쪽에도 분포가 있어야 한다.
초기 관절각에 작은 잡음을 넣어 롤아웃을 여러 번 돌리고
왼다리/오른다리 오차비의 분포를 본다.
"""
import argparse, numpy as np, onnx
from sim2sim import Sim2Sim

ap = argparse.ArgumentParser()
ap.add_argument("seq")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--noise", type=float, default=0.01, help="초기 관절각 잡음 rad")
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--motion", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

sim = Sim2Sim(a.seq, motion_npz=a.motion)
md = {e.key: e.value for e in
      onnx.load(f"/home/sehoon/colcon_ws/policies/{a.seq}.onnx").metadata_props}
jn = md["joint_names"].split(",")
LEG = ("hip", "knee", "ankle")
L = [i for i, n in enumerate(jn) if n.startswith("left_") and any(k in n for k in LEG)]
R = [i for i, n in enumerate(jn) if n.startswith("right_") and any(k in n for k in LEG)]

orig_reset = sim.reset_to_reference
ratios, eL, eR = [], [], []
for s in range(a.n):
    rng = np.random.default_rng(s)
    def reset(step, _o=orig_reset, _r=rng, _s=s):
        _o(step)
        if _s:  # seed 0 은 잡음 없는 원래 롤아웃
            sim.data.qpos[sim.qadr] += _r.normal(0, a.noise, len(sim.qadr))
    sim.reset_to_reference = reset
    sim.last_action = np.zeros_like(sim.last_action)
    log = sim.run(0, int(a.seconds / 0.02), None)
    e = np.abs(log["ref_joint_pos"] - log["rob_joint_pos"])
    l, r = e[:, L].mean(), e[:, R].mean()
    ratios.append(l / r); eL.append(l); eR.append(r)
    print(f"  seed {s:2d}  왼 {l:.4f}  오른 {r:.4f}  비율 {l/r:.3f}  "
          f"{len(log['step'])} 스텝", flush=True)

ratios = np.array(ratios)
print(f"\n{a.seq}  비율 평균 {ratios.mean():.3f}  표준편차 {ratios.std():.3f}  "
      f"최소 {ratios.min():.3f}  최대 {ratios.max():.3f}  "
      f"1.0 초과 {int((ratios > 1).sum())}/{len(ratios)}")
if a.out:
    np.savez(a.out, ratios=ratios, eL=eL, eR=eR)
