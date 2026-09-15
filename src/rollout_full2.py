"""전장 롤아웃을 다시 돌리되 자세와 관절속도까지 남긴다.

앞선 outputs/sim2sim_full 은 위치만 남겨서 BeyondMimic 종료조건 세 개와
mjlab R-MPKPE 를 하나도 못 냈다. 두 기준을 나란히 놓으려면 이게 필요하다.
"""
import glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sim2sim import Sim2Sim

OUT = "outputs/sim2sim_full2"
os.makedirs(OUT, exist_ok=True)
seqs = sorted(os.path.basename(f)[:-4] for f in glob.glob("outputs/sim2sim_full/*.npz"))
for seq in seqs:
    dst = f"{OUT}/{seq}.npz"
    if os.path.exists(dst):
        print(f"건너뜀 {seq}", flush=True); continue
    m = f"/home/sehoon/motions_fixed/{seq}.npz"
    sim = Sim2Sim(seq, motion_npz=m)
    n = len(np.load(m)["joint_pos"]) - 1
    d = sim.run(0, n, None)
    np.savez_compressed(dst, **d)
    e = np.linalg.norm(d["ref_body_pos_w"] - d["rob_body_pos_w"], axis=-1).mean(axis=1)
    print(f"{seq:<22} {n/50:.0f}초 · MPKPE {e.mean()*1000:.1f} mm · "
          f"PolySim {'성공' if not (e>0.5).any() else '실패'}", flush=True)
print("완료")
