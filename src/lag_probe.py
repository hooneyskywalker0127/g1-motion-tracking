"""표류가 공간 오차인지 시간 지연인지 가른다.

KungfuBot2/VMS 는 짧은 미래 창 안에서 최소를 취하는 보상을 쓴다.
그 발상이 우리 데이터에서도 유효하려면, 로봇이 레퍼런스보다 늦거나 이른 것이어야 한다.
레퍼런스를 tau 만큼 밀어가며 오차를 재고, 어느 tau 에서 최소인지 본다.
tau 를 아무리 밀어도 안 줄면 그건 지연이 아니라 공간 이탈이다.
"""
import glob, os, numpy as np

H = 50  # 1초
print(f"{'시퀀스':<22}{'tau=0 전역':>11}{'최소 전역':>10}{'최적 tau':>9}"
      f"{'tau=0 루트상대':>14}{'최소':>8}")
for f in sorted(glob.glob("outputs/sim2sim_full/*.npz")):
    s = os.path.basename(f)[:-4]
    d = np.load(f)
    ref, rob = d["ref_body_pos_w"], d["rob_body_pos_w"]
    n = len(ref) - H
    g = np.stack([np.linalg.norm(ref[t:t + n] - rob[:n], axis=-1).mean(axis=1)
                  for t in range(H + 1)])          # (H+1, n)
    rr = np.stack([np.linalg.norm(
        (ref[t:t + n] - ref[t:t + n, :1]) - (rob[:n] - rob[:n, :1]),
        axis=-1).mean(axis=1) for t in range(H + 1)])
    best = g.min(axis=0)
    tau = int(np.median(g.argmin(axis=0)))
    print(f"{s:<22}{g[0].mean():>11.3f}{best.mean():>10.3f}{tau:>9d}"
          f"{rr[0].mean():>14.3f}{rr.min(axis=0).mean():>8.3f}")
