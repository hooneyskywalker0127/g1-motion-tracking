"""평가 지평 H 에 따라 성공률이 어떻게 변하는지 곡선으로 낸다.

논문들은 성공률을 표로 비교하면서 지평 H 를 거의 밝히지 않는다.
같은 롤아웃이라도 H 를 바꾸면 성공률이 크게 움직인다는 걸 보인다.
"""
import glob, os, numpy as np

FAIL_M = 0.5
rows = []
for f in sorted(glob.glob("outputs/sim2sim_full/*.npz")):
    d = np.load(f)
    e = np.linalg.norm(d["ref_body_pos_w"] - d["rob_body_pos_w"], axis=-1).mean(axis=1)
    rows.append((os.path.basename(f)[:-4], e))

lens = np.array([len(e) / 50 for _, e in rows])
print(f"시퀀스 {len(rows)}개 · 길이 {lens.min():.0f}~{lens.max():.0f}초")
print()
print(f"{'지평H(초)':>9} {'대상':>4} {'성공률':>7}   생존 시퀀스")
for H in (5, 10, 20, 30, 60, 90, 120, 131, 150, 180, 210, 240, 261):
    ok = n = 0
    for s, e in rows:
        k = int(H * 50)
        if len(e) < k:            # H 까지 못 가는 클립은 대상에서 뺀다
            continue
        n += 1
        ok += int(not (e[:k] > FAIL_M).any())
    if n:
        print(f"{H:>9} {n:>4} {ok/n:>7.3f}")
