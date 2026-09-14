"""저자세 구간을 제외하고 성공률을 다시 낸다.

LAFAN1 클립 후반의 앉기·눕기 구간은 G1 이 재현할 수 없다.
그 구간을 평가에서 빼면 남는 실패가 무엇인지 보려는 것이다.
평가는 이미 저장된 전체 길이 롤아웃을 자르기만 한다. 재실행 없음.
"""
import glob, os, numpy as np

LOW = 0.35
rows = []
for f in sorted(glob.glob("outputs/sim2sim_full/*.npz")):
    s = os.path.basename(f)[:-4]
    z = np.load(f"/home/sehoon/motions_fixed/{s}.npz")["body_pos_w"][:, 0, 2]
    d = np.load(f)
    e = np.linalg.norm(d["ref_body_pos_w"] - d["rob_body_pos_w"], axis=-1).mean(axis=1)
    cut = int(np.argmax(z < LOW)) if (z < LOW).any() else len(e)
    cut = min(cut if cut > 0 else len(e), len(e))
    full = not (e > 0.5).any()
    trim = not (e[:cut] > 0.5).any()
    rows.append((s, full, trim, cut / 50, len(e) / 50))

print(f"{'시퀀스':<22}{'전체':>6}{'저자세 앞':>10}{'자른 길이(초)':>13}{'원래(초)':>10}")
for s, a, b, c, L in rows:
    print(f"{s:<22}{'O' if a else 'X':>6}{'O' if b else 'X':>10}{c:>13.1f}{L:>10.1f}")
print(f"\n전체 길이 성공률      {sum(r[1] for r in rows)}/{len(rows)} = "
      f"{sum(r[1] for r in rows)/len(rows):.3f}")
print(f"저자세 앞까지 성공률   {sum(r[2] for r in rows)}/{len(rows)} = "
      f"{sum(r[2] for r in rows)/len(rows):.3f}")
print(f"잘려나간 시간 합계     {sum(r[4]-r[3] for r in rows):.0f}초 / "
      f"{sum(r[4] for r in rows):.0f}초")
