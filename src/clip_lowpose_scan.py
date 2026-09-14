"""리타게팅한 클립 전체에서 저자세(앉기·눕기) 구간을 찾는다.

260912 확인: LAFAN1 클립 후반에 배우가 앉거나 눕는 구간이 있고,
그 구간은 G1 이 재현할 수 없어 정책 학습이 실패한다.
선별 단계에서 걸러야 하므로 전수 조사한다.
기준은 루트 높이다. 정상 보행은 0.6~0.8 m, 깊은 앉기 0.25~0.35, 눕기 0.1 이하.
"""
import glob, os, pickle, numpy as np

LOW, VERY_LOW = 0.35, 0.15
rows = []
for f in sorted(glob.glob("outputs/retarget_fixed/*.pkl")):
    d = pickle.load(open(f, "rb"))
    z = np.asarray(d["root_pos"])[:, 2]
    fps = float(d["fps"])
    low = z < LOW
    first = int(np.argmax(low)) if low.any() else -1
    rows.append((os.path.basename(f)[:-4], z.min(), float(low.mean()),
                 float((z < VERY_LOW).mean()), first / fps if first >= 0 else float("nan"),
                 len(z) / fps))

rows.sort(key=lambda r: -r[2])
print(f"{'클립':<24}{'루트z 최소':>10}{'0.35 미만':>10}{'0.15 미만':>10}"
      f"{'첫 저자세(초)':>13}{'길이(초)':>9}")
for n, mn, a, b, t, L in rows:
    if a == 0:
        continue
    print(f"{n:<24}{mn:>10.3f}{a:>10.1%}{b:>10.1%}{t:>13.1f}{L:>9.1f}")
a = np.array([r[2] for r in rows])
print(f"\n{len(rows)}개 중 저자세가 있는 클립 {(a > 0).sum()}개 "
      f"· 1% 넘는 클립 {(a > 0.01).sum()}개")
sel = "outputs/metrics/selected.txt"
if os.path.exists(sel):
    s = set(open(sel).read().split())
    bad = [n for n, mn, x, b, t, L in rows if x > 0.01 and n in s]
    print(f"선별 목록 {len(s)}개 중 저자세 1% 초과 {len(bad)}개: {bad}")
