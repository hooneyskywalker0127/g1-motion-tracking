"""평가 지평 H 를 쓸어가며 조건 사이의 순위가 뒤집히는지 본다.

조건 A: MuJoCo (시드 10개) · 조건 B: Isaac (환경 100개)
판정은 양쪽 동일하게 PolySim 기준 — 전역 바디 위치 오차가 [0,H] 안에서
한 번이라도 0.5 m 를 넘으면 실패.
Isaac 자체 종료조건(수직·자세)으로 잰 성공률도 같이 내서, 같은 롤아웃이라도
판정 기준에 따라 H 의존성이 어떻게 달라지는지 본다.
"""
import glob, os, numpy as np

FAIL_M = 0.5
MJ = "outputs/metrics/horizon_trials.npz"
IS = "outputs/eval_gt"

mj = dict(np.load(MJ)) if os.path.exists(MJ) else {}
isa = {}
for f in sorted(glob.glob(f"{IS}/*_dump.npz")):
    d = np.load(f)
    if "g_t" not in d.files:
        continue
    isa[os.path.basename(f)[:-9]] = (d["g_t"].astype(np.float32), d["alive_t"])

seqs = sorted(set(mj) & set(isa))
print(f"MuJoCo {len(mj)}개 · Isaac {len(isa)}개 · 공통 {len(seqs)}개")
if not seqs:
    raise SystemExit("아직 둘 다 있는 시퀀스가 없다")

def rate(curves, H):
    """curves: (n, T) 시행별 오차. H 초까지 한 번도 0.5 m 를 안 넘으면 성공."""
    k = int(H * 50)
    if curves.shape[1] < k:
        return None
    return float((~(curves[:, :k] > FAIL_M).any(axis=1)).mean())

grid = [5, 10, 20, 30, 60, 90, 120, 150, 180, 210, 240]
print(f"\n{'H(초)':>6} {'대상':>4} {'MuJoCo':>7} {'Isaac':>7} {'Isaac자체':>9}  우세")
flips = []
for H in grid:
    a, b, c, n = [], [], [], 0
    for s in seqs:
        ra = rate(mj[s].astype(np.float32), H)
        g, alive = isa[s]
        rb = rate(g.T, H)
        if ra is None or rb is None:
            continue
        k = int(H * 50)
        c.append(float(alive[k - 1].mean()))     # Isaac 자체 종료조건 생존율
        a.append(ra); b.append(rb); n += 1
    if not n:
        continue
    ma, mb, mc = np.mean(a), np.mean(b), np.mean(c)
    win = "MuJoCo" if ma > mb else ("Isaac" if mb > ma else "동률")
    flips.append(win)
    print(f"{H:>6} {n:>4} {ma:>7.3f} {mb:>7.3f} {mc:>9.3f}  {win}")

ch = [(grid[i], flips[i - 1], flips[i]) for i in range(1, len(flips))
      if flips[i] != flips[i - 1] and "동률" not in (flips[i], flips[i - 1])]
print()
if ch:
    for H, x, y in ch:
        print(f"순위 뒤집힘: H={H}초 에서 {x} → {y}")
else:
    print("순위 뒤집힘 없음 — 이 조건쌍으로는 후보가 죽는다")
