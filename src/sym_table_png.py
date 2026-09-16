"""대칭 프로토콜 결과 표를 그림 파일로 낸다.

표 A(17 시퀀스 평균)와 표 B(시퀀스별)를 한 장에 세로로 쌓는다.
읽는 데이터와 정의는 src/sym_table.py 와 같다.

    python src/sym_table_png.py --out 표.png
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

KFONT = FontProperties(family="Noto Sans CJK JP")
MONO = FontProperties(family="Noto Sans Mono")
C_BM, C_POLY, C_MPKPE, C_RMPKPE, C_JL2, C_JVEL = range(6)

ORDER = [  # 17표.png 와 같은 순서. 완주율 내림차순, 같으면 전역 오차 오름차순이다.
    "walk4_subject1", "walk3_subject2", "walk1_subject2", "walk3_subject5",
    "aiming1_subject1", "walk1_subject5", "dance2_subject3", "walk2_subject1",
    "walk1_subject1", "walk2_subject4", "run2_subject4", "jumps1_subject1",
    "obstacles3_subject3", "walk2_subject3", "walk3_subject4",
    "obstacles2_subject1", "walk3_subject1",
]
rank = lambda s: (ORDER.index(s) if s in ORDER else len(ORDER), s)

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="outputs/sym")
ap.add_argument("--out", default="outputs/sym/표.png")
a = ap.parse_args()

seqs = sorted({os.path.basename(f).rsplit("_sim2sim", 1)[0]
               for f in glob.glob(f"{a.dir}/*_sim2sim.npz")}, key=rank)
rows = []
for s in seqs:
    r = {"seq": s}
    for cond in ("sim", "simdr"):
        p = f"{a.dir}/{s}_{cond}.json"
        if os.path.exists(p):
            j = json.load(open(p))
            r[cond] = dict(bm=j["success_rate"], poly=j["success_rate_polysim"],
                           mpkpe=j["e_g_mpbpe_mm"], rmpkpe=j["e_mpbpe_mm"],
                           jl2=j["e_joint_l2_rad"],
                           jvel=j.get("e_jointvel_l2", float("nan")))
    z = np.load(f"{a.dir}/{s}_sim2sim.npz")
    A = z[s] if s in z else list(z.values())[0]
    ok = A[:, C_BM].astype(bool)
    f = lambda c: float(A[ok, c].mean()) if ok.any() else float("nan")
    r["sim2sim"] = dict(bm=float(A[:, C_BM].mean()), poly=float(A[:, C_POLY].mean()),
                        mpkpe=f(C_MPKPE), rmpkpe=f(C_RMPKPE), jl2=f(C_JL2),
                        jvel=f(C_JVEL), n=len(A))
    rows.append(r)

full = [r for r in rows if {"sim", "simdr", "sim2sim"} <= set(r)]
if not full:
    raise SystemExit("세 조건이 다 있는 시퀀스가 없다. 측정이 안 끝났다.")

# --- 표 A ------------------------------------------------------------------
LAB = [("S_bm 성공률", "bm", True, "{:.3f}"), ("S_poly 성공률", "poly", True, "{:.3f}"),
       ("MPKPE (mm)", "mpkpe", False, "{:.1f}"), ("R-MPKPE (mm)", "rmpkpe", False, "{:.1f}"),
       ("E_joint (rad)", "jl2", False, "{:.3f}"), ("E_jvel (rad/s)", "jvel", False, "{:.3f}")]
cellA, keeps = [], []
for name, k, hb, fmt in LAB:
    v = [np.nanmean([r[c][k] for r in full]) for c in ("sim", "simdr", "sim2sim")]
    keep = (v[2] / v[1] if hb else v[1] / v[2]) * 100 if v[1] else float("nan")
    keeps.append(keep)
    cellA.append([name] + [fmt.format(x) for x in v] + [f"{keep:.1f} %"])
colA = ["", "sim\n(교란 없음)", "sim-dr\n(Isaac, 교란)", "sim2sim\n(MuJoCo, 같은 교란)", "유지율"]

# --- 표 B ------------------------------------------------------------------
cellB = []
for r in rows:
    if not {"simdr", "sim2sim"} <= set(r):
        continue
    d, m = r["simdr"], r["sim2sim"]
    cellB.append([r["seq"], f"{d['bm']:.2f}", f"{m['bm']:.2f}",
                  f"{d['poly']:.2f}", f"{m['poly']:.2f}",
                  f"{d['mpkpe']:.1f}", f"{m['mpkpe']:.1f}",
                  f"{d['rmpkpe']:.1f}", f"{m['rmpkpe']:.1f}"])
colB = ["시퀀스", "S_bm\nIsaac", "S_bm\nMuJoCo", "S_poly\nIsaac", "S_poly\nMuJoCo",
        "MPKPE mm\nIsaac", "MPKPE mm\nMuJoCo", "R-MPKPE mm\nIsaac", "R-MPKPE mm\nMuJoCo"]

BG, HD, ALT, LINE = "#ffffff", "#e8eef5", "#f6f8fa", "#c9d2dc"
hA, hB = 0.42 + 0.34 * len(cellA), 0.42 + 0.30 * len(cellB)
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(13.2, hA + hB + 1.9),
    gridspec_kw=dict(height_ratios=[hA, hB], hspace=0.30))
fig.patch.set_facecolor(BG)


def draw(ax, cells, cols, title, sub, widths, hi=None, axh=1.0):
    ax.axis("off")
    ax.set_title(title, fontproperties=KFONT, fontsize=16, loc="left", pad=30)
    ax.text(0, 1.0 + 0.16 / axh, sub, transform=ax.transAxes, fontproperties=KFONT,
            fontsize=9.5, color="#5b6b7c", va="bottom")
    t = ax.table(cellText=cells, colLabels=cols, cellLoc="center",
                 colWidths=widths, loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(LINE); c.set_linewidth(0.6)
        c.set_height(1.0 / (len(cells) + 1.7))
        txt = c.get_text()
        txt.set_fontproperties(KFONT if (i == 0 or j == 0) else MONO)
        if i == 0:
            c.set_facecolor(HD); txt.set_fontsize(9.5); c.set_height(1.5 / (len(cells) + 1.7))
        else:
            c.set_facecolor(ALT if i % 2 else BG)
            if j == 0:
                txt.set_ha("left"); c.PAD = 0.04
        if hi is not None and j == len(cols) - 1 and i > 0:
            c.set_facecolor("#dff0e3" if hi[i - 1] >= 97 else "#fdecea")


nseq = len(full)
sub_a = (f"17 시퀀스 중 {nseq}개 · 시행 sim-dr 100환경 / sim2sim "
         f"{full[0]['sim2sim']['n']}회 · 창 = 모션 전 길이 · "
         "오차는 완주 시행의 살아 있는 프레임 평균")
draw(axA, cellA, colA, "표 A — Isaac Lab → MuJoCo 전이, 17 시퀀스 평균", sub_a,
     [0.24, 0.17, 0.19, 0.24, 0.16], hi=keeps, axh=hA)
draw(axB, cellB, colB, "표 B — 시퀀스별 (sim-dr → sim2sim)",
     "같은 교란 분포 · 같은 채점 정의 · 같은 시점 정렬",
     [0.21] + [0.0975] * 8, axh=hB)

fig.text(0.5, 0.014,
         "성공률 유지율 = MuJoCo / Isaac, 오차 유지율 = Isaac / MuJoCo. 둘 다 100 % 가 손실 없음을 뜻한다.",
         ha="center", fontproperties=KFONT, fontsize=9.5, color="#5b6b7c")
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out, dpi=170, bbox_inches="tight", facecolor=BG)
print(a.out)
