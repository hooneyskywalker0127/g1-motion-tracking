"""3단계 그림을 그린다. 선별 기준이 무엇을 걸러내고 무엇을 놓쳤는지 보인다.

    python scripts/make_stage3_figure.py --out docs/stage_3.png

outputs/metrics/quality.csv 의 발 오차로 77개를 세 무리로 나눈다. 기준을
통과한 19개 중 셋은 나중에 완주율 0% 로 드러났다. 그 셋을 따로 칠하는 이유는
발 오차만으로는 다른 통과분과 구분되지 않기 때문이다 — 기준의 맹점이 그림에
남아야 한다.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
MEAN_MAX, P95_MAX, MAX_MAX = 1.2, 3.5, 10.0

# 기준은 통과했지만 학습 뒤 완주율이 0% 였던 셋. 지면 관통과 공중 비율 때문이고
# 선별 기준은 그 둘을 보지 않았다.
UNUSABLE = {"obstacles2_subject1", "walk3_subject1", "walk3_subject4"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(csv_path, out):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()

    rows = list(csv.DictReader(open(csv_path)))
    passed, failed, bad = [], [], []
    for r in rows:
        pt = (float(r["feet_mean"]), float(r["feet_p95"]))
        if (pt[0] < MEAN_MAX and pt[1] < P95_MAX and float(r["feet_max"]) < MAX_MAX):
            (bad if r["seq"] in UNUSABLE else passed).append(pt)
        else:
            failed.append(pt)

    fig, ax = plt.subplots(figsize=(11.3, 6.4), dpi=100)
    for pts, color, edge, marker, size, label in [
        (failed, "#d9d9d9", "#9a9a9a", "o", 90, f"기준 탈락 ({len(failed)}개)"),
        (passed, "#5aa02c", "#3d6f1e", "o", 110, f"기준 통과 ({len(passed)}개)"),
        (bad, "#e8820c", "#a85c05", "X", 150, f"기준 통과, 학습 뒤 완주율 0% ({len(bad)}개)"),
    ]:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=size, c=color,
                   edgecolors=edge, linewidths=1.0, marker=marker, label=label, zorder=3)

    ax.axvline(MEAN_MAX, color="#b03030", linestyle="--", linewidth=1.4, zorder=2)
    ax.axhline(P95_MAX, color="#b03030", linestyle="--", linewidth=1.4, zorder=2)
    ax.text(MEAN_MAX - 0.02, ax.get_ylim()[0], f"평균 {MEAN_MAX} cm", color="#b03030",
            fontsize=12, ha="right", va="bottom")
    ax.text(ax.get_xlim()[1], P95_MAX + 0.08, f"p95 {P95_MAX} cm", color="#b03030",
            fontsize=12, ha="right", va="bottom")

    ax.set_title("발 오차로 77개 중 19개를 고른다. 쓸 수 없는 셋은 이 기준에 걸리지 않는다",
                 fontsize=15, pad=14)
    ax.set_xlabel("발 오차 평균 (cm)", fontsize=13)
    ax.set_ylabel("발 오차 p95 (cm)", fontsize=13)
    ax.grid(True, color="#ededed", linewidth=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=12, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"저장: {out}  탈락 {len(failed)} · 선별 {len(passed) + len(bad)} · 그중 못 쓴 것 {len(bad)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=os.path.join(ROOT, "outputs/metrics/quality.csv"))
    p.add_argument("--out", default=os.path.join(ROOT, "docs/stage_3.png"))
    a = p.parse_args()
    main(a.csv, a.out)
