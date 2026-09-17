"""README의 파이프라인 그림을 그린다.

    python scripts/make_pipeline_figure.py --out docs/pipeline.png

칸마다 docs/stage_N.png 를 썸네일로 넣는다. 단계가 늘면 STAGES 에 한 줄을
추가하고 docs/stage_N.png 를 두면 된다.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
GREEN, GREEN_FILL = "#5aa02c", "#e8f2dc"
BLUE, BLUE_FILL = "#1f5fa9", "#dce8f7"

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

# physics=False 는 자세만 계산하는 구간, True 는 로봇이 스스로 버티는 구간이다.
STAGES = [
    dict(no="1단계", name="Human mocap", physics=False, thumb="stage_1.png",
         body=["LAFAN1 (Ubisoft 공개)",
               "bvh 77개 · 30 fps",
               "배우 5명 · 4.6시간"]),
    dict(no="2단계", name="Retargeting", physics=False, thumb="stage_2.png",
         body=["GMR · IK로 G1 관절값 역산",
               "손끝·발끝 위치를 목표로",
               "77개 · 496,672 프레임"]),
    dict(no="3단계", name="Reference motion", physics=False, thumb="stage_3.png",
         body=["발 오차로 19개 선별",
               "평지 불가 3개 제외 → 16개",
               "npz 변환 · 50 fps"]),
    dict(no="4단계", name="Motion tracking policy", physics=True, thumb="stage_4.png",
         body=["BeyondMimic · PPO",
               "모션 1개당 정책 1개",
               "16개 학습"]),
    dict(no="5단계", name="Policy distillation", physics=True, thumb="stage_5.png",
         body=["HOVER · DAgger",
               "교사 14개 → 학생 1개",
               "정책 하나로 통합"]),
]

NOTE = ("4단계까지는 모션 1개당 정책 1개다. 5단계가 그것들을 정책 하나로 합친다. "
        "평가 기준은 GMR 논문(arXiv 2510.02252)을 따른다.")

W, GAP, X0 = 382, 20, 75
BOX_BOTTOM, BOX_H = 150, 500
HEAD_BOTTOM, HEAD_H = 520, 130


def _draw_thumb(ax, path, left, width):
    """칸 안쪽 썸네일. 비율을 지키면서 가로를 칸에 맞춘다."""
    if not os.path.isfile(path):
        return
    img = mpimg.imread(path)
    iw, ih = img.shape[1], img.shape[0]
    tw = width - 44
    th = tw * ih / iw
    top = 500  # 머리글 아래
    max_h = 190
    if th > max_h:  # 너무 세로로 길면 높이에 맞춘다
        th = max_h
        tw = th * iw / ih
    cx = left + width / 2
    ax.imshow(img, extent=(cx - tw / 2, cx + tw / 2, top - th, top), aspect="auto", zorder=3)


def main(out):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()

    n = len(STAGES)
    total_w = X0 + n * W + (n - 1) * GAP + 75
    fig, ax = plt.subplots(figsize=(total_w / 100.0, 8.6), dpi=100)
    ax.set_xlim(0, total_w); ax.set_ylim(0, 860); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(65, 745, "LAFAN1 → Unitree G1 모션 트래킹 파이프라인",
            fontsize=33, color="#222222")

    n_nophys = sum(1 for s in STAGES if not s["physics"])
    span_nophys = n_nophys * W + (n_nophys - 1) * GAP
    n_phys = n - n_nophys
    span_phys = n_phys * W + (n_phys - 1) * GAP

    ax.add_patch(FancyBboxPatch((55, 95), span_nophys + 60, 585,
                                boxstyle="round,pad=6,rounding_size=8",
                                linewidth=0, facecolor="#f2f2f2"))
    phys_left = X0 + n_nophys * (W + GAP) - 22
    ax.add_patch(FancyBboxPatch((phys_left, 95), span_phys + 60, 585,
                                boxstyle="round,pad=6,rounding_size=8",
                                linewidth=0, facecolor="#eaf1fa"))
    ax.text(55 + (span_nophys + 60) / 2, 120, "물리 없음 · 자세를 계산하고 고르는 구간",
            fontsize=16, color="#666666", ha="center")
    ax.text(phys_left + (span_phys + 60) / 2, 120, "물리 있음 · 로봇이 버티는 구간",
            fontsize=16, color=BLUE, ha="center")

    for i, st in enumerate(STAGES):
        color = BLUE if st["physics"] else GREEN
        fill = BLUE_FILL if st["physics"] else GREEN_FILL
        left = X0 + i * (W + GAP)
        ax.add_patch(FancyBboxPatch((left, BOX_BOTTOM), W, BOX_H,
                                    boxstyle="round,pad=4,rounding_size=6",
                                    linewidth=2, edgecolor=color, facecolor="white"))
        ax.add_patch(FancyBboxPatch((left, HEAD_BOTTOM), W, HEAD_H,
                                    boxstyle="round,pad=4,rounding_size=6",
                                    linewidth=2, edgecolor=color, facecolor=fill))
        ax.text(left + 24, 606, st["no"], fontsize=16, color=color)
        # 이름이 길면 칸 밖으로 넘친다. 글자 수에 따라 줄인다
        name_size = 20 if len(st["name"]) <= 18 else 17
        ax.text(left + 22, 553, st["name"], fontsize=name_size, color="#222222")

        _draw_thumb(ax, os.path.join(DOCS, st["thumb"]), left, W)

        for j, line in enumerate(st["body"]):
            ax.text(left + 22, 260 - j * 44, line, fontsize=15, color="#333333")

        if i < n - 1:
            ax.add_patch(FancyArrowPatch((left + W + 2, 400), (left + W + GAP - 2, 400),
                                         arrowstyle="-|>", mutation_scale=20,
                                         linewidth=3, color="#9a9a9a", zorder=4))

    ax.text(65, 55, NOTE, fontsize=15.5, color="#666666")

    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print(f"저장: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/pipeline.png")
    main(p.parse_args().out)
