"""README의 파이프라인 그림을 그린다.

    python scripts/make_pipeline_figure.py --out docs/pipeline.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
GREEN, GREEN_FILL = "#5aa02c", "#e8f2dc"
BLUE, BLUE_FILL = "#1f5fa9", "#dce8f7"

STAGES = [
    dict(no="1단계", name="Human mocap", color=GREEN, fill=GREEN_FILL,
         body=["LAFAN1 (Ubisoft 공개)",
               "bvh 77개 · 30 fps",
               "사람 관절 24개의 3D 위치",
               "",
               "배우 5명 · 걷기 뛰기 춤",
               "격투 장애물 넘어지기"]),
    dict(no="2단계", name="Retargeting", color=GREEN, fill=GREEN_FILL,
         body=["GMR · IK로 역산",
               "사람 손끝·발끝 위치를 목표로",
               "G1 관절값을 찾는다",
               "",
               "pkl 77개 · 496,672 프레임",
               "한 프레임에 숫자 36개",
               "골반 3 + 방향 4 + 관절각 29"]),
    dict(no="3단계", name="Reference motion", color=GREEN, fill=GREEN_FILL,
         body=["IK 목표 추적 오차 측정",
               "발·손·골반이 목표에서 몇 cm",
               "",
               "발 오차로 19개 선별",
               "mean<1.2 p95<3.5 max<10",
               "",
               "npz 변환 · 50 fps · 속도 추가"]),
    dict(no="4단계", name="Motion tracking policy", color=BLUE, fill=BLUE_FILL,
         body=["BeyondMimic · PPO",
               "환경 4096개 · 30000회 반복",
               "모션 1개당 정책 1개 → 19개",
               "",
               "평가는 완주율과 추적 오차",
               "E_g-mpbpe · E_mpbpe (mm)",
               "E_mpjpe (rad) · 기준은 GMR"]),
]

NOTE = ("1~3단계 완료. 4단계는 19개 중 1개 학습을 마쳤고 나머지가 순서를 기다린다. "
        "평가 기준은 GMR 논문(arXiv 2510.02252)을 따른다.")


def main(out):
    fm.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()

    fig, ax = plt.subplots(figsize=(18.0, 8.6), dpi=100)
    ax.set_xlim(0, 1800); ax.set_ylim(0, 860); ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(65, 745, "LAFAN1 → Unitree G1 모션 트래킹 파이프라인",
            fontsize=33, color="#222222")

    x, w, gap = 75, 382, 20
    span13 = 3 * w + 2 * gap
    ax.add_patch(FancyBboxPatch((55, 95), span13 + 60, 585, boxstyle="round,pad=6,rounding_size=8",
                                linewidth=0, facecolor="#f2f2f2"))
    ax.add_patch(FancyBboxPatch((x + 3 * (w + gap) - 22, 95), w + 60, 585,
                                boxstyle="round,pad=6,rounding_size=8",
                                linewidth=0, facecolor="#eaf1fa"))
    ax.text(55 + (span13 + 60) / 2, 120, "물리 없음 · 자세를 계산하고 고르는 구간",
            fontsize=16, color="#666666", ha="center")
    ax.text(x + 3 * (w + gap) - 22 + (w + 60) / 2, 120, "물리 있음 · 로봇이 버티는 구간",
            fontsize=16, color=BLUE, ha="center")

    for i, st in enumerate(STAGES):
        left = x + i * (w + gap)
        ax.add_patch(FancyBboxPatch((left, 150), w, 500, boxstyle="round,pad=4,rounding_size=6",
                                    linewidth=2, edgecolor=st["color"], facecolor="white"))
        ax.add_patch(FancyBboxPatch((left, 520), w, 130, boxstyle="round,pad=4,rounding_size=6",
                                    linewidth=2, edgecolor=st["color"], facecolor=st["fill"]))
        ax.text(left + 24, 606, st["no"], fontsize=16, color=st["color"])
        ax.text(left + 22, 553, st["name"], fontsize=20, color="#222222")
        for j, line in enumerate(st["body"]):
            ax.text(left + 22, 465 - j * 44, line, fontsize=15, color="#333333")
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((left + w + 2, 400), (left + w + gap - 2, 400),
                                         arrowstyle="-|>", mutation_scale=20,
                                         linewidth=3, color="#9a9a9a"))

    ax.text(65, 55, NOTE, fontsize=15.5, color="#666666")

    fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print(f"저장: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/pipeline.png")
    main(p.parse_args().out)
