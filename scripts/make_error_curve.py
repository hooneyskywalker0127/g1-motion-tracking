"""학습 로그(tfevents)에서 추종 오차 곡선 그림을 만든다.

wandb 화면을 캡처하지 않아도 같은 그림이 나온다. 값은 학습 중 기록된
Metrics/motion/error_body_pos 이고, 레퍼런스 대비 추종 대상 링크 위치
오차의 평균이다. 미터로 기록되므로 cm로 바꿔 그린다.

쓰기: make_error_curve.py <시퀀스> <출력 png>
"""

import sys
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGROOT = "/home/sehoon/Projects/whole_body_tracking/logs/rsl_rl/g1_flat"
TAG = "Metrics/motion/error_body_pos"
KFONT = FontProperties(family="Noto Sans CJK JP")


def load(seq):
    """같은 시퀀스의 학습 폴더를 모두 읽어 step 순으로 이어붙인다.

    중간에 끊겨 재개한 학습은 폴더가 둘로 나뉘고 뒤 폴더가 앞 구간을 이어받는다.
    그대로 두면 그래프가 두 동강 나므로 여기서 합친다.
    """
    steps, vals = [], []
    resumes = []  # 재개해서 이어 붙은 구간의 시작 step
    for d in sorted(glob.glob(os.path.join(LOGROOT, f"*_{seq}*"))):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if TAG not in ea.Tags()["scalars"]:
            continue
        first = True
        for s in ea.Scalars(TAG):
            if first:
                # 첫 폴더가 아닐 때만 재개로 본다. 플래그는 어느 경우든 내려야 한다.
                if steps:
                    resumes.append(s.step)
                first = False
            steps.append(s.step)
            vals.append(s.value)
    if not steps:
        raise SystemExit(f"{seq}: {TAG} 없음")
    steps = np.asarray(steps)
    vals = np.asarray(vals) * 100.0  # m -> cm
    order = np.argsort(steps, kind="stable")
    steps, vals = steps[order], vals[order]
    # 재개 지점은 앞뒤 폴더가 같은 step을 하나씩 갖는다. 뒤엣것을 남긴다.
    keep = np.concatenate([steps[1:] != steps[:-1], [True]])
    return steps[keep], vals[keep], sorted(set(resumes))


def smooth(v, w=101):
    if len(v) < w:
        return v
    k = np.ones(w) / w
    pad = np.r_[np.full(w // 2, v[0]), v, np.full(w // 2, v[-1])]
    return np.convolve(pad, k, mode="valid")[: len(v)]


def main():
    seq, out = sys.argv[1], sys.argv[2]
    steps, vals, resumes = load(seq)
    sm = smooth(vals)

    fig, ax = plt.subplots(figsize=(12.2, 7.0), dpi=100)
    # 회색과 파란색이 무엇인지 그림 안에서 알 수 있어야 한다.
    ax.plot(steps, vals, color="0.75", lw=0.8, zorder=1, label="기록값 (환경 4,096개 평균)")
    ax.plot(steps, sm, color="#1f5fbf", lw=2.4, zorder=2, label="이동평균 (101회)")

    # 재개 지점은 세로선으로 표시한다. 학습이 끊겼다 이어진 자리라 값이 한 번 튄다.
    # 표시가 없으면 그림만 보고 원인을 알 수 없다.
    for r in resumes:
        ax.axvline(r, color="#c0392b", lw=1.0, ls="--", alpha=0.55, zorder=0)
        ax.annotate(
            f"{r:,}회 재개",
            xy=(r, 0), xytext=(4, 6), textcoords="offset points",
            fontproperties=KFONT, fontsize=12, color="#c0392b", alpha=0.9,
        )

    # 짚어 줄 세 지점: 초기 최대, 2,000회, 최종. 수치는 평활선이 아니라 기록된 값이다.
    # 최댓값은 초반(2,000회 이내)에서만 찾는다. 전 구간에서 찾으면 재개 때 튄 값이
    # 잡혀 "초기 최대"라는 이름이 한가운데 붙는다.
    early = steps <= 2000
    i_max = int(np.argmax(np.where(early, vals, -np.inf)))
    i_2k = int(np.argmin(np.abs(steps - 2000)))
    marks = [
        (i_max, f"초기 최대 {vals[i_max]:.1f} cm", (70, -25)),
        (i_2k, f"{steps[i_2k]:,}회 {vals[i_2k]:.1f} cm", (60, 75)),
        (len(steps) - 1, f"최종 {vals[-1]:.1f} cm", (-120, 85)),
    ]
    for i, text, off in marks:
        ax.plot(steps[i], vals[i], "o", color="#c0392b", ms=9, zorder=3)
        ax.annotate(
            text,
            xy=(steps[i], vals[i]),
            xytext=(off[0], off[1]),
            textcoords="offset points",
            fontproperties=KFONT,
            fontsize=15,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=1.2),
        )

    ax.set_title(f"추종 오차의 변화 · {seq}", fontproperties=KFONT, fontsize=19, pad=16)
    ax.set_xlabel("학습 반복", fontproperties=KFONT, fontsize=15, labelpad=10)
    ax.set_ylabel("링크 위치 오차 (cm)", fontproperties=KFONT, fontsize=15, labelpad=10)
    ax.set_xlim(0, steps[-1] + 1)
    # 재개 스파이크에 맞추면 정작 봐야 할 초반 하강이 눌린다. 초반 최댓값 기준으로 잡되,
    # 스파이크가 잘려 보이지 않도록 회색 원자료는 그대로 둔다.
    ax.set_ylim(0, vals[i_max] * 1.25)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=13)
    for t in leg.get_texts():
        t.set_fontproperties(KFONT)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=13)

    fig.text(
        0.985,
        0.012,
        "레퍼런스 대비 추종 대상 링크 위치 오차의 평균. 환경 4096개 평균값.",
        ha="right",
        fontproperties=KFONT,
        fontsize=12,
        color="0.55",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out, facecolor="white")
    print(f"[INFO]: {out} · {len(steps)}점 · 최대 {sm[i_max]:.1f} cm · 최종 {sm[-1]:.1f} cm")


if __name__ == "__main__":
    main()
