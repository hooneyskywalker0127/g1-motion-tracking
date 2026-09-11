#!/usr/bin/env python3
"""Isaac 영상과 MuJoCo 영상의 시작 시점 차이를 초 단위로 낸다.

Isaac 정책 영상은 레퍼런스 타임라인 위에 있다 (같은 렌더러의 레퍼런스 영상과
실루엣 상관 0.507, 지연 0.00초로 확인).
MuJoCo 롤아웃은 기록이 그보다 늦게 시작됐다. 그 차이를 찾는다.

**시작 구간만 본다.** 두 시뮬은 시간이 갈수록 벌어지므로(루트 오차 수 m),
전체 구간으로 맞추면 최소가 평평해져 오프셋이 흔들린다.
"""
import pathlib
import sys

import numpy as np

REPO = pathlib.Path("/home/sehoon/Documents/GitHub/g1-motion-tracking")
BASE = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")
REF_FPS = 30
WINDOW = 600      # 20초. 드리프트가 쌓이기 전 구간
SEARCH = 420      # 14초까지 찾는다


def _z(x):
    x = x - x.mean(0)
    s = x.std(0)
    s[s < 1e-9] = 1
    return x / s


def offset_seconds(seq):
    csv = REPO / "outputs/csv" / f"{seq}.csv"
    npz = BASE / seq / "sim2sim" / f"{seq}_state.npz"
    if not (csv.exists() and npz.exists()):
        return None
    d = np.load(npz)
    t = d["t"]
    mq = d["q"][:: int(round(1 / REF_FPS / np.median(np.diff(t))))]
    ref = np.loadtxt(csv, delimiter=",")[:, 7:]
    if len(mq) < WINDOW:
        return None

    a = _z(mq[:WINDOW])
    corr = []
    for off in range(SEARCH):
        b = ref[off : off + WINDOW]
        if len(b) < WINDOW:
            break
        corr.append(float((a * _z(b)).mean()))
    if not corr:
        return None
    c = np.array(corr)
    k = int(np.argmax(c))
    second = c.copy()
    second[max(0, k - 12) : k + 12] = -np.inf
    return k / REF_FPS, float(c[k]), float(second.max())


def sharp(r):
    """상관이 뚜렷한가. 무효 롤아웃은 여기서 걸린다."""
    return r is not None and r[1] > 0.45 and r[1] > r[2] * 1.3


if __name__ == "__main__":
    every = sorted(p.name for p in BASE.iterdir() if (p / "sim2sim").is_dir())
    res = {s: offset_seconds(s) for s in every}
    good = [r[0] for r in res.values() if sharp(r)]
    fallback = float(np.median(good)) if good else 0.0

    for s in (sys.argv[1:] or every):
        r = res.get(s) if s in res else offset_seconds(s)
        if sharp(r):
            print(f"{s} {r[0]:.2f}")
        else:
            # 기록이 무효라 상관이 안 잡히는 경우. 기록 시작 지연은 거의 일정하므로
            # 뚜렷하게 잡힌 것들의 중앙값을 쓴다.
            print(f"{s} {fallback:.2f}  # 중앙값 대체")
