#!/usr/bin/env python3
"""MuJoCo sim2sim 롤아웃이 물리적으로 유효한지 골반 높이만으로 판정한다.

시뮬을 다시 돌리지 않는다. 기록된 state.npz 만 읽는다.
G1 의 선 자세 골반 높이는 약 0.79 m 다.

    초기상태 이상  시작 골반 z 가 0.6~0.95 밖        → 롤아웃 시작부터 틀렸다
    수치 발산      골반 z 가 1.5 위 또는 -1.0 아래   → 사람이 넘어지는 운동이 아니다
    넘어짐(정상)   골반 z 가 0.45 아래로 내려가되 범위 안
    정상 완주      위 어디에도 안 걸림
"""
import json
import pathlib
import sys

import numpy as np

BASE = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")
Z_STAND = (0.60, 0.95)
Z_BLOWUP = (-1.0, 1.5)
Z_FALL = 0.45


def verdict(pz):
    z0, zmin, zmax = float(pz[0]), float(pz.min()), float(pz.max())
    if not (Z_STAND[0] < z0 < Z_STAND[1]):
        v = "초기상태 이상"
    elif zmax > Z_BLOWUP[1] or zmin < Z_BLOWUP[0]:
        v = "수치 발산"
    elif zmin < Z_FALL:
        v = "넘어짐(정상)"
    else:
        v = "정상 완주"
    return v, z0, zmin, zmax


def scan():
    out = []
    for d in sorted(BASE.iterdir()):
        f = d / "sim2sim" / f"{d.name}_state.npz"
        if not f.exists():
            continue
        pz = np.load(f)["body"][:, 0, 2]
        v, z0, zmin, zmax = verdict(pz)
        out.append((d.name, v, z0, zmin, zmax))
    return out


if __name__ == "__main__":
    rows = scan()
    print(f"{'시퀀스':<22}{'시작 pz':>9}{'pz 최소':>10}{'pz 최대':>10}  판정")
    for n, v, z0, zmin, zmax in rows:
        print(f"{n:<22}{z0:>9.3f}{zmin:>10.3f}{zmax:>10.3f}  {v}")
    bad = [r for r in rows if r[1] in ("초기상태 이상", "수치 발산")]
    print(f"\n유효 {len(rows)-len(bad)} / {len(rows)}  ·  무효 {len(bad)}")
    for n, v, *_ in bad:
        print(f"  무효  {n}  ({v})")
    if "--json" in sys.argv:
        (BASE / "sim2sim_유효성.json").write_text(json.dumps(
            {n: v for n, v, *_ in rows}, ensure_ascii=False, indent=2))
        print("\n저장:", BASE / "sim2sim_유효성.json")
