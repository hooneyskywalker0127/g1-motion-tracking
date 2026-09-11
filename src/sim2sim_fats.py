#!/usr/bin/env python3
"""추종 오차와 생존을 ThorArena 의 FATS 로 묶어 Isaac 과 MuJoCo 를 나란히 놓는다.

    S_i = 100 exp(-E_i / sigma) * s_i ,  s_i = min(T_i / T_ref, 1) ,  sigma = 0.15 m
    (ThorArena, arXiv:2607.06052, 식 3. 외력 재생 없는 no-force 설정)

E 는 루트 상대 바디 위치 오차(E-mpbpe, m)를 쓴다. Isaac 쪽은 eval.py 가 내는
`e_mpbpe_mm`, MuJoCo 쪽은 `src/sim2sim_mpbpe.py` 와 같은 방식으로 계산한 값이다.
생존은 Isaac 은 `mean_alive_frames / motion_frames`, MuJoCo 는
`src/sim2sim_isaac_criteria.py` 가 같은 종료 조건으로 낸 비율이다.

    python3 src/sim2sim_fats.py [--stride N]
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIGMA = 0.15
PY_ = sys.executable


def fats(err_m, alive_ratio):
    return 100.0 * np.exp(-err_m / SIGMA) * min(alive_ratio, 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    import sim2sim_isaac_criteria as crit
    import sim2sim_mpbpe  # noqa: F401  (같은 계산을 재사용하려 import 만 해 둔다)

    rows = []
    for p in sorted((ROOT / "outputs" / "policy_io").glob("*.npz")):
        seq = p.stem
        ev = ROOT / "outputs" / "eval" / f"{seq}.json"
        if not ev.exists():
            continue
        isaac = json.loads(ev.read_text())
        try:
            r = crit.evaluate(seq, a.stride)
        except Exception as exc:
            print(f"{seq:<22} 건너뜀 ({type(exc).__name__})")
            continue

        out = subprocess.run(
            [PY_, str(ROOT / "src" / "sim2sim_mpbpe.py"), seq, "--stride", "20"],
            capture_output=True, text=True)
        try:
            mj_mm = float(out.stdout.split("E-mpbpe")[1].split("mm")[0])
        except Exception:
            print(f"{seq:<22} E-mpbpe 계산 실패")
            continue

        iso_mm = isaac.get("e_mpbpe_mm")
        iso_alive = isaac["mean_alive_frames"] / isaac["motion_frames"]
        rows.append((seq, mj_mm, r["alive_ratio"], iso_mm, iso_alive))

    print(f"{'시퀀스':<22}{'MuJoCo E':>10}{'생존':>8}{'FATS':>8}"
          f"{'Isaac E':>10}{'생존':>8}{'FATS':>8}{'격차':>8}")
    for seq, mj_mm, mj_a, iso_mm, iso_a in rows:
        f_mj = fats(mj_mm / 1000.0, mj_a)
        if iso_mm is None or not np.isfinite(iso_mm):
            print(f"{seq:<22}{mj_mm:>10.1f}{mj_a*100:>7.1f}%{f_mj:>8.1f}"
                  f"{'-':>10}{iso_a*100:>7.1f}%{'-':>8}{'-':>8}")
            continue
        f_is = fats(iso_mm / 1000.0, iso_a)
        print(f"{seq:<22}{mj_mm:>10.1f}{mj_a*100:>7.1f}%{f_mj:>8.1f}"
              f"{iso_mm:>10.1f}{iso_a*100:>7.1f}%{f_is:>8.1f}{f_mj-f_is:>+8.1f}")
