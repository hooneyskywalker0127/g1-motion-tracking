#!/usr/bin/env python3
"""Isaac 과 MuJoCo 에서 정책 순위가 보존되는지 본다.

SIMPLER (arXiv:2405.05941) 가 시뮬-실기 상관을 절대 성공률이 아니라 순위 위반으로
재는 데서 가져왔다. 절대값을 맞추는 것보다 "정책 A 가 B 보다 낫다" 가 유지되는지가
평가 도구로서는 더 중요하다.

Isaac 에서 수렴하지 않은 정책(success_rate 0.0, 지표가 nan)은 뺀다.

    python3 src/sim2sim_rank.py
"""
import glob
import json
import os
import pathlib

import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    isaac = {}
    for f in sorted(glob.glob(str(ROOT / "outputs" / "eval" / "*.json"))):
        j = json.load(open(f))
        v = j.get("e_mpjpe_rad")
        if v is not None and np.isfinite(v):
            isaac[os.path.basename(f)[:-5]] = v

    mj_path = ROOT / "outputs" / "metrics" / "sim2sim.json"
    if not mj_path.exists():
        raise SystemExit("src/sim2sim_metrics.py --out outputs/metrics/sim2sim.json 먼저 실행할 것")
    mj = {r["motion"]: r["e_mpjpe_rad"] for r in json.loads(mj_path.read_text())
          if r["e_mpjpe_rad"] == r["e_mpjpe_rad"]}

    names = [n for n in isaac if n in mj]
    a = np.array([isaac[n] for n in names])
    b = np.array([mj[n] for n in names])
    ra, rb = np.argsort(np.argsort(a)) + 1, np.argsort(np.argsort(b)) + 1

    print(f"공통 시퀀스 {len(names)}개 (Isaac 미수렴 제외)")
    print(f"Spearman {spearmanr(a, b).statistic:+.3f}  "
          f"Kendall {kendalltau(a, b).statistic:+.3f}  "
          f"Pearson {np.corrcoef(a, b)[0, 1]:+.3f}")
    print(f"\n{'시퀀스':<22}{'Isaac':>8}{'순위':>5}{'MuJoCo':>9}{'순위':>5}{'차':>5}")
    for i in np.argsort(a):
        d = int(rb[i] - ra[i])
        print(f"{names[i]:<22}{a[i]:>8.3f}{ra[i]:>5}{b[i]:>9.3f}{rb[i]:>5}"
              f"{d:>+5}" if d else f"{names[i]:<22}{a[i]:>8.3f}{ra[i]:>5}"
              f"{b[i]:>9.3f}{rb[i]:>5}{'':>5}")


if __name__ == "__main__":
    main()
