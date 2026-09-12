#!/usr/bin/env python3
"""표준 구조 sim2sim 결과를 mjlab 지표 정의로 표에 낸다.

지표는 mjlab `tasks/tracking/mdp/metrics.py` 를 따른다.
  MPKPE    전역 좌표 키바디 위치 오차. 전역 이동·방향 드리프트를 모두 포함
  R-MPKPE  루트 상대. body_pos_relative_w 를 쓴다 — 전역 이동과 yaw 드리프트를 빼고
           국소 자세 오차만 남긴다
여기서는 로그에 body_pos_w 만 있으므로 R-MPKPE 는 골반 기준으로 재구성한다.

Isaac 쪽은 outputs/eval_fixed (프레임 짝짓기를 고친 뒤 다시 낸 값)를 쓴다.

    python3 src/sim2sim_std_table.py
"""
import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def metrics(path):
    d = np.load(path)
    ref, rob = d["ref_body_pos_w"], d["rob_body_pos_w"]
    mpkpe = np.linalg.norm(ref - rob, axis=-1).mean()
    # 루트 상대: 각자의 골반(index 0)을 뺀다
    r_mpkpe = np.linalg.norm((ref - ref[:, :1]) - (rob - rob[:, :1]), axis=-1).mean()
    mpjpe = np.abs(d["ref_joint_pos"] - d["rob_joint_pos"]).mean()
    return mpkpe * 1000, r_mpkpe * 1000, mpjpe


if __name__ == "__main__":
    print(f"{'시퀀스':<22}{'MPKPE':>9}{'R-MPKPE':>10}{'MPJPE':>9}"
          f"{'Isaac 전역':>11}{'Isaac 상대':>11}{'Isaac MPJPE':>12}")
    for f in sorted(glob.glob(str(ROOT / "outputs" / "sim2sim_std" / "*.npz"))):
        seq = pathlib.Path(f).stem
        a, b, c = metrics(f)
        ev = ROOT / "outputs" / "eval_fixed" / f"{seq}.json"
        if ev.exists():
            j = json.loads(ev.read_text())
            g = j.get("e_g_mpbpe_mm_all")
            r = j.get("e_mpbpe_mm")
            m = j.get("e_mpjpe_rad")
            fmt = lambda v, w, p=1: (f"{v:>{w}.{p}f}" if v is not None and np.isfinite(v)
                                     else f"{'-':>{w}}")
            print(f"{seq:<22}{a:>9.1f}{b:>10.1f}{c:>9.3f}"
                  f"{fmt(g,11)}{fmt(r,11)}{fmt(m,12,3)}")
        else:
            print(f"{seq:<22}{a:>9.1f}{b:>10.1f}{c:>9.3f}{'-':>11}{'-':>11}{'-':>12}")
