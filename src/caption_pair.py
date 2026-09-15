"""비교 영상 자막에 쓸 네 줄을 만든다.

영상의 목적이 sim-to-sim 비교이므로 양쪽이 같은 양이어야 한다.
앞선 자막은 그렇지 않았다.
  같은 양을 Isaac 은 E_g-mpbpe, MuJoCo 는 E_g-mpjpe 로 불렀다 (다른 계보 이름)
  E_mpjpe 에 mm 를 붙였다. 그 이름을 쓰는 계보에서는 관절각 1e-3 rad 다
  MuJoCo 상대 오차가 골반만 뺀 값이라 Isaac 의 재정렬 값과 다른 양이었다
  Isaac 은 자체 종료조건, MuJoCo 는 0.5 m 기준인데 나란히 놓았다

이제 네 줄 다 양쪽 같은 정의로 내고 판정 기준을 이름으로 밝힌다.
Isaac 은 eval_polysim json, MuJoCo 는 sim2sim_full2 롤아웃에서 읽는다.
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from score_standard import score

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seq = sys.argv[1]

d = json.load(open(f"{R}/outputs/eval_polysim/{seq}.json"))
r = score(np.load(f"{R}/outputs/sim2sim_full2/{seq}.npz"))
# Isaac 의 관절 오차는 관절축 평균 절대값이다 (eval_dump.py). 같게 맞춘다.
q = np.load(f"{R}/outputs/sim2sim_full2/{seq}.npz")
jabs = np.abs(q["ref_joint_pos"] - q["rob_joint_pos"]).mean()

pf = lambda ok: "pass" if ok else "fail"
# 헤더를 값 열에 맞춘다. 눈대중으로 띄웠더니 둘 다 왼쪽으로 치우쳐 있었다.
rows = [
    f"global body error        {d['e_g_mpbpe_mm']:>7.0f} mm{r['mpkpe']:>12.0f} mm",
    f"local pose, re-anchored  {d['e_mpbpe_mm']:>7.0f} mm{r['r_mpkpe']:>12.0f} mm",
    f"joint angle              {d['e_mpjpe_rad']:>7.3f} rad{jabs:>11.3f} rad",
    f"survives termination     {pf(d['success_rate'] > 0.5):>10}"
    f"{pf(r['bm_success']):>14}",
    f"stays within 0.5 m       {pf(d['success_rate_polysim'] > 0.5):>10}"
    f"{pf(r['poly_success']):>14}",
]
a, b = rows[0].index(" mm"), rows[0].rindex(" mm")
hdr = (" " * (a + 3 - len("Isaac Lab")) + "Isaac Lab").ljust(b + 3 - len("MuJoCo")) + "MuJoCo"
print("|".join([hdr] + rows))
