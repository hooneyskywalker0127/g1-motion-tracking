#!/usr/bin/env python3
"""사이드바이사이드 자막 두 줄을 만든다. 공백은 ~ 로 바꿔 한 줄에 담아 넘긴다.

왼쪽  Isaac 지표
오른쪽 MuJoCo 지표 + Isaac 대비 배수
"""
import json, math, pathlib, sys

REPO = pathlib.Path("/home/sehoon/Documents/GitHub/g1-motion-tracking")
ROOT = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")

seq = sys.argv[1]
a = json.loads((REPO / "outputs/eval" / f"{seq}.json").read_text())
b = json.loads((ROOT / seq / "sim2sim" / f"{seq}_mujoco.json").read_text())

nan = lambda v: isinstance(v, float) and math.isnan(v)
ratio = lambda m, i: "" if nan(i) or i == 0 else f" ({m/i:.1f}x)"

if nan(a["e_mpbpe_mm"]):
    left = f"E_mpbpe -  ·  MPJPE -  ·  success {a['success_rate']*100:.0f}%"
else:
    left = (f"E_mpbpe {a['e_mpbpe_mm']:.0f} mm  ·  "
            f"MPJPE {a['e_mpjpe_rad']:.3f} rad  ·  "
            f"success {a['success_rate']*100:.0f}%")

right = (f"E_mpbpe {b['e_mpbpe_mm']:.0f} mm{ratio(b['e_mpbpe_mm'], a['e_mpbpe_mm'])}  ·  "
         f"MPJPE {b['e_mpjpe_rad']:.3f} rad{ratio(b['e_mpjpe_rad'], a['e_mpjpe_rad'])}  ·  "
         f"{'생존' if not b['fell'] else '넘어짐'} {b['alive_ratio']*100:.0f}%")

print(left.replace(" ", "~"), right.replace(" ", "~"))
