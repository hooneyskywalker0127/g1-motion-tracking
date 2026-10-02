"""IGRIS-C 클립별 평가(처음부터 학습)를 G1 교사와 나란히 표로 만들어 docs/igris(.en).md 에 넣는다.

표는 두 문서의 <!-- igris-table --> ... <!-- /igris-table --> 사이를 통째로 바꾼다.
평가는 프레임 0부터 100 롤아웃, 도메인 랜덤화 끔. 끝난 클립만 들어간다.

    python scripts/igris/results_table.py
"""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WBT = "/home/sehoon/Projects/whole_body_tracking/logs"
# run2_subject4 는 igris_transfer.sh(A)에서, 나머지는 igris_all.sh 에서 평가했다.
IGRIS = {"run2_subject4": f"{WBT}/igris_transfer/eval_a_29999.json"}
IGRIS.update({os.path.basename(p)[5:-5]: p for p in glob.glob(f"{WBT}/igris_all/eval_*.json")})

rows = []
for seq in sorted(IGRIS):
    i = json.load(open(IGRIS[seq]))
    g = json.load(open(f"{REPO}/outputs/eval/{seq}.json"))
    rows.append(f"| {seq} | {g['success_rate']:.0%} | {i['success_rate']:.0%} | "
                f"{g['e_mpbpe_mm']:.0f} | {i['e_mpbpe_mm']:.0f} | "
                f"{g['e_mpjpe_rad']:.3f} | {i['e_mpjpe_rad']:.3f} |")

HEAD = {"ko": "| 클립 | G1 완주 | IGRIS 완주 | G1 E_mpbpe (mm) | IGRIS E_mpbpe (mm) | G1 E_mpjpe (rad) | IGRIS E_mpjpe (rad) |",
        "en": "| Clip | G1 completion | IGRIS completion | G1 E_mpbpe (mm) | IGRIS E_mpbpe (mm) | G1 E_mpjpe (rad) | IGRIS E_mpjpe (rad) |"}
for lang, path in (("ko", "docs/igris.md"), ("en", "docs/igris.en.md")):
    table = "\n".join([HEAD[lang], "|---|---|---|---|---|---|---|", *rows])
    p = os.path.join(REPO, path)
    s = open(p).read()
    s, n = re.subn(r"<!-- igris-table -->.*?<!-- /igris-table -->",
                   f"<!-- igris-table -->\n{table}\n<!-- /igris-table -->", s, flags=re.S)
    assert n == 1, path
    open(p, "w").write(s)
print(f"{len(rows)} clips")
