#!/usr/bin/env python3
"""Isaac 평가 json 과 MuJoCo sim2sim json 을 나란히 놓아 sim-to-sim 표를 만든다.

두 시뮬 모두 같은 레퍼런스 모션을 기준으로 오차를 재므로,
같은 지표의 차이가 곧 시뮬 간 전이 손실이다.
"""
import json, math, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim2sim_validity import scan as validity_scan

REPO = pathlib.Path("/home/sehoon/Documents/GitHub/g1-motion-tracking")
MUJ = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")

VALID = {n: v for n, v, *_ in validity_scan()}

rows = []
for ij in sorted((REPO / "outputs/eval").glob("*.json")):
    seq = ij.stem
    mj = MUJ / seq / "sim2sim" / f"{seq}_mujoco.json"
    if not mj.exists():
        continue
    a, b = json.loads(ij.read_text()), json.loads(mj.read_text())
    # Isaac 은 성공 에피소드에서만 오차를 재므로 success_rate 0 이면 NaN 이 된다.
    # 그 경우는 전체 에피소드 기준 e_g_mpbpe_mm_all 만 있고 나머지는 값이 없다.
    nan = any(isinstance(a[k], float) and math.isnan(a[k])
              for k in ("e_g_mpbpe_mm", "e_mpbpe_mm", "e_mpjpe_rad"))
    rows.append({
        "valid": VALID.get(seq, "?") in ("정상 완주", "넘어짐(정상)"),
        "verdict": VALID.get(seq, "?"),
        "nan": nan, "i_gp_all": a.get("e_g_mpbpe_mm_all", float("nan")),
        "seq": seq,
        "i_gp": a["e_g_mpbpe_mm"], "m_gp": b["e_g_mpbpe_mm"],
        "i_lp": a["e_mpbpe_mm"],   "m_lp": b["e_mpbpe_mm"],
        "i_jr": a["e_mpjpe_rad"],  "m_jr": b["e_mpjpe_rad"],
        "i_sr": a["success_rate"], "m_al": b["alive_ratio"],
        "fell": b["fell"],
    })

if not rows:
    sys.exit("짝이 맞는 json 이 없다")

hdr = ("| 시퀀스 | Eg-mpbpe(mm) Isaac→MuJoCo | E-mpbpe(mm) Isaac→MuJoCo | "
       "MPJPE(rad) Isaac→MuJoCo | 배수(E-mpbpe) | 생존 |")
sep = "| " + " | ".join(["---"] * 6) + " |"
out = [hdr, sep]
for r in rows:
    if not r["valid"]:
        out.append(f"| ~~{r['seq']}~~ | — | — | — | — | **무효 ({r['verdict']})** |")
        continue
    live = f"{'버팀' if not r['fell'] else '넘어짐'} ({r['m_al']*100:.0f}%)"
    if r["nan"]:
        out.append(
            f"| {r['seq']} | (성공0) {r['i_gp_all']:.0f} → {r['m_gp']:.0f} | "
            f"— → {r['m_lp']:.0f} | — → {r['m_jr']:.3f} | — | {live} |")
        continue
    out.append(
        f"| {r['seq']} | {r['i_gp']:.0f} → {r['m_gp']:.0f} | "
        f"{r['i_lp']:.0f} → {r['m_lp']:.0f} | "
        f"{r['i_jr']:.3f} → {r['m_jr']:.3f} | "
        f"{r['m_lp']/r['i_lp']:.1f}× | {live} |")

ok = [r for r in rows if not r["nan"] and r["valid"]]
n = len(ok)
avg = lambda k: sum(r[k] for r in ok) / n
inv = [r for r in rows if not r["valid"]]
out += ["",
        f"**물리적으로 무효한 롤아웃 {len(inv)}개는 제외한다** "
        "(시작 자세가 땅속이거나 골반이 수십 m 로 튄 경우).",
        "판정은 `src/sim2sim_validity.py`, 골반 높이만 본다.",
        "",
        f"표본 {n}개 · 평균",
        f"- Eg-mpbpe  {avg('i_gp'):.0f} → {avg('m_gp'):.0f} mm  ({avg('m_gp')/avg('i_gp'):.1f}×)",
        f"- E-mpbpe   {avg('i_lp'):.0f} → {avg('m_lp'):.0f} mm  ({avg('m_lp')/avg('i_lp'):.1f}×)",
        f"- MPJPE     {avg('i_jr'):.3f} → {avg('m_jr'):.3f} rad ({avg('m_jr')/avg('i_jr'):.1f}×)",
        f"- MuJoCo 에서 넘어짐  {sum(r['fell'] for r in ok)}/{n} (유효분만)"]
print("\n".join(out))
