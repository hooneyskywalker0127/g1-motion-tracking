"""전장 롤아웃 기준으로 Isaac 과 MuJoCo 를 한 표에 놓는다.

기존 sim2sim_std_table.py 는 outputs/sim2sim_std 의 옛 롤아웃을 읽는다.
지금 기준이 되는 건 모션 전체 길이로 돌린 outputs/sim2sim_full 이다.
판정은 PolySim 기준 — 전역 바디 위치 오차 평균이 한 번이라도 0.5 m 를 넘으면 실패.
"""
import glob, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from score_standard import score as _bm

FAIL_M, FPS = 0.5, 50
MJ_TRIALS = "outputs/metrics/horizon_trials.npz"
tr = dict(np.load(MJ_TRIALS)) if os.path.exists(MJ_TRIALS) else {}

rows = []
for f in sorted(glob.glob("outputs/sim2sim_full2/*.npz")):
    seq = os.path.basename(f)[:-4]
    d = np.load(f)
    e = np.linalg.norm(d["ref_body_pos_w"] - d["rob_body_pos_w"], axis=-1).mean(axis=1)
    # 골반만 빼면 Isaac 쪽 e_mpbpe 와 다른 양이 된다. Isaac 은 torso 앵커에
    # yaw 까지 재정렬한 body_pos_relative_w 를 쓴다 (commands.py 284-294).
    # 정의를 맞춘다.
    rel = _bm(d)["r_mpkpe"] / 1000.0
    jnt = np.abs(d["ref_joint_pos"] - d["rob_joint_pos"]).mean()
    over = e > FAIL_M
    fail_s = float(np.argmax(over)) / FPS if over.any() else None
    j = f"outputs/eval_polysim/{seq}.json"
    iso = json.load(open(j)) if os.path.exists(j) else {}
    n10 = (f"{(~(tr[seq].astype(np.float32) > FAIL_M).any(axis=1)).mean():.1f}"
           if seq in tr else "—")
    rows.append(dict(
        seq=seq, sec=len(e) / FPS,
        i_jnt=iso.get("e_mpjpe_rad"), m_jnt=float(jnt),
        i_rel=iso.get("e_mpbpe_mm"), m_rel=float(rel * 1000),
        i_ok=iso.get("success_rate_polysim"),
        m_ok=0.0 if over.any() else 1.0, m_ok10=n10,
        fail=fail_s, over=float(over.mean())))

fmt = lambda v, p=3: ("—" if v is None else f"{v:.{p}f}")
out = ["| 시퀀스 | 길이(초) | MPJPE Isaac→MuJoCo (rad) | 상대오차 Isaac→MuJoCo (mm) "
       "| Isaac 성공 | MuJoCo 성공 1회 | MuJoCo 성공 10시드 | 첫 이탈(초) | 이탈 구간 |",
       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in sorted(rows, key=lambda r: -(r["m_ok"])):
    fs = "—" if r["fail"] is None else f"{r['fail']:.0f}"
    out.append(f"| {r['seq']} | {r['sec']:.0f} | {fmt(r['i_jnt'])} → {r['m_jnt']:.3f} "
               f"| {fmt(r['i_rel'],1)} → {r['m_rel']:.1f} | {fmt(r['i_ok'],2)} "
               f"| {r['m_ok']:.0f} | {r['m_ok10']} "
               f"| {fs} "
               f"| {r['over']*100:.1f}% |")
note = (
    "전장 롤아웃 기준. 판정은 PolySim — 전역 바디 위치 오차 평균이 한 번이라도\n"
    "0.5 m 를 넘으면 실패로 센다. Isaac 쪽은 환경 30개 평균이고 도메인 랜덤화가\n"
    "켜져 있으며, MuJoCo 쪽은 잡음 없는 결정론 1회다. 그래서 상대오차가 MuJoCo 에서\n"
    "더 작게 나오는 것은 전이가 더 정확해서가 아니라 표본 조건이 다르기 때문이다.\n"
    "10시드 열은 초기 관절각·루트 자세에 0.02 잡음을 준 10회 중 성공 비율이다.\n"
    "이탈 구간은 전체 프레임 중 0.5 m 를 넘은 비율이다.\n\n")
txt = note + "\n".join(out) + "\n"
print(txt)
open("outputs/metrics/sim2sim_full_table.md", "w").write(txt)
