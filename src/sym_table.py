"""대칭 프로토콜 결과를 표 A·B 로 낸다.

읽는 것
  outputs/sym/<seq>_sim.json        Isaac, 교란 끔
  outputs/sym/<seq>_simdr.json      Isaac, 초기 교란 켬
  outputs/sym/<seq>_sim2sim.npz     MuJoCo, 같은 교란, 시행 100회

유지율은 sim-dr 과 sim2sim 의 비다. 둘이 같은 교란 분포·같은 시행 수라
이 비만 전이 손실로 읽을 수 있다. sim 열은 교란이 성능을 얼마나 깎는지
보여주는 참고값이다.

성공률은 MuJoCo/Isaac, 오차는 Isaac/MuJoCo 로 적는다. 둘 다 100% 가
손실 없음을 뜻하게 하기 위해서다.

    python src/sym_table.py [--out outputs/sym/표.md]
"""
import argparse, glob, json, os, numpy as np

# sim2sim npz 한 행의 열 순서 (sim2sim_trials.py 와 같아야 한다)
C_BM, C_POLY, C_MPKPE, C_RMPKPE, C_JL2, C_JVEL, C_ALIVE = range(7)

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="outputs/sym")
ap.add_argument("--out", default="")
a = ap.parse_args()

seqs = sorted({os.path.basename(f).rsplit("_sim2sim", 1)[0]
               for f in glob.glob(f"{a.dir}/*_sim2sim.npz")})
if not seqs:
    seqs = sorted({os.path.basename(f).rsplit("_", 1)[0]
                   for f in glob.glob(f"{a.dir}/*_simdr.json")})

rows = []
for s in seqs:
    r = {"seq": s}
    for cond in ("sim", "simdr"):
        p = f"{a.dir}/{s}_{cond}.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        r[cond] = dict(bm=j["success_rate"], poly=j["success_rate_polysim"],
                       mpkpe=j["e_g_mpbpe_mm"], rmpkpe=j["e_mpbpe_mm"],
                       jl2=j["e_joint_l2_rad"], jvel=j.get("e_jointvel_l2", float("nan")))
    p = f"{a.dir}/{s}_sim2sim.npz"
    if os.path.exists(p):
        A = np.load(p)[s] if s in np.load(p) else list(np.load(p).values())[0]
        ok = A[:, C_BM].astype(bool)
        f = lambda c: float(A[ok, c].mean()) if ok.any() else float("nan")
        r["sim2sim"] = dict(bm=float(A[:, C_BM].mean()), poly=float(A[:, C_POLY].mean()),
                            mpkpe=f(C_MPKPE), rmpkpe=f(C_RMPKPE),
                            jl2=f(C_JL2), jvel=f(C_JVEL), n=len(A))
    rows.append(r)

full = [r for r in rows if {"sim", "simdr", "sim2sim"} <= set(r)]
out = []
w = out.append

w("# sim-to-sim 전이 — 대칭 프로토콜로 다시 잰 결과\n")
w("주장은 하나다. 같은 정책을 Isaac Lab 에서 MuJoCo 로 옮겼을 때 손실이 없다.")
w("남의 논문 숫자를 이기는 문장은 안 낸다 (아래 '버린 것').\n")

w("## 어떻게 했나\n")
w("전이 손실을 읽으려면 두 열이 같은 양이어야 한다. 그래서 넷을 맞췄다.\n")
w("```")
w("채점기    src/score_standard.py 하나로 고정. Isaac 쪽 scripts/rsl_rl/eval_sym.py 가")
w("          같은 식을 쓴다. 관절 오차는 KungfuBot 식 (23) 의 L2 노름이다")
w("          (고치기 전 Isaac 은 관절당 평균 절댓값, MuJoCo 는 RMS 로 서로 달랐다)")
w("판정      S_bm  BeyondMimic 종료조건 — 앵커 z 0.25 m · 중력 z 성분 0.8 · 말단 4곳 z 0.25 m")
w("          S_poly PolySim — 전역 바디 오차 평균이 한 번도 0.5 m 초과 안 함. S_bm 과 독립")
w("          (고치기 전 Isaac 은 S_poly 에 S_bm 을 AND 로 걸어 MuJoCo 보다 엄했다)")
w("교란      양쪽 다 tracking_env_cfg.py 의 MotionCommandCfg 값")
w("          루트 위치 x,y ±0.05 z ±0.01 m · 자세 roll,pitch ±0.1 yaw ±0.2 rad")
w("          루트 속도 x,y ±0.5 z ±0.2 m/s · roll,pitch ±0.52 yaw ±0.78 rad/s")
w("          관절 ±0.1 rad, 전부 균등분포. 시드 0~99")
w("          (고치기 전 MuJoCo 는 근거 없는 가우시안 0.02 였다)")
w("          물성·질량중심·밀치기는 안 켠다. MuJoCo 에서 재현할 수 없으면 대칭이 깨진다")
w("정렬      레퍼런스와 로봇을 같은 순간에 읽는다")
w("          (고치기 전 Isaac 은 레퍼런스 스텝 전, 로봇 스텝 후를 짝지었다)")
w("평균      완주한 시행의 살아 있는 프레임에만. 창은 모션 전 길이")
w("```\n")

w("## 근거 — 두 구현이 같은 값을 내는지 직접 대조했다\n")
w("Isaac env 0 롤아웃을 통째로 덤프해 MuJoCo 채점기로 다시 채점하고,")
w("Isaac 이 온라인으로 낸 값과 맞댔다. walk1_subject1, 13064 프레임.\n")
w("```")
w("              score_standard(env0)   eval_sym 온라인      차이")
w("MPKPE mm              77.1335           77.1335      0.000024")
w("R-MPKPE mm            33.7814           33.7809      0.000543")
w("E_joint rad            0.4495            0.4495      0.000001")
w("E_jvel                 3.2635            3.2636      0.000009")
w("S_bm                     True              True")
w("```")
w("남은 차이는 덤프가 float16 이라 생긴 것이다. 두 구현은 같은 양을 잰다.")
w("이 대조를 통과하기 전에는 두 열을 같은 표에 올리지 않았다.\n")
w("부수로 나온 것 — 반 스텝 정렬을 고치니 MPKPE 가 71.2 에서 77.1 mm 로 올라갔다.")
w("6 mm 는 보행 속도 x 20 ms 규모라 크기가 맞는다. 즉 그전까지 쓰던 74.3 mm 는")
w("레퍼런스를 한 프레임 앞세워 잰 값이었다.\n")

w("그리고 MuJoCo 초기 교란이 실제로 Isaac 규격대로 걸리는지 60회 표본으로 쟀다.\n")
w("```")
w("항목               min      max     규격 ±")
w("dx            -0.050    0.046    0.050")
w("dy            -0.050    0.049    0.050")
w("dz            -0.010    0.010    0.010")
w("droll         -0.102    0.098    0.100")
w("dpitch        -0.098    0.097    0.100")
w("dyaw          -0.196    0.195    0.200")
w("dvx           -0.495    0.491    0.500")
w("dvy           -0.497    0.496    0.500")
w("dvz           -0.191    0.199    0.200")
w("dwx           -0.526    0.512    0.520")
w("dwy           -0.491    0.560    0.520")
w("dwz           -0.733    0.748    0.780")
w("djoint max     0.087    0.100    0.100")
w("```")
w("각속도 셋만 규격을 조금 넘는데 이건 오류가 아니다. Isaac 은 월드 프레임에서")
w("델타를 더하고 MuJoCo free joint 의 각속도는 바디 로컬 규약이라, 월드 델타를")
w("로컬로 돌려 넣으면 성분이 섞여 축별 범위가 달라진다. 벡터 크기는 보존된다.\n")

w("독립 확인 하나 더 — MuJoCo 쪽을 kobe 를 포함한 18 시퀀스로 따로 한 번 더 돌려")
w("S_bm 0.788 · S_poly 0.631 을 얻었다. 아래 표 A 의 17 시퀀스 값(0.775 · 0.609)과")
w("kobe 한 개 차이만큼 다르고 그 외에는 같다.\n")

w("## 반 스텝·조기 종료를 고치기 전에 나왔던 값 — 왜 버렸나\n")
w("고치기 전 판은 이렇게 나왔다.\n")
w("```")
w("obstacles2_subject1   S_bm 0.00   S_poly 0.28")
w("```")
w("전부 실패인데 PolySim 기준으로는 28% 가 성공이다. Isaac 이 넘어지는 순간")
w("에피소드를 끝내 버려서 넘어진 뒤의 전역 오차가 기록되지 않기 때문이다.")
w("1136 / 12204 프레임에서 죽으니 0.5 m 를 넘을 기회 자체가 없었다.")
w("MuJoCo 는 종료가 없어 끝까지 굴러가므로 넘어지면 반드시 넘는다.")
w("같은 이름의 두 숫자가 다른 것을 재고 있었고, 그대로 표를 냈으면 MuJoCo 쪽")
w("S_poly 만 낮게 나와 '전이에 손실이 있다' 는 반대 결론이 나왔을 것이다.\n")
w("고친 뒤 같은 시퀀스가 양쪽 다 0.00 이다.\n")

if full:
    w("## 표 A — 17 시퀀스 평균\n")
    w("```")
    w(f"{'':<14}{'sim':>10}{'sim-dr':>10}{'sim2sim':>10}{'유지율':>10}")
    lab = [("S_bm 성공률", "bm", True), ("S_poly 성공률", "poly", True),
           ("MPKPE mm", "mpkpe", False), ("R-MPKPE mm", "rmpkpe", False),
           ("E_joint rad", "jl2", False), ("E_jvel", "jvel", False)]
    for name, k, higher_better in lab:
        v = [np.nanmean([r[c][k] for r in full]) for c in ("sim", "simdr", "sim2sim")]
        # 성공률은 높을수록 좋으므로 MuJoCo/Isaac, 오차는 낮을수록 좋으므로 Isaac/MuJoCo
        keep = (v[2] / v[1] if higher_better else v[1] / v[2]) * 100 if v[1] else float("nan")
        fmt = "{:>10.3f}" if k in ("bm", "poly", "jl2") else "{:>10.1f}"
        w(f"{name:<14}" + "".join(fmt.format(x) for x in v) + f"{keep:>9.1f}%")
    w(f"\n시행  sim 100환경(교란 없어 사실상 1표본) · sim-dr 100환경 · sim2sim {full[0]['sim2sim']['n']}회")
    w("```\n")

w("## 표 B — 시퀀스별 (sim-dr → sim2sim)\n")
w("```")
w(f"{'시퀀스':<22}{'S_bm':>14}{'S_poly':>14}{'MPKPE mm':>16}{'R-MPKPE mm':>16}")
w(f"{'':<22}{'Isaac  MuJoCo':>14}{'Isaac  MuJoCo':>14}{'Isaac   MuJoCo':>16}{'Isaac   MuJoCo':>16}")
for r in rows:
    if not {"simdr", "sim2sim"} <= set(r):
        w(f"{r['seq']:<22}  (미완)")
        continue
    d, m = r["simdr"], r["sim2sim"]
    w(f"{r['seq']:<22}{d['bm']:>7.2f}{m['bm']:>7.2f}{d['poly']:>7.2f}{m['poly']:>7.2f}"
      f"{d['mpkpe']:>8.1f}{m['mpkpe']:>8.1f}{d['rmpkpe']:>8.1f}{m['rmpkpe']:>8.1f}")
w("```\n")

w("## 표 C — 바깥 잣대. 같은 계보만\n")
w("```")
w("Retargeting Matters  LAFAN1 · G1 · BeyondMimic · IsaacSim · DR 없음 · 100회")
w("  전역 바디 위치 오차 중앙값")
w("  Unitree 73.4 / GMR 91.2 / ProtoMotions 101.9 / PHC 111.9 mm")
w("SONIC 이 MuJoCo 에서 잰 BeyondMimic 의 루트 상대 오차   40.9 mm")
w("PHUMA 부록 D.3 의 Isaac Gym → MuJoCo 유지율            90.5 % · 93.2 %")
w("```")
w("이 셋만 남긴 이유는 셋 다 같은 계보이거나 같은 양이기 때문이다.\n")

w("## 버린 것 — kobe 와 PolySim 표 맞대기\n")
w("세 가지가 따로 틀렸다.\n")
w("하나. 맞댄 숫자가 PolySim 의 성적이 아니다. Table III 원문이다 (v3 PDF 직접 확인).")
w("```")
w("Training Env.                 Genesis Succ  Eg-mpjpe   MuJoCo Succ  Eg-mpjpe")
w("IsaacGymDR                       1.000      163.135      0.100      295.877")
w("IsaacSimDR                       1.000      130.936      0.100      272.610")
w("IsaacSim+IsaacGym                1.000      103.941      0.100      178.190")
w("IsaacSim+IsaacGym+Genesis           -            -       1.000      199.166")
w("```")
w("인용했던 0.100 / 272.6 은 IsaacSimDR 행이다. 이 표의 목적 자체가 '파라미터 DR 로는")
w("안 되고 시뮬레이터를 섞어야 된다' 를 보이는 것이고, DR 행들은 그 대조군이다.")
w("PolySim 자신의 값은 마지막 행 1.000 / 199.166 이다.\n")
w("둘. 같은 모션 집합을 맞출 수 없다. Table I 은 ASAP 14개 평균, V-D 절은 5개인데")
w("둘 다 이름이 논문에 없다. 전문에 나오는 모션 이름은 Kobe 하나뿐이다.")
w("로컬 ASAP 저장소에 52개 pkl 이 있지만 어느 14개인지 고를 근거가 없다.\n")
w("셋. 둘을 다 고쳐도 비교가 안 선다. 학습기가 다르다 — PolySim 은 HumanoidVerse 에")
w("ASAP 보상, teacher-student 이고 이쪽은 BeyondMimic 이다. 같은 LAFAN1 · G1 ·")
w("BeyondMimic 으로 돌린 Retargeting Matters 가 sim2sim 성공률 대부분 100% 를 받았다.")
w("즉 1.000 은 PolySim 을 이긴 것이 아니라 BeyondMimic 계보의 통상값이다.")
w("학습량(10k/15k/20k 대 30k), 창 길이(미기재), 시행 수(추정 140 대 10)도 다르다.\n")
w("kobe 롤아웃 자체는 버리지 않는다. 뒷받침하는 문장을 바꾼다 —")
w("'PolySim 보다 낫다' 가 아니라 'ASAP 고난도 단발 모션에서도 전이 손실이 없다' 다.\n")

w("## 이 결과가 못 닫는 것\n")
w("```")
w("실기가 없다. sim-to-sim 은 sim-to-real 의 대리지표다")
w("17 시퀀스가 전부 LAFAN1 이다. 고난도 단발 동작은 kobe 하나뿐이고 하나는 표본이 아니다")
w("Isaac 안쪽 성공률이 분야 기준보다 낮다. 같은 LAFAN1 · G1 · BeyondMimic 으로")
w("  Retargeting Matters 가 96~100% 를 받는데 이쪽은 0.00 이 다섯이다.")
w("  전이가 손실 없다는 결론은 이 자리에서 오히려 강해지지만 — 남들이 100% 받는 데서")
w("  0.00 이 나오고 MuJoCo 는 잘 따라간다면 그 0.00 은 종료조건과 모션 선별의 문제다 —")
w("  표에 같이 적어야 한다. 숨기면 심사에서 먼저 물린다")
w("```")

text = "\n".join(out)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
    print(f"\n{a.out}")
