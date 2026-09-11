#!/usr/bin/env python3
"""기록한 policy_io 에서 sim-to-sim 지표를 낸다.

해석을 넣지 않는다. 쓰는 값은 컨트롤러가 발행한 것과 ONNX 가 돌려준 것뿐이다.

  관측 배치 (tracking_env_cfg.py ObservationsCfg 순서, 총 160)
    0: 29   command  = 레퍼런스 관절 위치      ← ONNX joint_pos 와 비트 일치
   29: 58   command  = 레퍼런스 관절 속도
   58: 61   motion_anchor_pos_b
   61: 67   motion_anchor_ori_b
   67: 70   base_lin_vel
   70: 73   base_ang_vel
   73:102   joint_pos_rel  = 측정 관절 위치 − default
  102:131   joint_vel_rel  = 측정 관절 속도
  131:160   이전 행동
  160:189   행동

지표 정의는 Isaac 쪽 scripts/rsl_rl/eval.py 와 같다.
실패 판정은 HumanTracker(arXiv:2608.13555) 를 따른다.
"""
import argparse
import json
import pathlib

import numpy as np
import onnx
import onnxruntime as ort

POLICY_DIR = pathlib.Path("/home/sehoon/colcon_ws/policies")
EVAL_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "eval"
CTRL_HZ = 50.0
VERT_LIMIT = 0.25      # m, 골반/발목/손목 수직 오차
ROT_LIMIT = 1.0        # rad, 골반 회전 오차
MEAN_LIMIT = 0.5       # m, OmniH2O 기준 평균 편차
VERT_BODIES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link",
               "left_wrist_yaw_link", "right_wrist_yaw_link")


def load_policy(seq):
    path = POLICY_DIR / f"{seq}.onnx"
    md = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess, md


def ref_at(sess, step, names):
    return sess.run(names, {"obs": np.zeros((1, 160), np.float32),
                            "time_step": np.array([[step]], np.float32)})


def find_start_step(sess, obs0, horizon=20000):
    """관측 앞 29개를 ONNX 레퍼런스와 대조해 time_step 을 **찾는다**(추정 아님)."""
    best = (None, np.inf)
    for k in range(horizon):
        r = ref_at(sess, k, ["joint_pos"])[0][0]
        e = float(np.abs(r - obs0).max())
        if e < best[1]:
            best = (k, e)
        if e < 1e-6:
            return k, e
    return best


def quat_angle(a, b):
    return 2.0 * np.arccos(np.clip(np.abs((a * b).sum(-1)), -1.0, 1.0))


def evaluate(seq, npz_path):
    d = np.load(npz_path, allow_pickle=True)
    io = d["io"]
    sess, md = load_policy(seq)
    qdef = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    bnames = md["body_names"].split(",")
    anchor_i = bnames.index(md["anchor_body_name"])

    k0, err = find_start_step(sess, io[0, :29])
    exact = err < 1e-6

    n = len(io)
    ref_q = io[:, :29]
    meas_q = io[:, 73:102] + qdef
    err_jnt = np.abs(ref_q - meas_q).mean(-1)

    # 레퍼런스 바디 자세를 ONNX 에서 가져온다. 측정 바디 자세는 없으므로
    # 앵커 상대 오차는 관측의 anchor 항으로 대신한다 (컨트롤러가 계산한 값).
    anchor_pos_b = io[:, 58:61]
    err_anchor = np.linalg.norm(anchor_pos_b, axis=-1)

    # 위상 일치 검증: 몇 군데를 뽑아 ONNX 와 정확히 맞는지 본다
    checks = [i for i in (0, n // 4, n // 2, 3 * n // 4, n - 1) if 0 <= i < n]
    match = sum(float(np.abs(ref_at(sess, k0 + i, ["joint_pos"])[0][0] - io[i, :29]).max()) < 1e-6
                for i in checks)

    # Isaac 의 eval.py 는 로봇을 레퍼런스 상태로 리셋하고 시작한다. MuJoCo 쪽은 그럴
    # 수단이 없어 기동 자세에서 레퍼런스로 수렴하는 구간이 앞에 붙는다. 그 구간을
    # 버리지 않고, 추종을 잡은 시각(acquire)을 따로 내고 그 뒤로 실패를 판정한다.
    finite = np.isfinite(io).all(-1)
    ok = finite & (err_anchor <= MEAN_LIMIT)
    acquire = int(np.argmax(ok)) if ok.any() else n
    bad = ~ok
    bad[:acquire] = False
    first = int(np.argmax(bad)) if bad[acquire:].any() else n

    # 추종을 한 번도 못 잡으면(acquire == n) 평균 낼 구간이 없다. 그대로 nan 으로 둔다.
    k = min(max(first, acquire + 1), n)
    empty = acquire >= k

    # Isaac 쪽 eval.py 결과를 그대로 옆에 둔다. 지표 정의가 같아서 바로 비교된다.
    ev = EVAL_DIR / f"{seq}.json"
    isaac = json.loads(ev.read_text()) if ev.exists() else {}

    return dict(
        motion=seq,
        messages=n,
        seconds=float(n / CTRL_HZ),
        start_time_step=int(k0),
        start_match_err=float(err),
        phase_exact=bool(exact),
        phase_checks_passed=f"{match}/{len(checks)}",
        acquire_s=float(acquire / CTRL_HZ),
        alive_messages=first - acquire,
        alive_ratio=float((first - acquire) / max(n - acquire, 1)),
        success=bool(first >= n),
        e_mpjpe_rad=float("nan") if empty else float(err_jnt[acquire:k].mean()),
        e_anchor_pos_m=float("nan") if empty else float(err_anchor[acquire:k].mean()),
        max_anchor_pos_m=float("nan") if empty else float(err_anchor[acquire:k].max()),
        isaac_e_mpjpe_rad=isaac.get("e_mpjpe_rad"),
        isaac_e_g_mpbpe_mm=isaac.get("e_g_mpbpe_mm_all"),
        isaac_alive_ratio=(isaac["mean_alive_frames"] / isaac["motion_frames"]
                           if isaac else None),
        isaac_success_rate=isaac.get("success_rate"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/sehoon/Documents/GitHub/g1-motion-tracking/outputs/policy_io")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    files = sorted(pathlib.Path(a.dir).glob("*.npz"))
    print(f"{'시퀀스':<22}{'초':>7}{'시작step':>9}{'위상':>7}{'획득s':>7}{'완주':>6}{'생존':>8}{'Isaac생존':>10}"
          f"{'MPJPE':>9}{'Isaac':>9}{'차이':>8}{'앵커오차m':>11}{'Isaac전역m':>11}")
    res = []
    for f in files:
        try:
            r = evaluate(f.stem, f)
        except Exception as e:          # 스윕이 아직 쓰는 중인 파일
            print(f"{f.stem:<22} 건너뜀 ({type(e).__name__})")
            continue
        res.append(r)
        ie = r["isaac_e_mpjpe_rad"]
        iso = f"{ie:.3f}" if ie else "-"
        dif = f"{r['e_mpjpe_rad'] - ie:+.3f}" if ie else "-"
        print(f"{r['motion']:<22}{r['seconds']:>7.0f}{r['start_time_step']:>9d}"
              f"{('정확' if r['phase_exact'] else '불일치'):>7}"
              f"{r['acquire_s']:>7.1f}{('O' if r['success'] else 'X'):>6}{r['alive_ratio']*100:>6.1f}%"
              f"{(r['isaac_alive_ratio'] or 0)*100:>9.1f}%"
              f"{r['e_mpjpe_rad']:>9.3f}{iso:>9}{dif:>8}"
              f"{r['e_anchor_pos_m']:>11.3f}{(r['isaac_e_g_mpbpe_mm'] or 0)/1000:>11.3f}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("저장", a.out)
