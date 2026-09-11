#!/usr/bin/env python3
"""Isaac 쪽 종료 조건을 그대로 MuJoCo 기록에 적용한다.

내가 쓰던 판정(3D 앵커 오차 0.5 m)은 Isaac 이 쓰는 것과 다르다. Isaac 은
수평 드리프트를 종료 사유로 보지 않는다. 같은 기준이 아니면 두 생존율을
나란히 놓을 수 없다. 그래서 `tracking_env_cfg.py` 의 TerminationsCfg 세 항을
그대로 옮긴다.

  anchor_pos    bad_anchor_pos_z_only        |ref_anchor_z - robot_anchor_z| > 0.25
  anchor_ori    bad_anchor_ori               |g_ref_b.z - g_robot_b.z|       > 0.8
  ee_body_pos   bad_motion_body_pos_z_only   발목·손목 |ref_z - robot_z|     > 0.25

`body_pos_relative_w` 의 z 성분은 재정렬(yaw only, x·y 만 로봇 앵커로 옮김)을
거쳐도 레퍼런스의 월드 z 그대로다 (commands.py:289-294). 그래서 z 비교는
양쪽 월드 높이를 직접 비교하는 것과 같다.

로봇 쪽 월드 자세는 odom(상태추정기)에서, 링크 높이는 MuJoCo 순기구학으로 낸다.
컨트롤러가 쓰는 것과 같은 추정기다.

    python3 src/sim2sim_isaac_criteria.py [--stride N]
"""
import argparse
import json
import pathlib

import mujoco as mj
import numpy as np
import onnx
import onnxruntime as ort

POLICY_DIR = pathlib.Path("/home/sehoon/colcon_ws/policies")
MJCF = "/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
ROOT = pathlib.Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "outputs" / "eval"
CTRL_HZ = 50.0

ANCHOR_Z = 0.25
ANCHOR_ORI = 0.8
EE_Z = 0.25
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_yaw_link", "right_wrist_yaw_link")
ANKLE_TO_SOLE = 0.02
CONTACT_TH = 150.0   # N, legged_estimation/LinearKalmanFilter.h

# ThorArena (arXiv:2607.06052, 식 3) 의 Force-Aware Tracking Score.
#   S_i = 100 exp(-E_i / sigma) * s_i ,  s_i = min(T_i / T_ref, 1) ,  sigma = 0.15 m
# 추종 오차와 생존을 곱으로 묶는다. 추종만 보면 안 갈리고 생존만 보면 오차를 버리는
# 문제를 한 수로 푼다. 여기서는 외력 재생 없이("no-force setting") 같은 식을 쓴다.
FATS_SIGMA = 0.15


def quat_to_mat(q):
    """(w, x, y, z) → 3x3"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def evaluate(seq, stride, height="estimator", gate_contact=False, from_dir="policy_io"):
    d = np.load(ROOT / "outputs" / from_dir / f"{seq}.npz", allow_pickle=True)
    io, io_t, od, od_t = d["io"], d["io_t"], d["od"], d["od_t"]

    # 부양 구간에서는 베이스 높이 추정을 믿을 수 없다 (arXiv:2210.02127).
    # 접촉 문턱은 구현값을 그대로 쓴다 (LinearKalmanFilter.h: contactForceThreshold).
    loaded = None
    if gate_contact:
        if "wl" not in d or d["wl"].size == 0:
            raise SystemExit(f"{seq}: 접촉 기록이 없다. policy_io_contact 쪽을 쓸 것.")
        wl = np.interp(io_t, d["wl_t"], d["wl"])
        wr = np.interp(io_t, d["wr_t"], d["wr_"])
        loaded = (wl >= CONTACT_TH) | (wr >= CONTACT_TH)

    path = POLICY_DIR / f"{seq}.onnx"
    md = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    qdef = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    jnames = md["joint_names"].split(",")
    bnames = md["body_names"].split(",")
    anchor = bnames.index(md["anchor_body_name"])
    ee = [bnames.index(b) for b in EE_BODIES]

    m = mj.MjModel.from_xml_path(MJCF)
    data = mj.MjData(m)
    qadr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in jnames]
    bid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in bnames]

    def ref(step):
        return sess.run(["joint_pos", "body_pos_w", "body_quat_w"],
                        {"obs": np.zeros((1, 160), np.float32),
                         "time_step": np.array([[step]], np.float32)})

    k0 = None
    for k in range(20000):
        if float(np.abs(ref(k)[0][0] - io[0, :29]).max()) < 1e-6:
            k0 = k
            break
    if k0 is None:
        raise SystemExit(f"{seq}: 시작 time_step 을 못 찾았다")

    # odom 을 policy_io 시각에 맞춘다
    base = np.stack([np.interp(io_t, od_t, od[:, i]) for i in range(7)], axis=1)

    idx = range(0, len(io), stride)
    first_fail, reason = None, None
    skipped = 0
    for i in idx:
        if loaded is not None and not loaded[i]:
            skipped += 1
            continue
        _, bpos, bquat = ref(k0 + i)
        bpos, bquat = bpos[0], bquat[0]

        q = base[i, 3:7]
        q = q / (np.linalg.norm(q) + 1e-12)
        data.qpos[:3] = base[i, :3]
        data.qpos[3:7] = q
        data.qpos[qadr] = io[i, 73:102] + qdef
        mj.mj_kinematics(m, data)
        rob = data.xpos[bid].copy()
        Rrob = data.xmat[bid[anchor]].reshape(3, 3)

        if height == "kinematics":
            # 추정기가 주는 월드 높이 대신, 낮은 발을 지면에 놓고 순기구학으로 높이를 잡는다.
            # 추정기 발산을 빼고 정책만의 전이 손실을 보기 위한 것이다.
            # 양발이 동시에 뜨는 구간에서는 높이를 낮게 잡는다는 한계가 있다.
            sole = min(rob[bnames.index(n), 2]
                       for n in ("left_ankle_roll_link", "right_ankle_roll_link"))
            rob[:, 2] += ANKLE_TO_SOLE - sole

        bad = None
        if abs(bpos[anchor, 2] - rob[anchor, 2]) > ANCHOR_Z:
            bad = "anchor_pos"
        else:
            Rref = quat_to_mat(bquat[anchor])
            # projected gravity 의 z 성분 = -R[2,2]
            if abs(-Rref[2, 2] - (-Rrob[2, 2])) > ANCHOR_ORI:
                bad = "anchor_ori"
            elif np.any(np.abs(bpos[ee, 2] - rob[ee, 2]) > EE_Z):
                bad = "ee_body_pos"
        if bad and not np.isfinite(base[i]).all():
            bad = "non_finite"
        if bad:
            first_fail, reason = i, bad
            break

    n = len(io)
    alive = first_fail if first_fail is not None else n
    ev = EVAL_DIR / f"{seq}.json"
    isaac = json.loads(ev.read_text()) if ev.exists() else {}
    return dict(
        motion=seq,
        alive_ratio=alive / n,
        fats=None,          # 호출부에서 E-mpbpe 를 받아 채운다
        fail_s=None if first_fail is None else first_fail / CTRL_HZ,
        reason=reason or "-",
        isaac_alive_ratio=(isaac["mean_alive_frames"] / isaac["motion_frames"]
                           if isaac else None),
        isaac_success=isaac.get("success_rate"),
        skipped_ratio=skipped / max(len(list(idx)), 1),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--height", choices=["estimator", "kinematics"], default="estimator",
                    help="월드 높이를 어디서 얻을지. 추정기 발산을 빼려면 kinematics.")
    ap.add_argument("--gate-contact", action="store_true",
                    help="한 발이라도 실려 있을 때만 판정한다. 접촉 기록이 필요하다.")
    ap.add_argument("--dir", default="policy_io", help="읽을 폴더 (outputs 아래).")
    a = ap.parse_args()
    seqs = a.seqs or sorted(p.stem for p in (ROOT / "outputs" / a.dir).glob("*.npz"))

    print(f"{'시퀀스':<22}{'MuJoCo생존':>12}{'Isaac생존':>11}{'Isaac성공':>11}"
          f"{'실패시각':>10}{'사유':>14}")
    for s in seqs:
        try:
            r = evaluate(s, a.stride, a.height, a.gate_contact, a.dir)
        except Exception as e:
            print(f"{s:<22} 건너뜀 ({type(e).__name__})")
            continue
        fs = "-" if r["fail_s"] is None else f"{r['fail_s']:.1f}초"
        print(f"{r['motion']:<22}{r['alive_ratio']*100:>11.1f}%"
              f"{(r['isaac_alive_ratio'] or 0)*100:>10.1f}%"
              f"{(r['isaac_success'] or 0)*100:>10.0f}%"
              f"{fs:>10}{r['reason']:>14}")
