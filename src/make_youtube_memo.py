"""Isaac/MuJoCo 비교 영상마다 유튜브 제목과 설명을 만든다.

기존 규약(g1_motion_tracking/<시퀀스>/유튜브_메모.md)과 같은 형식으로 쓴다.
수치는 지어내지 않고 outputs 의 측정 결과에서만 읽는다.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from score_standard import score

R = "/home/sehoon/Documents/GitHub/g1-motion-tracking"
DEST = "/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking"

# 저자세 구간(루트 0.35 m 미만) 진입 시점. 있으면 영상에서 갈리는 지점이다
def low_onset(seq):
    z = np.load(f"/home/sehoon/motions_fixed/{seq}.npz")["body_pos_w"][:, 0, 2]
    return int(np.argmax(z < 0.35)) / 50.0 if (z < 0.35).any() else None

for f in sorted(glob.glob(f"{R}/outputs/sim2sim_full2/*.npz")):
    seq = os.path.basename(f)[:-4]
    d = np.load(f)
    r = score(d)
    n = r["frames"]; dur = n / 50.0
    ip = f"{R}/outputs/eval_polysim/{seq}.json"
    iso = json.load(open(ip)) if os.path.exists(ip) else {}
    onset = low_onset(seq)

    # 유튜브 제목은 100자까지다. 가장 긴 시퀀스 이름으로도 넘지 않게 짧게 쓴다.
    title = f"sim-to-sim: Isaac Lab vs MuJoCo, frame-matched on {seq} (Unitree G1)"
    assert len(title) <= 100, f"{seq} 제목 {len(title)}자"

    lines = [
        "Left: Isaac Lab, where the policy was trained.",
        "Right: MuJoCo, running the same exported policy zero-shot.",
        "Same reference frame 0, same 50 Hz step, same number of frames,",
        "so the same frame number is the same instant in both panels.",
        "",
        f"Motion: LAFAN1 {seq}, {n:,} frames at 50 Hz ({dur:.0f} s), full",
        "sequence, no cuts. Domain randomization is off in both panels.",
        "",
        "Pipeline",
        "- Mocap: LAFAN1 BVH",
        "- Retargeting: GMR (General Motion Retargeting) to Unitree G1, 29 DoF",
        "- Training: BeyondMimic whole-body tracking in Isaac Lab, RSL-RL PPO,",
        "  4096 environments, 30,000 iterations",
        "- MuJoCo: the exported ONNX policy stepped at 50 Hz control / 1 kHz physics,",
        "  PD torques clipped to the same joint limits, no ROS layer",
        "",
        "Measured over the full sequence",
    ]
    jabs = float(np.abs(d["ref_joint_pos"] - d["rob_joint_pos"]).mean())
    pf = lambda ok: "pass" if ok else "fail"
    # 양쪽을 같은 정의로 적는다. 이름을 계보마다 다르게 쓰면 다른 지표로 읽힌다.
    lines += [
        "                          Isaac Lab      MuJoCo",
        f"global body error         {iso.get('e_g_mpbpe_mm', float('nan')):>7.0f} mm"
        f"{r['mpkpe']:>10.0f} mm",
        f"local pose, re-anchored   {iso.get('e_mpbpe_mm', float('nan')):>7.0f} mm"
        f"{r['r_mpkpe']:>10.0f} mm",
        f"joint angle               {iso.get('e_mpjpe_rad', float('nan')):>7.3f} rad"
        f"{jabs:>9.3f} rad",
        f"survives termination      {pf(iso.get('success_rate', 0) > 0.5):>10}"
        f"{pf(r['bm_success']):>12}",
        f"stays within 0.5 m        {pf(iso.get('success_rate_polysim', 0) > 0.5):>10}"
        f"{pf(r['poly_success']):>12}",
        "",
        "Two pass criteria, because there is no single one.",
        "BeyondMimic ends the episode when the anchor height, the anchor",
        "orientation or an ankle or wrist height leaves its threshold; all three",
        "look at z alone and never see horizontal drift.",
        "PolySim counts a rollout failed once the mean global body position error",
        "crosses 0.5 m, which is built to catch exactly that.",
        "Local pose is the reference re-anchored to the robot's torso with yaw",
        "removed, the same quantity on both sides.",
    ]
    if onset is not None:
        lines += ["",
                  f"At about {onset:.0f} s the clip enters a low-posture segment"
                  " (pelvis below 0.35 m).",
                  "That part of LAFAN1 has the actor sitting or lying down, which the",
                  "G1 cannot reproduce, so both simulators lose the reference there."]
    lines += ["",
              "Criteria and definitions: BeyondMimic (arXiv:2508.08241) termination",
              "thresholds, PolySim (arXiv:2510.01708) 0.5 m, mjlab MPKPE/R-MPKPE.",
              "Code: github.com/hooneyskywalker0127/g1-motion-tracking",
              "Policies: huggingface.co/hooneyskywalker/g1-motion-tracking-policies"]

    body = "\n".join(lines)
    out_dir = f"{DEST}/{seq}/isaac_mujoco"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{seq}_유튜브_메모.md", "w") as fp:
        # 코드펜스를 쓰면 복사할 때 따라붙는다. 제목과 설명만 남긴다.
        fp.write(f"[제목]\n{title}\n\n[설명]\n{body}\n")
    old_txt = f"{out_dir}/{seq}_youtube.txt"
    if os.path.exists(old_txt):
        os.remove(old_txt)      # 같은 내용을 두 파일로 두던 것을 정리한다
    print(f"{seq:<22} {title[:60]}")
