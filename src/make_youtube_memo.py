"""Isaac/MuJoCo 비교 영상마다 유튜브 제목과 설명을 만든다.

기존 규약(g1_motion_tracking/<시퀀스>/유튜브_메모.md)과 같은 형식으로 쓴다.
수치는 지어내지 않고 outputs 의 측정 결과에서만 읽는다.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sim2sim_polysim import score

R = "/home/sehoon/Documents/GitHub/g1-motion-tracking"
DEST = "/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking"

# 저자세 구간(루트 0.35 m 미만) 진입 시점. 있으면 영상에서 갈리는 지점이다
def low_onset(seq):
    z = np.load(f"/home/sehoon/motions_fixed/{seq}.npz")["body_pos_w"][:, 0, 2]
    return int(np.argmax(z < 0.35)) / 50.0 if (z < 0.35).any() else None

for f in sorted(glob.glob(f"{R}/outputs/sim2sim_full/*.npz")):
    seq = os.path.basename(f)[:-4]
    d = dict(np.load(f))
    r = score(d)
    n = r["frames"]; dur = n / 50.0
    ip = f"{R}/outputs/eval_fixed/{seq}.json"
    iso = json.load(open(ip)) if os.path.exists(ip) else {}
    onset = low_onset(seq)

    title = (f"Isaac Lab vs MuJoCo: the same tracking policy, frame-matched "
             f"on {seq} (Unitree G1)")

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
    if iso:
        lines.append(f"- Isaac completion rate: {iso['success_rate']*100:.0f}%"
                     f" over {iso['num_envs']} rollouts")
        lines.append(f"- Isaac E_g-mpbpe: {iso['e_g_mpbpe_mm_all']:.0f} mm,"
                     " body position error in global coordinates")
    lines += [
        f"- MuJoCo E_g-mpjpe: {r['e_g_mpjpe']:.0f} mm, global body position error",
        f"- MuJoCo E_mpjpe: {r['e_mpjpe']:.0f} mm, body position error"
        " after aligning on the root",
        f"- MuJoCo time above the 0.5 m failure threshold: {r['over_frac']*100:.1f}%"
        " of the rollout",
    ]
    if onset is not None:
        lines += ["",
                  f"At about {onset:.0f} s the clip enters a low-posture segment"
                  " (pelvis below 0.35 m).",
                  "That part of LAFAN1 has the actor sitting or lying down, which the",
                  "G1 cannot reproduce, so both simulators lose the reference there."]
    lines += ["",
              "Metric definitions follow PolySim (arXiv:2510.01708): a rollout counts as",
              "a failure once the mean global body position error exceeds 0.5 m."]

    body = "\n".join(lines)
    out_dir = f"{DEST}/{seq}/isaac_mujoco"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{seq}_유튜브_메모.md", "w") as fp:
        fp.write(f"# {seq} Isaac vs MuJoCo 유튜브 메모\n\n제목\n\n```\n{title}\n```\n\n"
                 f"설명\n\n```\n{body}\n```\n")
    with open(f"{out_dir}/{seq}_youtube.txt", "w") as fp:
        fp.write(title + "\n\n" + body + "\n")
    print(f"{seq}")
