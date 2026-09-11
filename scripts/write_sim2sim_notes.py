"""sim-to-sim 영상의 유튜브 제목·설명문을 시퀀스마다 써 둔다.

    python scripts/write_sim2sim_notes.py [시퀀스 ...]

Isaac 쪽 수치는 outputs/eval/<시퀀스>.json,
MuJoCo 쪽은 <영상폴더>/<시퀀스>/sim2sim/<시퀀스>_mujoco.json 에서 읽는다.
그리고 각 시퀀스의 유튜브_메모.md 에 같은 내용을 한 절로 덧붙인다.
"""

import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")
HEAD = "## Sim-to-Sim (MuJoCo)"


def note(seq, a, b):
    frames = a.get("motion_frames", 0)
    isnan = lambda v: isinstance(v, float) and math.isnan(v)
    fell = b["fell"]

    lines = [
        f"Sim-to-Sim Transfer: Isaac Lab Policy Running in MuJoCo ({seq}, Unitree G1)",
        "",
        "Left: the policy in Isaac Lab, where it was trained.",
        "Right: the same ONNX policy, unchanged, running in MuJoCo.",
        "No retraining, no fine-tuning, no gain retuning between the two.",
        "Both cameras track the pelvis, so the robot stays in frame in both panels.",
        "",
        "Why this matters: a tracking policy can look perfect in the simulator it",
        "was trained in and fall apart the moment the contact model changes. Running",
        "it in a second simulator with a different solver is the cheapest honest test",
        "of how much of the behaviour is the policy and how much is the simulator.",
        "",
        "Setup",
        "- Policy: exported to ONNX from the Isaac Lab checkpoint at 30,000 iterations",
        "- MuJoCo: mujoco_sim_ros2 driven through ros2_control, same controller stack",
        "  as the hardware bring-up",
        "- Rendering: offscreen from the recorded rollout state, tracking camera on the",
        "  pelvis. The root orientation is recovered from 14 body positions by Kabsch",
        "  alignment (0.2 mm mean reconstruction error).",
        "- Domain randomization is off on both sides.",
        "",
        f"Motion: LAFAN1 {seq}" + (f", {frames:,} frames at 50 Hz" if frames else ""),
        "",
        "Numbers on the caption bar",
        "- E_mpbpe: body position error after aligning on the anchor, in mm",
        "- MPJPE: joint angle error, in rad",
        "- The multiplier on the MuJoCo side is that panel's error divided by Isaac's",
        "",
    ]

    if isnan(a.get("e_mpbpe_mm")):
        lines += [
            "Isaac Lab: completion rate 0% on this sequence, so the per-frame errors",
            "are not defined there. Only the MuJoCo numbers are shown.",
        ]
    else:
        lines += [
            f"Isaac Lab   E_mpbpe {a['e_mpbpe_mm']:.0f} mm   "
            f"MPJPE {a['e_mpjpe_rad']:.3f} rad   "
            f"completion {a['success_rate']*100:.0f}%",
            f"MuJoCo      E_mpbpe {b['e_mpbpe_mm']:.0f} mm   "
            f"MPJPE {b['e_mpjpe_rad']:.3f} rad   " + (
                f"survived {b['alive_ratio']*100:.0f}% of the clip" if not fell
                else f"fell after {b['alive_ratio']*100:.0f}% of the clip"),
        ]
    mins = frames / 50.0 / 60.0
    lines += [
        "",
        "Global body position error (E_g-mpbpe) goes from "
        f"{a.get('e_g_mpbpe_mm_all', a.get('e_g_mpbpe_mm', 0)):.0f} mm in Isaac Lab",
        f"to {b['e_g_mpbpe_mm']:.0f} mm in MuJoCo. That gap is dominated by root drift,",
        f"which accumulates over a {mins:.0f}-minute clip; the anchor-aligned E_mpbpe",
        "above is the fairer read on posture.",
        "",
    ]
    if fell:
        lines += [
            "This policy does not survive the transfer. In MuJoCo it stays up for "
            f"{b['alive_ratio']*100:.0f}% of the clip",
            "before losing the reference. It is included as-is rather than dropped:",
            "the sequences that fail are the informative ones for sim-to-sim.",
            "",
        ]
    lines += [
        "Pipeline",
        "- Mocap: LAFAN1 BVH",
        "- Retargeting: GMR (General Motion Retargeting) to Unitree G1, 29 DoF",
        "- Training: BeyondMimic whole-body tracking in Isaac Lab, RSL-RL PPO,",
        "  4096 environments, 30,000 iterations",
        "- Sim-to-sim: MuJoCo through mujoco_sim_ros2 + ros2_control",
    ]
    return "\n".join(lines)


def main(seqs):
    for seq in seqs:
        ij = REPO / "outputs/eval" / f"{seq}.json"
        mj = BASE / seq / "sim2sim" / f"{seq}_mujoco.json"
        if not (ij.exists() and mj.exists()):
            print("건너뜀", seq, "(json 없음)")
            continue
        text = note(seq, json.loads(ij.read_text()), json.loads(mj.read_text()))

        out = BASE / seq / "sim2sim" / f"{seq}_sim2sim_youtube.txt"
        out.write_text(text + "\n")

        memo = BASE / seq / "유튜브_메모.md"
        title, body = text.split("\n", 1)
        block = f"{HEAD}\n\n제목\n\n```\n{title}\n```\n\n설명\n\n```\n{body.lstrip()}\n```\n"
        if memo.exists():
            cur = memo.read_text()
            cur = cur.split(HEAD)[0].rstrip() + "\n\n" if HEAD in cur else cur.rstrip() + "\n\n"
            memo.write_text(cur + block)
        else:
            memo.write_text(f"# {seq} 유튜브 메모\n\n" + block)
        print("작성:", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = sorted(d.name for d in BASE.iterdir()
                      if (d / "sim2sim").is_dir())
    main(args)
