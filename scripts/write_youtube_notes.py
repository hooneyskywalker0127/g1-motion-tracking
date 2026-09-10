"""각 영상 폴더에 유튜브 제목과 설명문을 영어로 써 둔다.

    python scripts/write_youtube_notes.py <시퀀스> [기준 폴더]

수치는 outputs/eval/<시퀀스>.json과 학습 로그에서 읽는다. 없으면 그 줄을 비운다.
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PIPELINE = """Pipeline
- Mocap: LAFAN1 BVH
- Retargeting: GMR (General Motion Retargeting) to Unitree G1, 29 DoF
- Training: BeyondMimic whole-body tracking in Isaac Lab, RSL-RL PPO,
  4096 environments, 30,000 iterations
- Rendering: Isaac Sim, identical lighting and camera in every panel"""


def metrics(seq):
    p = REPO / "outputs" / "eval" / f"{seq}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def compare_note(seq, m, frames, seconds):
    lines = [
        f"Humanoid Motion Tracking: Policy vs Reference on {seq} (Unitree G1)",
        "",
        "Left: an RL policy trained to track a reference motion.",
        "Right: the retargeting reference the policy is trying to follow.",
        "Same clip, same start frame, same camera, played side by side.",
        "",
        f"Motion: LAFAN1 {seq}, {frames:,} frames at 50 Hz ({seconds:.0f} s), full",
        "sequence, no cuts. Domain randomization is off in this video, so the",
        "policy runs under the same condition the numbers below were measured in.",
        "",
        PIPELINE,
        "",
    ]
    if m:
        lines += [
            f"Measured over {m['num_envs']} rollouts, domain randomization off,",
            "following the protocol in Retargeting Matters (arXiv:2510.02252)",
            f"- Completion rate: {m['success_rate'] * 100:.0f}%",
        ]
        # 세 지표는 완주한 롤아웃에서만 계산된다. 완주가 없으면 NaN이 되므로
        # 그대로 쓰면 "nan mm"이 그대로 찍힌다. 그때는 실제로 잰 값을 적는다.
        finished = m["success_rate"] > 0 and m["e_g_mpbpe_mm"] == m["e_g_mpbpe_mm"]
        if finished:
            lines += [
                f"- E_g-mpbpe: {m['e_g_mpbpe_mm']:.0f} mm, body position error in global coordinates",
                f"- E_mpbpe: {m['e_mpbpe_mm']:.0f} mm, body position error after aligning on the anchor",
                f"- E_mpjpe: {m['e_mpjpe_rad']:.3f} rad, joint angle error",
            ]
        else:
            alive = m.get("mean_alive_frames", 0.0)
            total = m.get("motion_frames", 0) or 1
            lines += [
                "",
                "No rollout reached the end of the clip, so the three errors that are",
                "defined over completed rollouts have no value here. What was measured:",
                f"- Mean tracked length: {alive / total * 100:.0f}% of the clip"
                f" ({alive:,.0f} of {total:,} frames)",
                f"- E_g-mpbpe over all rollouts: {m.get('e_g_mpbpe_mm_all', float('nan')):.0f} mm",
            ]
    return "\n".join(lines)


def progression_note(seq, m, frames, seconds):
    return "\n".join([
        f"Watching a Humanoid Learn: 1k to 30k Iterations on {seq} (Unitree G1)",
        "",
        "Six panels, one motion, one training run.",
        "",
        "Five checkpoints of the same policy, plus the reference it was trained to",
        "follow. Every panel starts from the same motion frame under the same camera",
        "and lighting, so the differences are the policy, not the setup. The plot",
        "below the panels is the tracking error over training.",
        "",
        "Top row     1,000 / 5,000 / 10,000 iterations",
        "Bottom row  20,000 / 30,000 iterations / retargeting reference",
        "",
        f"Motion: LAFAN1 {seq}, {frames:,} frames at 50 Hz ({seconds:.0f} s).",
        "Domain randomization is off in every panel.",
        "",
        PIPELINE,
    ])


def randomization_note(seq, frames, seconds):
    return "\n".join([
        f"What Domain Randomization Does to a Humanoid Policy ({seq}, Unitree G1)",
        "",
        "The same policy as the side-by-side video, this time with domain",
        "randomization left on. The red arrow marks every push: its direction and",
        "length follow the velocity that is written into the torso.",
        "",
        "What is randomized",
        "- A push every 1 to 3 s: linear velocity up to 0.5 m/s in x and y, 0.2 m/s",
        "  in z, angular velocity up to 0.52 rad/s in roll and pitch, 0.78 rad/s in yaw",
        "- Ground friction: static 0.3 to 1.6, dynamic 0.3 to 1.2, restitution 0 to 0.5",
        "- Torso centre of mass: 2.5 cm in x, 5 cm in y and z",
        "- Default joint positions, standing in for joint calibration error",
        "- Reset state: torso pose, torso velocity and joint angles are perturbed",
        "",
        "These are the terms BeyondMimic trains with (arXiv:2508.08241, Sec. III-G).",
        "When the policy loses the reference the episode ends and restarts from frame",
        "zero, so the two panels fall out of phase. That is the failure, not a render",
        "artifact.",
        "",
        f"Motion: LAFAN1 {seq}, {frames:,} frames at 50 Hz ({seconds:.0f} s).",
        "",
        PIPELINE,
    ])


def main(seq, base):
    base = pathlib.Path(base) / seq
    m = metrics(seq)
    frames = m["motion_frames"] if m else 0
    seconds = frames / 50.0
    for sub, text in (
        ("compare", compare_note(seq, m, frames, seconds)),
        ("progression", progression_note(seq, m, frames, seconds)),
        ("randomization", randomization_note(seq, frames, seconds)),
    ):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
        # 드라이브에서는 시퀀스 폴더 하나에 셋이 나란히 올라가므로 이름을 구분해 둔다.
        p = d / f"{seq}_{sub}_youtube.txt"
        p.write_text(text + "\n")
        print("작성:", p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")
