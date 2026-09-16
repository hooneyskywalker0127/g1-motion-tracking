"""학습된 정책을 모션 전체에 대해 굴려 완주율과 추적 오차를 잰다.

지표와 성공 판정은 GMR 논문(arXiv:2510.02252)을 따른다. 성공은 앵커 바디의
높이와 방향이 레퍼런스에서 임계 이상 벗어나지 않고 에피소드 끝까지 도달한
경우이고, 오차는 전역 바디 위치(E_g-mpbpe), 루트 기준 상대 바디 위치
(E_mpbpe), 관절 각도(E_mpjpe) 셋이다.

    python scripts/rsl_rl/eval.py --task=Tracking-Flat-G1-v0 \
      --load_run=<run> --checkpoint=model_29999.pt \
      --motion_file=/path/motion.npz --num_envs=100 --out eval.json
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a motion tracking policy.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of rollouts run in parallel.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the motion file.")
parser.add_argument("--out", type=str, default=None, help="Where to write the result json.")
parser.add_argument(
    "--randomize", action="store_true", default=False,
    help="Keep domain randomization on (the paper's sim-dr condition). Off by default.",
)
parser.add_argument(
    "--init_randomize", action="store_true", default=False,
    help=(
        "초기 상태 교란만 켠다. 물성·질량중심·밀치기는 끈 채로 둔다. "
        "MuJoCo 쪽에서 똑같이 재현할 수 있는 것만 켜야 sim-dr 열과 sim2sim 열이 "
        "같은 조건이 된다. --randomize 는 셋을 다 켜므로 이 비교에 쓸 수 없다."
    ),
)
parser.add_argument(
    "--start_standup", action="store_true", default=False,
    help=(
        "레퍼런스 자세가 아니라 배포 때와 같은 기동 자세에서 시작한다. "
        "MuJoCo sim-to-sim 은 로봇을 레퍼런스 상태로 써넣을 수단이 없어 "
        "기동 자세에서 레퍼런스로 올라가는 구간이 앞에 붙는다. 그 프로토콜을 맞춘다. "
        "값은 unitree_description/urdf/g1/ros2_control.xacro 의 initial_pos."
    ),
)
parser.add_argument(
    "--settle_steps", type=int, default=0,
    help=(
        "앞쪽 N 스텝은 종료 판정을 걸지 않는다. --start_standup 과 짝이다. "
        "기동 자세에서 레퍼런스로 수렴하는 구간을 실패로 세지 않기 위한 것으로, "
        "MuJoCo 쪽에서 '추종 획득' 이후부터 판정하는 것과 같은 처리다."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import numpy as np
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def _motion_name(path):
    """W&B artifact로 받은 파일은 이름이 전부 motion.npz라 상위 폴더에서 이름을 딴다."""
    name = os.path.splitext(os.path.basename(path))[0]
    if name == "motion":
        name = os.path.basename(os.path.dirname(path)).split(":")[0]
    return name


def _disable_randomization(env_cfg):
    """논문의 sim 조건. 시작 자세 흔들기, 물성 랜덤화, 밀치기를 모두 끈다."""
    env_cfg.events.physics_material = None
    # 이 이벤트를 통째로 끄면 default_joint_pos_nominal이 만들어지지 않아
    # onnx 메타데이터 export가 깨진다. 범위만 0으로 둔다.
    env_cfg.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)


STANDUP_POS = {
    "left_hip_pitch_joint": -0.1, "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1, "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 1.28,
}
STANDUP_BASE_Z = 0.793  # unitree_description/mjcf/g1.xml 의 pelvis 스폰 높이


def _apply_standup_start(env, command):
    """로봇을 배포 때의 기동 자세로 놓는다. 나열되지 않은 관절은 0."""
    robot = env.unwrapped.scene["robot"]

    jp = torch.zeros_like(robot.data.joint_pos)
    for name, value in STANDUP_POS.items():
        idx = robot.joint_names.index(name)
        jp[:, idx] = value
    jv = torch.zeros_like(robot.data.joint_vel)
    robot.write_joint_state_to_sim(jp, jv)

    root = robot.data.root_state_w.clone()
    root[:, 2] = STANDUP_BASE_Z + env.unwrapped.scene.env_origins[:, 2]
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False

    motion = np.load(args_cli.motion_file)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    total_frames = int(motion["joint_pos"].shape[0])
    # 모션이 끝나기 전에 시간 만료로 끊기지 않도록 에피소드를 클립보다 길게 잡는다.
    env_cfg.episode_length_s = total_frames / fps + 5.0

    if not args_cli.randomize:
        _disable_randomization(env_cfg)
    if args_cli.init_randomize:
        # 초기 교란만 되돌린다. physics_material · base_com · push_robot 은
        # _disable_randomization 이 끈 그대로 둔다.
        env_cfg.commands.motion.pose_range = {
            "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
        }
        env_cfg.commands.motion.velocity_range = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
        }
        env_cfg.commands.motion.joint_position_range = (-0.1, 0.1)
        env_cfg.seed = 0

    # 종료조건을 끄고 끝까지 굴린다. 판정은 강제하지 않고 계산만 한다.
    #
    # 이유 — 켜 두면 두 판정이 비대칭이 된다. PolySim 기준은 "전역 바디 오차 평균이
    # 한 번이라도 0.5 m 를 넘으면 실패" 인데, Isaac 은 넘어지는 순간 에피소드를
    # 끝내 버려서 넘어진 뒤의 전역 오차가 아예 기록되지 않는다. 그래서 실패한
    # 롤아웃이 PolySim 기준으로는 성공으로 잡힌다. MuJoCo 는 종료가 없어 끝까지
    # 굴러가므로 넘어지면 반드시 0.5 m 를 넘는다. 같은 이름의 두 숫자가 다른 것을
    # 재게 된다 (실측 — obstacles2_subject1 이 S_bm 0.00 인데 S_poly 0.28 로 나왔다).
    #
    # PolySim 자신도 조기 종료 없이 굴린 뒤 판정한다. 양쪽 다 끝까지 굴리고
    # 종료조건은 사후에 계산하는 것이 두 기준 모두에 맞는 유일한 방식이다.
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    command = env.unwrapped.command_manager.get_term("motion")

    # 모든 환경이 같은 지점에서 출발해야 완주 여부를 프레임 수로 판정할 수 있다.
    def _fixed_sampling(env_ids, term=command):
        term.time_steps[env_ids] = 0

    command._adaptive_sampling = _fixed_sampling
    env.reset()

    if args_cli.start_standup:
        _apply_standup_start(env, command)

    # 리셋만으로는 커맨드의 상대 좌표가 갱신되지 않는다. IsaacLab은 한 스텝 안에서
    # 종료 판정을 커맨드 갱신보다 먼저 하므로, 갱신 없이 첫 스텝을 밟으면 직전
    # 에피소드의 좌표와 비교되어 전부 종료로 잡힌다. time_steps를 -1로 두고 한 번
    # 갱신시켜 0프레임 기준으로 맞춘다.
    command.time_steps[:] = -1
    env.unwrapped.command_manager.compute(env.unwrapped.step_dt)

    device = env.unwrapped.device
    n = args_cli.num_envs
    # 말단 넷 — 발목·손목. body_names 안의 색인이다 (flat_env_cfg.py 16~30행).
    EE_IDX = [command.cfg.body_names.index(b) for b in
              ("left_ankle_roll_link", "right_ankle_roll_link",
               "left_wrist_yaw_link", "right_wrist_yaw_link")]
    grav_w = env.unwrapped.scene["robot"].data.GRAVITY_VEC_W
    alive = torch.ones(n, dtype=torch.bool, device=device)
    steps = torch.zeros(n, device=device)
    err_g = torch.zeros(n, device=device)
    err_rel = torch.zeros(n, device=device)
    err_jnt = torch.zeros(n, device=device)
    err_jvel = torch.zeros(n, device=device)
    # 관절별 오차도 누적한다. 좌우 비대칭 같은 것을 보려면 평균만으로는 부족하다.
    err_per_joint = torch.zeros(n, 29, device=device)
    dump_ref, dump_rob = [], []   # env 0 의 프레임별 관절각을 그대로 남긴다
    # 프레임별·환경별 다섯 지표. 창 길이를 바꿔가며 다시 셀 수 있어야 한다.
    dump_g, dump_r, dump_j, dump_jv, dump_alive = [], [], [], [], []
    # env 0 의 바디 자세 전체. MuJoCo 채점기와의 대조용이다.
    xref_bp, xref_bq, xrob_bp, xrob_bq, xref_jv, xrob_jv = [], [], [], [], [], []
    # PolySim 기준: 전역 바디 위치 오차 평균이 한 번이라도 0.5 m 를 넘으면 실패.
    # Isaac 종료조건은 수직·자세만 보므로 수평 표류를 놓친다. 같이 잰다.
    ever_far = torch.zeros(n, dtype=torch.bool, device=device)

    obs = env.get_observations()
    for _step in range(total_frames - 1):
        with torch.inference_mode():
            # 스텝이 보상을 현재 레퍼런스로 계산한 뒤에야 모션 프레임을 넘긴다.
            # 그래서 스텝 후에 레퍼런스를 읽으면 로봇을 다음 프레임과 짝짓게 된다.
            # mjlab 의 tracking/scripts/evaluate.py 가 같은 이유로 스텝 전에 스냅샷한다.
            ref_body_pos_w = command.body_pos_w.clone()
            ref_body_pos_relative_w = command.body_pos_relative_w.clone()
            ref_body_quat_w = command.body_quat_w.clone()
            ref_joint_pos = command.joint_pos.clone()
            ref_joint_vel = command.joint_vel.clone()
            # 로봇 상태도 스텝 전에 읽는다. 레퍼런스는 프레임 t 이고
            # body_pos_relative_w 는 직전 _update_command 가 프레임 t 와 상태 s_t 로
            # 재정렬해 둔 값이라, 로봇을 스텝 뒤에 읽으면 s_{t+1} 과 짝지어진다.
            # MuJoCo 쪽은 같은 순간의 ref(t)·rob(t) 를 남긴다 (sim2sim.py 217~226행).
            # 이 반 스텝이 R-MPKPE 를 1.9 mm 어긋나게 했다 (실측, walk1_subject1).
            robot_body_pos = command.robot_body_pos_w.clone()
            robot_body_quat = command.robot_body_quat_w.clone()
            robot_joint_pos = command.robot_joint_pos.clone()
            robot_joint_vel = command.robot_joint_vel.clone()

            # BeyondMimic 종료조건 세 항을 직접 계산한다. 식은 terminations.py 의
            # bad_anchor_pos_z_only · bad_anchor_ori · bad_motion_body_pos_z_only 그대로다.
            bad_pos = (command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]).abs() > 0.25
            gz_ref = math_utils.quat_rotate_inverse(command.anchor_quat_w, grav_w)[:, 2]
            gz_rob = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, grav_w)[:, 2]
            bad_ori = (gz_ref - gz_rob).abs() > 0.8
            bad_ee = ((ref_body_pos_relative_w[:, EE_IDX, -1]
                       - robot_body_pos[:, EE_IDX, -1]).abs() > 0.25).any(dim=-1)
            dones = bad_pos | bad_ori | bad_ee

            obs, _, _, _ = env.step(policy(obs))
            g = torch.norm(ref_body_pos_w - robot_body_pos, dim=-1).mean(dim=-1)
            rel = torch.norm(ref_body_pos_relative_w - robot_body_pos, dim=-1).mean(dim=-1)
            # 관절 오차는 KungfuBot 식 (23) 의 L2 노름으로 낸다. MuJoCo 쪽
            # score_standard.py 와 같은 식이어야 두 열을 맞댈 수 있다.
            # 예전 정의(관절당 평균 절댓값)는 MuJoCo 쪽 RMS 와도 달랐다.
            jnt = torch.norm(ref_joint_pos - robot_joint_pos, dim=-1)
            jvel = torch.norm(ref_joint_vel - robot_joint_vel, dim=-1)

            # 먼저 살아 있는지를 갱신하고 나서 누적한다. score_standard.py 가
            # 첫 위반 프레임을 평균에서 빼므로 (alive = arange < first) 순서를 맞춘다.
            if _step >= args_cli.settle_steps:
                alive &= ~dones
            m = alive.float()
            err_g += g * m
            err_rel += rel * m
            err_jnt += jnt * m
            err_jvel += jvel * m
            err_per_joint += (ref_joint_pos - robot_joint_pos).abs() * m.unsqueeze(-1)
            steps += m
            # PolySim 기준은 롤아웃 전체에서 본다. 살아 있는 구간으로 자르면
            # 넘어진 뒤의 표류가 안 잡혀 MuJoCo 쪽과 다른 것을 재게 된다.
            ever_far |= g > 0.5
            dump_g.append(g.cpu().numpy().astype("float16"))
            dump_r.append(rel.cpu().numpy().astype("float16"))
            dump_j.append(jnt.cpu().numpy().astype("float16"))
            dump_jv.append(jvel.cpu().numpy().astype("float16"))
            dump_alive.append(alive.cpu().numpy().copy())
            dump_ref.append(ref_joint_pos[0].cpu().numpy().copy())
            dump_rob.append(robot_joint_pos[0].cpu().numpy().copy())
            # env 0 만 바디 자세를 통째로 남긴다. MuJoCo 쪽 score_standard.py 로
            # 같은 롤아웃을 채점해 위 온라인 값과 맞는지 대조하기 위한 것이다.
            # 이 대조를 통과하기 전에는 두 열을 같은 표에 올리지 않는다.
            xref_bp.append(ref_body_pos_w[0].cpu().numpy().copy())
            xref_bq.append(ref_body_quat_w[0].cpu().numpy().copy())
            xrob_bp.append(robot_body_pos[0].cpu().numpy().copy())
            xrob_bq.append(robot_body_quat[0].cpu().numpy().copy())
            xref_jv.append(ref_joint_vel[0].cpu().numpy().copy())
            xrob_jv.append(robot_joint_vel[0].cpu().numpy().copy())
        # 다 죽어도 끊지 않는다. PolySim 기준이 롤아웃 끝까지를 보고,
        # MuJoCo 쪽도 종료 없이 전 길이를 굴리기 때문이다.

    import numpy as _np
    base = os.path.splitext(args_cli.out)[0]
    _np.savez(base + "_dump.npz",
              ref_joint_pos=_np.array(dump_ref), rob_joint_pos=_np.array(dump_rob),
              g_t=_np.array(dump_g), r_t=_np.array(dump_r), j_t=_np.array(dump_j),
              jv_t=_np.array(dump_jv), alive_t=_np.array(dump_alive))
    # env 0 의 롤아웃을 MuJoCo 쪽 npz 와 같은 키 이름으로 남긴다.
    # src/score_standard.py 의 score() 를 그대로 먹일 수 있다.
    _np.savez(base + "_env0.npz",
              ref_body_pos_w=_np.array(xref_bp), rob_body_pos_w=_np.array(xrob_bp),
              ref_body_quat_w=_np.array(xref_bq), rob_body_quat_w=_np.array(xrob_bq),
              ref_joint_pos=_np.array(dump_ref), rob_joint_pos=_np.array(dump_rob),
              ref_joint_vel=_np.array(xref_jv), rob_joint_vel=_np.array(xrob_jv))
    ok = alive
    denom = steps.clamp(min=1.0)
    result = dict(
        motion=_motion_name(args_cli.motion_file),
        checkpoint=os.path.basename(resume_path),
        run=os.path.basename(os.path.dirname(resume_path)),
        num_envs=n,
        motion_frames=total_frames,
        randomized=bool(args_cli.randomize),
        start_standup=bool(args_cli.start_standup),
        settle_steps=int(args_cli.settle_steps),
        init_randomized=bool(args_cli.init_randomize),
        # S_bm — BeyondMimic 종료조건을 끝까지 안 건드림
        success_rate=float(ok.float().mean().item()),
        # S_poly — 전역 바디 오차가 한 번도 0.5 m 를 안 넘음. S_bm 과 독립 판정.
        # 예전에는 ok 와 AND 를 걸어 MuJoCo 쪽보다 엄했다. 원문에는 그 조건이 없다.
        success_rate_polysim=float((~ever_far).float().mean().item()),
        # 완주한 롤아웃만 모아 평균낸다. 도중에 무너진 롤아웃의 오차는 섞지 않는다.
        e_g_mpbpe_mm=float((err_g[ok] / denom[ok]).mean().item() * 1000.0) if ok.any() else float("nan"),
        e_mpbpe_mm=float((err_rel[ok] / denom[ok]).mean().item() * 1000.0) if ok.any() else float("nan"),
        e_joint_l2_rad=float((err_jnt[ok] / denom[ok]).mean().item()) if ok.any() else float("nan"),
        e_jointvel_l2=float((err_jvel[ok] / denom[ok]).mean().item()) if ok.any() else float("nan"),
        e_per_joint_rad=(
            (err_per_joint[ok] / denom[ok].unsqueeze(-1)).mean(dim=0).tolist()
            if ok.any() else None),
        # 실패한 것까지 포함한 값. 완주율이 낮을 때 위 숫자만 보면 오해한다.
        e_g_mpbpe_mm_all=float((err_g / denom).mean().item() * 1000.0),
        e_mpbpe_mm_all=float((err_rel / denom).mean().item() * 1000.0),
        e_joint_l2_rad_all=float((err_jnt / denom).mean().item()),
        mean_alive_frames=float(steps.mean().item()),
    )
    print("[RESULT] " + json.dumps(result, ensure_ascii=False))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO]: Wrote {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
