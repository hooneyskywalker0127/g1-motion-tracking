#!/usr/bin/env python3
"""BeyondMimic 정책을 MuJoCo 에서 돌린다. 구조는 표준 sim2sim 을 그대로 따른다.

참고한 구현 (넷이 같은 구조다)
  Humanoid-Gym    humanoid/scripts/sim2sim.py
  unitree_rl_gym  deploy/deploy_mujoco/deploy_mujoco.py
  GMT             sim2sim.py                     ← 이 파일이 가장 가깝다
  HumanTracker    src/humantracker/eval/core/mj_sim.py

공통 골격
  물리 1000 Hz, 제어 50 Hz (decimation 20)
  관측을 스크립트가 MuJoCo 상태로부터 만든다
  PD 를 스크립트가 계산하고 torque_limits 로 클립한 뒤 data.ctrl 에 넣는다
  레퍼런스는 curr_timestep 으로 인덱싱한다
  카메라는 로봇을 따라가고, 제어 스텝마다 한 프레임씩 50 fps 로 녹화한다

BeyondMimic 쪽에서 오는 것
  게인·기본자세·행동 스케일   ONNX 메타데이터 (joint_stiffness / joint_damping /
                              default_joint_pos / action_scale)
  레퍼런스                    ONNX 에 time_step 을 넣으면 돌려준다
                              (joint_pos, joint_vel, body_pos_w, body_quat_w, ...)
  관측 배치 160               tracking_env_cfg.py ObservationsCfg 순서
    0: 29  레퍼런스 관절 위치      29: 58  레퍼런스 관절 속도
   58: 61  motion_anchor_pos_b     61: 67  motion_anchor_ori_b (회전행렬 앞 두 열)
   67: 70  base_lin_vel            70: 73  base_ang_vel
   73:102  joint_pos_rel          102:131  joint_vel_rel
  131:160  이전 행동

토크 제한은 표준값을 쓴다. Isaac 의 effort_limit_sim, GMT 의 torque_limits,
HumanTracker 의 GMT_TORQUE_LIMIT 이 모두 같은 값이다.

    python3 src/sim2sim.py <시퀀스> [--seconds N] [--video out.mp4]
"""
from __future__ import annotations

import argparse
import pathlib

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent
POLICY_DIR = pathlib.Path("/home/sehoon/colcon_ws/policies")
# 오버레이 사본을 쓴다. 원본과 물리는 같고 <visual><global offwidth=...> 만 더해
# 오프스크린 렌더 해상도를 키웠다. 기본 640x480 으로는 영상이 너무 작다.
MJCF = "/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"

SIM_DT = 0.001
SIM_DECIMATION = 20          # 제어 50 Hz
CONTROL_DT = SIM_DT * SIM_DECIMATION

# effort_limit_sim (whole_body_tracking/robots/g1.py) = GMT torque_limits
# = HumanTracker GMT_TORQUE_LIMIT. 관절 이름으로 붙인다.
TORQUE_LIMIT = {"hip_yaw": 88.0, "hip_roll": 139.0, "hip_pitch": 88.0,
                "knee": 139.0, "ankle": 50.0, "waist_yaw": 88.0,
                "waist_roll": 50.0, "waist_pitch": 50.0,
                "shoulder": 25.0, "elbow": 25.0, "wrist": 5.0}


def limit_of(name: str) -> float:
    for key, value in TORQUE_LIMIT.items():
        if key in name:
            return value
    raise KeyError(name)


class Sim2Sim:
    def __init__(self, seq: str, headless: bool = True):
        path = POLICY_DIR / f"{seq}.onnx"
        md = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.joint_names = md["joint_names"].split(",")
        self.body_names = md["body_names"].split(",")
        self.anchor = self.body_names.index(md["anchor_body_name"])
        arr = lambda k: np.array([float(x) for x in md[k].split(",")])
        self.default_dof_pos = arr("default_joint_pos")
        self.stiffness = arr("joint_stiffness")
        self.damping = arr("joint_damping")
        self.action_scale = arr("action_scale")
        self.torque_limits = np.array([limit_of(n) for n in self.joint_names])
        self.num_dofs = len(self.joint_names)

        self.model = mujoco.MjModel.from_xml_path(MJCF)
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        # ONNX 관절 순서 → MuJoCo 주소
        self.qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names]
        self.vadr = [self.model.jnt_dofadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names]
        self.bid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                    for n in self.body_names]
        self.headless = headless
        self.last_action = np.zeros(self.num_dofs, dtype=np.float32)

    # --- 레퍼런스 -----------------------------------------------------------
    def reference(self, step: int):
        out = self.sess.run(
            ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w"],
            {"obs": np.zeros((1, 160), np.float32),
             "time_step": np.array([[step]], np.float32)})
        return [o[0] for o in out]

    def policy(self, obs: np.ndarray, step: int) -> np.ndarray:
        out = self.sess.run(["actions"],
                            {"obs": obs.astype(np.float32)[None],
                             "time_step": np.array([[step]], np.float32)})
        return out[0][0]

    # --- 초기 상태 ----------------------------------------------------------
    def reset_to_reference(self, step: int):
        """표준 평가와 같이 로봇을 레퍼런스 상태에 놓고 시작한다.

        Isaac 쪽 MotionCommand._resample_command 가 하는 것과 같다
        (commands.py:269-277 이 관절과 루트 상태를 레퍼런스로 써넣는다).
        """
        jp, jv, bpos, bquat = self.reference(step)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[:3] = bpos[0]          # body_names[0] 는 pelvis
        self.data.qpos[3:7] = bquat[0]
        self.data.qpos[self.qadr] = jp
        self.data.qvel[self.vadr] = jv
        mujoco.mj_forward(self.model, self.data)

    # --- 관측 ---------------------------------------------------------------
    def observation(self, step: int) -> np.ndarray:
        jp, jv, bpos, bquat = self.reference(step)
        q = self.data.qpos[self.qadr]
        dq = self.data.qvel[self.vadr]

        # 로봇 앵커(torso) 자세
        rob_pos = self.data.xpos[self.bid[self.anchor]]
        rob_mat = self.data.xmat[self.bid[self.anchor]].reshape(3, 3)
        # 레퍼런스 앵커 자세
        ref_pos = bpos[self.anchor]
        w, x, y, z = bquat[self.anchor]
        ref_mat = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

        # 레퍼런스 앵커를 로봇 앵커 프레임에서 본 것 (observations.py motion_anchor_*)
        anchor_pos_b = rob_mat.T @ (ref_pos - rob_pos)
        rel_mat = rob_mat.T @ ref_mat
        # observations.py:82-83 은 mat[..., :2].reshape(N, -1) 이다.
        # 마지막 축에서 앞 둘을 취하므로 각 행의 앞 두 성분이 순서대로 들어간다.
        anchor_ori_b = rel_mat[:, :2].reshape(-1)

        base_mat = self.data.xmat[self.bid[0]].reshape(3, 3)
        base_lin_vel = base_mat.T @ self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]            # MuJoCo free joint 는 이미 body-local

        return np.concatenate([
            jp, jv, anchor_pos_b, anchor_ori_b,
            base_lin_vel, base_ang_vel,
            q - self.default_dof_pos, dq, self.last_action,
        ])

    # --- 실행 ---------------------------------------------------------------
    def run(self, start_step: int, steps: int, video: str | None, fps: int = 50):
        self.reset_to_reference(start_step)
        writer = renderer = None
        if video:
            import imageio
            writer = imageio.get_writer(video, fps=fps, macro_block_size=1)
            renderer = mujoco.Renderer(self.model, height=720, width=960)

        log = {"ref_body_pos_w": [], "rob_body_pos_w": [],
               "ref_joint_pos": [], "rob_joint_pos": [], "step": []}
        pd_target = self.default_dof_pos.copy()

        for i in range(steps * SIM_DECIMATION):
            if i % SIM_DECIMATION == 0:
                t = start_step + i // SIM_DECIMATION
                obs = self.observation(t)
                action = self.policy(obs, t)
                self.last_action = action.copy()
                pd_target = self.default_dof_pos + self.action_scale * action

                jp, _, bpos, _ = self.reference(t)
                log["step"].append(t)
                log["ref_body_pos_w"].append(bpos.copy())
                log["rob_body_pos_w"].append(self.data.xpos[self.bid].copy())
                log["ref_joint_pos"].append(jp.copy())
                log["rob_joint_pos"].append(self.data.qpos[self.qadr].copy())

                if writer is not None:
                    cam = mujoco.MjvCamera()
                    mujoco.mjv_defaultFreeCamera(self.model, cam)
                    cam.lookat[:] = self.data.qpos[:3]   # GMT sim2sim.py 와 같다
                    cam.distance = 3.0
                    cam.azimuth = 135
                    cam.elevation = -20
                    renderer.update_scene(self.data, camera=cam)
                    writer.append_data(renderer.render())

            q = self.data.qpos[self.qadr]
            dq = self.data.qvel[self.vadr]
            torque = (pd_target - q) * self.stiffness - dq * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.qfrc_applied[self.vadr] = torque   # MJCF 에 액추에이터가 없다
            mujoco.mj_step(self.model, self.data)

        if writer is not None:
            writer.close()
            renderer.close()
        return {k: np.asarray(v) for k, v in log.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--start_step", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sim = Sim2Sim(a.seq)
    log = sim.run(a.start_step, int(a.seconds / CONTROL_DT), a.video)

    rel = np.linalg.norm(log["ref_body_pos_w"] - log["rob_body_pos_w"], axis=-1).mean()
    jnt = np.abs(log["ref_joint_pos"] - log["rob_joint_pos"]).mean()
    print(f"{a.seq}  {len(log['step'])} 스텝  "
          f"MPKPE {rel*1000:.1f} mm  MPJPE {jnt:.3f} rad")
    if a.out:
        np.savez_compressed(a.out, **log)
        print("저장", a.out)
