#!/usr/bin/env python3
"""증류한 학생 정책 하나를 MuJoCo 에서 돌린다. 교사용 src/sim2sim.py 를 상속한다.

    python3 src/sim2sim_student.py <시퀀스> --onnx student.onnx \
        [--seconds N] [--video out.mp4] [--out result.npz]

교사와 다른 점 셋. 나머지(물리 1000Hz, 제어 50Hz, PD 계산, 토크 제한, 카메라)는
그대로 쓴다.

1. 정책이 하나다. 교사는 시퀀스마다 policies/<seq>.onnx 를 골랐다.
   학생은 14클립을 한 망에 담았으므로 --onnx 로 하나만 받는다.

2. 관측이 260 이다. 앞 160 은 교사와 순서까지 같고 뒤 100 이 더 붙는다
   (distill_env_cfg.StudentCfg).

     160:189  motion_joint_pos_diff   레퍼런스 관절각 - 현재 관절각
     189:218  motion_joint_vel_diff   레퍼런스 관절속도 - 현재 관절속도
     218:260  motion_body_pos_diff_b  앵커 프레임에서 레퍼런스 - 로봇 바디 위치 (14x3)

   그리고 속도 항에 vel_scale(0.05)을 곱한다. 곱하는 자리는
   env_adapter.HoverEnvAdapter._velocity_scale 과 같다 —
   command 의 뒤 절반, base_lin_vel, base_ang_vel, joint_vel, motion_joint_vel_diff.

3. 레퍼런스를 npz 에서 읽는다. 학생 ONNX 에는 모션을 굽지 않았다
   (클립이 14개이고, 학습에 없던 클립도 넣어 볼 것이기 때문이다).
   교사용의 --motion_npz 경로는 ONNX 가 body_pos_w 를 돌려줘야 바디 매핑을
   찾을 수 있었다. 학생 ONNX 는 그걸 안 내므로, 매핑을 교사 ONNX 로 한 번
   구해서 쓴다(로봇이 같으므로 같은 매핑이다).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np
import onnx
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim2sim import SIM_DECIMATION, Sim2Sim  # noqa: E402

MOTION_ROOT = pathlib.Path("/home/sehoon/motions_fixed")
META_TEACHER = pathlib.Path("/home/sehoon/colcon_ws/policies/walk4_subject1.onnx")


class Sim2SimStudent(Sim2Sim):
    def __init__(self, seq: str, onnx_path: str, headless: bool = True):
        # 부모는 policies/<seq>.onnx 를 찾는다. 바디 매핑을 구하는 데에만 쓰고
        # 정책은 곧바로 학생 것으로 바꾼다.
        super().__init__(seq, headless=headless, motion_npz=str(self._npz(seq)))

        self.student = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        md = {e.key: e.value for e in onnx.load(onnx_path).metadata_props}
        self.num_obs = int(md.get("num_obs", 260))
        self.vel_scale = float(md.get("vel_scale", 1.0))
        self.scale_vec = self._velocity_scale()
        print(f"[student] 관측 {self.num_obs}  vel_scale {self.vel_scale}  {onnx_path}")

    @staticmethod
    def _npz(seq: str) -> pathlib.Path:
        p = MOTION_ROOT / f"{seq}.npz"
        if not p.is_file():
            p = pathlib.Path(f"/home/sehoon/motions/{seq}.npz")
        assert p.is_file(), f"npz 를 못 찾았다: {seq}"
        return p

    def _velocity_scale(self) -> np.ndarray:
        """속도 항만 vel_scale, 나머지는 1. env_adapter._velocity_scale 과 같은 자리."""
        n = self.num_dofs
        s = np.ones(self.num_obs, dtype=np.float32)
        s[n : 2 * n] = self.vel_scale                      # command 뒤 절반 (레퍼런스 관절속도)
        s[2 * n + 9 : 2 * n + 9 + 3] = self.vel_scale      # base_lin_vel
        s[2 * n + 12 : 2 * n + 15] = self.vel_scale        # base_ang_vel
        s[2 * n + 15 + n : 2 * n + 15 + 2 * n] = self.vel_scale   # joint_vel
        s[160 + n : 160 + 2 * n] = self.vel_scale          # motion_joint_vel_diff
        return s

    def observation(self, step: int) -> np.ndarray:
        """교사 160 + 학생 전용 100."""
        base = super().observation(step)                   # 160, 스케일 안 걸린 값
        jp, jv, bpos, bquat = self.reference(step)
        q = self.data.qpos[self.qadr]
        dq = self.data.qvel[self.vadr]

        # 앵커(torso) 프레임. observations.motion_body_pos_diff_b 와 같은 정의다 —
        # 레퍼런스와 로봇 바디를 둘 다 로봇 앵커 프레임으로 옮긴 뒤 뺀다.
        rob_pos = self.data.xpos[self.bid[self.anchor]]
        rob_mat = self.data.xmat[self.bid[self.anchor]].reshape(3, 3)
        rob_bodies = self.data.xpos[self.bid]              # (14,3)
        ref_b = (bpos - rob_pos) @ rob_mat                 # rob_mat.T @ v 를 행벡터로
        rob_b = (rob_bodies - rob_pos) @ rob_mat
        body_diff_b = (ref_b - rob_b).reshape(-1)          # 42

        obs = np.concatenate([base, jp - q, jv - dq, body_diff_b]).astype(np.float32)
        assert obs.shape[0] == self.num_obs, f"관측 길이가 {obs.shape[0]} 이다"
        return obs * self.scale_vec

    def policy(self, obs: np.ndarray, step: int) -> np.ndarray:
        return self.student.run(["actions"], {"obs": obs.astype(np.float32)[None]})[0][0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("seq")
    p.add_argument("--onnx", required=True)
    p.add_argument("--seconds", type=float, default=0.0, help="0 이면 클립 전체")
    p.add_argument("--start_step", type=int, default=0)
    p.add_argument("--video", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    sim = Sim2SimStudent(a.seq, a.onnx)
    total = len(sim.ref_npz[0])
    steps = total - a.start_step if a.seconds <= 0 else int(a.seconds * 50)
    steps = min(steps, total - a.start_step)
    print(f"[student] {a.seq}  {steps} 스텝 ({steps / 50:.0f}초)")
    log = sim.run(a.start_step, steps, a.video)
    if a.out:
        np.savez_compressed(a.out, **{k: np.asarray(v) for k, v in log.items()})
        print(f"[student] 기록 {a.out}")


if __name__ == "__main__":
    main()
