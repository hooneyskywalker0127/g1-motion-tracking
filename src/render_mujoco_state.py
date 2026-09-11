#!/usr/bin/env python3
"""MuJoCo sim2sim 롤아웃을 기록된 state.npz 에서 오프스크린으로 다시 렌더한다.

기존 영상은 카메라가 고정이라 로봇이 걸어 나가면 화면에서 사라졌다.
여기서는 Isaac 쪽처럼 카메라가 로봇을 따라간다 (mjCAMERA_TRACKING, pelvis 추적).

state.npz 에는 루트 자세가 직접 들어 있지 않다. body 위치 14개로부터
Kabsch 정렬로 루트 회전을 복원한다 (검증 결과 body 위치 재현 오차 0.2mm).

  미리보기 한 장     --preview
  영상 한 편         (옵션 없이)
"""
import argparse
import pathlib

import imageio.v2 as imageio
import mujoco
import numpy as np

XML = "/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
ROOT = pathlib.Path("/home/sehoon/Desktop/참고/할일/26/09/g1_motion_tracking")


def kabsch(P, Q):
    """로컬 좌표 P 를 월드 좌표 Q 로 보내는 최적 회전."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def build_qpos(model, data, npz, idx):
    """기록된 관절각과 body 위치에서, 실제로 그릴 프레임의 qpos 만 복원한다."""
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(b))
                for b in npz["body_names"]]
    q, body = npz["q"], npz["body"]
    out = np.zeros((len(idx), model.nq))
    quat = np.zeros(4)
    for n, k in enumerate(idx):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        data.qpos[7:] = q[k]
        mujoco.mj_forward(model, data)
        local = data.xpos[body_ids] - data.xpos[body_ids[0]]
        mujoco.mju_mat2Quat(quat, kabsch(local, body[k] - body[k, 0]).flatten())
        out[n, 0:3] = body[k, 0]
        out[n, 3:7] = quat
        out[n, 7:] = q[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--distance", type=float, default=3.5)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 이면 전체")
    ap.add_argument("--preview", action="store_true", help="가운데 한 프레임만 png 로")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sim2sim = ROOT / args.seq / "sim2sim"
    npz = np.load(sim2sim / f"{args.seq}_state.npz")
    model = mujoco.MjModel.from_xml_path(XML)
    # MJCF 기본 오프스크린 버퍼가 작아서 그대로면 Renderer 가 거부한다.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)

    t = npz["t"]
    dt = float(np.median(np.diff(t)))
    step = max(1, int(round(1.0 / args.fps / dt)))
    last = len(t) if args.seconds <= 0 else min(len(t), int(args.seconds / dt))
    idx = np.arange(0, last, step)

    qpos = build_qpos(model, data, npz, idx)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    def frame(n):
        data.qpos[:] = qpos[n]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        return renderer.render()

    if args.preview:
        out = args.out or str(sim2sim / f"{args.seq}_preview.png")
        imageio.imwrite(out, frame(len(idx) // 2))
        print(f"미리보기 {out}")
        return

    out = args.out or str(sim2sim / f"{args.seq}_mujoco_track.mp4")
    with imageio.get_writer(out, fps=args.fps, codec="libx264",
                            quality=7, macro_block_size=2) as w:
        for n in range(len(idx)):
            w.append_data(frame(n))
            if n % 300 == 0:
                print(f"  {n}/{len(idx)}", flush=True)
    print(f"완료 {out}  ({len(idx)} 프레임, {len(idx)/args.fps:.1f}초)")


if __name__ == "__main__":
    main()
