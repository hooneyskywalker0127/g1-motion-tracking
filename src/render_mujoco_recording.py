#!/usr/bin/env python3
"""기록한 sim-to-sim 상태를 MuJoCo 렌더러로 그린다.

물리를 다시 돌리지 않는다. 공식 컨트롤러(`motion_tracking_controller`)가 실제로
만들어 낸 상태를 그대로 그린다.

  관절각   기록한 관측의 joint_pos_rel + default_joint_pos (측정값)
  베이스   /odom (컨트롤러가 쓰는 것과 같은 상태추정기 값)

베이스가 추정기 값이라는 점이 중요하다. 완주하는 시퀀스에서는 순기구학과
0.017 m 안쪽으로 일치하므로 충실하지만, 부양 구간이 많은 시퀀스에서는 추정기가
튀어 화면에서도 튄다. 그건 실제로 정책이 본 값이다.

카메라는 오버레이 MJCF 의 `track`(mode="trackcom")을 쓴다. Isaac 쪽이
`viewer.origin_type = "asset_root"` 로 로봇을 따라가는 것과 같은 개념이다
(tracking_env_cfg.py:319-322).

    python3 src/render_mujoco_recording.py <시퀀스> [--dir policy_io] [--fps 30]
"""
import argparse
import pathlib
import subprocess

import mujoco as mj
import numpy as np
import onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
MJCF = "/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
CTRL_HZ = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--dir", default="policy_io")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--camera", default="track")
    a = ap.parse_args()

    d = np.load(ROOT / "outputs" / a.dir / f"{a.seq}.npz", allow_pickle=True)
    io, io_t, od, od_t = d["io"], d["io_t"], d["od"], d["od_t"]
    md = {e.key: e.value for e in
          onnx.load(f"/home/sehoon/colcon_ws/policies/{a.seq}.onnx").metadata_props}
    qdef = np.array([float(x) for x in md["default_joint_pos"].split(",")])
    jnames = md["joint_names"].split(",")

    m = mj.MjModel.from_xml_path(MJCF)
    data = mj.MjData(m)
    qadr = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)] for n in jnames]

    base = np.stack([np.interp(io_t, od_t, od[:, i]) for i in range(7)], axis=1)
    step = max(int(round(CTRL_HZ / a.fps)), 1)
    end = len(io) if a.seconds is None else min(len(io), int(a.seconds * CTRL_HZ))

    out = a.out or str(ROOT / "outputs" / "videos_mujoco" / f"{a.seq}_mujoco.mp4")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE)

    with mj.Renderer(m, height=a.height, width=a.width) as r:
        for i in range(0, end, step):
            q = base[i, 3:7]
            n = np.linalg.norm(q)
            data.qpos[:3] = base[i, :3]
            data.qpos[3:7] = q / n if n > 1e-9 else np.array([1.0, 0, 0, 0])
            data.qpos[qadr] = io[i, 73:102] + qdef
            mj.mj_forward(m, data)
            r.update_scene(data, camera=a.camera)
            ff.stdin.write(r.render().tobytes())

    ff.stdin.close()
    ff.wait()
    print(f"저장 {out}  {end/CTRL_HZ:.0f}초  카메라 {a.camera}")


if __name__ == "__main__":
    main()
