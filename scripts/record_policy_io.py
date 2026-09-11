#!/usr/bin/env python3
"""공식 sim-to-sim 경로가 돌아가는 동안 컨트롤러가 내보내는 값만 받아 적는다.

물리도 관측도 다시 구현하지 않는다. motion_tracking_controller 가 발행하는
`/walking_controller/policy_io` (관측 160 + 행동 29) 와 로봇 상태를 그대로 저장한다.

관측 앞 29개가 그 스텝의 **레퍼런스 관절 위치**이고, ONNX 에 time_step 을 넣어 나오는
값과 비트 단위로 일치한다(확인함). 그래서 위상을 추정할 필요가 없다.

    python3 scripts/record_policy_io.py <출력.npz> [--seconds N]
"""
import argparse
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

TOPIC = "/walking_controller/policy_io"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=300.0)
    a = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("policy_io_recorder")
    io, js, od = [], [], []
    node.create_subscription(
        Float64MultiArray, TOPIC,
        lambda m: io.append((time.time(), np.asarray(m.data))), 500)
    node.create_subscription(
        JointState, "/joint_states",
        lambda m: js.append((time.time(), list(m.name), np.asarray(m.position))), 500)
    node.create_subscription(
        Odometry, "/odom",
        lambda m: od.append((time.time(), np.array([
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z,
            m.pose.pose.orientation.w, m.pose.pose.orientation.x,
            m.pose.pose.orientation.y, m.pose.pose.orientation.z]))), 500)

    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    rclpy.shutdown()

    if not io:
        raise SystemExit("policy_io 를 한 건도 못 받았다. 컨트롤러가 활성인지 확인할 것.")
    np.savez_compressed(
        a.out,
        io_t=np.array([t for t, _ in io]),
        io=np.stack([v for _, v in io]),
        js_t=np.array([t for t, _, _ in js]) if js else np.zeros(0),
        js=np.stack([v for _, _, v in js]) if js else np.zeros((0, 0)),
        js_names=np.array(js[0][1]) if js else np.zeros(0, dtype=str),
        od_t=np.array([t for t, _ in od]) if od else np.zeros(0),
        od=np.stack([v for _, v in od]) if od else np.zeros((0, 7)),
    )
    print(f"저장 {a.out} · policy_io {len(io)} · joint_states {len(js)} · odom {len(od)}")


if __name__ == "__main__":
    main()
