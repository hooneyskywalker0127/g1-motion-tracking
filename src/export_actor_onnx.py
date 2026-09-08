"""내보낸 policy.onnx에서 액터만 남긴다.

    python src/export_actor_onnx.py <입력.onnx> <출력.onnx>
    python src/export_actor_onnx.py --run <학습 폴더> <출력.onnx>

학습이 내보내는 policy.onnx에는 레퍼런스 모션 전체가 Constant 노드로 들어간다.
time_step 입력으로 그 모션을 조회해 joint_pos, body_pos_w 같은 출력을 함께 내주기
때문이다. LAFAN1은 CC BY-NC-ND라 2차 저작물을 배포할 수 없으므로, 공개하기 전에
obs -> actions 경로만 남긴다.

계산은 건드리지 않는다. 잘라낸 그래프는 같은 입력에 같은 액션을 낸다.
"""

import argparse
import pathlib

import numpy as np
import onnx


def strip(src, dst, checks=5):
    onnx.utils.extract_model(str(src), str(dst), ["obs"], ["actions"])

    # 모션 배열이 남아 있으면 안 된다. 프레임 축이 있는 상수를 찾는다.
    left = [
        list(a.t.dims)
        for n in onnx.load(dst).graph.node
        for a in n.attribute
        if a.type == onnx.AttributeProto.TENSOR and len(a.t.dims) > 1 and a.t.dims[0] > 1000
    ]
    if left:
        raise SystemExit(f"모션 상수가 남았다: {left}")

    # 액션이 그대로인지 대조한다.
    import onnxruntime as ort

    a, b = ort.InferenceSession(str(src)), ort.InferenceSession(str(dst))
    dim = a.get_inputs()[0].shape[1]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(checks):
        obs = rng.standard_normal((1, dim)).astype(np.float32)
        ra = a.run(["actions"], {"obs": obs, "time_step": np.zeros((1, 1), np.float32)})[0]
        rb = b.run(["actions"], {"obs": obs})[0]
        worst = max(worst, float(np.abs(ra - rb).max()))
    if worst != 0.0:
        raise SystemExit(f"액션이 달라졌다: 최대 차이 {worst}")

    print(f"{src.stat().st_size / 1e6:.1f}MB -> {dst.stat().st_size / 1e6:.2f}MB · 차이 0")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", type=pathlib.Path, nargs="?")
    p.add_argument("dst", type=pathlib.Path)
    p.add_argument("--run", type=pathlib.Path, help="학습 폴더. exported/policy.onnx를 쓴다.")
    a = p.parse_args()
    src = a.run / "exported" / "policy.onnx" if a.run else a.src
    if src is None:
        p.error("입력 onnx나 --run 중 하나가 필요하다")
    strip(src, a.dst)
