"""C++ 루프가 읽을 상수와 레퍼런스를 평평한 바이너리로 뽑는다.

C++ 쪽에서 npz(zip) 와 ONNX 메타데이터를 파싱하지 않으려고 한 번만 내보낸다.
값의 출처는 전부 src/sim2sim.py 와 같다.
"""
import json, pathlib, sys
import numpy as np, onnx

SEQ = sys.argv[1] if len(sys.argv) > 1 else "walk4_subject1"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".")
POL = pathlib.Path("/home/sehoon/colcon_ws/policies") / f"{SEQ}.onnx"
NPZ = pathlib.Path("/home/sehoon/motions_fixed") / f"{SEQ}.npz"

md = {e.key: e.value for e in onnx.load(str(POL)).metadata_props}
arr = lambda k: [float(x) for x in md[k].split(",")]
jn = md["joint_names"].split(",")
bn = md["body_names"].split(",")

LIM = {"hip_yaw": 88.0, "hip_roll": 139.0, "hip_pitch": 88.0, "knee": 139.0,
       "ankle": 50.0, "waist_yaw": 88.0, "waist_roll": 50.0, "waist_pitch": 50.0,
       "shoulder": 25.0, "elbow": 25.0, "wrist": 5.0}
lim = [next(v for k, v in LIM.items() if k in n) for n in jn]

m = np.load(NPZ)
fps = float(np.asarray(m["fps"]).reshape(-1)[0])
jp, jv = m["joint_pos"], m["joint_vel"]
bp, bq = m["body_pos_w"], m["body_quat_w"]
# npz 의 바디 30개 중 정책이 쓰는 14개를 고른다. 이름 순서는 ONNX 가 갖고 있다.
names = [str(x) for x in m["body_names"]] if "body_names" in m else None
idx = [names.index(n) for n in bn] if names else list(range(len(bn)))

meta = dict(seq=SEQ, fps=fps, frames=int(jp.shape[0]),
            joint_names=jn, body_names=bn,
            anchor=bn.index(md["anchor_body_name"]),
            default_joint_pos=arr("default_joint_pos"),
            stiffness=arr("joint_stiffness"), damping=arr("joint_damping"),
            action_scale=arr("action_scale"), torque_limit=lim,
            num_dofs=len(jn), num_bodies=len(bn))
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))

def w(name, a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.tofile(OUT / name)
    print(f"  {name} {a.shape}")

w("ref_jp.bin", jp)
w("ref_jv.bin", jv)
w("ref_bp.bin", bp[:, idx, :])
w("ref_bq.bin", bq[:, idx, :])
print(f"{SEQ}: {meta['frames']} 프레임, {meta['num_dofs']} dof, {meta['num_bodies']} 바디, 앵커 {meta['anchor']}")
