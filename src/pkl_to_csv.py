"""리타게팅 결과 pkl을 BeyondMimic이 받는 Unitree 규격 csv로 바꾼다.

csv 36열 = root_pos(3) + root_quat_xyzw(4) + dof_pos(29).
retarget_all.py가 저장할 때 이미 wxyz를 xyzw로 돌려놨으므로 이어 붙이면 된다.

BeyondMimic의 scripts/csv_to_npz.py가 이 csv를 받아 50 fps npz로 만든다.
"""
import argparse
import glob
import os
import pickle

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="outputs/retarget")
parser.add_argument("--dst", default="outputs/csv")
parser.add_argument("--seq", nargs="*", default=None,
                    help="시퀀스 이름들. 없으면 src의 전부")
args = parser.parse_args()

os.makedirs(args.dst, exist_ok=True)
if args.seq:
    paths = [os.path.join(args.src, f"{s}.pkl") for s in args.seq]
else:
    paths = sorted(glob.glob(os.path.join(args.src, "*.pkl")))

for p in paths:
    seq = os.path.basename(p).replace(".pkl", "")
    d = pickle.load(open(p, "rb"))
    m = np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1)
    assert m.shape[1] == 36, f"{seq}: 열 {m.shape[1]}개, 36이어야 한다"
    out = os.path.join(args.dst, f"{seq}.csv")
    np.savetxt(out, m, delimiter=",", fmt="%.6f")
    print(f"{seq:28s} {m.shape[0]:6d} frames  {d['fps']} fps  -> {out}")
