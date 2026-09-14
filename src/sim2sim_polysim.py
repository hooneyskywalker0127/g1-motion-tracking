"""PolySim(arXiv:2510.01708) 의 5개 지표로 sim2sim 로그를 채점한다.

발표된 표와 나란히 놓으려고 정의를 그대로 따른다.
  Success   전역 바디 위치 오차 평균이 한 번이라도 0.5 m 를 넘으면 실패
  E_g-mpjpe 전역 바디 위치 오차 (mm)
  E_mpjpe   루트 상대 바디 위치 오차 (mm)
  E_acc     바디 가속도 오차 (mm/frame^2)
  E_vel     루트 속도 오차 (mm/frame)
frame 은 로그 주기 50 Hz 이다.
"""
import argparse, glob, os, numpy as np

FAIL_M = 0.5

def score(d):
    ref, rob = d["ref_body_pos_w"], d["rob_body_pos_w"]   # (T, B, 3)
    n = min(len(ref), len(rob)); ref, rob = ref[:n], rob[:n]

    dist = np.linalg.norm(ref - rob, axis=-1)             # (T, B) m
    per_t = dist.mean(axis=1)
    fail_at = int(np.argmax(per_t > FAIL_M)) if (per_t > FAIL_M).any() else -1

    ref_r = ref - ref[:, :1]; rob_r = rob - rob[:, :1]
    dist_r = np.linalg.norm(ref_r - rob_r, axis=-1)

    a_ref = np.diff(ref, 2, axis=0); a_rob = np.diff(rob, 2, axis=0)
    e_acc = np.linalg.norm(a_ref - a_rob, axis=-1).mean()

    v_ref = np.diff(ref[:, 0], axis=0); v_rob = np.diff(rob[:, 0], axis=0)
    e_vel = np.linalg.norm(v_ref - v_rob, axis=-1).mean()

    return dict(frames=n, success=fail_at < 0, fail_at=fail_at,
                over_frac=float((per_t > FAIL_M).mean()),
                final_err=float(per_t[-1]),
                e_g_mpjpe=dist.mean() * 1000, e_mpjpe=dist_r.mean() * 1000,
                e_acc=e_acc * 1000, e_vel=e_vel * 1000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="outputs/sim2sim_fixed")
    ap.add_argument("--frames", type=int, default=0,
                    help="앞에서 이만큼만 채점한다 (0 이면 전체)")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "*.npz"))):
        d = dict(np.load(f))
        if a.frames:
            d = {k: v[:a.frames] for k, v in d.items()}
        r = score(d); r["seq"] = os.path.basename(f)[:-4]; rows.append(r)

    print(f"{'시퀀스':<22}{'성공':>5}{'초과시간':>9}{'E_g-mpjpe':>11}{'E_mpjpe':>10}"
          f"{'E_acc':>9}{'E_vel':>9}{'프레임':>8}")
    for r in rows:
        print(f"{r['seq']:<22}{'O' if r['success'] else 'X':>5}"
              f"{r['over_frac']:>9.1%}"
              f"{r['e_g_mpjpe']:>11.1f}{r['e_mpjpe']:>10.1f}"
              f"{r['e_acc']:>9.3f}{r['e_vel']:>9.3f}{r['frames']:>8}")
    if rows:
        ok = [r for r in rows if r["success"]]
        print(f"\n{len(rows)}개 · 성공률 {len(ok)/len(rows):.3f}")
        if ok:
            print(f"성공한 것만 평균  E_g-mpjpe {np.mean([r['e_g_mpjpe'] for r in ok]):.1f} mm"
                  f" · E_mpjpe {np.mean([r['e_mpjpe'] for r in ok]):.1f} mm"
                  f" · E_acc {np.mean([r['e_acc'] for r in ok]):.3f}"
                  f" · E_vel {np.mean([r['e_vel'] for r in ok]):.3f}")
        print("\n비교 · PolySim MuJoCo 제로샷(3시뮬 학습)  성공 0.564 · "
              "E_g-mpjpe 151.9 · E_mpjpe 65.6 · E_acc 4.308 · E_vel 9.329")
