"""두 표준으로 같은 롤아웃을 채점한다.

BeyondMimic (whole_body_tracking) 종료조건 — tracking_env_cfg.py 255-275
  anchor_pos   |ref_anchor_z - rob_anchor_z| > 0.25
  anchor_ori   |ref_gravity_z - rob_gravity_z| > 0.8
  ee_body_pos  발목·손목 4곳 중 |ref_rel_z - rob_z| > 0.25 가 하나라도
  성공 = 끝까지 셋 다 안 걸림
PolySim
  성공 = 전역 바디 오차 평균이 한 번도 0.5 m 를 안 넘음
mjlab 지표 — tracking/mdp/metrics.py
  MPKPE     전역 바디 위치 오차 평균
  R-MPKPE   레퍼런스를 로봇 앵커에 재정렬한 뒤의 위치 오차 평균
  관절속도  관절 축 RMS

재정렬 식은 commands.py 284-294 를 그대로 옮긴 것이다.
  delta_pos  = 로봇 앵커 xy + 레퍼런스 앵커 z
  delta_ori  = yaw only (로봇 앵커 * inv(레퍼런스 앵커))
"""
import glob, os, numpy as np

ANCHOR = 7                    # torso_link
EE = [3, 6, 10, 13]           # 발목 좌우, 손목 좌우
TH_POS, TH_ORI, TH_EE, FAIL_M = 0.25, 0.8, 0.25, 0.5


def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=-1)


def qinv(q):
    o = q.copy(); o[:, 1:] *= -1; return o


def yaw_only(q):
    """회전을 z 축 성분만 남긴다 (isaaclab yaw_quat)."""
    o = np.zeros_like(q); o[:, 0] = q[:, 0]; o[:, 3] = q[:, 3]
    n = np.linalg.norm(o, axis=-1, keepdims=True)
    return o / np.where(n < 1e-9, 1.0, n)


def qrot(q, v):
    """쿼터니언으로 벡터를 돌린다. q (T,4), v (T,B,3)."""
    w = q[:, None, 0:1]; u = q[:, None, 1:4]
    return (v + 2.0 * (w * np.cross(u, v) + np.cross(u, np.cross(u, v))))


def gravity_z(q):
    """중력벡터 (0,0,-1) 을 바디 프레임에서 본 z 성분."""
    g = np.zeros((len(q), 1, 3)); g[:, 0, 2] = -1.0
    return qrot(qinv(q), g)[:, 0, 2]


def score(d):
    rp, bp = d["ref_body_pos_w"], d["rob_body_pos_w"]
    rq, bq = d["ref_body_quat_w"], d["rob_body_quat_w"]
    mpkpe = np.linalg.norm(rp - bp, axis=-1).mean(axis=1)

    # 레퍼런스를 로봇 앵커에 재정렬
    dpos = bp[:, ANCHOR].copy(); dpos[:, 2] = rp[:, ANCHOR, 2]
    dori = yaw_only(qmul(bq[:, ANCHOR], qinv(rq[:, ANCHOR])))
    rel = dpos[:, None, :] + qrot(dori, rp - rp[:, ANCHOR][:, None, :])
    r_mpkpe = np.linalg.norm(rel - bp, axis=-1).mean(axis=1)

    bad_pos = np.abs(rp[:, ANCHOR, 2] - bp[:, ANCHOR, 2]) > TH_POS
    bad_ori = np.abs(gravity_z(rq[:, ANCHOR]) - gravity_z(bq[:, ANCHOR])) > TH_ORI
    bad_ee = (np.abs(rel[:, EE, 2] - bp[:, EE, 2]) > TH_EE).any(axis=1)
    term = bad_pos | bad_ori | bad_ee
    first = int(np.argmax(term)) if term.any() else -1

    jrms = np.sqrt(((d["ref_joint_pos"] - d["rob_joint_pos"]) ** 2).mean(axis=1))
    jvel = np.sqrt(((d["ref_joint_vel"] - d["rob_joint_vel"]) ** 2).mean(axis=1))
    return dict(
        frames=len(mpkpe),
        bm_success=not term.any(), bm_first=first,
        bm_cause=("pos" if first >= 0 and bad_pos[first] else
                  "ori" if first >= 0 and bad_ori[first] else
                  "ee" if first >= 0 else "-"),
        poly_success=not (mpkpe > FAIL_M).any(),
        mpkpe=mpkpe.mean() * 1000, r_mpkpe=r_mpkpe.mean() * 1000,
        jrms=jrms.mean(), jvel=jvel.mean())


if __name__ == "__main__":
    hdr = (f"{'시퀀스':<22}{'MPKPE':>8}{'R-MPKPE':>9}{'관절':>7}{'관절속도':>9}"
           f"{'BM성공':>8}{'BM이탈(초)':>11}{'원인':>6}{'PolySim':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for f in sorted(glob.glob("outputs/sim2sim_full2/*.npz")):
        s = os.path.basename(f)[:-4]
        r = score(np.load(f)); rows.append((s, r))
        first = "—" if r["bm_first"] < 0 else f"{r['bm_first'] / 50:.0f}"
        print(f"{s:<22}{r['mpkpe']:>8.1f}{r['r_mpkpe']:>9.1f}{r['jrms']:>7.3f}"
              f"{r['jvel']:>9.3f}{'성공' if r['bm_success'] else '실패':>8}"
              f"{first:>11}{r['bm_cause']:>6}"
              f"{'성공' if r['poly_success'] else '실패':>9}")
    if rows:
        n = len(rows)
        print(f"\n{n}개 · BeyondMimic 성공 {sum(r['bm_success'] for _, r in rows)}/{n}"
              f" · PolySim 성공 {sum(r['poly_success'] for _, r in rows)}/{n}")
        print(f"평균 MPKPE {np.mean([r['mpkpe'] for _, r in rows]):.1f} mm"
              f" · R-MPKPE {np.mean([r['r_mpkpe'] for _, r in rows]):.1f} mm")
