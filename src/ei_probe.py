"""변형 야코비안으로 노드별 ego 가중 e_i 를 재 본다. 로봇도 정책도 없다.

케이블 한쪽 끝을 mocap 으로 잡고, 중력으로 늘어뜨려 바닥에 닿게 한 뒤,
잡은 끝을 여섯 방향으로 미세 변위시켜 각 노드가 얼마나 따라오는지 잰다.
같은 것을 강체 막대에서도 재서 나란히 놓는다.
"""
import numpy as np, mujoco

N = 40          # 캡슐 수
L = 2.0         # 케이블 길이 m
EPS = 2e-3      # 미세 변위 m
SETTLE = 4.0    # 안정화 시간 s

def make(kind):
    body = ""
    if kind == "cable":
        comp = f'''
      <composite type="cable" curve="s" count="{N+1} 1 1" size="{L}"
                 offset="0 0 0.95" initial="free">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="1e6"/>
          <config key="bend" value="1e5"/>
          <config key="vmax" value="0"/>
        </plugin>
        <joint kind="main" damping="0.02"/>
        <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
              friction="0.6 0.005 0.0001" density="1200"/>
      </composite>'''
    else:  # 같은 길이·질량의 강체 막대
        comp = f'''
      <body name="rod" pos="0 0 0.95">
        <freejoint/>
        <geom type="capsule" fromto="0 0 0 {L} 0 0" size=".004" density="1200"
              condim="3" friction="0.6 0.005 0.0001" rgba=".2 .4 .8 1"/>
      </body>'''
    return f'''
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" pos="0 0 0"
          friction="0.6 0.005 0.0001"/>
    <body name="grip" mocap="true" pos="0 0 0.95"/>
    {comp}
  </worldbody>
  <equality><weld body1="grip" body2="{'B_first' if kind=='cable' else 'rod'}"/></equality>
</mujoco>'''

def probe(kind):
    m = mujoco.MjModel.from_xml_string(make(kind))
    d = mujoco.MjData(m)
    for _ in range(int(SETTLE / m.opt.timestep)):
        mujoco.mj_step(m, d)
    ids = [i for i in range(m.nbody) if m.body(i).name not in ("world", "grip")]
    base = d.xpos[ids].copy()
    g = m.body("grip").mocapid[0]
    p0 = d.mocap_pos[g].copy()
    state = np.concatenate([d.qpos, d.qvel])
    J = np.zeros((len(ids), 3, 3))
    for k in range(3):                     # 병진 3방향만 본다
        d.qpos[:], d.qvel[:] = state[:m.nq], state[m.nq:]
        d.mocap_pos[g] = p0
        mujoco.mj_forward(m, d)
        d.mocap_pos[g] = p0 + EPS * np.eye(3)[k]
        for _ in range(int(0.4 / m.opt.timestep)):   # 준정적으로 따라가게
            mujoco.mj_step(m, d)
        J[:, :, k] = (d.xpos[ids] - base) / EPS
    return np.array([np.linalg.svd(J[i], compute_uv=False)[0] for i in range(len(ids))]), base

for kind in ("rod", "cable"):
    s, base = probe(kind)
    e = s / max(s.max(), 1e-9)
    n = len(e)
    idx = [0, n//8, n//4, 3*n//8, n//2, 5*n//8, 3*n//4, 7*n//8, n-1]
    print(f"\n=== {kind}  (노드 {n}개) ===")
    print("  노드번호 :", "  ".join(f"{i:5d}" for i in idx))
    print("  e_i      :", "  ".join(f"{e[i]:5.2f}" for i in idx))
    print("  높이 z   :", "  ".join(f"{base[i,2]:5.3f}" for i in idx))
    print(f"  e_i  평균 {e.mean():.3f}  최소 {e.min():.3f}  "
          f"0.5미만 노드 {int((e<0.5).sum())}/{n}  0.1미만 {int((e<0.1).sum())}/{n}")
