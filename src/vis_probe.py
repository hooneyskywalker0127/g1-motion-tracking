"""e_i 와 '머리 카메라에서 보이는가'의 상관을 본다. 기하만 쓴다.

케이블을 늘어뜨린 뒤, 사람 키 높이의 머리 카메라를 손 근처에 두고
각 노드가 (1) 시야각 안에 있고 (2) 다른 것에 가리지 않는지 판정한다.
"""
import numpy as np, mujoco

N, L, EPS, SETTLE = 40, 2.0, 2e-3, 4.0
FOV_DEG, FAR = 70.0, 5.0          # D435i 수평 화각 약 69도

XML = f'''
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" pos="0 0 0" friction="0.6 0.005 0.0001"/>
    <body name="grip" mocap="true" pos="0 0 0.95"/>
    <composite type="cable" curve="s" count="{N+1} 1 1" size="{L}"
               offset="0 0 0.95" initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="1e6"/><config key="bend" value="1e5"/>
        <config key="vmax" value="0"/>
      </plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="0.6 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''

m = mujoco.MjModel.from_xml_string(XML); d = mujoco.MjData(m)
for _ in range(int(SETTLE/m.opt.timestep)): mujoco.mj_step(m, d)
ids = [i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
base = d.xpos[ids].copy()

# --- e_i (35회차와 같은 방식)
g = m.body("grip").mocapid[0]; p0 = d.mocap_pos[g].copy()
st = np.concatenate([d.qpos, d.qvel]); J = np.zeros((len(ids),3,3))
for k in range(3):
    d.qpos[:], d.qvel[:] = st[:m.nq], st[m.nq:]; d.mocap_pos[g] = p0
    mujoco.mj_forward(m, d); d.mocap_pos[g] = p0 + EPS*np.eye(3)[k]
    for _ in range(int(0.4/m.opt.timestep)): mujoco.mj_step(m, d)
    J[:,:,k] = (d.xpos[ids]-base)/EPS
sig = np.array([np.linalg.svd(J[i],compute_uv=False)[0] for i in range(len(ids))])
e = sig/sig.max()

# --- 가시성: 머리 카메라를 손 위 30cm, 사람 키 높이에 두고 케이블 쪽을 본다
def visible(cam_pos, look_at):
    fwd = look_at-cam_pos; fwd/= np.linalg.norm(fwd)
    half = np.deg2rad(FOV_DEG)/2
    vis = np.zeros(len(ids), bool)
    geomid = np.zeros(1, np.int32)
    for n,i in enumerate(ids):
        v = base[n]-cam_pos; dist = np.linalg.norm(v)
        if dist > FAR: continue
        if np.arccos(np.clip(v@fwd/dist,-1,1)) > half: continue     # 화각 밖
        # 가림: 카메라에서 노드로 광선을 쏴 처음 맞는 것이 그 노드인가
        frac = mujoco.mj_ray(m, d, cam_pos, v/dist, None, 1, -1, geomid)
        if geomid[0] < 0: vis[n] = True
        else:
            hit_body = m.geom_bodyid[geomid[0]]
            vis[n] = (hit_body == i) or (frac > dist-0.02)
    return vis

hand = base[0]
for name, cam, look in [
    ("정면 응시(손을 봄)",  hand+np.array([-0.35,0,0.40]), hand),
    ("전방 응시(수평)",     hand+np.array([-0.35,0,0.40]), hand+np.array([2.0,0,0.35])),
    ("아래 응시(바닥 봄)",  hand+np.array([-0.35,0,0.40]), hand+np.array([1.0,0,-0.9])),
]:
    v = visible(cam, look)
    hi, lo = e>=0.5, e<0.5
    print(f"\n--- {name}")
    print(f"  보이는 노드 {v.sum()}/{len(v)}")
    print(f"  e_i>=0.5 (ego)   중 보이는 비율  {v[hi].mean()*100:5.1f}%  ({hi.sum()}개)")
    print(f"  e_i< 0.5 (world) 중 보이는 비율  {v[lo].mean()*100:5.1f}%  ({lo.sum()}개)")
    print(f"  보이는 노드의 평균 e_i  {e[v].mean() if v.any() else float('nan'):.3f}")
    print(f"  안 보이는 노드 평균 e_i {e[~v].mean() if (~v).any() else float('nan'):.3f}")
