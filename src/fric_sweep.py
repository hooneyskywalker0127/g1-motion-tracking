"""마찰을 쓸어 가며 '경계가 시야 밖에서 이동하는' 현상이 사는 영역을 찾는다."""
import numpy as np, mujoco

N, L, SETTLE, SPEED, DUR = 40, 2.0, 4.0, 0.15, 6.0
FOV_DEG, FAR, MOVED = 70.0, 5.0, 0.01

def build(mu, bend):
    return f'''
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 .1" pos="0 0 0" friction="{mu} 0.005 0.0001"/>
    <body name="grip" mocap="true" pos="0 0 0.95"/>
    <composite type="cable" curve="s" count="{N+1} 1 1" size="{L}"
               offset="0 0 0.95" initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="1e6"/><config key="bend" value="{bend}"/>
        <config key="vmax" value="0"/>
      </plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="{mu} 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''

def run(mu, bend):
    m = mujoco.MjModel.from_xml_string(build(mu, bend)); d = mujoco.MjData(m)
    for _ in range(int(SETTLE/m.opt.timestep)): mujoco.mj_step(m, d)
    ids = [i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
    rest = d.xpos[ids].copy()
    g = m.body("grip").mocapid[0]; p0 = d.mocap_pos[g].copy(); gid = np.zeros(1,np.int32)
    worst, hidden_steps, b_first, b_last = 0, 0, None, None
    steps = int(DUR/m.opt.timestep); rec = int(0.25/m.opt.timestep)
    for s in range(steps):
        d.mocap_pos[g] = p0 + np.array([-SPEED*s*m.opt.timestep,0,0])
        mujoco.mj_step(m, d)
        if s % rec: continue
        pos = d.xpos[ids]; moved = np.linalg.norm(pos-rest,axis=1) > MOVED
        if not moved.any(): continue
        b = int(np.flatnonzero(moved).max())
        if b_first is None: b_first = b
        b_last = b
        cam = pos[0]+np.array([-0.35,0,0.40]); fwd = pos[0]-cam; fwd/=np.linalg.norm(fwd)
        half = np.deg2rad(FOV_DEG)/2; v = np.zeros(len(ids),bool)
        for n,i in enumerate(ids):
            w = pos[n]-cam; dist = np.linalg.norm(w)
            if dist>FAR or np.arccos(np.clip(w@fwd/dist,-1,1))>half: continue
            frac = mujoco.mj_ray(m,d,cam,w/dist,None,1,-1,gid)
            v[n] = gid[0]<0 or m.geom_bodyid[gid[0]]==i or frac>dist-0.02
        worst = max(worst, int((moved & ~v).sum()))
        if b >= 0 and not v[b] and b < len(ids)-1: hidden_steps += 1
    return b_first, b_last, worst, hidden_steps

print(f"{'마찰':>5} {'bend':>7} {'첫경계':>7} {'끝경계':>7} "
      f"{'움직였는데안보임(최대)':>22} {'경계가안보인구간(회)':>20}")
for bend in (1e5, 1e4):
    for mu in (0.15, 0.3, 0.6, 1.0):
        a,b,w,h = run(mu, bend)
        print(f"{mu:5.2f} {bend:7.0e} {str(a):>7} {str(b):>7} {w:22d} {h:20d}")
