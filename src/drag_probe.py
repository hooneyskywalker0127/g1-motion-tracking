"""손을 끌면서 '움직인 구간의 경계'가 케이블을 따라 이동하는지, 그것이 보이는지 본다."""
import numpy as np, mujoco

N, L, SETTLE = 40, 2.0, 4.0
SPEED, DUR = 0.15, 6.0            # 손을 -x 로 0.15 m/s, 6초
FOV_DEG, FAR, MOVED = 70.0, 5.0, 0.01   # 1cm 이상 움직였으면 '움직인 것'

XML = f'''
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 .1" pos="0 0 0" friction="0.6 0.005 0.0001"/>
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
rest = d.xpos[ids].copy()
g = m.body("grip").mocapid[0]; p0 = d.mocap_pos[g].copy()
gid = np.zeros(1, np.int32)

def vis_mask(pos, hand):
    cam = hand + np.array([-0.35, 0, 0.40]); look = hand
    fwd = look-cam; fwd /= np.linalg.norm(fwd); half = np.deg2rad(FOV_DEG)/2
    out = np.zeros(len(ids), bool)
    for n,i in enumerate(ids):
        v = pos[n]-cam; dist = np.linalg.norm(v)
        if dist > FAR or np.arccos(np.clip(v@fwd/dist,-1,1)) > half: continue
        frac = mujoco.mj_ray(m, d, cam, v/dist, None, 1, -1, gid)
        out[n] = gid[0] < 0 or m.geom_bodyid[gid[0]] == i or frac > dist-0.02
    return out

print(f"{'t':>5} {'손이동':>7} {'경계노드':>8} {'경계 보임':>9} {'움직인수':>8} "
      f"{'움직였는데 안보임':>16} {'손장력N':>8}")
steps = int(DUR/m.opt.timestep); rec = int(0.5/m.opt.timestep)
for s in range(steps):
    d.mocap_pos[g] = p0 + np.array([-SPEED*s*m.opt.timestep, 0, 0])
    mujoco.mj_step(m, d)
    if s % rec: continue
    pos = d.xpos[ids]
    disp = np.linalg.norm(pos-rest, axis=1)
    moved = disp > MOVED
    b = int(np.flatnonzero(moved).max()) if moved.any() else -1   # 움직인 구간의 끝
    v = vis_mask(pos, pos[0])
    # weld 제약력 = 손에 걸리는 힘
    f = np.linalg.norm(d.efc_force[:3]) if d.nefc >= 3 else 0.0
    print(f"{s*m.opt.timestep:5.1f} {SPEED*s*m.opt.timestep:7.3f} {b:8d} "
          f"{str(bool(v[b])) if b>=0 else '-':>9} {int(moved.sum()):8d} "
          f"{int((moved & ~v).sum()):16d} {f:8.1f}")
