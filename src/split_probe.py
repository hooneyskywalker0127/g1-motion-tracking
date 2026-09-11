"""기하(물체가 시야보다 길다) 와 동역학(안 보이는 쪽이 변한다) 의 기여를 가른다.

길이와 화각을 바꿔 가며 재고, 두 지표를 나눠 본다.
  안보임비율   전체 노드 중 시야 밖 비율            ← 순수 기하
  변화율_은닉  안 보이는 노드 중 '움직인' 비율       ← 동역학. 기하로 정규화됨
  변화율_가시  보이는 노드 중 '움직인' 비율          ← 대조
"""
import numpy as np, mujoco

SETTLE, SPEED, DUR, MOVED, FAR = 4.0, 0.15, 6.0, 0.01, 5.0

def build(L, N, mu=0.6):
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
        <config key="twist" value="1e6"/><config key="bend" value="1e5"/>
        <config key="vmax" value="0"/>
      </plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="{mu} 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''

def run(L, N, fov):
    m = mujoco.MjModel.from_xml_string(build(L, N)); d = mujoco.MjData(m)
    for _ in range(int(SETTLE/m.opt.timestep)): mujoco.mj_step(m, d)
    ids=[i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
    rest=d.xpos[ids].copy(); g=m.body("grip").mocapid[0]; p0=d.mocap_pos[g].copy()
    gid=np.zeros(1,np.int32); half=np.deg2rad(fov)/2
    acc_hid_moved, acc_hid, acc_vis_moved, acc_vis, acc_unvis = 0,0,0,0,0
    steps=int(DUR/m.opt.timestep); rec=int(0.25/m.opt.timestep); n_s=0
    for s in range(steps):
        d.mocap_pos[g]=p0+np.array([-SPEED*s*m.opt.timestep,0,0]); mujoco.mj_step(m,d)
        if s % rec: continue
        pos=d.xpos[ids]; moved=np.linalg.norm(pos-rest,axis=1)>MOVED
        cam=pos[0]+np.array([-0.35,0,0.40]); fwd=pos[0]-cam; fwd/=np.linalg.norm(fwd)
        v=np.zeros(len(ids),bool)
        for n,i in enumerate(ids):
            w=pos[n]-cam; dist=np.linalg.norm(w)
            if dist>FAR or np.arccos(np.clip(w@fwd/dist,-1,1))>half: continue
            frac=mujoco.mj_ray(m,d,cam,w/dist,None,1,-1,gid)
            v[n]= gid[0]<0 or m.geom_bodyid[gid[0]]==i or frac>dist-0.02
        acc_unvis += (~v).mean(); n_s += 1
        if (~v).any(): acc_hid_moved += moved[~v].mean(); acc_hid += 1
        if v.any():    acc_vis_moved += moved[v].mean();  acc_vis += 1
    f=lambda a,b: (a/b*100 if b else float('nan'))
    return f(acc_unvis,n_s), f(acc_hid_moved,acc_hid), f(acc_vis_moved,acc_vis)

print(f"{'길이m':>6} {'노드':>5} {'화각':>5} | {'안보임비율%':>11} "
      f"{'은닉중움직임%':>13} {'가시중움직임%':>13}")
for L,N in ((2.0,40),(1.0,20),(0.5,10)):
    for fov in (70, 140):
        u,h,vs = run(L,N,fov)
        print(f"{L:6.1f} {N:5d} {fov:5d} | {u:11.1f} {h:13.1f} {vs:13.1f}")
