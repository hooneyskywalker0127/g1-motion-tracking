"""38회차 미결: 기하(길이) 기여분과 동역학 기여분을 가른다.
짧은 케이블(시야에 다 들어옴)에서도 '안 보이는 쪽이 움직이는' 현상이 남는가."""
import numpy as np, mujoco
SETTLE,SPEED,DUR,MOVED,FAR,FOV = 4.0,0.15,6.0,0.01,6.0,70.0

def build(L,N):
    return f'''
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
        <config key="vmax" value="0"/></plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="0.6 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''

def run(L,N):
    m=mujoco.MjModel.from_xml_string(build(L,N)); d=mujoco.MjData(m)
    for _ in range(int(SETTLE/m.opt.timestep)): mujoco.mj_step(m,d)
    ids=[i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
    rest=d.xpos[ids].copy(); g=m.body("grip").mocapid[0]; p0=d.mocap_pos[g].copy()
    gid=np.zeros(1,np.int32); half=np.deg2rad(FOV)/2
    unvis=[]; hidmoved=[]; vismoved=[]
    steps=int(DUR/m.opt.timestep); rec=int(0.25/m.opt.timestep)
    for s in range(steps):
        d.mocap_pos[g]=p0+np.array([-SPEED*s*m.opt.timestep,0,0]); mujoco.mj_step(m,d)
        if s%rec: continue
        pos=d.xpos[ids]; moved=np.linalg.norm(pos-rest,axis=1)>MOVED
        cam=pos[0]+np.array([-0.35,0,0.40]); fwd=pos[0]-cam; fwd/=np.linalg.norm(fwd)
        v=np.zeros(len(ids),bool)
        for n,i in enumerate(ids):
            w=pos[n]-cam; dist=np.linalg.norm(w)
            if dist>FAR or np.arccos(np.clip(w@fwd/dist,-1,1))>half: continue
            fr=mujoco.mj_ray(m,d,cam,w/dist,None,1,-1,gid)
            v[n]= gid[0]<0 or m.geom_bodyid[gid[0]]==i or fr>dist-0.02
        unvis.append((~v).mean())
        if (~v).any(): hidmoved.append(moved[~v].mean())
        if v.any():    vismoved.append(moved[v].mean())
    f=lambda a: (np.mean(a)*100 if len(a) else float('nan'))
    return f(unvis), f(hidmoved), f(vismoved)

print(f"{'길이m':>6}{'노드':>5} | {'안보임%':>9}{'은닉중움직임%':>14}{'가시중움직임%':>14}")
for L,N in ((2.0,40),(1.2,24),(0.8,16),(0.5,10),(0.3,6)):
    u,h,v=run(L,N)
    print(f"{L:6.1f}{N:5d} | {u:9.1f}{h:14.1f}{v:14.1f}")
