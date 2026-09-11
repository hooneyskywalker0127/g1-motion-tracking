"""케이블을 물리적으로 밀어 쌓아(똬리) 놓고, 그 상태에서 방위각 가시성을 다시 잰다."""
import numpy as np, mujoco
L,N,FOV,FAR = 2.0,40,70.0,6.0
XML=f'''
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 .1" pos="0 0 0" friction="0.6 0.005 0.0001"/>
    <body name="grip" mocap="true" pos="0 0 0.05"/>
    <composite type="cable" curve="s" count="{N+1} 1 1" size="{L}"
               offset="0 0 0.05" initial="free">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="1e5"/><config key="bend" value="3e3"/>
        <config key="vmax" value="0"/></plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="0.6 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''
m=mujoco.MjModel.from_xml_string(XML); d=mujoco.MjData(m)
ids=[i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
g=m.body("grip").mocapid[0]; gid=np.zeros(1,np.int32); half=np.deg2rad(FOV)/2
for _ in range(int(2.0/m.opt.timestep)): mujoco.mj_step(m,d)

def report(label):
    pos=d.xpos[ids].copy(); c=pos.mean(0)
    def seen(cam):
        fwd=c-cam; fwd/=np.linalg.norm(fwd); v=np.zeros(len(ids),bool)
        for n,i in enumerate(ids):
            w=pos[n]-cam; dist=np.linalg.norm(w)
            if dist>FAR or np.arccos(np.clip(w@fwd/dist,-1,1))>half: continue
            fr=mujoco.mj_ray(m,d,cam,w/dist,None,1,-1,gid)
            v[n]= gid[0]<0 or m.geom_bodyid[gid[0]]==i or fr>dist-0.02
        return v.mean()*100
    r=1.2
    vals=[seen(c+np.array([r*np.cos(np.deg2rad(a)),r*np.sin(np.deg2rad(a)),0.8]))
          for a in range(0,360,30)]
    top=seen(c+np.array([0,0,2.0]))
    sp=float(np.ptp(pos[:,:2],axis=0).max())
    print(f"{label:16s} 퍼짐 {sp:4.2f}m | 방위각 최고 {max(vals):5.1f}% "
          f"최저 {min(vals):5.1f}% 평균 {np.mean(vals):5.1f}% | 위에서 {top:5.1f}%")

report("직선(초기)")
# 잡은 끝을 반대쪽 끝 쪽으로 밀어 쌓는다
p0=d.mocap_pos[g].copy()
for s in range(int(9.0/m.opt.timestep)):
    d.mocap_pos[g]=p0+np.array([min(1.7, 0.20*s*m.opt.timestep),0,0])
    mujoco.mj_step(m,d)
for _ in range(int(2.0/m.opt.timestep)): mujoco.mj_step(m,d)
report("밀어 쌓은 뒤")
