"""길이별로 카메라 방위각을 쓸어, 짧은 케이블은 옆에서 보면 다 보이는지 확인한다."""
import numpy as np, mujoco
FOV,FAR=70.0,6.0
def build(L,N):
    return f'''
<mujoco>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 .1" pos="0 0 0" friction="0.6 0.005 0.0001"/>
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
</mujoco>'''
print(f"{'길이m':>6} | {'손 응시 최고':>12}{'최저':>8} | {'무게중심 최고':>14}{'최저':>8}")
for L,N in ((2.0,40),(1.2,24),(0.8,16),(0.5,10)):
    m=mujoco.MjModel.from_xml_string(build(L,N)); d=mujoco.MjData(m)
    for _ in range(int(6.0/m.opt.timestep)): mujoco.mj_step(m,d)
    ids=[i for i in range(m.nbody) if m.body(i).name!="world"]
    pos=d.xpos[ids].copy(); gid=np.zeros(1,np.int32); half=np.deg2rad(FOV)/2
    hand=pos[0]; c=pos.mean(0)
    def seen(cam,look):
        fwd=look-cam; fwd/=np.linalg.norm(fwd); v=np.zeros(len(ids),bool)
        for n,i in enumerate(ids):
            w=pos[n]-cam; dist=np.linalg.norm(w)
            if dist>FAR or np.arccos(np.clip(w@fwd/dist,-1,1))>half: continue
            fr=mujoco.mj_ray(m,d,cam,w/dist,None,1,-1,gid)
            v[n]= gid[0]<0 or m.geom_bodyid[gid[0]]==i or fr>dist-0.02
        return v.mean()*100
    r=max(0.55, L*0.6)
    hv=[seen(hand+np.array([r*np.cos(np.deg2rad(a)),r*np.sin(np.deg2rad(a)),0.40]),hand)
        for a in range(0,360,30)]
    cv=[seen(hand+np.array([r*np.cos(np.deg2rad(a)),r*np.sin(np.deg2rad(a)),0.40]),c)
        for a in range(0,360,30)]
    print(f"{L:6.1f} | {max(hv):11.1f}%{min(hv):7.1f}% | {max(cv):13.1f}%{min(cv):7.1f}%")
