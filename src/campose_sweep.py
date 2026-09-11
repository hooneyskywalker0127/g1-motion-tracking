"""카메라 배치를 쓸어서, 어느 위치에서 보아도 절반이 안 보이는지 확인한다."""
import numpy as np, mujoco

L, N, SETTLE, FOV, FAR = 2.0, 40, 4.0, 70.0, 6.0
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
        <config key="vmax" value="0"/></plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="0.6 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''
m = mujoco.MjModel.from_xml_string(XML); d = mujoco.MjData(m)
for _ in range(int(SETTLE/m.opt.timestep)): mujoco.mj_step(m,d)
ids=[i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
pos=d.xpos[ids].copy(); gid=np.zeros(1,np.int32); half=np.deg2rad(FOV)/2
hand=pos[0]; centroid=pos.mean(0)

def seen(cam, look):
    fwd=look-cam; fwd/=np.linalg.norm(fwd); v=np.zeros(len(ids),bool)
    for n,i in enumerate(ids):
        w=pos[n]-cam; dist=np.linalg.norm(w)
        if dist>FAR or np.arccos(np.clip(w@fwd/dist,-1,1))>half: continue
        frac=mujoco.mj_ray(m,d,cam,w/dist,None,1,-1,gid)
        v[n]= gid[0]<0 or m.geom_bodyid[gid[0]]==i or frac>dist-0.02
    return v.mean()*100

print("케이블 방향: 손(x=0)에서 +x 로 뻗음.  카메라는 손 기준 반경 0.55m, 높이 +0.40m")
print(f"{'방위각':>7} {'손 응시':>9} {'무게중심 응시':>13}")
best=(None,-1)
for az in range(0,360,30):
    r=0.55; c=hand+np.array([r*np.cos(np.deg2rad(az)), r*np.sin(np.deg2rad(az)), 0.40])
    a,b = seen(c,hand), seen(c,centroid)
    if b>best[1]: best=(az,b)
    print(f"{az:7d} {a:9.1f} {b:13.1f}")
print(f"\n최고: 방위각 {best[0]}도 에서 {best[1]:.1f}% (무게중심 응시)")
# 높은 데서 내려다보기
for h,lab in ((1.2,"손 위 1.2m"),(2.0,"손 위 2.0m")):
    c=hand+np.array([0,0,h]); print(f"{lab} 에서 내려다보기: {seen(c,centroid):.1f}%")
