"""공정한 강체 대조군: 같은 길이·질량·마찰의 케이블을, 굽힘강성만 1000배 올려 굳힌다.
노드 수가 같으므로 e_i 분포를 나란히 비교할 수 있다."""
import numpy as np, mujoco
N,L,EPS,SETTLE = 40,2.0,2e-3,4.0

def build(bend, twist):
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
        <config key="twist" value="{twist}"/><config key="bend" value="{bend}"/>
        <config key="vmax" value="0"/></plugin>
      <joint kind="main" damping="0.02"/>
      <geom type="capsule" size=".004" rgba=".8 .2 .1 1" condim="3"
            friction="0.6 0.005 0.0001" density="1200"/>
    </composite>
  </worldbody>
  <equality><weld body1="grip" body2="B_first"/></equality>
</mujoco>'''

def ei(bend, twist):
    m=mujoco.MjModel.from_xml_string(build(bend,twist)); d=mujoco.MjData(m)
    for _ in range(int(SETTLE/m.opt.timestep)): mujoco.mj_step(m,d)
    ids=[i for i in range(m.nbody) if m.body(i).name not in ("world","grip")]
    base=d.xpos[ids].copy(); g=m.body("grip").mocapid[0]; p0=d.mocap_pos[g].copy()
    st=np.concatenate([d.qpos,d.qvel]); J=np.zeros((len(ids),3,3))
    for k in range(3):
        d.qpos[:],d.qvel[:]=st[:m.nq],st[m.nq:]; d.mocap_pos[g]=p0
        mujoco.mj_forward(m,d); d.mocap_pos[g]=p0+EPS*np.eye(3)[k]
        for _ in range(int(0.4/m.opt.timestep)): mujoco.mj_step(m,d)
        J[:,:,k]=(d.xpos[ids]-base)/EPS
    s=np.array([np.linalg.svd(J[i],compute_uv=False)[0] for i in range(len(ids))])
    return s/s.max(), base[:,2]

print(f"{'조건':>22} {'e_i 평균':>9} {'표준편차':>9} {'최소':>7} "
      f"{'e<0.1 개수':>11} {'바닥 닿은 노드':>14}")
for bend,twist,lab in ((1e5,1e6,"유연 (케이블)"),
                       (1e7,1e8,"뻣뻣 (100배)"),
                       (1e9,1e10,"매우 뻣뻣 (1e4배)")):
    e,z = ei(bend,twist)
    print(f"{lab:>22} {e.mean():9.3f} {e.std():9.3f} {e.min():7.3f} "
          f"{int((e<0.1).sum()):11d} {int((z<0.02).sum()):14d}")
