"""체공과 부양 아티팩트를 가른다.

점프 모션은 발이 떠 있는 게 정상이다. 문제는 서 있어야 할 국면에서도 떠 있는 것이다.
루트 높이가 그 클립의 하위 30% 인 프레임(=가라앉은 국면 = 서 있거나 착지)만 골라
그때 발바닥 높이를 본다. 정상이면 0 근처여야 한다.
"""
import numpy as np, mujoco, onnx, sys, joblib, glob, os
X="/home/sehoon/mtc_overlay/opt/ros/jazzy/share/unitree_description/mjcf/g1.xml"
m=mujoco.MjModel.from_xml_path(X); d=mujoco.MjData(m)
jn={e.key:e.value for e in onnx.load("/home/sehoon/colcon_ws/policies/walk1_subject1.onnx").metadata_props}["joint_names"].split(",")
qadr=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in jn]
verts={}
for i in range(m.ngeom):
    if not (m.geom_contype[i] or m.geom_conaffinity[i]): continue
    t,s=m.geom_type[i],m.geom_size[i]
    if t==mujoco.mjtGeom.mjGEOM_MESH:
        k=m.geom_dataid[i]; v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]].reshape(-1,3)
        verts[i]=v[::max(1,len(v)//300)]
    elif t==mujoco.mjtGeom.mjGEOM_SPHERE: verts[i]=np.array([[0,0,-s[0]]])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE,mujoco.mjtGeom.mjGEOM_CYLINDER):
        verts[i]=np.array([[0,0,s[1]],[0,0,-s[1]],[s[0],0,s[1]],[-s[0],0,s[1]]])
def low():
    return min(((v@d.geom_xmat[i].reshape(3,3).T)[:,2].min()+d.geom_xpos[i][2]) for i,v in verts.items())

def run(name, npz):
    z=np.load(npz); n=len(z["joint_pos"]); step=max(1,n//200)
    idx=np.arange(0,n,step)
    rootz=z["body_pos_w"][idx,0,2]
    lo=[]
    for f in idx:
        d.qpos[:3]=z["body_pos_w"][f,0]; d.qpos[3:7]=z["body_quat_w"][f,0]
        d.qpos[qadr]=z["joint_pos"][f]; mujoco.mj_forward(m,d); lo.append(low())
    lo=np.array(lo)
    thr=np.percentile(rootz,30)
    st=lo[rootz<=thr]
    print(f"{name:<20} 전체중앙 {np.median(lo):+.3f}  "
          f"가라앉은국면 중앙 {np.median(st):+.3f} 최소 {st.min():+.3f}  "
          f"그 국면 0.03 초과 {100*(st>0.03).mean():3.0f}%")

for n in ("walk1_subject1","jumps1_subject1","kobe_level1"):
    run(n, f"/home/sehoon/motions_fixed/{n}.npz")
