"""Sit-to-stand pieces that work in the runtime (see the chat notes): MoveBrain = two-phase leg move with soles kept flat
(lean hips forward first, then move hips+knees; ankle = home rule + IMU trunk pitch, sign -1). From the measured perch seat
(runs/perch_cands.json, hip -0.349 knee 2.182 ankle -1.047) MoveBrain(rt, home, lean=-1.3, 0.5, 1.5) stood 20/20 on
randomized bodies. Sitting DOWN into that exact seat is not solved yet: see the chat handoff.
"""
import json, sys, numpy as np, mujoco
from pathlib import Path
from s2s_perch import perch_ctrl
from runtime import DuckRuntime, PoseBrain, SimBackend, load_brains
brains,cfg=load_brains(Path("../models")); c=json.load(open("runs/perch_cands.json"))
names=cfg["servo_names"]; home=np.array(cfg["home"])
L={k:[i for i,n in enumerate(names) if n.endswith(k)] for k in ("hip_pitch","knee","ankle")}
mirror={i:(1.0 if j==0 else float(np.sign(home[i]*home[L["hip_pitch"][0]]) or 1)) for j,i in enumerate(L["hip_pitch"])}
class RiseBrain:
    def __init__(s, rt, ctrl, lean, t1, t2, sign): s.rt,s.ctrl,s.lean,s.t1,s.t2,s.sign,s.t0=rt,np.array(ctrl),lean,t1,t2,sign,None
    def __call__(s, obs):
        rt=s.rt
        if s.t0 is None: s.t0=rt.steps
        t=(rt.steps-s.t0)*rt.dt; g=obs[:3]
        tilt=float(np.arctan2(g[0], -g[2]))
        tgt=s.ctrl.copy()
        for hp,kn,an in zip(L["hip_pitch"],L["knee"],L["ankle"]):
            h0,k0=s.ctrl[hp],s.ctrl[kn]; hL=home[hp]+mirror[hp]*(s.lean-home[L["hip_pitch"][0]])
            if t<s.t1: f=t/s.t1; tgt[hp]=h0+(hL-h0)*f; tgt[kn]=k0
            else: f=min(1,(t-s.t1)/s.t2); tgt[hp]=hL+(home[hp]-hL)*f; tgt[kn]=k0+(home[kn]-k0)*f
            tgt[an]=home[an]-mirror[hp]*(tgt[hp]-home[hp])-(tgt[kn]-home[kn])+s.sign*tilt
        return np.clip((tgt[rt.pol]-home[rt.pol])/rt.scale,-1,1)
def run(r, lean, t1, t2, sign, seed=0, trace=False):
    be=SimBackend(names); m,d=be.model,be.data
    rt=DuckRuntime(backend=be, brains=dict(brains), config=cfg, log=None)
    ctrl=perch_ctrl(m,r); q=np.array(r["qpos"])
    mujoco.mj_resetData(m,d); d.qpos[:]=q; d.ctrl[:]=ctrl; mujoco.mj_forward(m,d)
    rt.brains["pose:perch"]=PoseBrain(ctrl,rt); rt.skill="pose:perch"; rt.last_action=np.clip((ctrl[rt.pol]-home[rt.pol])/rt.scale,-1,1); rt.run(0.5)
    rt.brains["pose:rise2"]=RiseBrain(rt, d.ctrl.copy(), lean, t1, t2, sign); rt.quiet.add("pose:rise2")
    rt.mode="rise"; rt.plan=[("pose:rise2",t1+t2+0.3),("stand",None)]; rt.plan_all=list(rt.plan); rt._next_in_plan()
    mn=1; rows=[]
    for i in range(int((t1+t2+2.5)*rt.hz)):
        rt.tick(); p=be.posture(); mn=min(mn,p["up"])
        if trace and i%10==0: rows.append((round(i/rt.hz,1),round(p["up"],2),round(p["height"],3)))
    if trace: print(rows)
    p=be.posture(); return round(p["up"],2),round(p["height"],3),round(mn,2)
if __name__=="__main__":
    r=[x for x in c if min(x["sole_flat"])>0.4][int(sys.argv[1]) if len(sys.argv)>1 else 2]
    print("seat",r["hip"],r["knee"],r["ankle"])
    for sign in (1,-1):
      for lean in (-1.3,-0.9,-0.5):
        for t1,t2 in ((0.5,1.5),):
            print("sign",sign,"lean",lean,"t",t1,t2, run(r,lean,t1,t2,sign))

def run_rand(r, lean, t1, t2, sign, seed):
    rng=np.random.default_rng(seed)
    be=SimBackend(names); m,d=be.model,be.data
    f=rng.uniform(0.9,1.1,m.nbody); m.body_mass[:]=m.body_mass*f; m.body_inertia[:]=m.body_inertia*f[:,None]; m.geom_friction[:,0]=rng.uniform(0.4,1.0)
    rt=DuckRuntime(backend=be, brains=dict(brains), config=cfg, log=None)
    ctrl=perch_ctrl(m,r); q=np.array(r["qpos"]); q[be.qadr]=np.clip(q[be.qadr]+rng.normal(0,0.02,m.nu),rt.lo,rt.hi)
    yaw=rng.uniform(-np.pi,np.pi); q[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    mujoco.mj_resetData(m,d); d.qpos[:]=q; d.ctrl[:]=ctrl; mujoco.mj_forward(m,d)
    rt.brains["pose:perch"]=PoseBrain(ctrl,rt); rt.skill="pose:perch"; rt.last_action=np.clip((ctrl[rt.pol]-home[rt.pol])/rt.scale,-1,1); rt.run(0.5)
    rt.brains["pose:rise2"]=RiseBrain(rt, d.ctrl.copy(), lean, t1, t2, sign); rt.quiet.add("pose:rise2")
    rt.mode="rise"; rt.plan=[("pose:rise2",t1+t2+0.3),("stand",None)]; rt.plan_all=list(rt.plan); rt._next_in_plan()
    mn=1
    for i in range(int((t1+t2+2.5)*rt.hz)):
        rt.tick(); p=be.posture(); mn=min(mn,p["up"])
    p=be.posture(); return p["up"]>0.9 and p["height"]>0.9*cfg["stand_height"] and mn>0.3

class MoveBrain:
    """Glide the legs from the current targets to `goal` (hip, knee per servo, from a ctrl vector) in two phases:
    lean the hips forward first (keep the knees), then move hips and knees together; the ankles keep the
    soles flat all the way (home rule + measured trunk pitch). At the end hold `goal` exactly."""
    def __init__(s, rt, goal, lean, t1, t2, sign=-1): s.rt,s.goal,s.lean,s.t1,s.t2,s.sign,s.t0,s.ctrl=rt,np.array(goal),lean,t1,t2,sign,None,None
    def __call__(s, obs):
        rt=s.rt
        if s.t0 is None: s.t0,s.ctrl=rt.steps,rt.targets.copy()
        t=(rt.steps-s.t0)*rt.dt; g=obs[:3]; tilt=float(np.arctan2(g[0],-g[2]))
        tgt=s.goal.copy() if t>=s.t1+s.t2 else s.ctrl.copy()
        if True:
            for hp,kn,an in zip(L["hip_pitch"],L["knee"],L["ankle"]):
                h0,k0=s.ctrl[hp],s.ctrl[kn]; hL=home[hp]+mirror[hp]*(s.lean-home[L["hip_pitch"][0]])
                if t>=s.t1+s.t2: pass
                elif t<s.t1: f=t/s.t1; tgt[hp]=h0+(hL-h0)*f; tgt[kn]=k0
                else: f=(t-s.t1)/s.t2; tgt[hp]=hL+(s.goal[hp]-hL)*f; tgt[kn]=k0+(s.goal[kn]-k0)*f
                tgt[an]=home[an]-mirror[hp]*(tgt[hp]-home[hp])-(tgt[kn]-home[kn])+s.sign*tilt
        return np.clip((tgt[rt.pol]-home[rt.pol])/rt.scale,-1,1)

def roundtrip(r, seed, lean=-1.3, t1=0.5, t2=1.5, sit_t=(0.5,1.5)):
    rng=np.random.default_rng(seed)
    be=SimBackend(names); m,d=be.model,be.data
    f=rng.uniform(0.9,1.1,m.nbody); m.body_mass[:]=m.body_mass*f; m.body_inertia[:]=m.body_inertia*f[:,None]; m.geom_friction[:,0]=rng.uniform(0.4,1.0)
    rt=DuckRuntime(backend=be, brains=dict(brains), config=cfg, log=None)
    ctrl=perch_ctrl(m,r)
    rt.brains["pose:perch"]=MoveBrain(rt,ctrl,lean,*sit_t); rt.quiet.add("pose:perch")
    rt.run(1.0)
    rt.mode="sit"; rt.plan=[("squat",2.0),("pose:perch",None)]; rt.plan_all=list(rt.plan); rt._next_in_plan()
    mn=1
    for _ in range(int(5.5*rt.hz)): rt.tick(); mn=min(mn,be.posture()["up"])
    p=be.posture(); thighs=any(m.body(m.geom_bodyid[g]).name in ("knee_and_ankle_assembly","knee_and_ankle_assembly_3") for cc in d.contact[:d.ncon] for g in cc.geom if be.floor in cc.geom)
    seated = p["up"]>0.85 and p["height"]<0.8*cfg["stand_height"] and thighs and mn>0.5
    rt.brains["pose:rise2"]=MoveBrain(rt, home, lean, t1, t2); rt.quiet.add("pose:rise2")
    rt.mode="rise"; rt.plan=[("pose:rise2",t1+t2+0.3),("stand",None)]; rt.plan_all=list(rt.plan); rt._next_in_plan()
    mn=1
    for _ in range(int(4.5*rt.hz)): rt.tick(); mn=min(mn,be.posture()["up"])
    p=be.posture(); stood=p["up"]>0.9 and p["height"]>0.9*cfg["stand_height"] and mn>0.3
    return seated, stood, round(p["up"],2)
