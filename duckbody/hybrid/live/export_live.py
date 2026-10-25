"""Model + brains for the LIVE match page (the simulation runs in the viewer's browser, MuJoCo WASM).

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/live/export_live.py OUT_DIR            # live_model_{5v5,3v3}.xml, live_meta_*.json, live_moves.json, policies/*.onnx
    $PY duckbody/hybrid/live/export_live.py OUT_DIR --check    # + the exported model against match.py's (Python)
    $PY duckbody/hybrid/live/export_live.py PAGE/live --page --wasm node_modules/@mujoco/mujoco/mujoco.wasm
        # the whole live page: PAGE/live.html + PAGE/live/ (JS, model, brains, mesh, the 10 MB MuJoCo WASM binary,
        # which must ship with the page: the artifact sandbox lets only scripts, not fetches, come from the CDN)

The model is match.py's (pitch, ball, 11 hybrids attached with MjSpec, policy servos turned into BAM torque motors),
written as ONE self-contained XML so the page can load it into MuJoCo WASM without any mesh files:
  - every collision mesh is inlined as the vertices of its convex hull. MuJoCo collides meshes by their convex hull
    anyway, and every body has an explicit inertial, so the physics is the same (--check compares 4 s of walking);
  - geoms that never collide (the visible meshes, groups 2 and 4) are dropped: the page draws duck_mesh_match.json;
  - the head camera 'eye' (follow.py's) is kept on every duck: the page's players "see" with its pose and field of
    view (no rendering, see live.js).
live_meta.json carries what the page needs to drive it: per robot the actuator / joint / qpos / dof indices in the
policy order, BAM's friction-row index quirk (match.MultiBam.quirk), sensor and body ids; BAM's XL330 m6 parameters;
the policies are Microduck's ONNX files, copied unchanged (the page runs them with onnx_mlp.js).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HYB = HERE.parent
sys.path.insert(0, str(HYB.parent / "microduck_port"))
sys.path.insert(0, str(HYB))
import mujoco  # noqa: E402
import match as M  # noqa: E402
import follow as F  # noqa: E402
from harness import BAM_VIN, BAM_KP_FW, BAM_VIN_DROP_GAIN, BAM_VIN_MIN, DEFAULT_POSE, JOINT_NAMES, TIMESTEP, POLICIES  # noqa: E402

VARIANTS = {"5v5": M.ROBOTS,
            "3v3": [("b1", "blue"), ("b2", "blue"), ("b3", "blue"), ("y1", "yellow"), ("y2", "yellow"), ("y3", "yellow"),
                    ("ref", "ref")]}
BRAINS = ("velstand", "ball_kick_left", "ball_kick_right", "roulade")


def live_spec(robots=M.ROBOTS, pitch=True):
    """match.build_model, but returning the MjSpec, without the visible-only geoms, with the head camera."""
    from bam.model import load_model
    bam = load_model(motor_name="xl330", model="m6")
    force_limit = BAM_VIN * bam.kt.value / bam.R.value
    world = mujoco.MjSpec()
    world.compiler.autolimits = True
    world.memory = 128 << 20
    world.option.timestep = TIMESTEP
    if pitch:
        M.add_pitch(world)
    else:
        g = world.worldbody.add_geom()
        g.name, g.type, g.size = "floor", mujoco.mjtGeom.mjGEOM_PLANE, [0, 0, 0.05]
    first = None
    for name, team in robots:
        child = M.robot_spec(force_limit, team, cam=False)
        for g in list(child.geoms):           # no-contact geoms (visible meshes, group 2 and 4): the page draws them
            if g.contype == 0 and g.conaffinity == 0:
                child.delete(g)
        used = {g.meshname for g in child.geoms}
        for mesh in list(child.meshes):
            if mesh.name not in used:
                child.delete(mesh)
        c = child.body("jaw_soft").add_camera()
        c.name, c.pos, c.quat, c.fovy = "eye", list(F.EYE_POS), list(F.EYE_QUAT), F.FOVY
        if first is not None:
            for mesh in list(child.meshes):
                child.delete(mesh)
        fr = world.worldbody.add_frame()
        fr.attach_body(child.body("trunk_base"), name + "_", "")
        if first is None:
            first = name + "_"
    if len(robots) > 1:
        for g in world.geoms:
            mn = g.meshname
            if mn and not mn.startswith(first):
                g.meshname = first + mn.split("_", 1)[1]
    return world, bam


def inline_hulls(make):
    """A spec from make() with every mesh file replaced by the vertices of its convex hull (same collision shape), in
    the file's frame. (Hulls are read from a first compiled copy: a compiled spec keeps its loaded mesh data.)"""
    m = make()[0].compile()
    spec = make()[0]
    n = 0
    for mesh in spec.meshes:
        mid = m.mesh(mesh.name).id
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        ga = m.mesh_graphadr[mid]                  # MuJoCo's own convex hull: numvert, numface, edgeadr, globalid ...
        assert ga >= 0, mesh.name
        nvh = int(m.mesh_graph[ga])
        gid = m.mesh_graph[ga + 2 + nvh: ga + 2 + 2 * nvh]
        hv = v[gid]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.mesh_quat[mid])
        orig = hv @ R.reshape(3, 3).T + m.mesh_pos[mid]        # back to the (scaled) file frame
        mesh.file = ""
        mesh.uservert = np.round(orig, 6).ravel().tolist()
        mesh.scale = [1.0, 1.0, 1.0]
        n += len(hv)
    return spec, n


def for_wasm(xml):
    """MuJoCo 3.17's XML writer adds actuator input="" / "pos" (3.17's actuator input types); the page's MuJoCo WASM
    is 3.15.0 (the newest npm release at least two weeks old) and rejects it. Both mean the plain actuator."""
    return re.sub(r' input="(pos)?"', "", xml)


def meta(m, robots=M.ROBOTS):
    from bam.model import load_model
    bam = load_model(motor_name="xl330", model="m6")
    rob = []
    for name, team in robots:
        free = m.joint(f"{name}_trunk_base_freejoint").id
        j0, d0 = free, int(m.jnt_dofadr[free])
        act, jnt, quirk = [], [], []
        for jn in JOINT_NAMES:
            a = m.actuator(f"{name}_{jn}").id
            j = int(m.actuator_trnid[a, 0])
            act.append(a)
            jnt.append(j)
            quirk.append(d0 + (j - j0))
        trunk = m.body(f"{name}_trunk_base").id
        bodies = [b for b in range(m.nbody) if m.body_rootid[b] == trunk]
        geoms = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == trunk]
        s = m.sensor(f"{name}_imu_ang_vel")
        rob.append(dict(name=name, team=team, act=act, jnt=jnt, quirk=quirk,
                        qadr=[int(m.jnt_qposadr[j]) for j in jnt], dof=[int(m.jnt_dofadr[j]) for j in jnt],
                        jrange=[[float(v) for v in m.jnt_range[j]] for j in jnt],
                        free_q=int(m.jnt_qposadr[free]), free_v=d0, trunk=trunk, gyro=int(s.adr[0]),
                        mouth=m.actuator(f"{name}_mouth").id, cam=m.camera(f"{name}_eye").id,
                        bodies=bodies, body_names=[m.body(b).name[len(name) + 1:] for b in bodies],
                        geom0=min(geoms), geom1=max(geoms), jaw=m.body(f"{name}_jaw_soft").id))
    p = {k: getattr(bam, k).value for k in ("kt", "R", "armature", "friction_base", "friction_stribeck",
                                             "load_friction_motor", "load_friction_external", "load_friction_motor_stribeck",
                                             "load_friction_external_stribeck", "load_friction_motor_quad",
                                             "load_friction_external_quad", "dtheta_stribeck", "alpha", "friction_viscous")}
    p.update(error_gain=bam.actuator.error_gain, max_pwm=bam.actuator.max_pwm, kp_fw=BAM_KP_FW, vin=BAM_VIN,
             vin_drop_gain=BAM_VIN_DROP_GAIN, vin_min=BAM_VIN_MIN, max_current=None)
    try:
        j = m.joint("ball_free")
        ball = dict(ball_q=int(m.jnt_qposadr[j.id]), ball_v=int(m.jnt_dofadr[j.id]), ball_geom=m.geom("ball_geom").id)
    except KeyError:                                   # live_model_one (no pitch, no ball)
        ball = dict(ball_q=-1, ball_v=-1, ball_geom=-1)
    return dict(**ball, robots=rob, bam=p, default_pose=[float(v) for v in DEFAULT_POSE], joint_names=list(JOINT_NAMES),
                nq=m.nq, nv=m.nv, nu=m.nu, nbody=m.nbody, ngeom=m.ngeom, timestep=float(m.opt.timestep),
                frictiondof=int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF.value), pitch=M.pitch_info(),
                eye=dict(fovy=F.FOVY, w=F.W, h=F.H), mujoco_py=mujoco.__version__)


def check(xml_path, seconds=4.0):
    """The exported model (hull meshes, from XML) against match.py's World physics: one duck, BAM, 4 s walk + turn."""
    import harness as Hm
    from bam.model import load_model
    # the page's model, one duck: export the same way with one robot and no pitch
    spec, _ = inline_hulls(lambda: live_spec(robots=[("b1", None)], pitch=False))
    xml = for_wasm(spec.to_xml())
    m2 = mujoco.MjModel.from_xml_string(xml)
    m2.opt.timestep = TIMESTEP
    w = M.World(robots=[("b1", None)], randomize=False, pitch=False, cams=False)
    out = []
    for m, d in ((w.m, w.d), (m2, mujoco.MjData(m2))):
        bam = load_model(motor_name="xl330", model="m6")
        bam.actuator.kp, bam.actuator.vin, bam.actuator.max_current = BAM_KP_FW, BAM_VIN, None
        mb = M.MultiBam(bam, m, d, ["b1"])
        fq = m.jnt_qposadr[m.joint("b1_trunk_base_freejoint").id]
        d.qpos[fq:fq + 7] = [0, 0, M.HOME_TRUNK_Z, 1, 0, 0, 0]
        d.qpos[mb.qadr] = DEFAULT_POSE
        mb.q_target[:] = DEFAULT_POSE
        mujoco.mj_forward(m, d)
        pol = Hm.Policy("velstand")
        trunk = m.body("b1_trunk_base").id
        gy = m.sensor("b1_imu_ang_vel").adr[0]
        last = np.zeros(14, np.float32)
        traj = []
        for k in range(int(seconds * 50)):
            c = np.zeros(13, np.float32)
            c[0] = 0.4 if k > 25 else 0.0
            c[2] = 0.3 if k > 100 else 0.0
            o = np.concatenate([d.sensordata[gy:gy + 3], -d.xmat[trunk].reshape(3, 3)[2, :], d.qpos[mb.qadr] - DEFAULT_POSE,
                                d.qvel[mb.dof], last, c]).astype(np.float32)
            last = pol(o)
            mb.q_target[:] = DEFAULT_POSE + last
            for _ in range(4):
                mb.update()
                mujoco.mj_step(m, d)
            traj.append(d.xpos[trunk].copy())
        out.append(np.array(traj))
    diff = np.abs(out[0] - out[1]).max(1)
    print(f"exported (hull meshes, XML) vs match.World, one duck {seconds} s walk + turn: max trunk diff "
          f"{diff.max() * 1000:.3f} mm (at 1 s {diff[49] * 1000:.3f} mm); end {out[0][-1].round(3)} vs {out[1][-1].round(3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--page", action="store_true", help="also copy live.html (to OUT/..), the JS and the live mesh")
    ap.add_argument("--wasm", default=None, help="mujoco.wasm of @mujoco/mujoco@3.15.0 (copied with --page)")
    ap.add_argument("--one", action="store_true", help="also live_model_one.xml: one duck, no pitch (walk comparison)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for tag, robots in VARIANTS.items():
        spec, nv = inline_hulls(lambda: live_spec(robots=robots))
        xml = for_wasm(spec.to_xml())
        (out / f"live_model_{tag}.xml").write_text(xml)
        m = mujoco.MjModel.from_xml_string(xml)
        (out / f"live_meta_{tag}.json").write_text(json.dumps(meta(m, robots), separators=(",", ":")))
        print(f"live_model_{tag}.xml {len(xml) / 1e6:.2f} MB ({nv} hull vertices), nbody {m.nbody} ngeom {m.ngeom} nq {m.nq} nv {m.nv}")
    if a.one:
        spec, _ = inline_hulls(lambda: live_spec(robots=[("b1", None)], pitch=False))
        xml = for_wasm(spec.to_xml())
        (out / "live_model_one.xml").write_text(xml)
        m = mujoco.MjModel.from_xml_string(xml)
        (out / "live_meta_one.json").write_text(json.dumps(meta(m, [("b1", None)]), separators=(",", ":")))
    mv = json.loads((HYB / "moves_hybrid.json").read_text())
    (out / "live_moves.json").write_text(json.dumps(dict(hop=dict(params=mv["hop"]["params"]), lie_down=mv["lie_down"])))
    pol = out / "policies"
    pol.mkdir(exist_ok=True)
    for b in BRAINS:
        shutil.copyfile(POLICIES / f"{b}.onnx", pol / f"{b}.onnx")       # unchanged (onnx_mlp.js reads them)
        (pol / f"{b}.onnx.b64.txt").write_text(base64.b64encode((POLICIES / f"{b}.onnx").read_bytes()).decode())   # the artifact host does not serve .onnx
    if a.page:
        for f in ("live_sim.js", "live_game.js", "tactics.js", "onnx_mlp.js"):
            shutil.copyfile(HERE / f, out / f)
        shutil.copyfile(HERE / "live.html", out.parent / "live.html")
        if a.wasm:
            shutil.copyfile(a.wasm, out / "mujoco.wasm")
        import subprocess
        subprocess.run([sys.executable, str(HERE / "make_live_mesh.py"), str(out)], check=True)
    if a.check:
        check(out / "live_model.xml")


if __name__ == "__main__":
    main()
