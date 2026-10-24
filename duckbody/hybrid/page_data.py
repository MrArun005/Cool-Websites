"""Data for the hybrid's live page: its mesh (per body, like duckbody/live.py's duck_mesh.json) and one replay per skill,
all with Microduck's own brains on the hybrid (owner's head, antennas drawn but not colliding: hybrid_noantenna).

    MUJOCO_GL=osmesa .venv/bin/python duckbody/hybrid/page_data.py OUT_DIR

Writes OUT_DIR/duck_mesh.json, OUT_DIR/rollout_<skill>.json and OUT_DIR/skills.json (per skill: success, note).
"""
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "microduck_port"))
sys.path.insert(0, str(HERE))
from harness import MicroduckSim, Policy  # noqa: E402
from evaluate import upright  # noqa: E402
from evaluate_hybrid import BODIES  # noqa: E402

BODY = "hybrid_noantenna"


def mesh(m, out, max_faces=1500):
    """Visible geoms per body in body coordinates (ints, x1e4), meshes simplified to about max_faces triangles."""
    parts = {}
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
        if b == 0 or rgba[3] < 0.05 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            if m.geom_group[g] not in (0, 1, 2):
                continue
        if m.geom_group[g] == 3:                                    # collision-only shapes
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid]: m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        if len(f) > max_faces:                                       # proper decimation (quadric), not dropped faces
            import fast_simplification
            v, f = fast_simplification.simplify(np.asarray(v, np.float32), np.asarray(f, np.int64), 1 - max_faces / len(f))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        parts.setdefault(m.body(b).name, []).append(dict(
            color="#%02x%02x%02x" % tuple(int(255 * c) for c in rgba[:3]),
            v=np.round(v * 1e4).astype(int).ravel().tolist(), f=f.ravel().tolist()))
    names = [m.body(b).name for b in range(1, m.nbody)]
    (out / "duck_mesh.json").write_text(json.dumps(dict(scale=1e-4, bodies=names, parts=parts), separators=(",", ":")))


def record(sim, run):
    frames = []
    ok, note = run(sim, frames)
    return dict(fps=25, seconds=len(frames) / 25, frames=frames, bodies=sim.body_names(), success=bool(ok), note=note)


def stepper(sim, frames, pol, n, cmd=None):
    for t in range(n):
        if cmd is not None:
            sim.command[:] = 0
            c = cmd(t)
            if c is not None:
                sim.command[0] = c
        sim.step(pol)
        if t % 2 == 0:
            frames.append(sim.frame())


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    vel, sit, roll = Policy("velstand"), Policy("alpha_sitstand"), Policy("roulade")
    full, ground = BODIES[BODY]
    skills = {}

    def stand(sim, fr):
        sim.place("standing"); stepper(sim, fr, vel, 300)
        return upright(sim), "standing still, Microduck's walk/stand brain (velstand)"

    def walk(sim, fr):
        sim.place("standing"); stepper(sim, fr, vel, 450, cmd=lambda t: 0.3 if t >= 50 else 0.0)
        return upright(sim), "walk command 0.3 m/s (in sim it covers ~0.1 m/s and drifts; same on Microduck's own body)"

    def getup_down(sim, fr):
        sim.place("face_down"); stepper(sim, fr, vel, 300)
        return upright(sim), "face down -> standing (velstand)"

    def getup_back(sim, fr):
        sim.place("face_up"); stepper(sim, fr, vel, 300)
        return upright(sim), "on its back -> standing (velstand)"

    def sitstand(sim, fr):
        sim.place("standing"); stepper(sim, fr, sit, 400, cmd=lambda t: 1.0 if 50 <= t < 250 else 0.0)
        return upright(sim), "sit at 1 s, stand at 5 s (alpha_sitstand)"

    def roulade(sim, fr):
        sim.place("standing"); stepper(sim, fr, vel, 75)
        sim.last_action[:] = 0
        rot = 0.0
        for t in range(int(5.0 * 50)):
            sim.step(roll if t < 100 else vel)
            rot += float(sim.d.sensordata[sim.gyro_adr][1]) * 0.02
            if t % 2 == 0:
                fr.append(sim.frame())
        return upright(sim) and abs(rot) > math.radians(300), f"forward roll (roulade brain, 2 s slot), rotated {abs(math.degrees(rot)):.0f} deg, then velstand"

    for name, fn, scene in (("stand", stand, full), ("walk", walk, full), ("getup_face_down", getup_down, full),
                            ("getup_back", getup_back, full), ("sit_stand", sitstand, ground), ("roll", roulade, ground)):
        for seed in range(6):
            sim = MicroduckSim(scene, seed=seed)
            r = record(sim, fn)
            if r["success"] or seed == 5:
                break
        (out / f"rollout_{name}.json").write_text(json.dumps(dict(r, task=name), separators=(",", ":")))
        skills[name] = dict(success=r["success"], note=r["note"], seed=seed)
        print(name, r["success"], r["note"], flush=True)
        if name == "stand":
            mesh(sim.m, out)
    (out / "skills.json").write_text(json.dumps(skills, indent=1))


if __name__ == "__main__":
    main()
