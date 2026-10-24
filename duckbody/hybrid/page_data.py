"""Data for the hybrid's live page: its mesh (per body, like duckbody/live.py's duck_mesh.json), the test table
and the skill queue, from test_skills_hybrid.py (every skill through runtime_hybrid.py, 10 randomized bodies).

    .venv/bin/python duckbody/hybrid/test_skills_hybrid.py --page OUT_DIR     # results + one replay per skill
    .venv/bin/python duckbody/hybrid/page_data.py OUT_DIR [--mesh]           # tests.json, queue.json, skills.json

Writes OUT_DIR/duck_mesh.json (if missing, or with --mesh), tests.json, queue.json, skills.json and rollout.json
(= the first skill's replay, what the page shows first).
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


def main_old(out):
    """The first page (Microduck's brains only, six skills). Kept for reference; main() now builds the page from
    test_skills_hybrid.py's results and replays."""
    out.mkdir(parents=True, exist_ok=True)
    vel, sit, roll = Policy("velstand"), Policy("alpha_sitstand"), Policy("roulade")
    full, ground = BODIES[BODY]
    skills = {}

    def stand(sim, fr):
        sim.place("standing"); stepper(sim, fr, vel, 300)
        return upright(sim), "standing still, Microduck's walk/stand brain (velstand)"

    def walk(sim, fr):
        sim.place("standing"); stepper(sim, fr, vel, 450, cmd=lambda t: 0.3 if t >= 50 else 0.0)
        return upright(sim), "walk command 0.3 m/s"

    for name, fn, scene in (("stand", stand, full), ("walk", walk, full)):
        sim = MicroduckSim(scene, seed=0)
        r = record(sim, fn)
        (out / f"rollout_{name}.json").write_text(json.dumps(dict(r, task=name), separators=(",", ":")))
        skills[name] = dict(success=r["success"], note=r["note"], seed=0)
        if name == "stand":
            mesh(sim.m, out)
    (out / "skills.json").write_text(json.dumps(skills, indent=1))


# skill -> (short metric for the table) from results_skills_hybrid.json runs
def metric(name, runs):
    ok = [r for r in runs if r["success"]] or runs
    med = lambda k: float(np.median([r[k] for r in ok if r.get(k) is not None])) if any(r.get(k) is not None for r in ok) else None
    if name in ("walk", "run"):
        return f"{med('speed_mps'):.2f} m/s"
    if name == "squat":
        return f"{100 * med('drop_m'):.1f} cm down"
    if name == "idle":
        return f"head {med('head_yaw_range_rad'):.1f} rad, jaw {med('jaw_max_rad'):.2f} rad"
    if name.startswith("getup"):
        return f"up in {med('t_up_s'):.2f} s"
    if name == "hop":
        return f"{med('hops_per_s'):.2f} hops/s, flight {med('flight_ms_med'):.0f} ms" if med('flight_ms_med') else f"{med('hops_per_s'):.2f} hops/s"
    if name == "roll":
        return f"{med('rotation_deg'):.0f} deg"
    if name == "lie_down":
        return f"landing {med('peak_g'):.1f} g"
    if name == "swag":
        return f"left knee opens {med('knee_open_rad'):.2f} rad"
    if name == "sleep_wake":
        return f"head droop {med('neck_droop_rad'):.2f} rad"
    if name == "sit_stand":
        return f"seat at {100 * med('seat_z'):.1f} cm"
    return "-"


BRAIN = {"idle": "velstand + head command; jaw from speech", "walk": "velstand 0.4 + heading hold",
         "run": "velstand 1.35 (past training) + heading hold", "squat": "scripted, IMU balance; velstand up",
         "sit_stand": "alpha_sitstand", "swag": "alpha_sitstand + leg offsets, head cocked",
         "sleep_wake": "alpha_sitstand, head drooped", "lie_down": "scripted, searched; velstand up",
         "getup_face_down": "velstand", "getup_back": "velstand", "hop": "scripted, searched; ~1 per s",
         "roll": "roulade, then velstand"}
ORDER = ["idle", "walk", "run", "squat", "sit_stand", "swag", "sleep_wake", "lie_down", "getup_face_down", "getup_back",
         "hop", "roll"]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    if "--mesh" in sys.argv or not (out / "duck_mesh.json").exists():
        mesh(MicroduckSim(str(HERE / "scene_hybrid_allcollisions_noantenna.xml"), seed=0).m, out)
    res = json.loads((HERE / "results_skills_hybrid.json").read_text())
    tests, queue, skills = {}, [], {}
    for name in ORDER:
        if name not in res:
            continue
        r = res[name]
        tests[name] = dict(success=r["success"] / r["n"], episodes=r["n"], mean_reward=metric(name, r["runs"]),
                           brain=BRAIN.get(name, ""), layout="hybrid")
        queue.append(dict(task=name, steps=1, state="done", exam={name: dict(success=r["success"] / r["n"])}))
        skills[name] = dict(success=r["success"], n=r["n"], note=r["desc"])
    (out / "tests.json").write_text(json.dumps(tests))
    (out / "queue.json").write_text(json.dumps(queue))
    (out / "skills.json").write_text(json.dumps(skills, indent=1))
    first = out / f"rollout_{ORDER[0]}.json"
    if first.exists():
        (out / "rollout.json").write_text(first.read_text())
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    main()
