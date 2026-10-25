"""Data for the hybrid's live page: its mesh (per body, like duckbody/live.py's duck_mesh.json), the test table
and the skill queue, from test_skills_hybrid.py (every skill through runtime_hybrid.py, 10 randomized bodies).

    .venv/bin/python duckbody/hybrid/test_skills_hybrid.py --page OUT_DIR     # results + one replay per skill
    .venv/bin/python duckbody/hybrid/page_data.py OUT_DIR [--mesh]           # tests.json, queue.json, skills.json
    MUJOCO_GL=osmesa .venv/bin/python duckbody/hybrid/page_data.py OUT_DIR --replays   # + hurdle / follow / soccer replays
    MUJOCO_GL=osmesa .venv/bin/python duckbody/hybrid/page_data.py OUT_DIR --soccer    # + only the soccer replay

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
         "hop", "roll", "hurdles", "hurdles_typical", "follow", "soccer"]


def games():
    """The hurdle game (course.py -> results_course_hybrid.json) and ball following (follow.py ->
    results_follow_hybrid.json): table rows for tests.json and the per-level summary for games.json."""
    tests, skills, g = {}, {}, {}
    cp, fp = HERE / "results_course_hybrid.json", HERE / "results_follow_hybrid.json"
    if cp.exists():
        c = json.loads(cp.read_text())
        S = c["summary"]
        levels = sorted({int(k[1]) for k in S})
        g["hurdles"] = {k: {kk: v[kk] for kk in ("n", "finished", "median_dist", "median_t_finished", "falls",
                                               "runs_with_falls", "median_score", "cleared", "failed_at")} for k, v in S.items()}
        g["hurdles_levels"] = levels
        best = S.get("L1_course_r") or S.get("L1_course")
        per = " · ".join(f"L{lv} {S[f'L{lv}_course_r']['finished']}/10" for lv in levels if f"L{lv}_course_r" in S)
        tests["hurdles"] = dict(success=best["finished"] / best["n"], episodes=best["n"], mean_reward=per,
                                brain="course mode: speed per obstacle, stumble reflex", layout="hybrid")
        tests["hurdles_typical"] = dict(tests["hurdles"], mean_reward="a middling run, level 2")
        skills["hurdles"] = dict(success=best["finished"], n=best["n"], note="hurdle game, level 1 (course mode)")
    if fp.exists():
        f = json.loads(fp.read_text())
        S = f["summary"]
        ok = sum(r["final_dist_cam"] < 0.25 and r["falls"] == 0 for r in f["runs"])
        tests["follow"] = dict(success=ok / S["n"], episodes=S["n"],
                               mean_reward=f"in view {100 * S['in_view_median']:.0f} %, ends {100 * S['final_dist_cam_median']:.0f} cm",
                               brain="head camera + OpenCV colour blob; velstand", layout="hybrid")
        skills["follow"] = dict(success=ok, n=S["n"], note="follow a ball with the head camera")
        g["follow"] = S
    sp = HERE / "results_soccer_hybrid.json"
    if sp.exists():
        S = json.loads(sp.read_text())["summary"]
        scored = S["matches_with_goals"]
        tests["soccer"] = dict(success=scored / S["n"], episodes=S["n"],
                               mean_reward=f"{S['goals_median']:g} goals / 2 min (median), kicks {S['connected']}/{S['kicks']} connect",
                               brain="head camera + goal sightings; velstand, ball_kick_left/right", layout="hybrid")
        skills["soccer"] = dict(success=scored, n=S["n"], note="duck soccer: goals in a 2-minute match")
        g["soccer"] = S
    return tests, skills, g


def replays(out):
    """Record the page replays: hurdles (best level-3 run, a typical level-2 run) and follow (a median run)."""
    import course as C
    import follow as F
    c = json.loads((HERE / "results_course_hybrid.json").read_text())["runs"]
    l3 = [r for r in c if r["level"] == 3 and r["mode"] == "course_r"]
    best = max(l3, key=lambda r: r["score"])
    l2 = sorted([r for r in c if r["level"] == 2 and r["mode"] == "course_r"], key=lambda r: r["score"])
    typical = l2[len(l2) // 2]
    for name, pick in (("hurdles", best), ("hurdles_typical", typical)):
        r = C.play(pick["level"], pick["seed"], pick["mode"], record=True)
        L = r["layout"]
        note = (f"hurdle game level {r['level']}, body {r['seed']}: {r['end']} in {r['t']:.1f} s, {r['falls']} falls, "
                f"score {r['score']}")
        (out / f"rollout_{name}.json").write_text(json.dumps(dict(
            fps=25, seconds=len(r["frames"]) / 25, frames=r["frames"], bodies=r["bodies"], success=r["finished"],
            task=name, seed=r["seed"], note=note, checkpoint="hybrid runtime, course mode: " + note,
            scene=dict(floor_z=L["floor_z"], boxes=[dict(pos=[round(v, 4) for v in b["pos"]], size=[round(v, 4) for v in b["size"]],
                                                         quat=[round(v, 5) for v in b["quat"]], color=b["color"]) for b in L["boxes"]],
                       stages=[[n, round(a, 3), round(b, 3)] for n, a, b in L["stages"]]),
            falls_t=[f["t"] for f in r["falls_at"]], level=r["level"], score=r["score"], curve=[]), separators=(",", ":")))
        print(name, note, len(r["frames"]))
    f = json.loads((HERE / "results_follow_hybrid.json").read_text())["runs"]
    good = [r for r in f if r["final_dist_cam"] < 0.25] or f        # a median run among those that got there
    med = sorted(good, key=lambda r: r["in_view"])[len(good) // 2]
    r = F.play(med["seed"], record=True)
    nb = len(r["bodies"]) - 1                         # the ball is the last body: drawn from `ball`, not the mesh
    note = (f"follow the ball, body {r['seed']}: in view {100 * r['in_view']:.0f} % of the time, ends "
            f"{100 * r['final_dist_cam']:.0f} cm away, {r['hops']} hops, {r['falls']} falls")
    (out / "rollout_follow.json").write_text(json.dumps(dict(
        fps=25, seconds=len(r["frames"]) / 25, frames=[fr[:nb * 7] for fr in r["frames"]], bodies=r["bodies"][:nb],
        success=r["falls"] == 0 and r["final_dist_cam"] < 0.25, task="follow", seed=r["seed"], note=note,
        checkpoint="hybrid runtime, follow: " + note, ball=r["ball"], ball_r=F.BALL_R, ball_color="#f2269e",
        cam=r["cam"], cam_fps=r["cam_fps"], det=r["det"], curve=[]), separators=(",", ":")))
    print("follow", note, len(r["frames"]), len(r["cam"]))
    soccer_replay(out)


def soccer_replay(out, max_mb=3.0):
    """The soccer replay: a good match (most goals, fewest falls) of soccer.py's 10, recorded again (same seed, same
    match); cut to the first 60 s if the file would be over max_mb."""
    import soccer as S
    runs = json.loads((HERE / "results_soccer_hybrid.json").read_text())["runs"]
    pick = max(runs, key=lambda r: (r["goals"], -r["falls"], r["connected"]))
    r = S.play(pick["seed"], record=True, cam_every=5)
    nb = len(r["bodies"]) - 1                         # the ball is the last body: drawn from `ball`
    fps = 25

    def pack(n):
        fr = [[round(v, 4) for v in f[:nb * 7]] for f in r["frames"][:n]]
        secs = len(fr) / fps
        ncam = int(secs * r["cam_fps"])
        sc = dict(floor_z=0.0, boxes=[dict(pos=[round(v, 4) for v in b["pos"]], size=[round(v, 4) for v in b["size"]],
                                           quat=b["quat"], color=b["color"]) for b in S.pitch_boxes()])
        goals = [t for t in r["goal_t"] if t <= secs]
        kicks = [dict(t=k["t"], foot=k["foot"], connected=k["connected"]) for k in r["kick_log"] if k["t"] <= secs]
        note = (f"duck soccer, body {r['seed']}: {len(goals)} goals in {secs:.0f} s, {sum(k['connected'] for k in kicks)}/"
                f"{len(kicks)} kicks connected, {sum(t <= secs for t in r['falls_t'])} falls")
        return dict(fps=fps, seconds=secs, frames=fr, bodies=r["bodies"][:nb], success=len(goals) > 0, task="soccer",
                    seed=r["seed"], note=note, checkpoint="hybrid runtime, soccer.py: " + note, ball=r["ball"][:n],
                    ball_r=S.BALL_R, ball_color="#f2269e", scene=sc, states=r["states"][:n], goals_t=goals, kicks=kicks,
                    falls_t=[t for t in r["falls_t"] if t <= secs], cam=r["cam"][:ncam], cam_fps=r["cam_fps"],
                    det=r["det"][:n], view=[-0.55, 1.25, 0.55], match_s=S.T_MATCH, curve=[])
    data = json.dumps(pack(len(r["frames"])), separators=(",", ":"))
    if len(data) > max_mb * 1e6:
        data = json.dumps(pack(60 * fps), separators=(",", ":"))
    (out / "rollout_soccer.json").write_text(data)
    print("soccer", pick["seed"], f"{len(data) / 1e6:.2f} MB")


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
    gt, gs, g = games()
    for name in ("hurdles", "hurdles_typical", "follow", "soccer"):
        if name in gt:
            tests[name] = gt[name]
            queue.append(dict(task=name, steps=1, state="done", exam={name: dict(success=gt[name]["success"])}))
            if name in gs:
                skills[name] = gs[name]
    (out / "games.json").write_text(json.dumps(g))
    if "--replays" in sys.argv:
        replays(out)
    elif "--soccer" in sys.argv:
        soccer_replay(out)
    (out / "tests.json").write_text(json.dumps(tests))
    (out / "queue.json").write_text(json.dumps(queue))
    (out / "skills.json").write_text(json.dumps(skills, indent=1))
    first = out / f"rollout_{ORDER[0]}.json"
    if first.exists():
        (out / "rollout.json").write_text(first.read_text())
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    main()
