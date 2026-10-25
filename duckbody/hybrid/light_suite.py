"""Lighter / smaller head study (README "Lighter head"): the full test suite on each head variant of build_hybrid.py.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/light_suite.py run [--jobs 3] [--bodies _noantenna,_m160,...] [--logs DIR]
    $PY duckbody/hybrid/light_suite.py merge          # -> light/results_light.json + the README table (printed)
    $PY duckbody/hybrid/light_suite.py promote --bodies _m175   # that body's runs -> the main results_*_hybrid.json

Per body (HYBRID_BODY=<suffix>, so runtime_hybrid.SCENE is scene_hybrid_allcollisions<suffix>.xml), single-process
units, `--jobs` at a time (each with one numpy thread):
  * test_skills_hybrid.py: the 12 skills x 10 randomized bodies (seeds 0-9), one unit per skill
  * course.py: hurdles, levels 1-3, course mode + stumble reflex (course_r), seeds 0-9, one unit per level
  * follow.py: seeds 0-9 (two units)
  * soccer.py: 10 matches of 2 min, seeds 0-9 (two units)
  * match.py --one: the 5v5, seeds 0 and 1, one half of 3 min each
Results: light/<body>/*.json (a unit whose file exists is skipped, so a run can be resumed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "light"
PY = sys.executable
SKILLS = ["idle", "walk", "run", "squat", "sit_stand", "swag", "sleep_wake", "lie_down", "getup_face_down",
          "getup_back", "hop", "roll"]
BODIES = {          # suffix -> label, head mass (g), head scale
    "_noantenna": ("baseline (189 g, x1)", 189, 1.0),
    "_m175": ("175 g", 175, 1.0),
    "_m160": ("160 g", 160, 1.0),
    "_m140": ("140 g", 140, 1.0),
    "_m120": ("120 g", 120, 1.0),
    "_m100": ("100 g", 100, 1.0),
    "_k85": ("x0.85 (159 g)", 159, 0.85),
    "_k75": ("x0.75 (142 g)", 142, 0.75),
    "_human": ("human-like head (175 g)", 175, None),             # build_shapes.py; mass build_hybrid.SHAPE_KG
    "_roundduck": ("round duck head (175 g)", 175, None),
    "_human_tall": ("human-like, first tall version (160 g, skills only)", 160, None),
    "_roundduck_tall": ("round duck, first tall version (160 g, skills only)", 160, None),
}
MATCH_SEEDS = (0, 1)
MORE_HURDLES = ("_noantenna", "_m175", "_m160")


def units(body):
    d = OUT / body
    u = []
    for s in MATCH_SEEDS:      # longest first
        u.append((f"match_s{s}", [PY, "match.py", "--one", str(s), "--t", "180", "--halves", "1", "--out", str(d / f"match_s{s}.json")]))
    for a, b in ((0, 4), (5, 9)):
        u.append((f"soccer_{a}-{b}", [PY, "soccer.py", "--seeds", f"{a}-{b}", "--jobs", "1", "--out", str(d / f"soccer_{a}-{b}.json")]))
        u.append((f"follow_{a}-{b}", [PY, "follow.py", "--seeds", f"{a}-{b}", "--jobs", "1", "--out", str(d / f"follow_{a}-{b}.json")]))
    for lv in (3, 2, 1):
        u.append((f"course_L{lv}", [PY, "course.py", "--levels", str(lv), "--modes", "course_r", "--jobs", "1",
                                    "--out", str(d / f"course_L{lv}.json")]))
    if body in MORE_HURDLES:   # 10 more bodies at the hard levels (10 per cell is noisy there)
        for lv in (2, 3):
            u.append((f"course_L{lv}_seeds10-19", [PY, "course.py", "--levels", str(lv), "--modes", "course_r", "--seeds", "10-19",
                                                   "--jobs", "1", "--out", str(d / f"course_L{lv}_seeds10-19.json")]))
    for s in SKILLS:
        u.append((f"skills_{s}", [PY, "test_skills_hybrid.py", "--skills", s, "--out", str(d / f"skills_{s}.json")]))
    return [(body, name, cmd, Path(cmd[-1])) for name, cmd in u]


def run(a):
    logs = Path(a.logs)
    logs.mkdir(parents=True, exist_ok=True)
    todo = []
    for body in a.bodies.split(","):
        (OUT / body).mkdir(parents=True, exist_ok=True)
        todo += [u for u in units(body) if not u[3].exists()]
    # interleave bodies so the long units (matches) of every body start early
    rank = {"skills": 0, "course": 1, "follow": 2, "soccer": 3, "match": 4}     # the skills of every body first
    todo.sort(key=lambda u: (rank[u[1].split("_")[0]], list(BODIES).index(u[0]), u[1]))
    print(f"{len(todo)} units", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MUJOCO_GL="osmesa")

    def one(u):
        body, name, cmd, out = u
        t = time.time()
        tmp = out.with_suffix(".part.json")
        cmd = cmd[:-1] + [str(tmp)]
        with open(logs / f"{body}_{name}.log", "w") as f:
            r = subprocess.run(cmd, cwd=HERE, env=dict(env, HYBRID_BODY=body), stdout=f, stderr=subprocess.STDOUT)
        if r.returncode == 0 and tmp.exists():
            tmp.rename(out)
        print(f"{time.strftime('%H:%M:%S')} {body:12s} {name:22s} rc {r.returncode} {time.time() - t:6.0f} s", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, todo))


# ---------------------------------------------------------------------------------------------- merge
def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def hop_height_mm(flight_ms):
    """Ballistic height of the trunk for a flight of t ms: g t^2 / 8."""
    return 9.81 * (flight_ms / 1000) ** 2 / 8 * 1000


def summarize(body):
    d = OUT / body
    S = dict(body=body, label=BODIES[body][0], head_g=BODIES[body][1], k=BODIES[body][2], missing=[])
    sk = {}
    for s in SKILLS:
        f = d / f"skills_{s}.json"
        if f.exists():
            sk[s] = json.loads(f.read_text())[s]
        else:
            S["missing"].append(f"skills_{s}")
    S["skills"] = {s: f"{r['success']}/{r['n']}" for s, r in sk.items()}
    S["skills_passed"] = sum(r["success"] for r in sk.values())
    S["skills_total"] = sum(r["n"] for r in sk.values())

    def runs(s, ok=True):
        return [r for r in sk.get(s, {}).get("runs", []) if r["success"] or not ok]
    S["walk_mps"] = med(r["speed_mps"] for r in runs("walk"))
    S["run_mps"] = med(r["speed_mps"] for r in runs("run"))
    S["walk_tilt_max"] = med(r.get("max_tilt") for r in runs("walk", False))
    S["run_tilt_max"] = med(r.get("max_tilt") for r in runs("run", False))
    S["idle_tilt_max"] = med(r.get("max_tilt") for r in runs("idle", False))
    hop = runs("hop", False)
    S["hop_flight_ms"] = med(r["flight_ms_med"] for r in hop if r["flight_ms"])
    S["hop_height_mm"] = hop_height_mm(S["hop_flight_ms"]) if S["hop_flight_ms"] else None
    S["hops_per_s"] = med(r["hops_per_s"] for r in hop)
    S["hop_landing_bw"] = med(r["landing_bw"] for r in hop)
    S["hop_tilt_max"] = med(r["max_tilt"] for r in hop)
    S["lie_down_peak_g"] = med(r.get("peak_g") for r in runs("lie_down", False))
    S["getup_back_s"] = med(r.get("t_up_s") for r in runs("getup_back"))
    S["getup_face_down_s"] = med(r.get("t_up_s") for r in runs("getup_face_down"))
    S["roll_deg"] = med(r.get("rotation_deg") for r in runs("roll", False))
    S["squat_tilt"] = med(r.get("tilt_squat") for r in runs("squat", False))
    # hurdles
    S["course"] = {}
    for lv in (1, 2, 3):
        f = d / f"course_L{lv}.json"
        if not f.exists():
            S["missing"].append(f"course_L{lv}")
            continue
        c = json.loads(f.read_text())["summary"][f"L{lv}_course_r"]
        S["course"][lv] = dict(finished=c["finished"], n=c["n"], falls=c["falls"], t_med=c.get("median_t_finished"))
        extra = sorted(d.glob(f"course_L{lv}_seeds*.json"))           # more bodies (light_suite.py extra)
        if extra:
            rs = [r for x in [f] + extra for r in json.loads(x.read_text())["runs"]]
            S["course"][lv]["more"] = dict(finished=sum(r["finished"] for r in rs), n=len(rs), falls=sum(r["falls"] for r in rs))
    # follow
    fr = []
    for part in ("0-4", "5-9"):
        f = d / f"follow_{part}.json"
        if f.exists():
            fr += json.loads(f.read_text())["runs"]
        else:
            S["missing"].append(f"follow_{part}")
    if fr:
        S["follow"] = dict(n=len(fr), in_view=med(r["in_view"] for r in fr), final_cm=100 * med(r["final_dist_cam"] for r in fr),
                           within_25=sum(r["final_dist_cam"] < 0.25 for r in fr), falls=sum(r["falls"] for r in fr),
                           hops=sum(r["hops"] for r in fr))
    # soccer
    sr = []
    for part in ("0-4", "5-9"):
        f = d / f"soccer_{part}.json"
        if f.exists():
            sr += json.loads(f.read_text())["runs"]
        else:
            S["missing"].append(f"soccer_{part}")
    if sr:
        S["soccer"] = dict(n=len(sr), goals=sum(r["goals"] for r in sr), goals_med=med(r["goals"] for r in sr),
                           with_goal=sum(r["goals"] > 0 for r in sr), kicks=sum(r["kicks"] for r in sr),
                           connected=sum(r["connected"] for r in sr), falls=sum(r["falls"] for r in sr))
    # match
    mr = []
    for s in MATCH_SEEDS:
        f = d / f"match_s{s}.json"
        if f.exists():
            mr.append(json.loads(f.read_text()))
        else:
            S["missing"].append(f"match_s{s}")
    if mr:
        T = lambda r, k: sum(r["teams"][t][k] for t in ("blue", "yellow"))  # noqa: E731
        S["match"] = dict(n=len(mr), scores=[r["score"] for r in mr], goals=sum(sum(r["score"]) for r in mr),
                          kicks=sum(T(r, "kicks") for r in mr), connected=sum(T(r, "connected") for r in mr),
                          shots=sum(T(r, "shots") for r in mr),
                          falls=sum(p["falls"] for r in mr for p in r["players"].values()),
                          rtf=med(r["rtf"] for r in mr))
    return S


def fmt(v, f="{:.2f}"):
    return "-" if v is None else f.format(v)


def table(rows):
    cols = [r["label"] for r in rows]
    L = ["| test | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]

    def line(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(r) for r in rows) + " |")
    line("12 skills x 10 bodies, passed", lambda r: f"**{r['skills_passed']}/{r['skills_total']}**")
    line("skills below 10/10", lambda r: ", ".join(f"{s} {v}" for s, v in r["skills"].items() if not v.startswith("10/")) or "none")
    line("walk speed (m/s, median)", lambda r: fmt(r["walk_mps"]))
    line("run speed (m/s, median)", lambda r: fmt(r["run_mps"]))
    line("hop: flight ms / height mm / hops per s", lambda r: f"{fmt(r['hop_flight_ms'], '{:.0f}')} / {fmt(r['hop_height_mm'], '{:.1f}')} / {fmt(r['hops_per_s'])}")
    line("hop landing (x body weight)", lambda r: fmt(r["hop_landing_bw"], "{:.1f}"))
    line("lie-down peak (g)", lambda r: fmt(r["lie_down_peak_g"], "{:.1f}"))
    line("max tilt: idle / walk / run / hop (deg)", lambda r: " / ".join(fmt(r[k], "{:.1f}") for k in ("idle_tilt_max", "walk_tilt_max", "run_tilt_max", "hop_tilt_max")))
    line("get up: face down / back (s)", lambda r: f"{fmt(r['getup_face_down_s'])} / {fmt(r['getup_back_s'])}")
    line("roll (deg)", lambda r: fmt(r["roll_deg"], "{:.0f}"))
    for lv in (1, 2, 3):
        line(f"hurdles L{lv}: finished, falls, median s",
             lambda r, lv=lv: (lambda c: f"{c['finished']}/{c['n']}, {c['falls']}, {fmt(c['t_med'], '{:.0f}')}"
                               + (f" ({c['more']['finished']}/{c['more']['n']} on {c['more']['n']} bodies)" if "more" in c else ""))(r["course"][lv]) if lv in r["course"] else "-")
    line("follow: in view, final cm, within 25 cm, falls",
         lambda r: (lambda f: f"{100 * f['in_view']:.0f} %, {f['final_cm']:.1f}, {f['within_25']}/{f['n']}, {f['falls']}")(r["follow"]) if "follow" in r else "-")
    line("soccer (10 x 2 min): goals, with goal, kicks connected, falls",
         lambda r: (lambda s: f"{s['goals']}, {s['with_goal']}/{s['n']}, {s['connected']}/{s['kicks']}, {s['falls']}")(r["soccer"]) if "soccer" in r else "-")
    line("5v5 (seeds 0, 1; 1 x 3 min): scores, kicks (connected), shots, falls",
         lambda r: (lambda m: f"{' '.join(f'{a}:{b}' for a, b in m['scores'])}, {m['kicks']} ({m['connected']}), {m['shots']}, {m['falls']}")(r["match"]) if "match" in r else "-")
    return "\n".join(L)


def merge(a):
    rows = [summarize(b) for b in a.bodies.split(",") if (OUT / b).exists()]
    (OUT / "results_light.json").write_text(json.dumps(rows, indent=1))
    print(table(rows))
    for r in rows:
        if r["missing"]:
            print(r["body"], "missing", r["missing"])


def promote(a):
    """The recommended body's runs -> the main results files (results_{skills,course,follow,soccer}_hybrid.json), the
    ones page_data.py reads. The match results come from the page's recorded matches (page_data.py --match-only)."""
    sys.path.insert(0, str(HERE))
    os.environ["HYBRID_BODY"] = a.bodies
    import course as C
    import soccer as SO
    d = OUT / a.bodies
    sk = {s: json.loads((d / f"skills_{s}.json").read_text())[s] for s in SKILLS}
    sk["_meta"] = dict(n=10, body=f"scene_hybrid_allcollisions{a.bodies}.xml", seeds="0-9", source=f"light/{a.bodies}",
                       randomization="harness.MicroduckSim.randomize_body (mass/inertia x0.95-1.05, trunk+head CoM +-3mm, "
                                     "friction x0.7-1.3, kp x0.85-1.15, 6.5-8.2 V, delay 3-6 steps) + obs noise")
    (HERE / "results_skills_hybrid.json").write_text(json.dumps(sk, indent=1))
    runs = [r for lv in (1, 2, 3) for r in json.loads((d / f"course_L{lv}.json").read_text())["runs"]]
    (HERE / "results_course_hybrid.json").write_text(json.dumps(dict(summary=C.summarize(runs), runs=runs), indent=1))
    fr = [r for part in ("0-4", "5-9") for r in json.loads((d / f"follow_{part}.json").read_text())["runs"]]
    fs = dict(n=len(fr), in_view_median=float(np.median([r["in_view"] for r in fr])),
              in_view_front_median=float(np.median([r["in_view_front"] for r in fr])),
              final_dist_cam_median=float(np.median([r["final_dist_cam"] for r in fr])),
              final_dist_trunk_median=float(np.median([r["final_dist_trunk"] for r in fr])),
              within_25cm=sum(r["final_dist_cam"] < 0.25 for r in fr), falls=sum(r["falls"] for r in fr),
              hops=sum(r["hops"] for r in fr), reacquired=sum(len(r["reacquire_s"]) > 0 for r in fr),
              reacquire_median_s=float(np.median([x for r in fr for x in r["reacquire_s"]])) if any(r["reacquire_s"] for r in fr) else None)
    (HERE / "results_follow_hybrid.json").write_text(json.dumps(dict(summary=fs, runs=fr), indent=1))
    sr = [r for part in ("0-4", "5-9") for r in json.loads((d / f"soccer_{part}.json").read_text())["runs"]]
    (HERE / "results_soccer_hybrid.json").write_text(json.dumps(dict(summary=SO.summarize(sr), runs=sr), indent=1))
    print("promoted", a.bodies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "merge", "promote"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--bodies", default=",".join(BODIES))
    ap.add_argument("--logs", default=str(OUT / "logs"))
    a = ap.parse_args()
    {"run": run, "merge": merge, "promote": promote}[a.what](a)


if __name__ == "__main__":
    main()
