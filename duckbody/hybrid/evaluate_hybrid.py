"""Microduck's official brains (ONNX, unchanged) on the hybrid duck vs on Microduck's own body.

    .venv/bin/python duckbody/hybrid/evaluate_hybrid.py [--n 10] [--seed 0] [--bodies microduck,hybrid,hybrid_ownermass]

Same harness (BAM XL330 servos, 61-D obs, 50 Hz, 5 ms physics, obs noise, 3-6 step command delay),
same tests, same seeds and same success rules as duckbody/microduck_port/evaluate.py, run on each body:

  microduck          Microduck's own MJCF (control, should reproduce microduck_port/results.json)
  hybrid             Microduck + owner's head, head mass/CoM/inertia matched to Microduck's head
  hybrid_ownermass   Microduck + owner's head with the owner's estimated head mass (0.158 kg vs 0.189 kg)
  hybrid_noantenna   as hybrid, antennas visual only (no collision)
  hybrid_ownermass_noantenna   owner's head mass and no antenna collision
  hybrid_mdcollision as hybrid, but Microduck's own head collision meshes (owner head drawn only): control
  hybrid_small       owner head x2/3 (Microduck's head width), antennas visual only
(variants are built by build_hybrid.py; see head_report.json)

Extra, not in evaluate.py (wobble numbers, so "works but wobbles" can be told apart from "works"):
  stand 10 s    trunk tilt mean / max (deg), trunk angular-rate RMS (rad/s), horizontal drift (m)
  walk 0.15     the same while walking, plus forward speed and sideways drift
  talking       stand 10 s with the mouth flapping (0 <-> 35 deg at 3 Hz), velstand: does it stay up?
Writes duckbody/hybrid/results_hybrid.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PORT = HERE.parent / "microduck_port"
sys.path.insert(0, str(PORT))
from harness import SCENE_FULL, SCENE_GROUND, MicroduckSim, Policy  # noqa: E402
from evaluate import getup, roulade, sitstand, summarize, walk  # noqa: E402

BODIES = {
    "microduck": (SCENE_FULL, SCENE_GROUND),
    "hybrid": (HERE / "scene_hybrid_allcollisions.xml", HERE / "scene_hybrid.xml"),
    "hybrid_ownermass": (HERE / "scene_hybrid_allcollisions_ownermass.xml", HERE / "scene_hybrid_ownermass.xml"),
    "hybrid_noantenna": (HERE / "scene_hybrid_allcollisions_noantenna.xml", HERE / "scene_hybrid_noantenna.xml"),
    "hybrid_ownermass_noantenna": (HERE / "scene_hybrid_allcollisions_ownermass_noantenna.xml",
                                   HERE / "scene_hybrid_ownermass_noantenna.xml"),
    "hybrid_mdcollision": (HERE / "scene_hybrid_allcollisions_mdcollision.xml", HERE / "scene_hybrid_mdcollision.xml"),
    "hybrid_small": (HERE / "scene_hybrid_allcollisions_small.xml", HERE / "scene_hybrid_small.xml"),
}


def wobble(sim, pol, vx=0.0, seconds=10.0, talk=False):
    """Stand (vx=0) or walk; 1 s settle, then measure trunk tilt, angular-rate RMS and drift."""
    sim.place("standing")
    mouth = sim.m.actuator("mouth").id if talk else None
    tilts, rates, fell = [], [], False
    for t in range(int(seconds * 50) + 50):
        sim.command[:] = 0
        if t >= 50:
            sim.command[0] = vx
        if t == 50:
            p0 = sim.d.xpos[sim.trunk][:2].copy()
            R0 = sim.d.xmat[sim.trunk].reshape(3, 3)
            fwd = R0[:2, 0] / np.linalg.norm(R0[:2, 0])
            left = np.array([-fwd[1], fwd[0]])
        if talk:
            sim.d.ctrl[mouth] = 0.61 * (0.5 - 0.5 * math.cos(2 * math.pi * 3.0 * t / 50))
        sim.step(pol)
        if t >= 50:
            tilts.append(sim.tilt_deg())
            rates.append(float(np.linalg.norm(sim.d.sensordata[sim.gyro_adr])))
        if sim.tilt_deg() > 45:
            fell = True
    dp = sim.d.xpos[sim.trunk][:2] - p0
    ok = not fell and sim.tilt_deg() < 25 and sim.trunk_z() > 0.10
    return dict(success=bool(ok), fell=fell, tilt_mean=round(float(np.mean(tilts)), 2), tilt_max=round(float(np.max(tilts)), 2),
                rate_rms=round(float(np.sqrt(np.mean(np.square(rates)))), 3),
                forward_m=round(float(dp @ fwd), 3), lateral_m=round(float(dp @ left), 3),
                drift_m=round(float(np.linalg.norm(dp)), 3), final_z=round(sim.trunk_z(), 4))


def wsum(name, runs):
    out = summarize(name, runs)
    keys = ("tilt_mean", "tilt_max", "rate_rms", "forward_m", "lateral_m", "drift_m")
    out["mean"] = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in keys}
    print(" " * 44, out["mean"], flush=True)
    return out


def run_body(body, n, seed):
    full_scene, ground_scene = BODIES[body]
    vel, sits, roll, walkp = Policy("velstand"), Policy("alpha_sitstand"), Policy("roulade"), Policy("alpha_walking")
    full = MicroduckSim(full_scene, seed=seed)
    ground = MicroduckSim(ground_scene, seed=seed + 1)
    res = {}
    print(f"== {body}", flush=True)
    for kind in ("face_down", "face_up", "sitting", "standing"):
        res[f"velstand getup {kind}"] = summarize(f"velstand get-up from {kind}", [getup(full, vel, kind) for _ in range(n)])
    for kind in ("face_down", "face_up"):
        res[f"velstand getup {kind} random joints"] = summarize(
            f"velstand get-up from {kind}, random joints", [getup(full, vel, kind, random_joints=True) for _ in range(n)])
    res["sitstand stand from sitting"] = summarize("alpha_sitstand stand-up from sitting (flag 0)",
                                                   [getup(ground, sits, "sitting") for _ in range(n)])
    res["sitstand sit then stand"] = summarize("alpha_sitstand sit (1 s..5 s) then stand", [sitstand(ground, sits) for _ in range(n)])
    for vx in (0.15, 0.3, 0.4):
        res[f"velstand walk vx={vx}"] = summarize(f"velstand walk vx={vx} 10 s", [walk(full, vel, vx) for _ in range(n)])
    res["alpha_walking walk vx=0.15"] = summarize("alpha_walking walk vx=0.15 10 s", [walk(ground, walkp, 0.15) for _ in range(n)])
    for dur in (1.0, 2.0):
        res[f"roulade {dur}s"] = summarize(f"roulade ({dur} s window) -> velstand", [roulade(ground, roll, vel, dur) for _ in range(n)])
    # wobble
    res["wobble stand velstand"] = wsum("wobble: velstand stand 10 s", [wobble(full, vel) for _ in range(n)])
    res["wobble walk velstand vx=0.15"] = wsum("wobble: velstand walk 0.15 10 s", [wobble(full, vel, 0.15) for _ in range(n)])
    if body != "microduck":
        res["talking stand velstand"] = wsum("talking: velstand stand, mouth flapping", [wobble(full, vel, talk=True) for _ in range(n)])
        res["talking walk velstand vx=0.15"] = wsum("talking: velstand walk 0.15, mouth flapping",
                                                    [wobble(full, vel, 0.15, talk=True) for _ in range(n)])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bodies", default=",".join(BODIES))
    a = ap.parse_args()
    out_path = HERE / "results_hybrid.json"
    allres = json.loads(out_path.read_text()) if out_path.exists() else {}
    for body in a.bodies.split(","):
        t0 = time.time()
        allres[body] = run_body(body, a.n, a.seed)
        allres[body]["_meta"] = dict(seed=a.seed, n=a.n, wall_s=round(time.time() - t0))
        out_path.write_text(json.dumps(allres, indent=1))
    # compact table
    keys = [k for k in allres[next(iter(allres))] if not k.startswith("_")]
    print(f"\n{'test':44s}" + "".join(f"{b:>28s}" for b in allres))
    for k in sorted(set().union(*[set(v) for v in allres.values()]) - {"_meta"}, key=lambda x: (x not in keys, x)):
        print(f"{k:44s}" + "".join(f"{(str(v[k]['success']) + '/' + str(v[k]['n'])) if k in v else '-':>28s}" for v in allres.values()))


if __name__ == "__main__":
    main()
