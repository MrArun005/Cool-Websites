"""10 randomized starts per test, Microduck body + official Microduck brains. Writes results.json.

    .venv/bin/python duckbody/microduck_port/evaluate.py [--n 10] [--seed 0]

Success ("upright & at standing height at the end, no fall"):
  get-up tests   tilt < 25 deg and trunk z > 0.10 m for the whole last 1 s, and after the first
                 recovery it never tips past 45 deg again.
  walk           never tips past 45 deg, ends upright & at standing height.
  sit            reaches trunk z < 0.08 with tilt < 25 deg (seated) and then, on flag 0, stands again.
  roulade        trunk turns >= 300 deg forward (integrated pitch rate) and ends upright & standing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import STAND_Z, UP_TILT_DEG, UP_Z, SCENE_FULL, SCENE_GROUND, MicroduckSim, Policy  # noqa: E402

HERE = Path(__file__).resolve().parent
FALL_DEG = 45.0


def upright(sim, z=STAND_Z):
    return sim.tilt_deg() < UP_TILT_DEG and sim.trunk_z() > z


def getup(sim, pol, kind, seconds=10.0, random_joints=False, record=False):
    sim.place(kind, random_joints=random_joints)
    steps = int(seconds * 50)
    t_rec, refell, tail_ok, frames, max_w = None, False, True, [], 0.0
    for t in range(steps):
        sim.step(pol)
        max_w = max(max_w, float(np.linalg.norm(sim.d.sensordata[sim.gyro_adr])))
        if record and t % 2 == 0:
            frames.append(sim.frame())
        if t_rec is None and sim.tilt_deg() < UP_TILT_DEG and sim.trunk_z() > UP_Z:
            t_rec = (t + 1) / 50
        if t_rec is not None and sim.tilt_deg() > FALL_DEG:
            refell = True
        if t >= steps - 50 and not upright(sim):
            tail_ok = False
    ok = t_rec is not None and tail_ok and not refell
    out = dict(success=bool(ok), t_recover=t_rec, refell=refell, final_tilt=round(sim.tilt_deg(), 1),
               final_z=round(sim.trunk_z(), 4), max_gyro=round(max_w, 1))
    return (out, frames) if record else out


def walk(sim, pol, vx=0.15, seconds=10.0):
    sim.place("standing")
    fell = False
    for t in range(int(seconds * 50) + 50):
        sim.command[:] = 0
        if t >= 50:
            sim.command[0] = vx
        if t == 50:
            p0 = sim.d.xpos[sim.trunk][:2].copy()
            R0 = sim.d.xmat[sim.trunk].reshape(3, 3)
            fwd = R0[:2, 0] / np.linalg.norm(R0[:2, 0])
            left = np.array([-fwd[1], fwd[0]])
        sim.step(pol)
        if sim.tilt_deg() > FALL_DEG:
            fell = True
    dp = sim.d.xpos[sim.trunk][:2] - p0
    ok = (not fell) and upright(sim)
    return dict(success=bool(ok), fell=fell, forward_m=round(float(dp @ fwd), 3), lateral_m=round(float(dp @ left), 3),
                mean_vx=round(float(dp @ fwd) / seconds, 3), final_tilt=round(sim.tilt_deg(), 1), final_z=round(sim.trunk_z(), 4))


def sitstand(sim, pol):
    """alpha_sitstand: command twist.vx is the posture flag (1 sit, 0 stand), manifest ramp_s 2."""
    sim.place("standing")
    seated, fell = False, False
    for t in range(int(8 * 50)):
        sim.command[:] = 0
        sim.command[0] = 1.0 if 50 <= t < 250 else 0.0      # sit at 1 s, stand at 5 s
        sim.step(pol)
        if 150 <= t < 250 and sim.trunk_z() < 0.08 and sim.tilt_deg() < UP_TILT_DEG:
            seated = True
        if sim.tilt_deg() > FALL_DEG:
            fell = True
    ok = seated and not fell and upright(sim)
    return dict(success=bool(ok), seated=seated, fell=fell, final_tilt=round(sim.tilt_deg(), 1), final_z=round(sim.trunk_z(), 4))


def roulade(sim, roll_pol, stand_pol, duration=1.0, settle=1.5, after=5.0):
    """Manifest: roulade kind=episodic duration_s=1.0, then the walk slot (velstand) takes back over."""
    sim.place("standing")
    for _ in range(int(settle * 50)):
        sim.step(stand_pol)
    sim.last_action[:] = 0   # policy switch; infer_policy keeps last_action, daemon too -> keep? see README
    rot, max_rot = 0.0, 0.0
    for t in range(int((duration + after) * 50)):
        sim.step(roll_pol if t < duration * 50 else stand_pol)
        rot += float(sim.d.sensordata[sim.gyro_adr][1]) * 0.02
        max_rot = max(max_rot, abs(rot))
    rolled = max_rot >= math.radians(300)
    ok = rolled and upright(sim)
    return dict(success=bool(ok), rolled=bool(rolled), rotation_deg=round(math.degrees(max_rot)),
                final_tilt=round(sim.tilt_deg(), 1), final_z=round(sim.trunk_z(), 4))


def summarize(name, runs):
    k = sum(r["success"] for r in runs)
    extra = ""
    trec = [r["t_recover"] for r in runs if r.get("t_recover")]
    if trec:
        extra = f"  t_recover median {np.median(trec):.2f}s"
    if "mean_vx" in runs[0]:
        extra = f"  mean vx {np.mean([r['mean_vx'] for r in runs]):.3f} m/s"
    if "rotation_deg" in runs[0]:
        extra = f"  rotation median {np.median([r['rotation_deg'] for r in runs]):.0f} deg"
    print(f"{name:42s} {k}/{len(runs)}{extra}", flush=True)
    return dict(success=k, n=len(runs), runs=runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    vel, sits, roll, walkp = Policy("velstand"), Policy("alpha_sitstand"), Policy("roulade"), Policy("alpha_walking")
    full = MicroduckSim(SCENE_FULL, seed=a.seed)
    ground = MicroduckSim(SCENE_GROUND, seed=a.seed + 1)
    res = {}
    for kind in ("face_down", "face_up", "sitting", "standing"):
        res[f"velstand getup {kind}"] = summarize(f"velstand get-up from {kind}", [getup(full, vel, kind) for _ in range(a.n)])
    for kind in ("face_down", "face_up"):
        res[f"velstand getup {kind} random joints"] = summarize(
            f"velstand get-up from {kind}, random joints", [getup(full, vel, kind, random_joints=True) for _ in range(a.n)])
    res["sitstand stand from sitting"] = summarize("alpha_sitstand stand-up from sitting (flag 0)",
                                                   [getup(ground, sits, "sitting") for _ in range(a.n)])
    res["sitstand sit then stand"] = summarize("alpha_sitstand sit (1 s..5 s) then stand", [sitstand(ground, sits) for _ in range(a.n)])
    for vx in (0.15, 0.3):
        res[f"velstand walk vx={vx}"] = summarize(f"velstand walk vx={vx} 10 s", [walk(full, vel, vx) for _ in range(a.n)])
    res["alpha_walking walk vx=0.15"] = summarize("alpha_walking walk vx=0.15 10 s", [walk(ground, walkp, 0.15) for _ in range(a.n)])
    for dur in (1.0, 2.0):
        res[f"roulade {dur}s"] = summarize(f"roulade ({dur} s window) -> velstand", [roulade(ground, roll, vel, dur) for _ in range(a.n)])
    res["_meta"] = dict(seed=a.seed, n=a.n, wall_s=round(time.time() - t0))
    (HERE / "results.json").write_text(json.dumps(res, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
