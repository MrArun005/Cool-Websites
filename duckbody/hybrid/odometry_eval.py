"""Score the contact odometry (odometry.py) against the simulator's ground truth and against soccer.py's dead
reckoning, on walk / turn courses with velstand (HybridRuntime.drive) on randomized bodies.

    PY=/home/user/cool-websites/.venv/bin/python
    $PY duckbody/hybrid/odometry_eval.py --seeds 0-9 [--t 45]     # -> results_odometry_hybrid.json

Estimators (all start at the true pose):
    odometry        ALPHA16 anchors, encoder noise U(+-0.001) rad, orientation from the simulated IMU fusion
                    (odometry.SimIMU: gyro quaternion + accelerometer tilt, gyro noise and residual bias)
    dead_reckoning  soccer.py Player.imu()'s position: the drive twist sent to velstand through its measured speed
                    map (soccer.VX_MAP / VY_MAP), rotated by the same IMU heading
    soccer_now      exactly what soccer.py does today: that dead reckoning with its heading (gyro . gravity at 200 Hz)
    odometry_gyro_up  the odometry with soccer.py's heading
    *_true_heading  odometry / dead reckoning with the simulator's true heading: the position part alone
    odometry_perfect  ALPHA16 with the true joint angles and trunk rotation (Pollen's --odom-compare setting)
    odometry_alpha4 / odometry_v15   the other anchor sets, IMU heading
Path length: the trunk's ground-truth X/Y sampled at 2 Hz (at 50 Hz the step sway adds ~2x).
Courses (per seed, `--t` s each):
    straight  drive 0.5 forward, heading held on the gyro (wz kept off velstand's turning dead band)
    square    0.6 forward 5 s, turn on the spot (0.8 rad/s) by 90 deg, repeat
    mixed     soccer-like: a new twist every 1.5-3 s from forward 0.35/0.5/0.8, back -0.3, side +-0.45,
              turn +-0.7, forward+turn, stop
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_hybrid as rh  # noqa: E402
from odometry import SimOdometry  # noqa: E402

# soccer.py's speed map (copied, so this script does not import the game): velstand's measured speed per command
VX_MAP = ((-0.3, -0.07), (-0.25, 0.0), (0.25, 0.0), (0.3, 0.11), (0.5, 0.23), (0.8, 0.35), (1.0, 0.41))
VY_MAP = ((-0.4, -0.058), (-0.3, 0.0), (0.3, 0.0), (0.4, 0.058))
ESTIMATORS = ("odometry", "dead_reckoning", "soccer_now", "odometry_gyro_up", "odometry_true_heading",
              "dead_reckoning_true_heading", "odometry_perfect", "odometry_alpha4", "odometry_v15")


def interp(m, x):
    xs, ys = zip(*m)
    return float(np.interp(x, xs, ys))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def course_cmd(course, t, st, psi, rng):
    if course == "straight":
        e = wrap(st.setdefault("h", psi) - psi)
        wz = 0.0 if abs(e) < 0.08 else math.copysign(0.55, e)
        return 0.5, 0.0, wz
    if course == "square":
        ph = st.setdefault("phase", ("walk", t, psi))
        if ph[0] == "walk":
            if t - ph[1] > 5.0:
                st["phase"] = ("turn", t, psi)
            return 0.6, 0.0, 0.0
        if abs(wrap(psi - ph[2])) > math.pi / 2 - 0.1:
            st["phase"] = ("walk", t, psi)
        return 0.0, 0.0, 0.8
    if course == "mixed":
        if t >= st.get("until", -1):
            st["until"] = t + rng.uniform(1.5, 3.0)
            opts = [(0.35, 0, 0), (0.5, 0, 0), (0.8, 0, 0), (-0.3, 0, 0), (0, 0.45, 0), (0, -0.45, 0),
                    (0, 0, 0.7), (0, 0, -0.7), (0.5, 0, 0.6), (0.5, 0, -0.6), (0, 0, 0)]
            st["cmd"] = opts[rng.integers(len(opts))]
        return st["cmd"]
    raise ValueError(course)


def run(seed, course, t_run=45.0):
    rt = rh.HybridRuntime(seed=seed, randomize=True, log=None)
    rt.reset("standing")
    sim = rt.sim
    rng = np.random.default_rng(500 + seed)
    p0 = sim.d.xpos[sim.trunk][:2].copy()
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    y0 = math.atan2(R[1, 0], R[0, 0])
    est = dict(odometry=SimOdometry(sim, p0, y0, seed=seed),
               odometry_gyro_up=SimOdometry(sim, p0, y0, seed=seed, heading="gyro_up"),
               odometry_alpha4=SimOdometry(sim, p0, y0, anchors="alpha4", seed=seed),
               odometry_v15=SimOdometry(sim, p0, y0, anchors="v15", seed=seed),
               odometry_perfect=SimOdometry(sim, p0, y0, perfect=True),
               odometry_true_heading=SimOdometry(sim, p0, y0, seed=seed, heading="gyro_up"))
    imu, gup = est["odometry"], est["odometry_gyro_up"]

    def substep(s):
        for e in est.values():
            if not e.perfect:
                e.substep(s)
    rt.on_substep = substep
    dr = {k: p0.copy() for k in ("dead_reckoning", "soccer_now", "dead_reckoning_true_heading")}

    def true_yaw():
        Rt = sim.d.xmat[sim.trunk].reshape(3, 3)
        return math.atan2(Rt[1, 0], Rt[0, 0])
    path, last, st = 0.0, p0.copy(), {}
    err_max = {k: 0.0 for k in ESTIMATORS}
    t0 = time.time()
    rt.run(1.0)                                  # settle standing (everyone tracks it)
    for e in est.values():
        e.update()
    n = int(t_run * rh.HZ)
    for k in range(n):
        t = k * rh.DT
        rt.drive(*course_cmd(course, t, st, imu.yaw, rng))
        rt.tick()
        est["odometry_true_heading"].psi = true_yaw()
        for e in est.values():
            e.update()
        if rt.mode == "drive" and not rt.lying():
            vx, vy = interp(VX_MAP, rt.vel[0]), interp(VY_MAP, rt.vel[1])
            for key, yaw in (("dead_reckoning", imu.yaw), ("soccer_now", gup.yaw),
                             ("dead_reckoning_true_heading", true_yaw())):
                c, s = math.cos(yaw), math.sin(yaw)
                dr[key] += np.array([c * vx - s * vy, s * vx + c * vy]) * rh.DT
        p = sim.d.xpos[sim.trunk][:2].copy()
        if k % (rh.HZ // 2) == 0:
            path += float(np.hypot(*(p - last)))
            last = p
        pos = {k2: e.xy for k2, e in est.items()}
        pos.update(dr)
        for k2 in ESTIMATORS:
            err_max[k2] = max(err_max[k2], float(np.hypot(*(pos[k2] - p))))
    R = sim.d.xmat[sim.trunk].reshape(3, 3)
    yaw_true = math.atan2(R[1, 0], R[0, 0])
    p = sim.d.xpos[sim.trunk][:2]
    path += float(np.hypot(*(p - last)))
    out = dict(seed=seed, course=course, t=t_run, path_m=round(path, 3), falls=rt.falls, body=sim.dr,
               net_m=round(float(np.hypot(*(p - p0))), 3),
               heading_err_deg=round(math.degrees(wrap(imu.yaw - yaw_true)), 2),
               heading_err_deg_gyro_up=round(math.degrees(wrap(gup.yaw - yaw_true)), 2),
               wall_s=None)
    for k2 in ESTIMATORS:
        e = float(np.hypot(*(pos[k2] - p)))
        out[k2] = dict(final_cm=round(100 * e, 1), max_cm=round(100 * err_max[k2], 1),
                       cm_per_m=round(100 * e / max(path, 1e-6), 2))
    out["odometry"]["z_err_mm"] = round(1000 * (est["odometry"].z - float(sim.d.xpos[sim.trunk][2])), 1)
    out["odometry"]["switches"] = est["odometry"].odo.switches
    out["wall_s"] = round(time.time() - t0, 1)
    return out


def summarize(runs):
    rows = {}
    for course in ("straight", "square", "mixed", "all"):
        rs = [r for r in runs if course == "all" or r["course"] == course]
        if not rs:
            continue
        row = dict(n=len(rs), path_m=round(float(np.mean([r["path_m"] for r in rs])), 2),
                   heading_err_deg_median=round(float(np.median([abs(r["heading_err_deg"]) for r in rs])), 2),
                   heading_err_deg_max=round(float(np.max([abs(r["heading_err_deg"]) for r in rs])), 2),
                   heading_err_deg_gyro_up_median=round(float(np.median([abs(r["heading_err_deg_gyro_up"]) for r in rs])), 2),
                   falls=int(sum(r["falls"] for r in rs)))
        for k in ESTIMATORS:
            row[k] = dict(cm_per_m_median=round(float(np.median([r[k]["cm_per_m"] for r in rs])), 2),
                          cm_per_m_max=round(float(np.max([r[k]["cm_per_m"] for r in rs])), 2),
                          final_cm_median=round(float(np.median([r[k]["final_cm"] for r in rs])), 1),
                          max_cm_max=round(float(np.max([r[k]["max_cm"] for r in rs])), 1))
        rows[course] = row
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--t", type=float, default=45.0)
    ap.add_argument("--courses", default="straight,square,mixed")
    ap.add_argument("--out", default=str(HERE / "results_odometry_hybrid.json"))
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)
    runs = []
    for s in seeds:
        for c in a.courses.split(","):
            r = run(s, c, a.t)
            runs.append(r)
            print(f"seed {s} {c:8s} path {r['path_m']:5.2f} m  falls {r['falls']}  heading {r['heading_err_deg']:+6.2f} / {r['heading_err_deg_gyro_up']:+6.2f} deg  " +
                  "  ".join(f"{k.replace('odometry', 'odo').replace('dead_reckoning', 'dr').replace('_true_heading', '_th')}: {r[k]['cm_per_m']:5.2f}/{r[k]['max_cm']:4.0f}" for k in ESTIMATORS),
                  flush=True)
    summ = summarize(runs)
    print(json.dumps(summ, indent=1))
    Path(a.out).write_text(json.dumps(dict(summary=summ, runs=runs), indent=1))


if __name__ == "__main__":
    main()
