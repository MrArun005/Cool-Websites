"""Every skill the owner asked for, through the hybrid runtime, on 10 randomized bodies each.

    /home/user/cool-websites/.venv/bin/python duckbody/hybrid/test_skills_hybrid.py [--n 10] [--skills walk,run] [--page DIR]

Body randomization per run (harness.MicroduckSim.randomize_body, Microduck's training ranges): mass and
inertia x0.95-1.05 per body, trunk and head CoM +-3 mm, contact friction x0.7-1.3, servo kp x0.85-1.15,
battery 6.5-8.2 V, command delay 3-6 physics steps; plus the observation noise of their actor.
Body: runtime_hybrid.SCENE (every part collides, antennas drawn only; HYBRID_BODY=<suffix> picks a build_hybrid.py variant).

Writes results_skills_hybrid.json (per skill: successes, per-run numbers). With --page DIR it also writes one
replay per skill (DIR/rollout_<skill>.json, the first successful seed) and DIR/tests.json + DIR/queue.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runtime_hybrid as rh  # noqa: E402

UP_TILT, UP_Z, FALL = 25.0, 0.10, 45.0


class Probe:
    """Per-tick measurements while a skill runs."""

    def __init__(self, rt, record):
        self.rt, self.record = rt, record
        s = rt.sim
        self.frames, self.max_tilt, self.min_z, self.k = [], 0.0, 1.0, 0
        self.mouth_q = s.m.jnt_qposadr[s.m.joint("mouth").id]
        self.yaw_q = s.qadr[7]
        self.jaw_max, self.yaw_lo, self.yaw_hi = 0.0, 0.0, 0.0
        self.pitch_rot, self.air, self.flights, self.peak_g = 0.0, 0, 0, 0.0
        self.flight_ms, self.land_bw, self.after_flight = [], 0.0, 0
        self.W = s.m.body_subtreemass[1] * 9.81
        rt.on_substep = self.substep

    def substep(self, s):
        """Contacts at 200 Hz: a flight = no part of the robot touching the floor for >= 15 ms."""
        tot, touching, f = 0.0, False, np.zeros(6)
        for i in range(s.d.ncon):
            c = s.d.contact[i]
            if s.m.geom_bodyid[c.geom1] == 0 or s.m.geom_bodyid[c.geom2] == 0:
                touching = True
                mujoco.mj_contactForce(s.m, s.d, i, f)
                tot += abs(f[0])
        if touching:
            if self.air >= 3:
                self.flights += 1
                self.flight_ms.append(self.air * 5)
                self.after_flight = 60                     # landing window: 0.3 s
            self.air = 0
        else:
            self.air += 1
        if self.after_flight > 0:
            self.after_flight -= 1
            self.land_bw = max(self.land_bw, tot / self.W)

    def __call__(self, rt):
        s = rt.sim
        self.k += 1
        if self.record and self.k % 2 == 0:
            self.frames.append(s.frame())
        self.max_tilt = max(self.max_tilt, s.tilt_deg())
        self.min_z = min(self.min_z, s.trunk_z())
        self.jaw_max = max(self.jaw_max, float(s.d.qpos[self.mouth_q]))
        y = float(s.d.qpos[self.yaw_q])
        self.yaw_lo, self.yaw_hi = min(self.yaw_lo, y), max(self.yaw_hi, y)
        self.pitch_rot += float(s.d.sensordata[s.gyro_adr][1]) * rh.DT
        a = s.m.sensor("imu_accel").adr[0]
        self.peak_g = max(self.peak_g, float(np.linalg.norm(s.d.sensordata[a:a + 3])) / 9.81)

    def mark(self):
        self.max_tilt, self.min_z = 0.0, 1.0


def up(rt):
    return rt.sim.tilt_deg() < UP_TILT and rt.sim.trunk_z() > UP_Z


def standing_through(rt, p, seconds):
    """Run `seconds`; True if the last second is all upright and nothing fell."""
    ok = True
    for i in range(int(seconds * rh.HZ)):
        rt.tick()
        p(rt)
        if i >= (seconds - 1.0) * rh.HZ and not up(rt):
            ok = False
    return ok and p.max_tilt < FALL


def heading_frame(rt):
    s = rt.sim
    R = s.d.xmat[s.trunk].reshape(3, 3)
    f = R[:2, 0] / np.linalg.norm(R[:2, 0])
    return s.d.xpos[s.trunk][:2].copy(), f


# ------------------------------------------------------------------ the skills
def t_idle(rt, p):
    rt.command("idle")
    n = rt.say("Hello! I am your duck. I can walk, run, sit and roll. Want to see?")
    ok = standing_through(rt, p, max(8.0, n * rh.DT + 1))
    head_range = p.yaw_hi - p.yaw_lo
    good = ok and head_range > 0.6 and p.jaw_max > 0.3
    return good, dict(head_yaw_range_rad=round(head_range, 2), jaw_max_rad=round(p.jaw_max, 2), max_tilt=round(p.max_tilt, 1))


def _go(rt, p, mode, need):
    rt.command(mode)
    rt.run(2.0, p)
    p0, f = heading_frame(rt)
    ok = standing_through(rt, p, 8.0)
    d = rt.sim.d.xpos[rt.sim.trunk][:2] - p0
    v = float(d @ f) / 8.0
    side = float(d @ np.array([-f[1], f[0]])) / 8.0
    return ok and v >= need, dict(speed_mps=round(v, 3), sideways_mps=round(side, 3), max_tilt=round(p.max_tilt, 1))


def t_walk(rt, p):
    return _go(rt, p, "walk", 0.12)


def t_run(rt, p):
    return _go(rt, p, "run", 0.30)


def t_squat(rt, p):
    rt.run(0.5, p)
    z0 = rt.sim.trunk_z()
    p.mark()
    rt.command("squat")
    rt.run(3.0, p)
    drop, tilt_held = z0 - rt.sim.trunk_z(), p.max_tilt
    rt.command("stand")
    p.mark()
    ok = standing_through(rt, p, 3.0)
    return ok and drop > 0.035 and tilt_held < UP_TILT, dict(drop_m=round(drop, 3), tilt_squat=round(tilt_held, 1),
                                                            tilt_rise=round(p.max_tilt, 1))


def _seated(rt):
    return rt.sim.trunk_z() < 0.08 and rt.sim.tilt_deg() < 30


def t_sit_stand(rt, p):
    rt.command("sit")
    rt.run(4.0, p)
    seated = _seated(rt)
    rt.command("stand")
    p.mark()
    ok = standing_through(rt, p, 4.0)
    return seated and ok, dict(seated=seated, seat_z=round(p.min_z, 3), tilt_rise=round(p.max_tilt, 1))


def t_swag(rt, p):
    rt.command("swag")
    rt.run(2.4, p)
    k0 = float(rt.sim.d.qpos[rt.sim.qadr[3]])
    rt.run(4.6, p)
    knee_open = k0 - float(rt.sim.d.qpos[rt.sim.qadr[3]])
    swag = rt.skill == "swag_hold" and rt.sim.trunk_z() < 0.09 and rt.sim.tilt_deg() < 60
    pose = dict(z=round(rt.sim.trunk_z(), 3), tilt=round(rt.sim.tilt_deg(), 1))
    rt.command("stand")
    p.mark()
    ok = standing_through(rt, p, 6.0)
    return swag and ok, dict(swag_pose=pose, in_swag=swag, knee_open_rad=round(knee_open, 2), tilt_rise=round(p.max_tilt, 1))


def t_sleep_wake(rt, p):
    rt.command("sleep")
    rt.run(6.0, p)
    neck = float(rt.sim.d.qpos[rt.sim.qadr[5]] - rh.DEFAULT_POSE[5])
    asleep = _seated(rt) and neck < -0.5
    rt.command("wake")
    p.mark()
    ok = standing_through(rt, p, 4.0)
    return asleep and ok, dict(asleep=asleep, neck_droop_rad=round(neck, 2), tilt_rise=round(p.max_tilt, 1))


def t_lie_down(rt, p):
    rt.command("lie down")
    rt.run(6.0, p)
    g = rt.sim.gravity()
    lying = bool(g[0] > 0.6 and rt.sim.trunk_z() < 0.08)      # belly down, resting on bill, chest and knees
    rest = float(np.linalg.norm(rt.sim.d.qvel[:6]))
    peak = p.peak_g
    rt.command("get up")
    t_up = None
    for i in range(int(3.0 * rh.HZ)):
        rt.tick()
        p(rt)
        if t_up is None and up(rt):
            t_up = (i + 1) * rh.DT
    p.mark()
    ok = t_up is not None and standing_through(rt, p, 2.0)
    return lying and ok, dict(lying=lying, face="down" if g[0] > 0 else "up", rest_speed=round(rest, 3),
                              peak_g=round(peak, 1), t_up_s=t_up, tilt_after=round(p.max_tilt, 1))


def _getup(kind):
    def f(rt, p):
        rt.reset(kind)
        t_up = None
        for i in range(int(5.0 * rh.HZ)):
            rt.tick()
            p(rt)
            if t_up is None and up(rt):
                t_up = (i + 1) * rh.DT
        p.mark()
        ok = t_up is not None and standing_through(rt, p, 2.0)
        return ok, dict(t_up_s=t_up)
    return f


def t_hop(rt, p):
    rt.run(0.5, p)
    p.flights, p.flight_ms, p.land_bw = 0, [], 0.0
    rt.command("jump")
    rt.run(6.0, p)                                   # crouched between hops: only "never tipped past 45 deg" here
    hops, fl = p.flights, list(p.flight_ms)
    rt.command("stand")
    ok = standing_through(rt, p, 2.0)
    return ok and hops >= 4, dict(flights=hops, hops_per_s=round(hops / 6.0, 2), flight_ms=fl,
                                  flight_ms_med=float(np.median(fl)) if fl else 0.0, landing_bw=round(p.land_bw, 2),
                                  max_tilt=round(p.max_tilt, 1))


def t_roll(rt, p):
    rt.run(0.5, p)
    p.pitch_rot = 0.0
    rt.command("roll")
    rt.run(2.6, p)
    rot = abs(math.degrees(p.pitch_rot))
    p.mark()
    ok = standing_through(rt, p, 3.0) and up(rt)
    return ok and rot >= 300, dict(rotation_deg=round(rot), tilt_after=round(p.max_tilt, 1))


SKILLS = {
    "idle": (t_idle, "stand and look around while talking (velstand + head command, jaw from speech)"),
    "walk": (t_walk, f"walk (velstand, command {rh.WALK_VX} m/s + IMU heading hold)"),
    "run": (t_run, f"run (velstand at command {rh.RUN_VX}, beyond its 0.4 training range, heading hold)"),
    "squat": (t_squat, "squat down (scripted, soles flat, IMU ankle balance), then stand (velstand)"),
    "sit_stand": (t_sit_stand, "sit down, then stand up (alpha_sitstand), velstand after"),
    "swag": (t_swag, "relaxed swag sit (alpha_sitstand seated + leg offsets: knees splayed, left knee open; head cocked, beak open), then stand"),
    "sleep_wake": (t_sleep_wake, "sleep sitting (head drooped, slow breathing nod), wake up and stand"),
    "lie_down": (t_lie_down, "lie down (scripted), then get up (velstand)"),
    "getup_face_down": (_getup("face_down"), "get up from lying face down (velstand)"),
    "getup_back": (_getup("face_up"), "get up from lying on its back (velstand)"),
    "hop": (t_hop, "hops, about one a second (scripted crouch-push-tuck-land), velstand between"),
    "roll": (t_roll, "forward roll (roulade, 2 s slot), then velstand"),
}


def run_skill(name, n, record=False):
    fn, desc = SKILLS[name]
    runs, best = [], None
    sim = None
    for seed in range(n):
        rt = rh.HybridRuntime(seed=seed, randomize=True, log=None) if sim is None else rh.HybridRuntime(sim=sim, seed=seed, log=None)
        sim = rt.sim
        sim.rng = np.random.default_rng(seed)
        rt.reset("standing")
        p = Probe(rt, record and best is None)
        ok, info = fn(rt, p)
        info = dict(success=bool(ok), seed=seed, body=dict(rt.sim.dr), **info)
        runs.append(info)
        if record and best is None and ok:
            best = dict(frames=p.frames, seed=seed, info=info, events=rt.events)
    k = sum(r["success"] for r in runs)
    print(f"{name:16s} {k}/{n}  {runs[0] if not k else next(r for r in runs if r['success'])}", flush=True)
    return dict(success=k, n=n, desc=desc, runs=runs), best, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--skills", default=",".join(SKILLS))
    ap.add_argument("--page", default=None)
    ap.add_argument("--out", default=str(HERE / "results_skills_hybrid.json"))
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    page = Path(a.page) if a.page else None
    for name in a.skills.split(","):
        t0 = time.time()
        r, best, sim = run_skill(name, a.n, record=page is not None)
        r["wall_s"] = round(time.time() - t0)
        res[name] = r
        out.write_text(json.dumps(res, indent=1))
        if page is not None and best is not None:
            (page / f"rollout_{name}.json").write_text(json.dumps(dict(
                fps=25, seconds=len(best["frames"]) / 25, frames=best["frames"], bodies=sim.body_names(), success=True,
                task=name, seed=best["seed"], note=r["desc"], info=best["info"], events=best["events"],
                checkpoint="hybrid runtime: " + r["desc"]), separators=(",", ":")))


if __name__ == "__main__":
    main()
