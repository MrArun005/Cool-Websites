"""Opt-in switch for duck soccer: the contact odometry (odometry.py) instead of soccer.py's dead reckoning.

soccer.py itself is unchanged (its default results stay what results_soccer_hybrid.json says). This runs the same
matches with soccer.Player swapped for a subclass whose pose between goal sightings comes from:
    position  the contact odometry (Microduck's, ALPHA16 sole anchors, encoder angles, IMU orientation)
    heading   the simulated IMU fusion (odometry.SimIMU: gyro quaternion + accelerometer tilt, noise + bias),
              instead of soccer.py's gyro . gravity integration (which drifts ~0.25 deg/s while walking)
Goal sightings still correct position and heading (and re-anchor the odometry); the boards still clamp the estimate;
while the duck is lying down the estimate is held (the sole anchors mean nothing then) and a tumble still invalidates
the heading as before.

Both modes record the duck's pose-estimate error against the simulator every tick (median / p90 / max, and the time
with > 20 cm / > 15 deg error), which soccer.py does not.

    PY=/home/user/cool-websites/.venv/bin/python
    MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer_odometry.py --seeds 0-9 --mode both   # -> results_soccer_odometry_hybrid.json
    MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer_odometry.py --trace 0                 # one match, odometry, printed

From other code: `import soccer_odometry; soccer_odometry.use_odometry()` before soccer.play(...).
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
import soccer  # noqa: E402
from odometry import SimOdometry  # noqa: E402

DefaultPlayer = soccer.Player


class _Measured:
    """Pose-estimate error against the simulator, every 50 Hz tick."""

    def _measure(self):
        sim = self.sim
        p = sim.d.xpos[sim.trunk][:2]
        R = sim.d.xmat[sim.trunk].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        if not hasattr(self, "err_xy"):
            self.err_xy, self.err_yaw = [], []
        self.err_xy.append(float(np.hypot(*(self.pos - p))))
        self.err_yaw.append(abs(soccer.wrap(self.psi - yaw)))


class MeasuredPlayer(_Measured, DefaultPlayer):
    """soccer.Player as it is, with the error log."""

    def imu(self):
        super().imu()
        self._measure()


class OdometryPlayer(_Measured, DefaultPlayer):
    def __init__(self, rt, cam_id, psi0=0.0, pos0=(0.0, 0.0)):
        super().__init__(rt, cam_id, psi0=psi0, pos0=pos0)
        self.so = SimOdometry(rt.sim, pos0=pos0, yaw0=psi0, seed=int(abs(pos0[0]) * 1e6) % 100_000)   # (not rt.rng: that would change the run)
        rt.on_substep = self._substep

    def _substep(self, sim):
        self.so.substep(sim)

    def see(self, img):
        pos, psi = self.pos.copy(), self.psi
        out = super().see(img)
        if np.any(self.pos != pos) or self.psi != psi:      # a goal sighting moved the estimate: re-anchor
            self.so.set_pose(self.pos, self.psi)
        return out

    def imu(self):
        rt = self.rt
        self.so.update()
        if rt.lying() or rt.mode == "get up":
            self.so.set_pose(self.pos)                     # hold the position while down
        else:
            self.pos = self.so.xy
        self.psi = self.so.yaw
        lim = np.array([soccer.PITCH_X - 0.06, soccer.PITCH_Y - 0.06])   # the boards, as soccer.Player.imu
        corr = np.clip(self.pos, -lim, lim) - self.pos
        if np.any(corr != 0):
            self.pos = self.pos + corr
            self.ball_f = self.ball_f + corr
            self.so.set_pose(self.pos)
        self._measure()


def use_odometry(on=True):
    soccer.Player = OdometryPlayer if on else MeasuredPlayer


def _job(a):
    seed, mode = a
    use_odometry(mode == "odometry")
    t = time.time()
    r = soccer.play(seed)
    pl_err = _LAST.get("err")
    r["wall_s"] = round(time.time() - t, 1)
    r["mode"] = mode
    if pl_err is not None:
        r["pose_err"] = pl_err
    return r


_LAST = {}
_orig_play = soccer.play


def _play_capture(seed=0, **kw):
    """soccer.play, keeping the player's error log (play() does not return the player)."""
    holder = {}
    cls = soccer.Player

    class Capture(cls):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            holder["pl"] = self
    soccer.Player = Capture
    try:
        r = _orig_play(seed, **kw)
    finally:
        soccer.Player = cls
    pl = holder["pl"]
    e, y = np.array(pl.err_xy), np.degrees(np.array(pl.err_yaw))
    _LAST["err"] = dict(xy_cm_median=round(100 * float(np.median(e)), 1), xy_cm_p90=round(100 * float(np.percentile(e, 90)), 1),
                        xy_cm_max=round(100 * float(e.max()), 1), frac_over_20cm=round(float(np.mean(e > 0.2)), 3),
                        yaw_deg_median=round(float(np.median(y)), 1), yaw_deg_p90=round(float(np.percentile(y, 90)), 1),
                        frac_over_15deg=round(float(np.mean(y > 15)), 3))
    return r


soccer.play = _play_capture


def summarize(runs):
    out = {}
    for mode in ("default", "odometry"):
        rs = [r for r in runs if r["mode"] == mode]
        if not rs:
            continue
        s = soccer.summarize(rs)
        for k in ("xy_cm_median", "xy_cm_p90", "frac_over_20cm", "yaw_deg_median", "yaw_deg_p90", "frac_over_15deg"):
            s["pose_" + k] = round(float(np.median([r["pose_err"][k] for r in rs])), 3)
        s["pose_xy_cm_max"] = max(r["pose_err"]["xy_cm_max"] for r in rs)
        out[mode] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--mode", default="both", choices=("both", "default", "odometry"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--trace", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results_soccer_odometry_hybrid.json"))
    a = ap.parse_args()
    if a.trace is not None:
        use_odometry(True)
        r = soccer.play(a.trace, trace=True)
        print({k: v for k, v in r.items() if k not in ("body", "kick_log")}, _LAST.get("err"))
        return
    lo, _, hi = a.seeds.partition("-")
    modes = ("default", "odometry") if a.mode == "both" else (a.mode,)
    jobs = [(s, m) for s in range(int(lo), int(hi or lo) + 1) for m in modes]
    runs = []
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.jobs) as ex:
            runs = list(ex.map(_job, jobs))
    else:
        for j in jobs:
            r = _job(j)
            runs.append(r)
            print({k: v for k, v in r.items() if k in ("seed", "mode", "goals", "kicks", "connected", "falls", "fixes",
                                                      "pose_err", "wall_s")}, flush=True)
    s = summarize(runs)
    print(json.dumps(s, indent=1))
    Path(a.out).write_text(json.dumps(dict(summary=s, runs=[{k: v for k, v in r.items() if k != "body"} for r in runs]),
                                      indent=1))


if __name__ == "__main__":
    main()
