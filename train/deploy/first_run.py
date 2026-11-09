"""First contact with a real Go2, without moving it: this program only LISTENS. It never creates a command publisher,
never releases sport mode and never sends a motor command, so it is safe with the robot lying down, powered on.

    python deploy/first_run.py --iface <eth> --domain 0            # real robot, guided: about 5 minutes, two people
    python deploy/first_run.py --iface lo --domain 1 --auto        # Unitree's simulator (sim_vendor.py): no hand moves
    python deploy/first_run.py --selftest                          # no robot, no SDK: the judging rules on made-up data

What it checks, each with a PASS / FAIL line (HARDWARE_DAY.md step 1, automated):
  link     state messages arrive fast enough and without long gaps
  still    lying flat and still: gravity points down in the body frame, gyro and joint speeds near 0, joints in range
  tilt     you lift the nose, then lower the left side: gravity must move the way it does in our simulator, and the
           gyro must agree with the orientation (same frame)
  legs     you move one joint at a time by hand: the joint that changes must be the one we expect, in our sign
           (our order: FL, FR, RL, RR x hip, thigh, calf; the expected signs come from our own leg kinematics)
  shadow   the robot brain runs on the live readings for --shadow-seconds, its answers logged and thrown away: the
           loop must keep its 20 ms step on this machine, the answers must be finite and inside the joint limits

It writes <out>/report.json (every number and verdict) and <out>/samples.npz (the raw readings). If anything FAILS,
do not run run_go2.py: send that folder back (it holds no personal data: readings, verdicts, OS and Python version).
The verdict "ready for the hanging test" means only that the wiring, frames and timing match what the brain was
trained on in simulation. It says nothing about how the robot will walk.
"""
import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import robot_bodies  # noqa: E402
from controller import DT, Go2Controller, quat_to_mat  # noqa: E402
from estimator import LegEstimator  # noqa: E402

LEG_NAMES = ("front left", "front right", "rear left", "rear right")
JOINT_NAMES = ("hip", "thigh", "calf")
MOVE_MIN = 0.15      # rad: a hand move smaller than this is not judged
TILT_MIN = 0.17      # change of the gravity reading (~10 degrees)


# ---------- the judging rules (pure functions of recorded arrays; --selftest runs them on made-up data) ----------

def gravity(quat):
    """Gravity direction in the body frame for a w,x,y,z quaternion: (0, 0, -1) when level."""
    return quat_to_mat(np.asarray(quat, float)).T @ np.array([0, 0, -1.0])


def judge_link(stamps, min_hz):
    stamps = np.asarray(stamps)
    if len(stamps) < 10:
        return dict(ok=False, why=f"{len(stamps)} state messages: is the robot on this interface and domain?")
    hz, gap = (len(stamps) - 1) / (stamps[-1] - stamps[0]), float(np.diff(stamps).max() * 1000)
    ok = hz >= min_hz and gap < 20.0
    return dict(ok=bool(ok), hz=round(float(hz), 1), largest_gap_ms=round(gap, 1),
                why=f"{hz:.0f} messages/s (need {min_hz:g}), largest gap {gap:.1f} ms (need under 20)")


def judge_still(quats, gyros, qs, dqs, lo, hi):
    g = np.array([gravity(x) for x in quats]).mean(0)
    tilt = float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
    gyro, speed = float(np.abs(gyros).max()), float(np.abs(dqs).max())
    out_of_range = [int(i) for i in np.where((qs.min(0) < lo - 0.05) | (qs.max(0) > hi + 0.05))[0]]
    ok = tilt < 20 and gyro < 0.2 and speed < 0.5 and not out_of_range
    return dict(ok=bool(ok), tilt_deg=round(tilt, 1), gyro_max=round(gyro, 3), joint_speed_max=round(speed, 3),
                joints_out_of_range=out_of_range, gravity=[round(float(x), 3) for x in g],
                why=f"gravity {tilt:.0f} degrees from straight down (need under 20), gyro up to {gyro:.2f} rad/s (under 0.2), "
                    f"joint speed up to {speed:.2f} rad/s (under 0.5), joints outside their range: {out_of_range or 'none'}")


def judge_tilt(times, quats, gyros, axis, sign):
    """The gravity reading must move by TILT_MIN along `axis` with `sign` (nose up: x negative; left side down: y
    positive, as in our simulator), and the gyro must agree with the orientation: d(gravity)/dt = -gyro x gravity."""
    g = np.array([gravity(x) for x in quats])
    d = g - g[:5].mean(0)
    k = int(np.argmax(np.abs(d).max(1)))
    moved = d[k]
    big = int(np.argmax(np.abs(moved)))
    dg = np.gradient(g, times, axis=0)
    pred = -np.cross(gyros, g)
    mask = np.linalg.norm(pred, axis=1) > 0.05
    agree = float(np.sum(dg[mask] * pred[mask]) / max(1e-9, np.linalg.norm(dg[mask]) * np.linalg.norm(pred[mask]))) if mask.sum() > 5 else 0.0
    ok = abs(moved[axis]) >= TILT_MIN and big == axis and np.sign(moved[axis]) == sign and agree > 0.5
    return dict(ok=bool(ok), gravity_change=[round(float(x), 3) for x in moved], gyro_agreement=round(agree, 2),
                why=f"gravity changed by {np.round(moved, 2)} (need {'+' if sign > 0 else '-'}{TILT_MIN} or more on "
                    f"{'xyz'[axis]}, the largest change), gyro agrees with the orientation {agree:.2f} (need over 0.5)")


def judge_move(qs, joint, sign):
    """One hand move: the joint that travelled furthest from where it started must be `joint` (0-11, our order), by
    MOVE_MIN or more, in the direction `sign`. Another joint of the same leg moving more is a FAIL too (a swapped
    joint inside a leg is as dangerous as a swapped leg), but named differently."""
    d = qs - qs[:5].mean(0)
    reach = np.abs(d).max(0)
    top = int(np.argmax(reach))
    k = int(np.argmax(np.abs(d[:, joint])))
    got = float(d[k, joint])
    if reach[top] < MOVE_MIN:
        return dict(ok=False, moved=False, why=f"nothing moved by {MOVE_MIN} rad (largest: joint {top}, {reach[top]:.2f} rad): repeat")
    name = lambda j: f"{LEG_NAMES[j // 3]} {JOINT_NAMES[j % 3]} (our joint {j})"   # noqa: E731
    if top != joint:
        kind = "another joint of the same leg" if top // 3 == joint // 3 else "ANOTHER LEG"
        return dict(ok=False, moved=True, top=top, why=f"expected {name(joint)}, but {name(top)} moved most "
                                                         f"({reach[top]:.2f} rad vs {reach[joint]:.2f}): {kind}")
    ok = np.sign(got) == sign
    return dict(ok=bool(ok), moved=True, top=top, change=round(got, 3),
                why=f"{name(joint)} changed by {got:+.2f} rad (expected {'+' if sign > 0 else '-'})")


def expected_signs(body="go2"):
    """For each of our 12 joints: the sign of its angle change for the hand move we ask for, from our own leg kinematics
    (LegEstimator.paws_rel), so the check follows the model the brain was trained on rather than a remembered rule.
    hip: foot outward (away from the midline); thigh: foot backward (toward the tail); calf: knee bent more (foot up)."""
    est, q0 = LegEstimator(body), np.asarray(robot_bodies.get(body).stance, float)
    p0, signs = est.paws_rel(np.eye(3), q0), []
    for j in range(12):
        q = q0.copy()
        q[j] += 0.05
        d = est.paws_rel(np.eye(3), q)[j // 3] - p0[j // 3]
        wanted = (np.sign(p0[j // 3][1]) * d[1], -d[0], d[2])[j % 3]   # outward, backward, up
        signs.append(int(np.sign(wanted)))
    return signs


MOVE_TEXT = ("move the FOOT OUTWARD, away from the robot's midline (the leg swings sideways at the hip)",
             "swing the WHOLE LEG BACKWARD, foot toward the tail (the upper leg turns at the thigh joint)",
             "BEND THE KNEE MORE, foot up toward the body (only the lower leg moves)")


# ---------- the robot side: a subscriber, nothing else ----------

class Listener:
    """rt/lowstate only. There is deliberately no publisher in this file."""

    def __init__(self, iface, domain, order):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        ChannelFactoryInitialize(domain, iface)
        self.order, self.lock, self.state, self.stamps = np.asarray(order), threading.Lock(), None, []
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on, 10)

    def _on(self, msg):
        with self.lock:
            self.state = msg
            self.stamps.append(time.perf_counter())

    def read(self):
        with self.lock:
            s, t = self.state, (self.stamps[-1] if self.stamps else 0.0)
        if s is None:
            return None
        q, dq = np.zeros(12), np.zeros(12)
        for i in range(12):
            q[self.order[i]], dq[self.order[i]] = s.motor_state[i].q, s.motor_state[i].dq
        temp = max(int(s.motor_state[i].temperature) for i in range(12))
        return np.array(s.imu_state.quaternion, float), np.array(s.imu_state.gyroscope, float), q, dq, time.perf_counter() - t, temp

    def record(self, seconds, hz=100):
        """(times, quats, gyros, qs, dqs) sampled at hz, and the message stamps inside the window."""
        with self.lock:
            n0 = len(self.stamps)
        rows, t0 = [], time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            r = self.read()
            if r is not None:
                rows.append(np.concatenate([[time.perf_counter() - t0], r[0], r[1], r[2], r[3]]))
            time.sleep(1.0 / hz)
        with self.lock:
            stamps = np.array(self.stamps[n0:])
        a = np.array(rows) if rows else np.zeros((0, 32))
        return a[:, 0], a[:, 1:5], a[:, 5:8], a[:, 8:20], a[:, 20:32], stamps


def shadow(bot, ctl, seconds):
    """The brain on live readings, answers thrown away: timing, finite answers inside the limits, a sane estimate."""
    est = LegEstimator(ctl.body.key)
    r = bot.read()
    ctl.reset("lie")
    ctl.command("lie", q=r[2])
    took, rows, late, bad = [], [], 0, 0
    for _ in range(int(seconds / DT)):
        t = time.perf_counter()
        quat, gyro, q, dq = bot.read()[:4]
        v_body, height = est.update(quat, q)
        tgt = ctl.step(quat, gyro, q, dq, v_body, height)
        took.append(time.perf_counter() - t)
        bad += not (np.all(np.isfinite(tgt)) and np.all(tgt >= ctl.lo - 1e-6) and np.all(tgt <= ctl.hi + 1e-6))
        rows.append(np.concatenate([quat, gyro, q, dq, tgt, v_body, [height]]))
        time.sleep(max(0.0, DT - (time.perf_counter() - t)))
        late += (time.perf_counter() - t) > 0.025
    rows, took = np.array(rows), np.array(took) * 1000
    v = float(np.abs(rows[:, 43:46]).max())
    ok = late == 0 and bad == 0 and took.max() < 15 and v < 0.3
    return dict(ok=bool(ok), steps=len(rows), late_steps=int(late), bad_answers=int(bad), brain_ms_mean=round(float(took.mean()), 2),
                brain_ms_max=round(float(took.max()), 2), estimated_speed_max=round(v, 3),
                estimated_height=round(float(rows[:, 46].mean()), 3),
                answer_minus_joints_max=round(float(np.abs(rows[:, 31:43] - rows[:, 7:19]).max()), 3),
                why=f"{late} late steps of {len(rows)} (need 0), brain step {took.mean():.1f} ms mean / {took.max():.1f} max "
                    f"(need under 15), {bad} answers outside the joint limits (need 0), estimated speed while still up to "
                    f"{v:.2f} m/s (need under 0.3)"), rows


# ---------- self-test: the rules on made-up readings ----------

def selftest():
    signs = expected_signs("go2")
    order = robot_bodies.get("go2").entry["sdk_order"]
    assert sorted(order) == list(range(12)), "sdk_order is not a permutation"
    assert signs[0::3] == [1, -1, 1, -1], f"hip signs {signs[0::3]}: outward should be + on left legs, - on right"
    assert len(set(signs[1::3])) == 1 and len(set(signs[2::3])) == 1, "thigh and calf signs should match across legs"
    n = 300
    ramp = np.concatenate([np.zeros(50), np.linspace(0, 0.4, 100), np.full(50, 0.4), np.linspace(0.4, 0, 100)])
    noise = lambda: np.random.default_rng(0).normal(0, 0.003, (n, 12))   # noqa: E731
    for j in range(12):
        qs = noise()
        qs[:, j] += signs[j] * ramp
        assert judge_move(qs, j, signs[j])["ok"], f"right joint, right sign failed for joint {j}"
        assert not judge_move(qs, j, -signs[j])["ok"], f"flipped sign passed for joint {j}"
        other = (j + 3) % 12
        r = judge_move(qs, other, signs[other])
        assert not r["ok"] and "ANOTHER LEG" in r["why"], f"a swapped leg passed (joint {j} moved, {other} expected)"
        r = judge_move(qs, 3 * (j // 3) + (j + 1) % 3, 1)
        assert not r["ok"] and "same leg" in r["why"], "a swapped joint inside a leg passed"
    assert not judge_move(noise(), 0, 1)["moved"], "no move was judged"
    t = np.arange(n) * 0.01

    def tilt(axis_vec, angle):   # the body turned by angle(t) about a body axis: quaternions and the matching gyro
        ang = angle * ramp / 0.4
        quats = np.array([np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * np.asarray(axis_vec, float)]) for a in ang])
        return quats, np.outer(np.gradient(ang, t), axis_vec)
    q_up, w_up = tilt((0, 1, 0), -0.5)        # nose up: a negative turn about y (left)
    assert judge_tilt(t, q_up, w_up, 0, -1)["ok"], "nose up failed"
    assert not judge_tilt(t, q_up, -w_up, 0, -1)["ok"], "a gyro with the wrong sign passed"
    assert not judge_tilt(t, q_up, w_up, 0, 1)["ok"], "nose down passed as nose up"
    q_left, w_left = tilt((1, 0, 0), -0.5)    # left side down: a negative turn about x (forward)
    assert judge_tilt(t, q_left, w_left, 1, 1)["ok"], "left side down failed"
    assert not judge_tilt(t, q_left, w_left, 0, -1)["ok"], "a roll passed as a pitch"
    lvl = np.tile([1.0, 0, 0, 0], (n, 1))
    lo, hi = -np.full(12, 3.0), np.full(12, 3.0)
    assert judge_still(lvl, np.zeros((n, 3)), np.zeros((n, 12)), np.zeros((n, 12)), lo, hi)["ok"]
    assert not judge_still(q_up[150:200], np.zeros((50, 3)), np.zeros((50, 12)), np.zeros((50, 12)), lo, hi)["ok"], "tilted passed as flat"
    assert judge_link(np.arange(1000) * 0.002, 400)["ok"] and not judge_link(np.arange(100) * 0.02, 400)["ok"]
    gappy = np.arange(1000) * 0.002
    gappy[500:] += 0.05
    assert not judge_link(gappy, 400)["ok"], "a 50 ms gap passed"
    print(f"selftest OK: expected signs (our order, hip/thigh/calf x FL FR RL RR) {signs}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=1, help="1: Unitree's simulator; 0: the real robot")
    ap.add_argument("--auto", action="store_true", help="skip the hand moves (tilt, legs): simulator rehearsal")
    ap.add_argument("--quick", action="store_true", help="legs: one move per leg (the knee) instead of three")
    ap.add_argument("--brain", default=str(HERE.parent.parent / "models" / "bolt-go2-torque-steer" / "params.pkl"))
    ap.add_argument("--shadow-seconds", type=float, default=10.0)
    ap.add_argument("--move-seconds", type=float, default=6.0)
    ap.add_argument("--out", default=time.strftime("first_run_%Y%m%d_%H%M%S"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    body = robot_bodies.get("go2")
    ctl = Go2Controller(a.brain, body="go2")
    bot = Listener(a.iface, a.domain, body.entry["sdk_order"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(program="first_run.py (listens only)", brain=Path(a.brain).parent.name, domain=a.domain,
                  system=platform.platform(), python=platform.python_version(), checks={})
    raw = {}

    def done(name, r):
        report["checks"][name] = r
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {name}: {r['why']}", flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1))
        return r["ok"]

    def ask(text):
        print(f"\n{text}", flush=True)
        if not a.auto:
            input("  press Enter, then do it within %.0f s and hold: " % a.move_seconds)

    print("This program only listens. Robot lying flat on the floor, powered on, nobody touching it.", flush=True)
    t0 = time.perf_counter()
    while bot.read() is None:
        if time.perf_counter() - t0 > 5:
            done("link", dict(ok=False, why="no state message in 5 s: check the cable, the interface name and --domain"))
            sys.exit(1)
        time.sleep(0.05)
    t, quats, gyros, qs, dqs, stamps = bot.record(5.0)
    raw["still"] = np.column_stack([t, quats, gyros, qs, dqs])
    done("link", judge_link(stamps, 400 if a.domain == 0 else 150))
    still = judge_still(quats, gyros, qs, dqs, ctl.lo, ctl.hi)
    still["motor_temperature_max_c"] = bot.read()[5]
    done("still", still)
    if not a.auto:
        for name, text, axis, sign in (("tilt nose up", "Lift the FRONT of the robot about a hand's width (nose up), hold it, put it back.", 0, -1),
                                       ("tilt left side down", "Lift the RIGHT side of the robot about a hand's width (left side down), hold it, put it back.", 1, 1)):
            for attempt in range(2):
                ask(text)
                t, quats, gyros, qs, dqs, _ = bot.record(a.move_seconds)
                raw[name.replace(" ", "_")] = np.column_stack([t, quats, gyros, qs, dqs])
                if done(name, judge_tilt(t, quats, gyros, axis, sign)) or attempt:
                    break
                print("  once more, slower and a little further", flush=True)
        signs = expected_signs("go2")
        for leg in range(4):
            for jt in ((2,) if a.quick else (0, 1, 2)):
                j, name = 3 * leg + jt, f"{LEG_NAMES[leg]} {JOINT_NAMES[jt]}"
                for attempt in range(2):
                    ask(f"{LEG_NAMES[leg].upper()} leg: {MOVE_TEXT[jt]}. Move ONLY this leg, then let go.")
                    t, quats, gyros, qs, dqs, _ = bot.record(a.move_seconds)
                    raw[f"leg_{j}"] = np.column_stack([t, quats, gyros, qs, dqs])
                    r = judge_move(qs, j, signs[j])
                    if done(name, r) or attempt or r["moved"]:
                        break
    print(f"\nShadow run: the brain on the live readings for {a.shadow_seconds:g} s, answers thrown away. Hands off.", flush=True)
    r, rows = shadow(bot, ctl, a.shadow_seconds)
    raw["shadow"] = rows
    done("shadow", r)
    np.savez_compressed(out / "samples.npz", **raw)
    fails = [k for k, v in report["checks"].items() if not v["ok"]]
    report["skipped_hand_checks"] = bool(a.auto)
    report["verdict"] = ("STOP: do not run run_go2.py" if fails else
                         "rehearsal only (hand checks skipped)" if a.auto else "ready for the hanging test (HARDWARE_DAY.md step 2)")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\n{report['verdict']}" + (f"; failed: {', '.join(fails)}" if fails else "") + f"\nreport: {out}/report.json", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
