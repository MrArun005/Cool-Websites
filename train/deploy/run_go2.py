"""Run a Bolt Go2 brain on a Go2 through unitree_sdk2 (PLAN.md phase 1 steps 5-7): the same program for Unitree's
simulator (sim_vendor.py, --iface lo --domain 1) and for the real robot (--iface <ethernet> --domain 0).

    python deploy/run_go2.py --brain ../models/bolt-go2-gpu-489M/params.pkl --plan "trot:6,sit:5,lie:6,getup:6,walk:6" \
        --iface lo --domain 1 --truth --log /tmp/run.npz

Loop: rt/lowstate in (our joint order via robot_bodies' sdk_order) -> every 20 ms Go2Controller.step -> joint targets
-> rt/lowcmd (q, dq 0, kp, kd, tau 0, CRC) re-sent every 2 ms by a writer thread, as Unitree's own examples do.

Startup: blend from the pose the robot is in to the stance over --blend s, hold --hold s, then hand over to the brain.
Safety (never in the brain): targets kept --margin rad inside the joint limits; damping mode (kp 0, kd --damp-kd) on
a tilt past 60 degrees, a joint faster than its rated speed, a state older than 60 ms, or the end of the plan.

Body velocity and height have no sensor on a real Go2 (PLAN.md phase 3). --truth takes them from rt/sportmodestate,
which Unitree's simulator fills with the true values: a rehearsal-only shortcut, refused with --domain 0. Without it a
leg-kinematics height and a gyro-free velocity of 0 are used (placeholders until phase 3's estimator).

On a real robot: release sport mode first (--release-sport); start lying down, robot clear, e-stop in hand.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from controller import DT, Go2Controller, quat_to_mat  # noqa: E402

FALL_TILT = np.cos(np.radians(60))   # body z-axis below this up-component: fallen


class Robot:
    """The SDK side: latest state in our joint order, a 500 Hz command writer."""

    def __init__(self, order, kp, kd):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
        from unitree_sdk2py.utils.crc import CRC
        self.order = np.asarray(order)            # our index for each SDK motor
        self.state, self.sport, self.t_state = None, None, 0.0
        self.target, self.kp, self.kd = None, kp, kd
        self.lock, self.crc = threading.Lock(), CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0], self.cmd.head[1], self.cmd.level_flag, self.cmd.gpio = 0xFE, 0xEF, 0xFF, 0
        for i in range(20):
            m = self.cmd.motor_cmd[i]
            m.mode, m.q, m.dq, m.kp, m.kd, m.tau = 0x01, 0.0, 0.0, 0.0, 0.0, 0.0
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_state, 10)
        ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(self._on_sport, 10)
        self.running = True
        threading.Thread(target=self._writer, daemon=True).start()

    def _on_state(self, msg):
        with self.lock:
            self.state, self.t_state = msg, time.perf_counter()

    def _on_sport(self, msg):
        self.sport = msg

    def read(self):
        """(quat wxyz, gyro, q, dq, age s) in our joint order, or None before the first state."""
        with self.lock:
            s, t = self.state, self.t_state
        if s is None:
            return None
        q, dq = np.zeros(12), np.zeros(12)
        for i in range(12):
            q[self.order[i]], dq[self.order[i]] = s.motor_state[i].q, s.motor_state[i].dq
        return np.array(s.imu_state.quaternion, float), np.array(s.imu_state.gyroscope, float), q, dq, time.perf_counter() - t

    def set(self, target, kp=None, kd=None):
        with self.lock:
            self.target = None if target is None else np.asarray(target, float).copy()
            if kp is not None:
                self.kp, self.kd = kp, kd

    def _writer(self):
        while self.running:
            t = time.perf_counter()
            with self.lock:
                tgt, kp, kd = self.target, self.kp, self.kd
            for i in range(12):
                m = self.cmd.motor_cmd[i]
                m.q = float(tgt[self.order[i]]) if tgt is not None else 0.0
                m.dq, m.tau = 0.0, 0.0
                m.kp, m.kd = (kp, kd) if tgt is not None else (0.0, kd)
            self.cmd.crc = self.crc.Crc(self.cmd)
            self.pub.Write(self.cmd)
            time.sleep(max(0.0, 0.002 - (time.perf_counter() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=str(HERE.parent.parent / "models" / "bolt-go2-gpu-489M" / "params.pkl"))
    ap.add_argument("--plan", default="trot:6,sit:5,lie:6,getup:6,walk:6", help="clip:seconds[,clip:seconds...]")
    ap.add_argument("--steer", type=float, default=0.0)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=1, help="1: Unitree's simulator; 0: the real robot")
    ap.add_argument("--kp", type=float, default=None, help="default: the body's trained kp (Go2 40)")
    ap.add_argument("--kd", type=float, default=None, help="default: the body's trained kd (Go2 1)")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="first floor runs: 0.8")
    ap.add_argument("--blend", type=float, default=2.5)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--damp-kd", type=float, default=2.0)
    ap.add_argument("--speed-stop", type=float, default=1.2, help="damping mode above this x rated joint speed (0: off)")
    ap.add_argument("--truth", action="store_true", help="simulator only: true velocity/height from rt/sportmodestate")
    ap.add_argument("--release-sport", action="store_true", help="real robot: release Unitree's sport mode first")
    ap.add_argument("--log")
    a = ap.parse_args()
    if a.truth and a.domain == 0:
        sys.exit("--truth is a simulator-only shortcut (the real robot has no true velocity/height)")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(a.domain, a.iface)
    if a.release_sport:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, r = msc.CheckMode()
        while r["name"]:
            msc.ReleaseMode()
            time.sleep(1)
            _, r = msc.CheckMode()

    ctl = Go2Controller(a.brain)
    body = ctl.body
    kp = (a.kp or body.entry["kp"]) * a.gain_scale
    kd = (a.kd or body.entry["kv"]) * a.gain_scale
    lo, hi = ctl.lo + a.margin, ctl.hi - a.margin
    rated = np.tile(np.asarray(body.entry["speed"], float), 4)
    bot = Robot(body.entry["sdk_order"], kp, kd)
    t0 = time.perf_counter()
    while bot.read() is None:
        if time.perf_counter() - t0 > 5:
            sys.exit("no rt/lowstate after 5 s: is the simulator / robot up on this interface and domain?")
        time.sleep(0.01)

    log, reason = [], "plan done"
    # startup: blend to the stance, then hold
    q0 = bot.read()[2]
    n_blend, n_hold = int(a.blend / DT), int(a.hold / DT)
    for k in range(n_blend + n_hold):
        s = min(1.0, (k + 1) / n_blend)
        bot.set(np.clip((1 - s) * q0 + s * ctl.stance, lo, hi))
        time.sleep(DT)
    plan = [(c.split(":")[0], float(c.split(":")[1])) for c in a.plan.split(",")]
    print(f"brain {Path(a.brain).parent.name}, kp {kp:g} kd {kd:g}, plan {plan}", flush=True)
    for name, secs in plan:
        quat, gyro, q, dq, age = bot.read()
        ctl.command(name, steer=a.steer, q=q)
        for _ in range(int(secs / DT)):
            t = time.perf_counter()
            quat, gyro, q, dq, age = bot.read()
            R = quat_to_mat(quat)
            if a.truth and bot.sport is not None:
                v_body = R.T @ np.array(bot.sport.velocity, float)
                height = float(bot.sport.position[2])
            else:
                v_body, height = np.zeros(3), body.stand_height   # placeholders until the phase 3 estimator
            if R[2, 2] < FALL_TILT:
                reason = "tilt past 60 degrees"
            elif age > 0.06:
                reason = f"state {age * 1000:.0f} ms old"
            elif a.speed_stop and np.any(np.abs(dq) > rated * a.speed_stop):
                k = int(np.argmax(np.abs(dq) / rated))
                reason = f"joint {k} at {dq[k]:+.1f} rad/s, over its rated {rated[k]:g}"
            if reason != "plan done":
                break
            tgt = np.clip(ctl.step(quat, gyro, q, dq, v_body, height), lo, hi)
            bot.set(tgt)
            log.append(np.concatenate([[time.perf_counter() - t0], quat, gyro, q, dq, tgt, v_body, [height],
                                       [ALL.index(name)]]))
            time.sleep(max(0.0, DT - (time.perf_counter() - t)))
        if reason != "plan done":
            break
    bot.set(None, 0.0, a.damp_kd)   # damping mode
    time.sleep(0.5)
    bot.running = False
    up = quat_to_mat(bot.read()[0])[2, 2]
    print(f"stopped: {reason}; {len(log) * DT:.1f} s under the brain; body up-component at the end {up:.2f}", flush=True)
    if a.log:
        np.savez(a.log, frames=np.array(log), reason=reason, plan=a.plan)


ALL = __import__("controller").ALL_CLIPS

if __name__ == "__main__":
    main()
