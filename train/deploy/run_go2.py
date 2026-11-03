"""Run a Bolt Go2 brain on a Go2 through unitree_sdk2 (PLAN.md phase 1 steps 5-7): the same program for Unitree's
simulator (sim_vendor.py, --iface lo --domain 1) and for the real robot (--iface <ethernet> --domain 0).

    python deploy/run_go2.py --brain ../models/bolt-go2-damp-798M/params.pkl --plan "trot:6,sit:5,lie:6,getup:6,walk:6" \
        --iface lo --domain 1 --truth --log /tmp/run.npz

Loop: rt/lowstate in (our joint order via robot_bodies' sdk_order) -> every 20 ms Go2Controller.step -> joint targets
-> rt/lowcmd (q, dq 0, kp, kd, tau 0, CRC) re-sent every 2 ms by a writer thread, as Unitree's own examples do.

Startup: blend from the pose the robot is in to the stance over --blend s, hold --hold s, then hand over to the brain.
Safety (never in the brain): targets kept --margin rad inside the joint limits; damping mode (kp 0, kd --damp-kd) on
a tilt past 60 degrees, a joint faster than its rated speed, a state older than 60 ms, or the end of the plan.

Body velocity and height have no sensor on a real Go2 (PLAN.md phase 3). --truth takes them from rt/sportmodestate,
which Unitree's simulator fills with the true values: a rehearsal-only shortcut, refused with --domain 0. Without it
estimator.LegEstimator gives them from the IMU rotation and the joints: kinematic height (0.3 cm off in the rehearsal)
and leg-odometry velocity (0.2-0.3 m/s off; a brain trained with --blind-vel ignores it).

--recover: a fall (tilt past 60 degrees) no longer stops the run: recovery.Recovery rolls the dog belly-down with the
self-right brain (numpy, models/bolt-go2-selfright-6.9M/brain.npz), stands it up with the getup clip and resumes the
command; the plan clock waits meanwhile. Damping mode if it is not righted in 4 s or falls 3 times in a row. Only with
a cleared cage on a real robot (HARDWARE_DAY.md).

--remote: drive it with the Go2's own wireless remote (deploy/remote.py: left stick walk/trot, R1 + full stick canter,
right stick steer, A sit, B lie, Y get up, L2+B or SELECT stop); starts sitting. Works together with --interactive.

--pet: talk to it (deploy/pet_go2.py: "hi Bolt", "sit then dance", "tell me a joke", "good boy"...): it answers
(--speak: aloud, if espeak-ng is installed) and acts with its trained skills; mood, memory and your feedback persist in
--pet-memory. "stop" cuts the current trick short. A line starting with "!" is a raw command ("!trot"). Starts sitting.

--interactive: type commands while it runs (a clip name, "steer 0.4", "stop"). Late brain steps (> 25 ms) are counted;
3 in a row over 40 ms put the robot in damping mode.

--body lite3: the DEEP Robotics Lite3, through DEEP's UDP joint protocol (deploy/lite3_link.py) instead of
unitree_sdk2: their own simulator (deploy/sim_lite3.py) or, on the robot, our bridge to their Lite3_MotionSDK
(deploy/lite3_bridge/). --host is where that listens for commands. Velocity and height from LegEstimator(lite3);
--truth and --remote are Go2-only; --recover uses models/bolt-lite3-selfright-28M.

On a real robot: release sport mode first (--release-sport); start lying down, robot clear, e-stop in hand.
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from controller import DT, Go2Controller, quat_to_mat  # noqa: E402
from estimator import LegEstimator  # noqa: E402

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
    ap.add_argument("--body", default="go2", choices=["go2", "lite3"])
    ap.add_argument("--host", default="127.0.0.1", help="lite3: the simulator's or bridge's address")
    ap.add_argument("--brain", default=None, help="default: the body's current brain (go2 damp-798M, lite3 dog-61.8M)")
    ap.add_argument("--plan", default="trot:6,sit:5,lie:6,getup:6,walk:6", help="clip:seconds[,clip:seconds...]")
    ap.add_argument("--steer", type=float, default=0.0)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=1, help="1: Unitree's simulator; 0: the real robot")
    ap.add_argument("--kp", type=float, default=None, help="default: the body's trained kp (Go2 40)")
    ap.add_argument("--kd", type=float, default=None, help="default: the body's trained kd (Go2 1)")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="first floor runs: 0.8")
    ap.add_argument("--blend", type=float, default=2.5)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--ramp", type=float, default=0.5, help="s to ease from the stance into the brain's targets at the start (0: none)")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--damp-kd", type=float, default=2.0)
    ap.add_argument("--speed-stop", type=float, default=1.2, help="damping mode when a joint stays above this x its rated speed "
                    "for --speed-steps brain steps in a row (0: off)")
    ap.add_argument("--speed-steps", type=int, default=3, help="consecutive 20 ms steps over --speed-stop that stop the run (60 ms)")
    ap.add_argument("--speed-hard", type=float, default=3.0, help="any single step over this x rated stops the run at once "
                    "(single 20 ms spikes of 1.5-2.4x come with switches between moves in the rehearsal; a runaway or a shake "
                    "stays fast, which --speed-steps catches)")
    ap.add_argument("--truth", action="store_true", help="simulator only: true velocity/height from rt/sportmodestate")
    ap.add_argument("--release-sport", action="store_true", help="real robot: release Unitree's sport mode first")
    ap.add_argument("--interactive", action="store_true", help="read commands from the keyboard (clip names, steer X, stop)")
    ap.add_argument("--pet", action="store_true", help="talk to it: the pet brain (deploy/pet_go2.py) reads what you type")
    ap.add_argument("--pet-memory", default="~/.bolt/pet.json")
    ap.add_argument("--speak", action="store_true", help="--pet: say the answers aloud with espeak-ng (local)")
    ap.add_argument("--remote", action="store_true", help="the Go2's wireless remote drives it (deploy/remote.py)")
    ap.add_argument("--recover", action="store_true", help="on a fall: self-right, get up, resume (instead of stopping)")
    ap.add_argument("--selfright", default=None, help="default: the body's (go2 selfright-6.9M, lite3 selfright-28M)")
    ap.add_argument("--log")
    a = ap.parse_args()
    if a.truth and a.domain == 0:
        sys.exit("--truth is a simulator-only shortcut (the real robot has no true velocity/height)")
    a.brain = a.brain or str(HERE.parent.parent / "models" / ("bolt-lite3-dog-61.8M" if a.body == "lite3" else "bolt-go2-damp-798M") / "params.pkl")
    a.selfright = a.selfright or str(HERE.parent.parent / "models" / ("bolt-lite3-selfright-28M" if a.body == "lite3" else "bolt-go2-selfright-6.9M"))
    if a.body == "lite3":
        if a.truth or a.remote or a.release_sport:
            sys.exit("--truth, --remote and --release-sport are Go2-only (Unitree's sport state, remote and mode switcher)")
    else:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(a.domain, a.iface)
    if a.release_sport and a.body == "go2":
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        _, r = msc.CheckMode()
        while r["name"]:
            msc.ReleaseMode()
            time.sleep(1)
            _, r = msc.CheckMode()

    ctl = Go2Controller(a.brain, body=a.body)
    body = ctl.body
    kp = (a.kp or body.entry["kp"]) * a.gain_scale
    kd = (a.kd or body.entry["kv"]) * a.gain_scale
    lo, hi = ctl.lo + a.margin, ctl.hi - a.margin
    rated = np.tile(np.asarray(body.entry["speed"], float), 4)
    if a.body == "lite3":
        from lite3_link import Lite3Robot
        bot = Lite3Robot(body.entry.get("sdk_sign", 1), kp, kd, host=a.host)
    else:
        bot = Robot(body.entry["sdk_order"], kp, kd)
    t0 = time.perf_counter()
    while bot.read() is None:
        if time.perf_counter() - t0 > 5:
            sys.exit("no robot state after 5 s: is the simulator / robot (lite3: the bridge) up on this interface and domain?")
        time.sleep(0.01)

    log, reason = [], "plan done"
    # startup: blend to the stance, then hold
    q0 = bot.read()[2]
    n_blend, n_hold = int(a.blend / DT), int(a.hold / DT)
    for k in range(n_blend + n_hold):
        s = min(1.0, (k + 1) / n_blend)
        bot.set(np.clip((1 - s) * q0 + s * ctl.stance, lo, hi))
        time.sleep(DT)
    plan = [(c.split(":")[0], float(c.split(":")[1])) for c in a.plan.split(",")] if a.plan else []
    cmds = queue.Queue()
    if a.pet:   # what you say -> the pet brain -> lines + timed moves (the same commands as --interactive)
        from pet_go2 import PetBrain
        pet = PetBrain(a.pet_memory, speak=a.speak)
        moves, moves_lock, cut = [], threading.Lock(), threading.Event()

        def listener():
            for line in sys.stdin:
                line = line.strip()
                if line.startswith("!"):
                    cmds.put(line[1:])
                    continue
                if not line:
                    continue
                steps = pet.understand(line)
                with moves_lock:
                    if any(st["trick"] == "stop" for st in steps):
                        moves.clear()
                        cut.set()   # cut the move being held short
                    for st in steps:
                        moves.append(("say", st["say"]))
                        moves.extend(st["moves"])

        def performer():   # says each line, then sends each move and holds it for its seconds
            while True:
                with moves_lock:
                    item = moves.pop(0) if moves else None
                if item is None:
                    time.sleep(0.05)
                elif item[0] == "say":
                    pet.speak(item[1])
                else:
                    cut.clear()
                    cmds.put(item[1])
                    cut.wait(float(item[0]))
        threading.Thread(target=listener, daemon=True).start()
        threading.Thread(target=performer, daemon=True).start()
        plan = [("sit", 1e9)]
    if a.interactive and not a.pet:   # type a clip name (trot, sit, ...), "steer 0.4" or "stop" + Enter while it runs
        def reader():
            for line in sys.stdin:
                cmds.put(line.strip())
        threading.Thread(target=reader, daemon=True).start()
        plan = plan or [("trot", 1e9)]
    if a.remote:   # the remote's buttons and sticks -> the same text commands, polled every 20 ms
        from remote import RemoteCommander
        rc = RemoteCommander(DT)

        def poller():
            while bot.running:
                with bot.lock:
                    raw = None if bot.state is None else list(bot.state.wireless_remote)
                for c in (rc.update(raw) if raw is not None else []):
                    print(f"remote: {c}", flush=True)
                    cmds.put(c)
                time.sleep(DT)
        threading.Thread(target=poller, daemon=True).start()
        plan = [("sit", 1e9)] if not (a.interactive or a.pet) else plan
    est = LegEstimator(body.key) if not a.truth else None
    rec = None
    if a.recover:
        from recovery import Recovery
        rec = Recovery(ctl, a.selfright)
    print(f"brain {Path(a.brain).parent.name}{' (no velocity input)' if ctl.brain.blind_vel else ''}, kp {kp:g} kd {kd:g}, "
          f"plan {plan}{', interactive' if a.interactive else ''}{', remote' if a.remote else ''}{', pet' if a.pet else ''}", flush=True)
    late = late_run = fast_run = n_fast = 0
    n_brain, n_ramp = 0, int(round(a.ramp / DT))
    for name, secs in plan:
        quat, gyro, q, dq, age = bot.read()
        ctl.command(name, steer=a.steer, q=q)
        n = 0
        while n < int(secs / DT):
            t = time.perf_counter()
            while not cmds.empty():   # interactive commands, between brain steps
                c = cmds.get().split()
                if not c:
                    continue
                if c[0] == "stop":
                    reason = "stop command"
                elif c[0] == "steer" and len(c) > 1:
                    ctl.steer = float(c[1])
                elif c[0] in ctl.clips:
                    ctl.command(c[0], q=bot.read()[2])
                    name = c[0]
                else:
                    print(f"? {' '.join(c)} (clips: {', '.join(ctl.clips)}; steer X; stop)", flush=True)
            quat, gyro, q, dq, age = bot.read()
            fast = bool(np.any(np.abs(dq) > rated * a.speed_stop)) if a.speed_stop else False
            fast_run, n_fast = (fast_run + 1 if fast else 0), n_fast + fast
            R = quat_to_mat(quat)
            if a.truth and bot.sport is not None:
                v_body = R.T @ np.array(bot.sport.velocity, float)
                height = float(bot.sport.position[2])
            else:   # what a real Go2 can know: kinematic height; leg-odometry velocity (a --blind-vel brain ignores it)
                v_body, height = est.update(quat, q)
            rtgt = rec.step(quat, gyro, q, dq, v_body, height, resume=name) if rec is not None else None
            if reason != "plan done":
                pass
            elif rec is not None and rec.failed:
                reason = f"recovery gave up: {rec.failed}"
            elif rec is None and R[2, 2] < FALL_TILT:
                reason = "tilt past 60 degrees"
            elif age > 0.06:
                reason = f"state {age * 1000:.0f} ms old"
            elif a.speed_stop and (fast_run >= a.speed_steps or np.any(np.abs(dq) > rated * a.speed_hard)):
                k = int(np.argmax(np.abs(dq) / rated))
                reason = (f"joint {k} at {dq[k]:+.1f} rad/s ({abs(dq[k]) / rated[k]:.2f}x its rated {rated[k]:g}; "
                          f"{fast_run} steps in a row over {a.speed_stop:g}x)")
            elif late_run >= 3:
                reason = "3 brain steps in a row over 40 ms (the loop is too slow for this machine)"
            if reason != "plan done":
                break
            if rtgt is not None:   # recovering: self-right or getup targets; the plan clock waits
                tgt = np.clip(rtgt, lo, hi)
                n -= 1
            else:
                tgt = np.clip(ctl.step(quat, gyro, q, dq, v_body, height), lo, hi)
                if n_brain < n_ramp:   # ease in, once, from the stance to the brain's targets: a gait's first targets from a
                    s = (n_brain + 1) / n_ramp   # standstill whip the knees past their rated speed in one step (sim_vendor
                    tgt = (1 - s) * ctl.stance + s * tgt   # rehearsal, 2 Nov). Not at later switches: blending a gait's
                n_brain += 1                               # pose into a posture's targets made it fall (tried, 2 Nov)
            bot.set(tgt)
            log.append(np.concatenate([[time.perf_counter() - t0], quat, gyro, q, dq, tgt, v_body, [height],
                                       [-1 if rec is not None and rec.mode == "selfright" else ALL.index(ctl.clip.name)]]))
            took = time.perf_counter() - t
            time.sleep(max(0.0, DT - took))
            step = time.perf_counter() - t
            late += step > 0.025
            late_run = late_run + 1 if step > 0.04 else 0
            n += 1
        if reason != "plan done":
            break
    bot.set(None, 0.0, a.damp_kd)   # damping mode
    time.sleep(0.5)
    bot.running = False
    up = quat_to_mat(bot.read()[0])[2, 2]
    print(f"stopped: {reason}; {len(log) * DT:.1f} s under the brain; body up-component at the end {up:.2f}; "
          f"late steps (> 25 ms) {late} of {len(log)}; steps with a joint over {a.speed_stop:g}x rated {n_fast}"
          + (f"; falls recovered {rec.recovered}, events {rec.events}" if rec is not None else ""), flush=True)
    if a.log:
        np.savez(a.log, frames=np.array(log), reason=reason, plan=a.plan)


ALL = __import__("controller").ALL_CLIPS

if __name__ == "__main__":
    main()
