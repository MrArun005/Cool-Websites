"""The skill-switch tests of test_skills.py (same sequences, seeds, windows) on the one MasterMind brain.

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/out/student.pt --n 10
    ... --brain teacher      the env's teacher chains (the skill manager inside the unified env, a sanity check)
Sim-to-real stress test (all off by default, so the numbers are today's): --obs-noise [K] (K x env.NOISE sensor noise),
--latency-ms 20|40 (action delay, in 2 ms physics steps), --motor-lag-ms T (first-order lag of the joint targets),
--gain-scale G (PD kp and kv x G), --motor-brake (bolt_env brake, 25 rad/s, 3.5 N·m·s/rad). Mass/friction/kp shifts
are the randomized physics already on in every test.
Also --exam: imitate_env.exam per clip (the student driven through the Master env on a clip command, scored by
the ImitateEnv tracking reward) and jump clean rates from a trot (the trot->jump rows of the switch tests).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

import bolt_env  # noqa: E402
from baseline import run, show  # noqa: E402
from drive import Driver  # noqa: E402
import env  # noqa: E402
from env import Master  # noqa: E402
from student import Student  # noqa: E402


def clip_exam(m, brain, episodes=4, seed=9000, clips=("walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie")):
    """imitate_env.exam's measure, through the Master env: 4 s per loop from evenly spaced points of the stride
    (sit/lie: from the start through the 1 s hold); per step the ImitateEnv tracking reward; 0 for every step lost
    after a fall; speed_ = forward distance (turns: angle) over the dog's."""
    from imitate_env import HOLD
    out = {}
    for name in clips:
        c = m.im.clips[name]
        scores, falls, pace = [], 0, []
        for k in range(episodes):
            m.start(name, tau=(k / episodes) * c.length if c.loop else 0.0, seed=seed + k)
            n = 200 if c.loop else int((c.length + HOLD) / m.dt)
            total, fell, done_by, yaw = 0.0, False, 0.0, 0.0
            for _ in range(n):
                R = m.w._rotation()
                p0, h0 = m.data.qpos[:2].copy(), np.arctan2(R[1, 0], R[0, 0])
                ctrl = brain.act(m.obs()) if brain is not None else None
                r = m.step(ctrl)
                R = m.w._rotation()
                done_by += float(np.dot(m.data.qpos[:2] - p0, [np.cos(h0), np.sin(h0)]))
                yaw += float(np.angle(np.exp(1j * (np.arctan2(R[1, 0], R[0, 0]) - h0))))
                terms, err = m.im.tracking()
                total += sum(m.im.weights[k2] * v for k2, v in terms.items())
                lost = abs(err["height_err"]) > 0.15 or err["tilt"] > 0.8
                if r["fell"] or lost:
                    fell = True
                    break
            falls += fell
            scores.append(total / n)
            if c.loop:
                turn = name.startswith("turn")
                want = n * m.dt * (c.w_body[:, 2].mean() if turn else c.v_body[:, 0].mean())
                pace.append((yaw if turn else done_by) / want)
        out[f"track_{name}"] = round(float(np.mean(scores)), 3)
        out[f"fall_{name}"] = falls / episodes
        if c.loop:
            out[f"speed_{name}"] = round(float(np.mean(pace)), 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="teacher")
    ap.add_argument("--imitate", help="override the teacher set's dog-motion brain")
    ap.add_argument("--teachers", default="v2", help="teacher set (teachers.SETS): v2 for the v2 student, v3 for v3")
    ap.add_argument("--with-50", action="store_true", help="also the 50 cm jump sequence")
    ap.add_argument("--over", default="{}", help='JSON overrides of the teacher set, e.g. {"jump_from": 1.3, "bridge": 0}')
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--obs-noise", type=float, nargs="?", const=1.0, default=0.0,
                    help="sensor noise, a multiple of env.NOISE (legged_gym levels); bare flag = 1")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="action delay in ms (a multiple of the 2 ms physics step; 20 ms = one control step)")
    ap.add_argument("--motor-lag-ms", type=float, default=0.0, help="first-order lag (time constant, ms) of the joint targets")
    ap.add_argument("--gain-scale", type=float, default=1.0, help="PD gains kp and kv x this (on top of the kp randomization)")
    ap.add_argument("--imu-bias", type=float, default=0.0, help="constant tilt bias of the gravity reading, up to this (rad), per try")
    ap.add_argument("--gyro-bias", type=float, default=0.0, help="constant gyro bias up to this (rad/s), per try")
    ap.add_argument("--vel-bias", type=float, default=0.0, help="constant body-speed estimate bias up to this (m/s), per try")
    ap.add_argument("--height-bias", type=float, default=0.0, help="constant body-height estimate bias up to this (m), per try")
    ap.add_argument("--quant", type=float, default=0.0, help="joint encoder step (rad)")
    ap.add_argument("--drop", type=float, default=0.0, help="chance a control packet is lost per step")
    ap.add_argument("--payload", type=float, default=0.0, help="extra load on the body, share of total mass")
    ap.add_argument("--friction", type=float, default=1.0, help="foot grip, multiple of the floor friction")
    bolt_env.add_brake_args(ap, default=False)
    a = ap.parse_args()
    over = json.loads(a.over)
    brake = bolt_env.set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain)
    m = Master(seed=a.seed, imitate=a.imitate, teachers=a.teachers, **over)
    step_ms = m.w.model.opt.timestep * 1000
    if abs(round(a.latency_ms / step_ms) * step_ms - a.latency_ms) > 1e-6:
        ap.error(f"--latency-ms must be a multiple of the {step_ms:g} ms physics step")
    m.real.update(noise=a.obs_noise, delay=a.latency_ms / 1000, lag=a.motor_lag_ms / 1000, gain=a.gain_scale,
                  imu_bias=a.imu_bias, gyro_bias=a.gyro_bias, vel_bias=a.vel_bias, height_bias=a.height_bias,
                  quant=a.quant, drop=a.drop, payload=a.payload, friction=a.friction)
    brain = None if a.brain == "teacher" else Student(a.brain, m)
    t0 = time.time()
    res = dict(brain=a.brain, teachers=a.teachers, over=over, imitate=a.imitate, n=a.n, seed=a.seed)
    if m.real != env.REAL_OFF or brake[0]:
        res["real"] = dict(m.real, latency_ms=a.latency_ms, brake=list(brake))
    if a.exam:
        res["exam"] = clip_exam(m, brain)
        print(res["exam"], flush=True)
    rows, secs = run(Driver(m, brain), a.n, a.seed, a.only, a.with_50)
    show(rows)
    res.update(seconds=round(time.time() - t0), switches=rows)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
