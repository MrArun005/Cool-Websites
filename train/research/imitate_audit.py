"""Motor audit of a dog-motion brain on the new training starts (HANDOFF.md "real body, real motors" pattern, as
research/jump_audit.py): landing starts, moving-start posture commands and the canter -> trot switch.

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/imitate_audit.py ../models/bolt-dog-imitate-gpu-360M/params.pkl \
        --motor-brake --out research/results/audit_imitate_360M_brake.json

Every physics substep (500 Hz, not only the 50 Hz policy steps) of every try is recorded: joint speeds, the torque each
motor gave (clipped by bolt_env's motor model), the braking torque (--motor-brake: bolt_env's back-EMF brake above
--brake-speed) and the motoring power (torque x speed where they have the same sign: the motor driving the joint;
braking against the motion is not limited by the 420 W). The body is bolt_env.MOTOR (BOLT_MOTOR_TORQUE, set from
--torque before bolt_env is imported). Fixed physics (no randomization), the brain's deterministic action. Kinds:
    landing   --episodes landing states (imitate_env landing_rate: mid-air to touch-down, canter, bridge to trot)
    moving    --episodes moving starts per posture clip (sit, lie, getup) from walk and trot at their clip pace
    switch    --episodes canter tries switched to trot after 1 s (the transition term's case), 3 s each
Per kind and joint type (abd, hip, knee): peak speed, share of substeps above the rated 21 rad/s and above the brake
speed, peak torque given, peak motoring power, peak braking torque, and the tries that ended early (fell).
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain", help="a numpy params.pkl (mjx_imitate) brain")
    ap.add_argument("--torque", type=float, default=33.5, help="motor torque cap, N·m (BOLT_MOTOR_TORQUE)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--kinds", nargs="+", default=["landing", "moving", "switch"])
    ap.add_argument("--latency-ms", type=float, default=0.0, help="fixed action latency (imitate_env latency_ms), ms")
    ap.add_argument("--obs-noise", type=float, default=0.0, help="sensor noise, x imitate_env.NOISE")
    ap.add_argument("--motor-lag-ms", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    os.environ["BOLT_MOTOR_TORQUE"] = str(ap.parse_known_args()[0].torque)
    sys.path.insert(0, str(HERE.parent))
    import bolt_env
    bolt_env.add_brake_args(ap, default=False)
    a = ap.parse_args()
    print("motor brake (on, rad/s, N·m per rad/s):", bolt_env.set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain), flush=True)
    import numpy as np
    import imitate_env
    from imitate_env import ALL_CLIPS, ImitateEnv
    from mjx_imitate import MjxImitatePolicy
    assert bolt_env.MOTOR["torque"] == a.torque, bolt_env.MOTOR
    rec = []

    def physics(model, data, n):   # bolt_env.physics, recording every substep
        on, bspeed, bgain = bolt_env.brake_settings(model)
        for _ in range(n):
            lo, hi = bolt_env.motor_limits(data.qvel[6:18])
            model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1] = lo, hi
            brake = bolt_env.brake_torque(data.qvel[6:18], bspeed, bgain) if on else np.zeros(12)
            if on:
                data.qfrc_applied[6:18] = brake
            qd = data.qvel[6:18].copy()
            bolt_env.mujoco.mj_step(model, data)
            rec.append(np.concatenate([qd, data.actuator_force, brake]))

    imitate_env.physics = physics
    pol = MjxImitatePolicy(a.brain)
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, landing_rate=1.0, moving_rate=1.0, latency_ms=(0.0, a.latency_ms),
                     obs_noise=a.obs_noise, motor_lag_ms=a.motor_lag_ms)
    env.exam_latency_ms = a.latency_ms
    rng = np.random.default_rng(a.seed)
    out = dict(brain=a.brain, torque=a.torque, latency_ms=a.latency_ms, obs_noise=a.obs_noise, motor_lag_ms=a.motor_lag_ms, motor=dict(bolt_env.MOTOR), episodes=a.episodes, seed=a.seed, kinds={})
    for kind in a.kinds:
        rec.clear()
        tries, fell_n = [], 0
        starts = []
        if kind == "landing":
            starts = [("landing", int(i)) for i in rng.choice(len(env.landing["tau0"]), a.episodes, replace=False)]
        elif kind == "moving":
            starts = [("moving", (post, gait, float(rng.uniform(0, env.clips[gait].length)), env.gait_speed(gait)))
                      for post in env.moving_clips for gait in env.moving_gaits for _ in range(max(1, a.episodes // 4))]
        else:
            starts = [("switch", float(rng.uniform(0, env.clips["canter"].length))) for _ in range(a.episodes)]
        for k, (how, arg) in enumerate(starts):
            env.exam_landing = arg if how == "landing" else None
            env.exam_moving = arg if how == "moving" else None
            env.exam_start = ("canter", arg) if how == "switch" else None
            obs, _ = env.reset(seed=a.seed + k)
            steps = 150 if how == "switch" else env.limit
            for t in range(steps):
                if how == "switch" and t == 50:
                    env._switch_to(env.clips["trot"])
                obs, r, fell, done, info = env.step(pol.act(obs))
                if fell or (done and how != "switch"):
                    break
            fell_n += bool(fell)
        r = np.array(rec)
        qd, f, brk = r[:, :12], r[:, 12:24], r[:, 24:36]
        v, tq = np.abs(qd), np.abs(f)
        motoring = np.where(f * qd > 0, tq * v, 0.0)
        joints = {}
        for j, name in enumerate(("abd", "hip", "knee")):
            idx = slice(j, 12, 3)
            joints[name] = dict(peak_speed=round(float(v[:, idx].max()), 1), over_21_pct=round(100 * float((v[:, idx] > 21).mean()), 2),
                                over_brake_speed_pct=round(100 * float((v[:, idx] > a.brake_speed).mean()), 2),
                                peak_torque=round(float(tq[:, idx].max()), 2), peak_power_w=round(float(motoring[:, idx].max()), 1),
                                peak_brake_torque=round(float(np.abs(brk[:, idx]).max()), 1))
        out["kinds"][kind] = dict(tries=len(starts), fell=fell_n, substeps=len(r), joints=joints)
        print(kind, json.dumps(out["kinds"][kind]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
