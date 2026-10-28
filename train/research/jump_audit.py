"""Motor audit of a jump brain's attempts at one bar height (HANDOFF.md "real body, real motors" pattern).

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/jump_audit.py ../models/bolt-dog-jump-runup-16i-34.5M \
        --torque 33.5 --height 0.60 --out research/results/audit60_34.5M_t33.5.json

Runs the fair exam's tries (jump_env.exam: fixed physics, straight approach, `--speed` m/s from `--dist` m,
seeds `--seed`+k, 200 steps) and records EVERY physics substep (not only the 50 Hz policy steps) of every
try: joint speeds, the torque each motor gave (clipped by the motor model), the torque the position servo
asked for (kp (target - q) - kv q'), and the motor model's limits at that speed. BOLT_MOTOR_TORQUE is set
from --torque BEFORE bolt_env is imported, so the whole run uses that body. --motor-brake (default off) adds
bolt_env's braking above --brake-speed; over_brake_speed_pct and peak_brake_torque per joint type then show it.

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/jump_audit.py ../models/bolt-dog-jump-runup-16i-34.5M \
        --height 0.60 --motor-brake --out research/results/audit60_34.5M_brake.json

Measured 28 Oct, 16i-34.5M at 60 cm, 30 tries (seed 7000, 200 steps; results/audit60_34.5M_*.json), peak joint
speed over all tries hip / knee (rad/s), share of substeps above 21 rad/s hip / knee, clean tries:
  brake off                        29.2 / 31.7   3.8% / 8.6%    4/30   (--out .../audit60_34.5M_nobrake.json)
  --motor-brake (gain 1.6, 21)     27.7 / 28.0  11.4% / 15.3%   2/30   (--out .../audit60_34.5M_brake.json)
  --motor-brake --brake-gain 3.5   25.1 / 26.0   9.2% / 14.5%   1/30   (--out .../audit60_34.5M_brake_g3.5.json)
  --motor-brake --brake-gain 8     24.0 / 25.4   3.3% / 7.2%    0/30   (--out .../audit60_34.5M_brake_g8.json)
The brake removes the landing back-drive spikes (above the 25 rad/s cutoff) but this brain still drives its joints
in the 21-25 rad/s band where the motor model lets the drive fade out, so speeds sit at 21 + ~4-7 rad/s, not at 21.

Per try: clean (agility rules: over, bar stays in its cups, lands upright), apex body height, body height
over the bar, the lowest paw's clearance over the bar while crossing, and per joint type (abd, hip, knee):
peak speed, share of substeps above 21 rad/s, peak torque, peak motoring power (torque x speed, same sign),
the same three over the take-off window only (takeoff_*: speeds above 21 rad/s outside it are landing impacts
back-driving the joint, not the motor driving it), and the share of the take-off push (the 0.3 s up to the moment the last paw leaves the floor before the bar is
cleared) spent AT the motor's limit
(asked for at least the available torque in the direction of motion). The best tries are ranked clean
first, then by paw clearance. Writes everything to --out.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--zip", default="policy.zip")
    ap.add_argument("--stats", default="vecnormalize.pkl")
    ap.add_argument("--torque", type=float, default=33.5, help="motor torque cap, N·m (BOLT_MOTOR_TORQUE)")
    ap.add_argument("--height", type=float, default=0.60)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--dist", type=float, default=1.5)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--best", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200, help="steps per try (default 200, as this audit always used)")
    ap.add_argument("--out", required=True)
    os.environ["BOLT_MOTOR_TORQUE"] = str(ap.parse_known_args()[0].torque)
    sys.path.insert(0, str(HERE.parent))
    import bolt_env
    bolt_env.add_brake_args(ap, default=False)
    a = ap.parse_args()
    print("motor brake (on, rad/s, N·m per rad/s):", bolt_env.set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain), flush=True)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    import jump_env
    from jump_env import JumpEnv, JumpPolicy
    assert bolt_env.MOTOR["torque"] == a.torque, bolt_env.MOTOR

    rec, env = [], None

    def physics(model, data, n):   # bolt_env.physics, recording every substep
        on, bspeed, bgain = bolt_env.brake_settings(model)
        for _ in range(n):
            lo, hi = bolt_env.motor_limits(data.qvel[6:18])
            model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1] = lo, hi
            brake = bolt_env.brake_torque(data.qvel[6:18], bspeed, bgain) if on else np.zeros(12)
            if on:
                data.qfrc_applied[6:18] = brake
            q, qd = data.qpos[7:19].copy(), data.qvel[6:18].copy()
            ask = model.actuator_gainprm[:, 0] * data.ctrl + model.actuator_biasprm[:, 1] * q + model.actuator_biasprm[:, 2] * qd
            bolt_env.mujoco.mj_step(model, data)
            down = float(env.feet_on_ground().sum()) if env is not None else 0.0
            rec.append(np.concatenate([qd, data.actuator_force, ask, lo, hi, [down], brake]))   # qd and limits as at the step

    jump_env.physics = physics
    pol = JumpPolicy(Path(a.folder), a.zip, a.stats)
    env = JumpEnv(randomize=False, real=pol.real, max_steps=a.max_steps)
    env.exam_speed, env.exam_gait = a.speed, False
    h, sub = a.height, env.frame_skip
    tries = []
    for k in range(a.episodes):
        env.exam_bar = (a.dist, h)
        env.exam_askew = (0.0, 0.0)
        obs, _ = env.reset(seed=a.seed + k)
        rec.clear()
        info, fell, apex, over_z, paw_clear, clear_step = {}, False, 0.0, None, None, None
        for t in range(env.max_steps):
            obs, _, fell, trunc, info = env.step(pol.act(obs))
            apex = max(apex, float(env.data.qpos[2]))
            past = env.paws_past()
            if abs(env.hurdle_ahead()) < 0.1:
                over_z = max(over_z or 0.0, float(env.data.qpos[2]))
            crossing = np.abs(past) < 0.05
            if crossing.any():
                z = np.array([env.data.geom_xpos[f][2] for f in env.foot_ids])[crossing] - 0.025 - h   # paw bottom over bar
                paw_clear = float(z.min()) if paw_clear is None else min(paw_clear, float(z.min()))
            if info.get("cleared") and clear_step is None:
                clear_step = len(rec)
            if fell or trunc:
                break
        r = np.array(rec)
        qd, f, ask, lo, hi = (r[:, 12 * i:12 * (i + 1)] for i in range(5))
        v, tq = np.abs(qd), np.abs(f)
        motoring = np.where(f * qd > 0, tq * v, 0.0)
        end = clear_step or len(r)
        down = r[:end, 60]
        brk = np.abs(r[:, 61:73])
        liftoff = int(np.nonzero(down > 0)[0].max()) + 1 if (down > 0).any() else end   # last paw leaves the floor
        w = slice(max(0, liftoff - int(0.3 / env.model.opt.timestep)), liftoff)   # the push: 0.3 s up to lift-off
        avail = np.where(ask >= 0, hi, -lo)      # what the motor could give in the asked direction
        at_limit = np.abs(ask[w]) >= avail[w] - 0.05
        joints = {}
        for j, name in enumerate(("abd", "hip", "knee")):
            idx = slice(j, 12, 3)
            joints[name] = dict(peak_speed=round(float(v[:, idx].max()), 1), over_21_pct=round(100 * float((v[:, idx] > 21).mean()), 2),
                                peak_torque=round(float(tq[:, idx].max()), 1), peak_power_w=round(float(motoring[:, idx].max())),
                                takeoff_at_limit_pct=round(100 * float(at_limit[:, idx].mean()), 1),
                                takeoff_peak_speed=round(float(v[w][:, idx].max()), 1),
                                takeoff_over_21_pct=round(100 * float((v[w][:, idx] > 21).mean()), 2),
                                takeoff_peak_power_w=round(float(motoring[w][:, idx].max())),
                                peak_asked_torque=round(float(np.abs(ask[:, idx]).max()), 1),
                                over_brake_speed_pct=round(100 * float((v[:, idx] > a.brake_speed).mean()), 2),
                                peak_brake_torque=round(float(brk[:, idx].max()), 1))
        clean = bool(info["cleared"] and not info.get("dropped", info["touched"]) and not fell and info["up"] > 0.7)
        tries.append(dict(seed=a.seed + k, clean=clean, cleared=bool(info["cleared"]), dropped=bool(info.get("dropped")),
                          touched=bool(info["touched"]), fell=bool(fell), apex=round(apex, 3),
                          body_over_bar=None if over_z is None else round(over_z, 3),
                          paw_clearance=None if paw_clear is None else round(paw_clear, 3), joints=joints))
        print(f"seed {a.seed + k}: clean={clean} cleared={info['cleared']} dropped={info.get('dropped')} apex={apex:.3f} "
              f"paw_clear={paw_clear if paw_clear is None else round(paw_clear, 3)} "
              + " ".join(f"{n}:{x['peak_speed']}rad/s,{x['peak_torque']}Nm,{x['peak_power_w']}W,lim{x['takeoff_at_limit_pct']}%"
                         for n, x in joints.items()), flush=True)
    best = sorted(tries, key=lambda x: (x["clean"], x["cleared"], x["paw_clearance"] if x["paw_clearance"] is not None else -9),
                  reverse=True)[:a.best]
    peak = {n: {m: max(t["joints"][n][m] for t in tries) for m in tries[0]["joints"][n]} for n in ("abd", "hip", "knee")}
    peak_best = {n: {m: max(t["joints"][n][m] for t in best) for m in best[0]["joints"][n]} for n in ("abd", "hip", "knee")}
    out = dict(args=vars(a), motor=bolt_env.MOTOR, clean=sum(t["clean"] for t in tries), cleared=sum(t["cleared"] for t in tries),
               tries=len(tries), apex_max=max(t["apex"] for t in tries),
               paw_clearance_max=max((t["paw_clearance"] for t in tries if t["paw_clearance"] is not None), default=None),
               peak_all=peak, peak_best=peak_best, best=best, all=tries)
    print(json.dumps({k: out[k] for k in ("motor", "clean", "cleared", "tries", "apex_max", "paw_clearance_max", "peak_all", "peak_best")}, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
