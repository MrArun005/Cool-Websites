"""How high can the mid duck hop with its XL330 servos? Scripted crouch-then-extend in physics.

    python duckbody/jump_probe.py                 # sweep crouch depth / push / landing posture
    python duckbody/jump_probe.py --best          # repeat the best hop on 20 randomized bodies

Every hop is commanded through SkillEnv's own action interface (home + scale × action, clipped
to ±1), so it is exactly what a policy could do. Measured per hop, at the 2 ms physics step:
  hop      highest trunk height above standing while both soles are off the floor (cm)
  air      longest time with both soles off the floor (ms)
  peak F   largest total sole contact force within 0.3 s after touchdown, × body weight
  peak az  largest trunk vertical acceleration after touchdown, in g
  up       uprightness 1 s after landing (1 = upright)
"""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np

from skills_env import SkillEnv

HERE = Path(__file__).resolve().parent


def leg_action(env, hip, knee, ankle):
    """Action that commands both legs to (hip, knee, ankle) in the left leg's sign convention."""
    m = env.model
    target = env.home.copy()
    for side, s in (("left", 1), ("right", -1)):
        target[m.actuator(f"{side}_hip_pitch").id] = s * hip
        target[m.actuator(f"{side}_knee").id] = knee
        target[m.actuator(f"{side}_ankle").id] = ankle
    a = (target - env.home)[env.pol] / env.action_scale
    return np.clip(a, -1, 1)


class Meter:
    """Substep-resolution flight and landing measurements."""

    def __init__(self, env):
        self.env = env
        self.W = env.model.body_subtreemass[1] * 9.81
        self.f6 = np.zeros(6)
        self.air = self.best_air = 0
        self.hop = 0.0
        self.landings = []           # (peak F/W, peak az/g) per touchdown after a flight ≥ 20 ms
        self.after = None            # substeps since touchdown, while measuring a landing
        self.vz_prev = 0.0

    def sole_force(self):
        e, d = self.env, self.env.data
        f = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if e.floor in (c.geom[0], c.geom[1]) and (c.geom[0] in e.sole_ids or c.geom[1] in e.sole_ids):
                mujoco.mj_contactForce(e.model, d, i, self.f6)
                f += self.f6[0]
        return f

    def substep(self):
        e, d = self.env, self.env.data
        feet = e.feet_on_ground()
        h = d.qpos[2] - e.stand_height
        az = (d.qvel[2] - self.vz_prev) / e.model.opt.timestep / 9.81
        self.vz_prev = d.qvel[2]
        if feet.sum() == 0:
            self.air += 1
            if self.air * e.model.opt.timestep >= 0.02:
                self.hop = max(self.hop, h)
            self.best_air = max(self.best_air, self.air)
        else:
            if self.air * e.model.opt.timestep >= 0.02:
                self.landings.append([0.0, 0.0, self.air * e.model.opt.timestep])
                self.after = 0
            self.air = 0
        if self.after is not None:
            L = self.landings[-1]
            L[0] = max(L[0], self.sole_force() / self.W)
            L[1] = max(L[1], az)
            self.after += 1
            if self.after * e.model.opt.timestep > 0.3:
                self.after = None


def run_hops(env, crouch, push, land, n_hops=3, seed=0, settle=0.6, crouch_t=0.4, push_t=0.16):
    """crouch: (hip, knee, ankle) to sink into; push: pose commanded to extend; land: pose held in the air."""
    env.reset(seed=seed)
    meter = Meter(env)
    home = leg_action(env, *HOME)
    plan = [(home, settle)]
    for _ in range(n_hops):
        plan += [(leg_action(env, *crouch), crouch_t), (leg_action(env, *push), push_t),
                 (leg_action(env, *land), 0.5), (home, 0.4)]
    up_min, fell = 1.0, False
    for a, secs in plan:
        for _ in range(int(round(secs * 50))):
            target = env.home.copy()                       # SkillEnv.step's action → servo target
            target[env.pol] += env.action_scale * a
            env._personality(target)
            env.pending.append(np.clip(target, env.lo, env.hi))
            env.data.ctrl[:] = env.pending[-1 - env.latency]
            env.pending = env.pending[-2:]
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, env.data)
                meter.substep()
            up = env.posture()["up"]
            up_min = min(up_min, up)
            fell |= up < 0.3
    return dict(hop_cm=100 * meter.hop, air_ms=1000 * meter.best_air * env.model.opt.timestep,
                landings=[[round(x, 2) for x in L] for L in meter.landings],
                peak_F=max([L[0] for L in meter.landings], default=0.0),
                peak_az=max([L[1] for L in meter.landings], default=0.0),
                up_end=env.posture()["up"], up_min=up_min, fell=fell)


HOME = (-0.63, 1.37, -0.79)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model" / "duck_mid.xml"))
    ap.add_argument("--best", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    env = SkillEnv(task="jump", randomize=False, model_path=a.model, practice=False)
    print(f"stand height {env.stand_height * 100:.1f} cm, crouch {env.crouch_height * 100:.1f} cm, "
          f"L={env.L:.3f} (1.5 cm×L = {1.5 * env.L:.2f} cm, 3 cm×L = {3 * env.L:.2f} cm), "
          f"weight {env.model.body_subtreemass[1] * 9.81:.2f} N")
    results = []
    # crouch depth: knee angle, hips/ankles keep the feet flat (env._crouch_pose recipe)
    for knee_c, knee_p, land_soft, crouch_t, push_t in itertools.product(
            (1.6, 1.85, 2.1), (0.37, 0.7, 1.0), (False, True), (0.3, 0.5), (0.08, 0.16)):
        crouch = (-0.46 * knee_c, knee_c, -(knee_c - 0.46 * knee_c))
        # extend: as straight as the action range allows (knee ≥ home − 1 rad), feet flat
        push = (-0.46 * knee_p, knee_p, -(knee_p - 0.46 * knee_p))
        land = HOME if land_soft else push
        r = run_hops(env, crouch, push, land, crouch_t=crouch_t, push_t=push_t)
        r.update(knee_crouch=knee_c, knee_push=knee_p, soft_land=land_soft, crouch_t=crouch_t, push_t=push_t)
        results.append(r)
        print(f"crouch knee {knee_c:.2f} push knee {knee_p:.2f} {'soft' if land_soft else 'stiff'} land "
              f"ct {crouch_t} pt {push_t}: hop {r['hop_cm']:5.2f} cm, air {r['air_ms']:4.0f} ms, "
              f"peak F {r['peak_F']:5.1f} W, peak az {r['peak_az']:5.1f} g, up_min {r['up_min']:.2f}, "
              f"{'FELL' if r['fell'] else 'ok'}  landings {r['landings']}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
