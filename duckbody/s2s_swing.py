"""Sit to stand by swinging up (owner's idea): rock back, then kick hard so the whole body comes up.

    python train.py --skills --env s2s_swing:SwingUpEnv --model model/duck_mid.xml --task sit_to_stand \\
        --name mid_s2s_swing --resume runs/mid_sit_to_stand_v2/checkpoints/policy_500000_steps.zip

Why: every slow push from the swag seat tips the duck backwards (the tail is on the floor and the weight
behind the feet; ~600 scripted variants, 0 rose). The earlier brains that "almost came up" pushed the body
to ~10 cm but past vertical backwards, and the up < 0.4 cut-off ended them there.
  * episodes are not cut off for rocking back (only when upside down, up < -0.6)
  * reward = max(up, 0)² × (height progress seat→stand)², so sitting still or lying back earns ~0, and
    + 2 per step when upright (up > 0.9) at squat height or above; landing shock penalties as before
  * success: upright (up > 0.95) at squat height or above, belly off the floor
"""
import numpy as np

from skills_env import IMPACT_FREE, IMPACT_W, SHOCK_W, SkillEnv

UP_BONUS = 2.0


class SwingUpEnv(SkillEnv):
    rise_term_up = -0.6

    def __init__(self, task=None, **kw):
        kw.setdefault("max_seconds", 6)
        super().__init__(task=task, **kw)

    def _rise_reward(self, action):
        p = self.posture()
        seat_h = float(self.seat_q[2])
        prog = float(np.clip((p["height"] - seat_h) / (self.stand_height - seat_h), 0, 1))
        up = max(p["up"], 0.0)
        reward = up ** 2 * prog ** 2
        if p["up"] > 0.9 and p["height"] > self.crouch_height - 0.01 * self.L:
            reward += UP_BONUS
        reward -= SHOCK_W * abs(self.data.qacc[2]) + IMPACT_W * max(0.0, self._belly_force() / self.weight - IMPACT_FREE)
        return float(reward), dict(h_ref=self.stand_height, up_ref=1.0, **p)

    def success(self, info):
        if self.task == "sit_to_stand" and self.sit_q is not None:
            return bool(info["up"] > 0.95 and info["height"] > self.crouch_height - 0.01 * self.L and not self._trunk_on_floor())
        return super().success(info)
