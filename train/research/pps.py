"""Policy-prior predictive sampling (PPS): plan around the trained dog-motion brain with short look-ahead
rollouts, and (pps_distill.py) distill what the planner does back into the brain.

The brain (run 18b) is a reflex: one forward pass per 1/50 s, no look-ahead. Most of its failures are
"walked into a state it cannot get out of" 0.5-1 s after a switch, or mid-canter. A planner that tries a
few dozen slightly different futures before committing can see those coming. This is the idea behind
TD-MPC2 / LOOP / MuZero-style "policy + planning", and predictive sampling / MPPI on legged robots, here
with the cheapest possible proposal distribution: the trained brain itself.

Every M control steps (default 5 = 0.1 s):
  * copy Bolt's body state (robot qpos/qvel only) into K "shadow" ImitateEnvs that have NOMINAL physics
    (no randomized motor strength, mass or friction, no sensor noise), so the model the planner uses is
    deliberately NOT the world Bolt lives in;
  * candidate k runs the brain closed-loop for H steps with a smooth open-loop perturbation e_k(t) added
    to its action (piecewise-linear knots every M steps, N(0, sigma)); candidate 0 is e = 0 (the plain
    brain) and candidate 1 is last plan shifted (warm start);
  * score = sum of the training tracking reward over the horizon, minus FALL_PENALTY if the shadow
    falls (ImitateEnv's own termination: 15 cm off the reference height, 0.8 rad tilt, body on floor);
  * the live Bolt then runs the brain with the winner's perturbation for the next M steps (the brain
    still closes the loop on the real state every step: only the small offset is planned).

Cost: K*H shadow steps per M live steps (K=16, H=30, M=5: ~100 shadow steps per live step, ~0.06 s of
one CPU core per 0.02 s of robot time on this machine: 3x slower than real time on one core).
"""
from pathlib import Path

import mujoco
import numpy as np
import torch

from imitate_env import ImitateEnv

FALL_PENALTY = 10.0


class BatchBrain:
    """The ImitatePolicy brain, evaluated for a batch of observations at once."""

    def __init__(self, policy):
        self.p = policy

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] > 66 + self.p.slots:
            obs = np.concatenate([obs[..., :48 + self.p.slots], obs[..., 57:]], axis=-1)
        x = torch.as_tensor(self.p.norm.normalize_obs(obs), dtype=torch.float32)
        with torch.no_grad():
            a = self.p.model.policy.get_distribution(x).mode() if hasattr(self.p.model.policy, "get_distribution") else None
        return np.clip(a.numpy(), -1, 1)


class Planner:
    def __init__(self, policy, clips, K=16, H=30, M=5, sigma=0.2, seed=0, robust=False):
        self.brain = BatchBrain(policy)
        self.K, self.H, self.M, self.sigma = K, H, M, sigma
        self.rng = np.random.default_rng(seed)
        self.shadows = [ImitateEnv(randomize=False, clips=clips, slots=policy.slots) for _ in range(K)]
        self.robust = robust   # shadows with their own randomized physics (a crude ensemble model)
        self.plan = np.zeros((H, 12))
        self.stats = dict(calls=0, chose_plain=0, plain_fell=0, best_fell=0)

    def _knots_to_seq(self, knots):
        """(K, nk, 12) knot values every M steps -> (K, H, 12) piecewise-linear perturbations."""
        nk = knots.shape[1]
        xs = np.arange(nk) * self.M
        t = np.arange(self.H)
        idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, nk - 2)
        w = ((t - xs[idx]) / self.M)[None, :, None]
        return knots[:, idx] * (1 - w) + knots[:, idx + 1] * w

    def _load(self, sh, live):
        """Copy Bolt's body and the brain's bookkeeping (clip, clip clock, last action) into a shadow."""
        d = sh.data
        mujoco.mj_resetData(sh.model, d)
        d.qpos[:19] = live.data.qpos[:19]
        d.qvel[:18] = live.data.qvel[:18]
        d.qacc_warmstart[:18] = live.data.qacc_warmstart[:18]
        sh.clip, sh.tau0, sh.t = sh.clips[live.clip.name], live.tau0, live.t
        sh.last_action = live.last_action.copy()
        sh.limit = 10 ** 9
        mujoco.mj_forward(sh.model, d)

    def new_plan(self, live):
        """Plan from the live ImitateEnv's current state. Returns (M, 12) perturbations for the next M steps."""
        K, H, M = self.K, self.H, self.M
        nk = H // M + 1
        knots = self.rng.normal(0, self.sigma, (K, nk, 12))
        knots[0] = 0.0                                             # the plain brain
        prev = np.concatenate([self.plan[M:], np.repeat(self.plan[-1:], M, 0)])   # warm start, shifted
        knots[1] = prev[::M][:nk] if len(prev[::M]) >= nk else np.pad(prev[::M], ((0, nk - len(prev[::M])), (0, 0)))
        knots[2:K // 2] += knots[1]                                # half the samples around the last plan
        seq = self._knots_to_seq(knots)
        for k, sh in enumerate(self.shadows):
            if self.robust and k > 0:
                m = sh.model
                kp = sh.nominal_kp * self.rng.uniform(0.85, 1.15)
                m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
                m.body_mass[sh.torso] = sh.nominal_mass[sh.torso] * self.rng.uniform(0.85, 1.15)
                m.geom_friction[:, 0] = sh.nominal_friction * self.rng.uniform(0.6, 1.3)
            self._load(sh, live)
        score, alive = np.zeros(K), np.ones(K, bool)
        obs = np.stack([sh._obs() for sh in self.shadows])
        for t in range(H):
            a = self.brain.act(obs)
            for k, sh in enumerate(self.shadows):
                if not alive[k]:
                    continue
                o, r, fell, _, _ = sh.step(np.clip(a[k] + seq[k, t], -1, 1))
                obs[k] = o
                score[k] += r
                if fell:
                    alive[k] = False
                    score[k] -= FALL_PENALTY
        best = int(np.argmax(score))
        self.plan = seq[best]
        st = self.stats
        st["calls"] += 1
        st["chose_plain"] += best == 0
        st["plain_fell"] += not alive[0]
        st["best_fell"] += not alive[best]
        return self.plan[:M]


class PlannedController:
    """Drop-in controller for a live ImitateEnv: brain action + planned perturbation, replanned every M steps."""

    def __init__(self, policy, planner):
        self.policy, self.planner = policy, planner
        self.queue = []

    def reset(self):
        self.queue = []
        self.planner.plan[:] = 0.0

    def act(self, env, obs):
        if not self.queue:
            self.queue = list(self.planner.new_plan(env))
        e = self.queue.pop(0)
        return np.clip(self.policy.act(obs) + e, -1, 1)
