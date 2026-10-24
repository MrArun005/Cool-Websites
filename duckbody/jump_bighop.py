"""Big hop: ONE deliberate hop with ≥ 1 cm of sole clearance, from standing, then a soft landing.

    python train.py --skills --env jump_bighop:BigHopEnv --model model/duck_mid.xml --task jump \\
        --name mid_jump --resume runs/mid_jump/checkpoints/policy_3000000_steps.zip
    python jump_bighop.py runs/mid_jump/checkpoints/policy_3500000_steps.zip [--episodes 5]   # exam

Same observation / action layout and physics as JumpEnv (XL330 back-EMF, randomized body). Why a
countermovement: a hop from a still stand clears only 5–8 mm; right after a small dip or prep bounce
the rebound stores energy and the same servos clear 9–15 mm (3M brain, paced runtime).

Episode: 2.5 s from standing (training, JumpEnv's practice starts: 20% crouched, 20% just above the floor to practise
landings). It ends 0.3 s after the hop lands; the runtime hands over to the stand brain there.
  the hop        the first flight whose soles clear ≥ 3 mm (smaller prep bounces are free)
  before it      0.05 ready + 0.3 × clip(upward CoM speed / 0.4 m/s)   (dip-and-push shaping)
                 in the air: 0.5 × clip(clearance / 12 mm) per step
  at landing     BONUS × clip((max clearance − 4 mm) / 8 mm): nothing under 4 mm, ¼ at 6 mm, full at 12 mm
  after landing  0.1 + 0.3 × standing still (CoM speed), −1 per step with both feet off the floor
  always         upright × no_spin gates the positive terms; JumpEnv's landing shock penalty
                 (1·x + 2·x², x = F/W − 2, plus 5 per W above 3.5)
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from jump_env import SHOCK_FREE, SHOCK_HARD, SHOCK_HARD_W, SHOCK_LIN, SHOCK_SQ, JumpEnv

HERE = Path(__file__).resolve().parent
SECONDS = 2.5
HOP_MIN = 0.003           # m: flights clearing less are prep bounces, not the hop
BONUS = 30.0
CLEAR_ZERO, CLEAR_FULL = 0.004, 0.012
AFTER_S = 0.3             # s the episode goes on after the hop lands (landing shock, settle)
PUSH_V, PUSH_W = 0.4, 0.3


class BigHopEnv(JumpEnv):
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.max_steps = int(SECONDS / self.dt)
        d = self.data
        in_air = self.feet_on_ground().sum() == 0 and self.practice and d.qpos[2] > 0
        # stage: 0 before the hop, 1 in the hop's flight, 2 landed
        self.stage, self.flight_clear, self.hop_clear, self.extra_air = 0, 0.0, 0.0, 0
        self.air_start = bool(in_air and min(self._sole_bottom(g) for g in self.sole_ids) > 0.0015)
        if self.air_start:
            self.stage = 1                      # practising the landing: no bonus for this "hop"
        self.air_now = 0
        return obs, info

    def _reward(self, action):
        if self.command != "jump":
            return super()._reward(action)
        m, d = self.model, self.data
        p = self.posture()
        feet = self.feet_on_ground()
        airborne = feet.sum() == 0
        mujoco.mj_subtreeVel(m, d)
        v_com = d.subtree_linvel[1].copy()
        vz = float(v_com[2])
        fw = 1.0 + (vz - self.com_vz) / self.dt / 9.81
        self.com_vz = vz
        x = max(0.0, fw - SHOCK_FREE)
        shock = SHOCK_LIN * x + SHOCK_SQ * x * x + SHOCK_HARD_W * max(0.0, fw - SHOCK_HARD)
        self.peak_fw = max(self.peak_fw, fw)
        clear = min(self._sole_bottom(g) for g in self.sole_ids)
        rise = p["height"] - self.stand_height

        up = 1.0 if p["up"] >= 0.9 else float(np.exp(-((p["up"] - 0.9) / 0.15) ** 2))
        spin = np.exp(-(p["yaw_rate"] / 0.8) ** 2) * np.exp(-float(v_com[:2] @ v_com[:2]) / (0.25 * self.L) ** 2)
        gate = up * spin
        r = 0.0
        if airborne:
            self.air_now += 1
            self.flight_clear = max(self.flight_clear, clear)
            if self.air_now == 2:
                self.hops += 1
            if self.stage == 0 and self.flight_clear >= HOP_MIN:
                self.stage = 1                                  # this flight is the hop
            if self.stage <= 1:
                self.best_hop = max(self.best_hop, rise)
                r += gate * 0.5 * float(np.clip(clear / CLEAR_FULL, 0, 1))
            else:
                self.extra_air += 1
                r -= 1.0                                        # no more hops after the landing
        else:
            if self.air_now > 0 and self.stage == 1:            # the hop just landed
                self.stage = 2
                if not self.air_start:
                    self.hop_clear = self.flight_clear
                    r += gate * BONUS * float(np.clip((self.flight_clear - CLEAR_ZERO) / (CLEAR_FULL - CLEAR_ZERO), 0, 1))
                self.max_steps = min(self.max_steps, self.t + int(AFTER_S / self.dt))
            self.air_now, self.flight_clear = 0, 0.0
            if self.stage == 0:
                r += gate * (0.05 + PUSH_W * float(np.clip(vz / PUSH_V, 0, 1)))
            elif self.stage == 2:
                still = float(np.exp(-float(v_com @ v_com) / 0.05 ** 2)) if feet.sum() == 2 else 0.0
                r += gate * (0.1 + 0.3 * still)
        reward = r - shock
        info = dict(h_ref=self.stand_height, up_ref=1.0, fw=fw, shock=shock, clear=clear, airborne=bool(airborne),
                    stage=self.stage, hop_clear=self.hop_clear, **p)
        return float(reward), info


def evaluate(ckpt, episodes=5, seed=100, back_emf=True):
    from jump_eval import load
    rows = []
    for ep in range(episodes):
        env = BigHopEnv(task="jump", randomize=True, seed=seed + ep, model_path=str(HERE / "model" / "duck_mid.xml"),
                        practice=False)
        env.back_emf = back_emf
        brain = load(ckpt, env)
        obs, _ = env.reset(seed=seed + ep)
        done, total, flights, air, cl, land_t = False, 0.0, [], 0, 0.0, None
        while not done:
            obs, r, term, trunc, info = env.step(brain(obs))
            total += r
            done = term or trunc
            if info["airborne"]:
                air += 1
                cl = max(cl, info["clear"])
            else:
                if air:
                    flights.append(round(1000 * cl, 1))
                air, cl = 0, 0.0
                if info["stage"] == 2 and land_t is None:
                    land_t = env.t
        rows.append(dict(hop_clear_mm=round(1000 * env.hop_clear, 1), flights_mm=flights, landed_s=land_t and land_t * env.dt,
                         peak_fw=round(env.peak_fw, 2), extra_air=env.extra_air, fell=bool(term),
                         up=round(info["up"], 2), reward=round(total, 1)))
    return rows


def check():
    """Sign check (scripted through BigHopEnv.step, nominal body, back-EMF): one hop must beat standing,
    a hop that keeps bouncing must score below the same hop that stops, the bigger hop more."""
    from jump_check import CROUCH, PUSH, REACH, SOFT, HOME, leg_action, pose
    env = BigHopEnv(task="jump", randomize=False, model_path=str(HERE / "model" / "duck_mid.xml"), practice=False)

    def run(plan):
        env.reset(seed=0)
        total, k = 0.0, 0
        for pose_, n in plan:
            for _ in range(n):
                _, r, term, trunc, info = env.step(leg_action(env, *pose_))
                total += r
                if term or trunc:
                    return dict(reward=round(total, 2), hop_mm=round(1000 * env.hop_clear, 1), peak_fw=round(env.peak_fw, 2),
                                extra_air=env.extra_air)
        return dict(reward=round(total, 2), hop_mm=round(1000 * env.hop_clear, 1), peak_fw=round(env.peak_fw, 2), extra_air=env.extra_air)
    once = [(CROUCH, 15), (PUSH, 7), (HOME, 2), (REACH, 8), (SOFT, 10), (HOME, 83)]
    dip = [(HOME, 5), (pose(1.2, 0.46), 3), (CROUCH, 4), (PUSH, 7), (HOME, 2), (REACH, 8), (SOFT, 10), (HOME, 86)]
    bounce = [(CROUCH, 15), (PUSH, 7), (HOME, 2)] * 5 + [(HOME, 5)]
    rows = dict(stand=run([(HOME, 125)]), hop_once=run(once), dip_hop=run(dip), keeps_bouncing=run(bounce))
    for k, v in rows.items():
        print(f"{k:15s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="reward sign check, scripted")
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--no-back-emf", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    for c in a.ckpts:
        rows = evaluate(c, a.episodes, back_emf=not a.no_back_emf)
        ok = sum(r["hop_clear_mm"] >= 10 and not r["fell"] for r in rows)
        print(f"{Path(c).name}: ≥10 mm {ok}/{len(rows)}, clear mean {np.mean([r['hop_clear_mm'] for r in rows]):.1f} mm, "
              f"falls {sum(r['fell'] for r in rows)}, peak F/W {max(r['peak_fw'] for r in rows):.2f}, "
              f"extra air steps {sum(r['extra_air'] for r in rows)}, reward {np.mean([r['reward'] for r in rows]):.1f}")
        for r in rows:
            print("   ", r)


if __name__ == "__main__":
    main()
