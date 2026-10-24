"""What a jump brain actually does: hops, airtime, sole clearance, landing force.

    python duckbody/jump_eval.py runs/mid_jump/checkpoints/policy_2000000_steps.zip [--episodes 5]
    python duckbody/jump_eval.py runs/mid_jump/best/policy.zip

Exam conditions (start standing, randomized body, practice starts off). Per episode:
  hops       flights with both soles off the floor for ≥ 2 control steps (40 ms)
  air        longest flight, ms (control-step resolution, 20 ms)
  clear      highest lowest-sole point in flight, mm
  rise       SkillEnv's hop measure: trunk height above standing while airborne, cm (exam: > 1.04 cm)
  F/W        largest mean ground force over a 20 ms step, × body weight (from the CoM acceleration)
  landings   share of landings above 2.5× and 3.5× body weight
  ok         SkillEnv's success: upright at the end and rise > 1.5 cm × L
"""
import argparse
import re
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from jump_env import SHOCK_FREE, JumpEnv

HERE = Path(__file__).resolve().parent


def stats_for(ckpt):
    ckpt = Path(ckpt)
    if ckpt.name == "policy.zip":
        return ckpt.with_name("vecnormalize.pkl")
    n = re.findall(r"\d+", ckpt.name)[0]
    return ckpt.with_name(f"policy_vecnormalize_{n}_steps.pkl")


def evaluate(ckpt, episodes=5, seed=100, model=HERE / "model" / "duck_mid.xml"):
    policy = PPO.load(str(ckpt), device="cpu")
    rows = []
    for ep in range(episodes):
        env = JumpEnv(task="jump", randomize=True, seed=seed + ep, model_path=str(model), practice=False)
        venv = VecNormalize.load(str(stats_for(ckpt)), DummyVecEnv([lambda: env]))
        venv.training = False
        obs, _ = env.reset(seed=seed + ep)
        done, total, air, best_air, clear, peaks, landing_peak, was_air = False, 0.0, 0, 0, 0.0, [], None, False
        while not done:
            a = policy.predict(venv.normalize_obs(obs[None]), deterministic=True)[0][0]
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            if info["airborne"]:
                air += 1
                best_air = max(best_air, air)
                clear = max(clear, info["clear"])
                if air >= 2:
                    was_air = True
            else:
                if was_air:
                    landing_peak = 0                       # measure the next 5 steps after touchdown
                    was_air = False
                air = 0
            if landing_peak is not None:
                peaks.append(info["fw"]) if landing_peak == 0 else None
                peaks[-1] = max(peaks[-1], info["fw"])
                landing_peak += 1
                if landing_peak >= 5:
                    landing_peak = None
        peaks = np.array(peaks) if peaks else np.zeros(0)
        rows.append(dict(hops=env.hops, air_ms=best_air * 20, clear_mm=1000 * clear, rise_cm=100 * env.best_hop,
                         peak_fw=env.peak_fw, mean_land_fw=float(peaks.mean()) if len(peaks) else 0.0,
                         hard=float((peaks > SHOCK_FREE).mean()) if len(peaks) else 0.0,
                         very_hard=float((peaks > 3.5).mean()) if len(peaks) else 0.0,
                         ok=bool(info["success"]), fell=bool(term), reward=total, up=info["up"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--episodes", type=int, default=5)
    a = ap.parse_args()
    for c in a.ckpts:
        rows = evaluate(c, a.episodes)
        m = lambda k: np.mean([r[k] for r in rows])
        print(f"{Path(c).parent.name}/{Path(c).name}: success {sum(r['ok'] for r in rows)}/{len(rows)}, fell {sum(r['fell'] for r in rows)}, "
              f"hops/ep {m('hops'):.1f}, air {m('air_ms'):.0f} ms, clear {m('clear_mm'):.1f} mm, rise {m('rise_cm'):.2f} cm, "
              f"landing F/W mean {m('mean_land_fw'):.2f} peak {max(r['peak_fw'] for r in rows):.2f}, "
              f">2.5W {100 * m('hard'):.0f}% >3.5W {100 * m('very_hard'):.0f}%, reward {m('reward'):.1f}")
        for r in rows:
            print("   ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
