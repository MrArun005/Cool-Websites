"""Can this body stand up from the floor seat at all? Search open-loop keyframe scripts before spending RL on it.

    python s2s_script.py                                  # mid duck, swag seat, 400 candidates
    python s2s_script.py --model model/probe_f2.xml --seat sit --candidates 800 --keys 5

A script = K keyframes of servo targets (both legs the same: hip pitch, knee, ankle; plus head pitch as a
counterweight: the head is ~19% of the mass) with a duration each, played from the measured seat through
SwingUpEnv (same physics, same success test as training). Round 1 samples the keyframes over the full servo
range; later rounds sample around the best. The score is the best moment of the rollout:
    up (clipped at 0) × height progress seat→squat   (1.0 = fully up at squat height)
and `rise` = did SwingUpEnv.success fire at any step.

If the best script on the nominal body never rises, no reward shaping will make PPO find it: change the body
(torque: probe_f1.5 / probe_f2, seat pose, knee range) first. If it rises on the nominal body, the winning
script is the teacher (clone it like walk_bc.py) and PPO only has to make it robust.
Writes runs/s2s_script_<model>_<seat>.json with the best script and its score on randomized bodies.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from s2s_swing import SwingUpEnv

HERE = Path(__file__).resolve().parent
LEG = ("hip_pitch", "knee", "ankle")


class Script:
    def __init__(self, env):
        names = [env.model.actuator(i).name for i in range(env.model.nu)]
        self.pol = list(env.pol)
        self.idx = {k: [i for i, n in enumerate(names) if n.endswith(k)] for k in LEG + ("head_pitch",)}
        self.env = env

    def nparams(self, keys):
        return keys * (len(LEG) + 1 + 1)       # per key: 3 leg targets, head pitch, duration

    def sample(self, rng, keys, centre=None, spread=1.0):
        e = self.env
        p = []
        for k in range(keys):
            for j, name in enumerate(LEG + ("head_pitch",)):
                i = self.idx[name][0]
                lo, hi = e.lo[i], e.hi[i]
                c = centre[k * 5 + j] if centre is not None else rng.uniform(lo, hi)
                p.append(float(np.clip(c + rng.normal(0, (hi - lo) / 3) * spread, lo, hi)))
            c = centre[k * 5 + 4] if centre is not None else rng.uniform(0.1, 0.8)
            p.append(float(np.clip(c + rng.normal(0, 0.25) * spread, 0.06, 1.2)))
        return np.array(p)

    def targets(self, p, t):
        """Servo targets at time t: linear ramps between keyframes, holding the last one."""
        e = self.env
        keys = len(p) // 5
        start = e.seat_ctrl if e.seat_ctrl is not None else e.seat_q[e.qadr]
        prev, t0 = np.array(start), 0.0
        for k in range(keys):
            key = prev.copy()
            for j, name in enumerate(LEG + ("head_pitch",)):
                for i in self.idx[name]:
                    key[i] = p[k * 5 + j]
            dur = p[k * 5 + 4]
            if t < t0 + dur:
                f = (t - t0) / dur
                return prev + (key - prev) * f
            prev, t0 = key, t0 + dur
        return prev

    def action(self, p, t):
        e = self.env
        tgt = self.targets(p, t)
        return np.clip((tgt[e.pol] - e.home[e.pol]) / e.action_scale, -1, 1)


def rollout(env, script, p, seed=0):
    env.reset(seed=seed)
    seat_h = float(env.seat_q[2])
    best, rose, done = 0.0, False, False
    while not done:
        _, _, term, trunc, info = env.step(script.action(p, env.t * env.dt))
        prog = np.clip((info["height"] - seat_h) / (env.crouch_height - seat_h), 0, 1)
        best = max(best, max(info["up"], 0.0) * prog)
        rose = rose or bool(info["success"])
        done = term or trunc
    return float(best), rose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model" / "duck_mid.xml"))
    ap.add_argument("--seat", default="swag", choices=("swag", "sit"))
    ap.add_argument("--keys", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--poses", default=str(HERE / "model" / "poses_mid.json"), help="seat poses for models without their own")
    ap.add_argument("--check-seeds", type=int, default=10, help="randomized bodies to confirm the winner on")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    def make(randomize, seed):
        env = SwingUpEnv(task="sit_to_stand", randomize=randomize, seed=seed, model_path=a.model, practice=False, max_seconds=a.seconds)
        if env.sit_q is None:
            # probe models (same body, other servos) have no poses file of their own: borrow the one named
            poses = json.loads(Path(a.poses).read_text())
            from poses import leg_q
            env.sit_q, env.seat_q = np.array(poses["sit"]["qpos"]), np.array(poses.get("swag", poses["sit"])["qpos"])
            env.seat_ctrl = np.array(poses["swag"]["ctrl"]) if "swag" in poses else None
            env.sit_cmd = env.sit_q.copy()
            c = poses["sit"]["ctrl_legs"]
            leg_q(env.model, env.sit_cmd, c["hip"], c["knee"], c["ankle"])
        if a.seat == "sit":
            env.seat_q, env.seat_ctrl = env.sit_q, env.sit_cmd[env.qadr]
        return env
    env = make(False, 0)
    sc = Script(env)
    best = (-1.0, False, None)
    for rd in range(a.rounds):
        spread = 1.0 if rd == 0 else 0.5 ** rd
        for _ in range(a.candidates):
            p = sc.sample(rng, a.keys, best[2] if rd else None, spread)
            s, rose = rollout(env, sc, p)
            if (rose, s) > (best[1], best[0]):
                best = (s, rose, p)
        print(f"round {rd + 1}: best score {best[0]:.3f} rose={best[1]}")
    renv = make(True, 1)
    rsc = Script(renv)
    res = [rollout(renv, rsc, best[2], seed=100 + s) for s in range(a.check_seeds)]
    print(f"randomized bodies: rose {sum(r for _, r in res)}/{len(res)}, mean score {np.mean([s for s, _ in res]):.3f}")
    out = HERE / "runs" / f"s2s_script_{Path(a.model).stem}_{a.seat}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(model=a.model, seat=a.seat, keys=a.keys, score=best[0], rose=best[1],
                                   script=best[2].tolist(), randomized_rises=int(sum(r for _, r in res)),
                                   layout="per key: hip_pitch, knee, ankle, head_pitch, seconds"), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
