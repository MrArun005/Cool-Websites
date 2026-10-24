"""Sit to stand as a person does it: plant the soles, pull the feet in, lean the nose over the toes, push.

    python s2s_plan.py [--model model/probe_f2.xml] [--seat sit] [--candidates 80] [--rounds 2]

s2s_script.py searched free keyframes and every winner just rolled onto its back (the back lies higher than
the squat, so "height" is not progress here, and the feet never touched the floor). This searches a script
with physical meaning instead, 8 numbers:
  phase A (tA s): both hips -> hpA, both knees -> knA, ankles keep the soles flat on the floor (+ offset),
                  head pitch -> headA (the head is ~19% of the mass: a counterweight)
  phase B (tB s): hips -> hpB, knees -> home, soles flat, head -> headB
  then hold the home pose.
Sole-flat ankle = home ankle - (hip - home hip) - (knee - home knee) + trunk tilt (measured in walk_gait's
frame on the model itself, not assumed). Score = did SwingUpEnv.success fire; tie-break on best up × feet down.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from s2s_swing import SwingUpEnv

HERE = Path(__file__).resolve().parent


def make_env(a, randomize, seed):
    env = SwingUpEnv(task="sit_to_stand", randomize=randomize, seed=seed, model_path=a.model, practice=False, max_seconds=a.seconds)
    if env.sit_q is None:
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


class Plan:
    def __init__(self, env):
        m = env.model
        names = [m.actuator(i).name for i in range(m.nu)]
        self.env = env
        self.leg = {k: [i for i, n in enumerate(names) if n.endswith(k)] for k in ("hip_pitch", "knee", "ankle")}
        self.head = [i for i, n in enumerate(names) if n == "head_pitch"]
        h = env.home
        self.h_hp, self.h_kn, self.h_an = h[self.leg["hip_pitch"][0]], h[self.leg["knee"][0]], h[self.leg["ankle"][0]]
        # Hip pitch is mirrored between the legs (home: left -36°, right +36°; the crouch folds both forward as
        # left -40°, right +40°). A hip parameter is the LEFT value; the right gets the mirrored offset.
        self.mirror = {i: (1.0 if j == 0 else float(np.sign(h[i] * h[self.leg["hip_pitch"][0]]) or 1.0))
                       for j, i in enumerate(self.leg["hip_pitch"])}
        self.h_hip = {i: h[i] for i in self.leg["hip_pitch"]}
        self.lo, self.hi = env.lo, env.hi
        self.names = ("hpA", "knA", "ankA", "headA", "tA", "hpB", "tB", "headB")

    def bounds(self):
        hp, kn, an = self.leg["hip_pitch"][0], self.leg["knee"][0], self.leg["ankle"][0]
        hd = self.head[0] if self.head else None
        hb = (self.lo[hd], self.hi[hd]) if hd is not None else (0.0, 0.0)
        return [(self.lo[hp], self.hi[hp]), (self.h_kn, self.hi[kn]), (-0.6, 0.6), hb, (0.1, 1.0),
                (self.lo[hp], self.hi[hp]), (0.1, 1.0), hb]

    def sample(self, rng, centre=None, spread=1.0):
        out = []
        for j, (lo, hi) in enumerate(self.bounds()):
            if centre is None:
                out.append(rng.uniform(lo, hi))
            else:
                out.append(float(np.clip(centre[j] + rng.normal(0, (hi - lo) / 3) * spread, lo, hi)))
        return np.array(out)

    def trunk_tilt(self):
        R = self.env.data.xmat[self.env.trunk].reshape(3, 3)
        return float(np.arctan2(-R[2, 0], R[2, 2]))         # pitch about y: + = nose up (leaning back)

    def targets(self, p, t):
        hpA, knA, ankA, headA, tA, hpB, tB, headB = p
        e = self.env
        start = np.array(e.seat_ctrl if e.seat_ctrl is not None else e.seat_q[e.qadr])
        tilt = self.trunk_tilt()
        keyA = start.copy()
        for i in self.leg["hip_pitch"]:
            keyA[i] = self.h_hip[i] + self.mirror[i] * (hpA - self.h_hp)
        for i in self.leg["knee"]:
            keyA[i] = knA
        for i in self.head:
            keyA[i] = headA
        keyB = keyA.copy()
        for i in self.leg["hip_pitch"]:
            keyB[i] = self.h_hip[i] + self.mirror[i] * (hpB - self.h_hp)
        for i in self.leg["knee"]:
            keyB[i] = self.h_kn
        for i in self.head:
            keyB[i] = headB
        keyC = e.home.copy()
        if t < tA:
            tgt = start + (keyA - start) * (t / tA)
        elif t < tA + tB:
            tgt = keyA + (keyB - keyA) * ((t - tA) / tB)
        else:
            tgt = keyC
        # ankles: soles flat on the floor for whatever the hip and knee are doing right now, closed loop on the
        # measured trunk tilt (the tilt sign was checked on the model: sole pitch = hip + knee + ankle, trunk-relative)
        for hp, kn, an in zip(self.leg["hip_pitch"], self.leg["knee"], self.leg["ankle"]):
            off = ankA if t < tA + tB else 0.0
            tgt[an] = self.h_an - self.mirror[hp] * (tgt[hp] - self.h_hip[hp]) - (tgt[kn] - self.h_kn) + tilt + off
        return np.clip(tgt, self.lo, self.hi)

    def action(self, p, t):
        e = self.env
        return np.clip((self.targets(p, t)[e.pol] - e.home[e.pol]) / e.action_scale, -1, 1)


def rollout(env, plan, p, seed=0):
    env.reset(seed=seed)
    best, rose, done = 0.0, False, False
    while not done:
        _, _, term, trunc, info = env.step(plan.action(p, env.t * env.dt))
        feet = env.feet_on_ground().mean()
        best = max(best, max(info["up"], 0.0) * (0.5 + 0.5 * feet) * (0.0 if env._trunk_on_floor() else 1.0))
        rose = rose or bool(info["success"])
        done = term or trunc
    return float(best), rose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model" / "duck_mid.xml"))
    ap.add_argument("--seat", default="swag", choices=("swag", "sit"))
    ap.add_argument("--poses", default=str(HERE / "model" / "poses_mid.json"))
    ap.add_argument("--candidates", type=int, default=80)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--check-seeds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    env = make_env(a, False, 0)
    plan = Plan(env)
    best = (-1.0, False, None)
    for rd in range(a.rounds):
        spread = 1.0 if rd == 0 else 0.5 ** rd
        for _ in range(a.candidates):
            p = plan.sample(rng, best[2] if rd else None, spread)
            s, rose = rollout(env, plan, p)
            if (rose, s) > (best[1], best[0]):
                best = (s, rose, p)
        print(f"round {rd + 1}: best score {best[0]:.3f} rose={best[1]}  {dict(zip(plan.names, np.round(best[2], 3).tolist()))}")
    renv = make_env(a, True, 1)
    rplan = Plan(renv)
    res = [rollout(renv, rplan, best[2], seed=100 + s) for s in range(a.check_seeds)]
    print(f"randomized bodies: rose {sum(r for _, r in res)}/{len(res)}, mean score {np.mean([s for s, _ in res]):.3f}")
    out = HERE / "runs" / f"s2s_plan_{Path(a.model).stem}_{a.seat}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(model=a.model, seat=a.seat, score=best[0], rose=best[1], params=dict(zip(plan.names, best[2].tolist())),
                                   randomized_rises=int(sum(r for _, r in res))), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
