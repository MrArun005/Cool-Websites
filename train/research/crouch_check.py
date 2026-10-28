"""Does the jump brain crouch (bend its legs to load them) before it takes off, and do its clean jumps crouch deeper
than its failures? Owner's question, 28 Oct: "bend the knees, create force, then jump".

Per try (fair agility rules, jump_env.exam's): body height while running, the lowest body height in the
--window s before take-off (take-off = the step of fastest upward body speed before the highest point), the dip
(running height minus that lowest height), upward speed at take-off, and the highest body point.

    cd train
    MUJOCO_GL=osmesa python research/crouch_check.py ../models/bolt-dog-jump-trot-55M --speed 2.5 --dist 1.5 \
        --height 0.60 --gait --episodes 30 --seed 9000 --motor-brake --out research/results/crouch_trot55_v25_60.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bolt_env import add_brake_args  # noqa: E402
from jump_env import JumpEnv, JumpPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--speed", type=float, default=2.5)
    ap.add_argument("--dist", type=float, default=1.5)
    ap.add_argument("--height", type=float, default=0.60)
    ap.add_argument("--gait", action="store_true", help="start mid-stride")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--window", type=float, default=0.4, help="seconds before take-off searched for the crouch")
    ap.add_argument("--out")
    add_brake_args(ap, default=False)
    a = ap.parse_args()
    pol = JumpPolicy(a.folder)
    env = JumpEnv(randomize=False, real=pol.real, max_steps=a.max_steps, motor_brake=a.motor_brake)
    env.exam_speed, env.exam_gait = a.speed, a.gait
    rows = []
    for k in range(a.episodes):
        env.exam_bar = (a.dist, a.height)
        obs, _ = env.reset(seed=a.seed + k)
        z, vz, info, fell = [], [], {}, False
        for _ in range(env.max_steps):
            obs, _, fell, trunc, info = env.step(pol.act(obs))
            z.append(float(env.data.qpos[2])); vz.append(float(env.data.qvel[2]))
            if fell or trunc:
                break
        z, vz = np.array(z), np.array(vz)
        apex = int(np.argmax(z))
        take = int(np.argmax(vz[:apex + 1]))
        w = max(1, round(a.window / env.dt))
        run_h = float(np.median(z[:max(1, take - w)]))
        low = float(z[max(0, take - w):take + 1].min())
        clean = bool(info["cleared"] and not info.get("dropped", info["touched"]) and not fell and info["up"] > 0.7)
        rows.append(dict(seed=a.seed + k, clean=clean, run_height=run_h, crouch_low=low, dip=run_h - low,
                         takeoff_vz=float(vz[take]), apex=float(z[apex]), takeoff_s=take * env.dt))
    summ = {}
    for name, sel in (("clean", [r for r in rows if r["clean"]]), ("failed", [r for r in rows if not r["clean"]])):
        if sel:
            summ[name] = dict(n=len(sel), **{f"mean_{q}": round(float(np.mean([r[q] for r in sel])), 3)
                                             for q in ("run_height", "dip", "takeoff_vz", "apex")})
    res = dict(settings=vars(a), summary=summ, tries=rows, command="python " + " ".join(sys.argv))
    print(json.dumps(summ, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
