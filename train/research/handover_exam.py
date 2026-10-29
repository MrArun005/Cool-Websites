"""Fair jump exam from REAL hand-over states (research/record_handover.py): the case MasterMind and Bolt's Day Out
hand the jump brain, not the exam's synthetic start. Use a state file recorded on seeds no training file used.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/handover_exam.py ../models/bolt-dog-jump-trot-55M \
        --states research/results/handover_states_test.npz --heights 0.20 0.30 0.40 0.50 [--motor-brake]

Per height: every state of the file once (fixed physics), clean on agility rules (jump_env.exam's fair rule: over,
bar not dropped, landed on its feet, not under the bar), 220-step tries.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from bolt_env import add_brake_args, set_motor_brake  # noqa: E402
from jump_env import JumpEnv, JumpPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--states", required=True)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--latency-ms", type=float, default=0.0, help="fixed control delay (ms) for every try")
    ap.add_argument("--out")
    add_brake_args(ap, default=False)
    a = ap.parse_args()
    brake = set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain)
    pol = JumpPolicy(a.folder)
    env = JumpEnv(randomize=False, real=True, max_steps=a.max_steps, handover_file=a.states, motor_brake=a.motor_brake)
    env.exam_latency_ms = a.latency_ms
    n = len(env.handover["ahead"])
    res = dict(folder=a.folder, states=a.states, n=n, motor_brake=brake, latency_ms=a.latency_ms)
    for h in a.heights:
        wins = 0
        for k in range(n):
            env.exam_bar, env.exam_handover = (0.0, h), k
            obs, _ = env.reset(seed=k)
            info, fell = {}, False
            for _ in range(env.max_steps):
                obs, _, fell, trunc, info = env.step(pol.act(obs))
                if fell or trunc:
                    break
            wins += bool(info["cleared"] and not info.get("dropped", info["touched"]) and not fell and info["up"] > 0.7)
        res[f"jump_{round(h * 100)}"] = round(wins / n, 3)
        print(f"{round(h * 100)} cm: {wins}/{n}", flush=True)
    res["command"] = "python " + " ".join(sys.argv)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
