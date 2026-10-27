"""Watch Bolt learn, for the Bolt Learning Live page: record every checkpoint of the get-up runs.

    python learnwatch.py --out <site>/data/learning.json --run getup17b:"Run 17b ..." [--run getup17:"..."]

For each new checkpoint: the exam result (from the run's train.log) and one real attempt per kind of fall
(side, back, half-tipped, standing), recorded as Bolt's body pose (qpos 0-18) at 25 Hz, plus how upright
it got (1 standing, -1 on its back). Keeps polling until every run has its final policy.zip; the page
draws the attempts in the browser physics (same robot model).
"""
import argparse
import ast
import datetime as dt
import json
import re
import time
from pathlib import Path

import numpy as np

from bolt_env import LIE_POSE, BoltEnv
from evaluate import Policy
from getup_env import GetUpEnv, GetUpPolicy

POSTURE = None   # the posture brain, loaded once, for the full recovery


def recovery(env, sr, obs):
    """The full recovery the games use: self-right until belly-down and still for 0.3 s (at most 4 s), tuck
    the legs with the posture brain's "lie" (until every joint is within 0.6 rad of the lying pose, at most
    2 s), then the posture brain's "stand" for 3 s. Returns per control step: the body pose and the phase."""
    global POSTURE
    POSTURE = POSTURE or Policy("../models/bolt-dog-posture-getup-3M")
    frames, phases, ups, calm = [], [], [], 0
    def keep(phase):
        frames.append(env.data.qpos[:19].copy()); phases.append(phase); ups.append(env._rotation()[2, 2])
    keep("fall")
    for _ in range(200):
        obs, _, _, _, info = env.step(sr.act(obs)); keep("self-righting")
        calm = calm + 1 if info["standing"] else 0
        if calm >= 15:
            break
    env.forced_command = None
    for cmd, steps in (("lie", 100), ("stand", 150)):
        env.set_command(cmd); env.last_action = np.zeros(12)
        for _ in range(steps):
            BoltEnv.step(env, POSTURE.act(BoltEnv._obs(env)))
            keep("tucking" if cmd == "lie" else "standing up")
            if cmd == "lie" and np.abs(env.data.qpos[7:19] - LIE_POSE).max() < 0.6:
                break
    p = env.posture()
    return frames, phases, ups, bool(p["up"] > 0.9 and p["height"] > 0.38)

KINDS = ("side", "back", "tipped", "standing")


def exams(log):
    out = {}
    if log.exists():
        for m in re.finditer(r"checkpoint (\d+) ready: (\{.*\})", log.read_text()):
            out[int(m.group(1))] = ast.literal_eval(m.group(2))
    return out


def attempts(ck, n, seed=4321, goal="stand"):
    p = GetUpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl")
    env = GetUpEnv(randomize=False, goal=goal, max_steps=10**6 if goal == "selfright" else 200)
    rec, prog = {}, {}
    for kind in KINDS:
        env.exam_start = kind
        obs, _ = env.reset(seed=seed)
        if goal == "selfright":   # show the whole recovery (self-right, tuck, stand) from 5 falls of this kind:
            tries = []            # the replay is the first that stood up (labelled "1 of the k that stood")
            for s in range(5):
                env.exam_start = kind
                obs, _ = env.reset(seed=seed + s)
                tries.append(recovery(env, p, obs))
            wins = sum(tr[3] for tr in tries)
            frames, phases, ups, stood = next((tr for tr in tries if tr[3]), tries[0])
            keep = slice(None, None, 2)   # 25 Hz
            rec[kind] = np.round(np.concatenate(frames[keep]), 4).tolist()
            rec[kind + "_phase"] = phases[keep]
            prog[kind] = dict(start=round(float(ups[0]), 2), best=round(float(max(ups)), 2), end=round(float(ups[-1]), 2),
                              stood=stood, wins=wins, tries=len(tries))
            continue
        frames, ups = [env.data.qpos[:19].copy()], [env._rotation()[2, 2]]
        for t in range(env.max_steps):
            obs, _, _, done, info = env.step(p.act(obs))
            ups.append(info["up"])
            if t % 2 == 1:
                frames.append(env.data.qpos[:19].copy())
            if done:
                break
        rec[kind] = np.round(np.concatenate(frames), 4).tolist()
        prog[kind] = dict(start=round(float(ups[0]), 2), best=round(float(max(ups)), 2), end=round(float(ups[-1]), 2),
                          stood=bool(info["standing"]))
    return rec, prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="name:label, e.g. getup17b:Run 17b")
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else dict(runs={})
    runs = [r.split(":", 1) for r in args.run]
    while True:
        changed = False
        for name, label in runs:
            d = Path(__file__).with_name("runs") / name
            R = data["runs"].setdefault(name, dict(label=label, checkpoints={}))
            R["label"] = label
            R["goal"] = "recovery" if name.startswith("selfright") else "stand"
            ex = exams(d / "train.log")
            for zp in sorted((d / "checkpoints").glob("policy_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1])):
                n = int(zp.stem.split("_")[1])
                if str(n) in R["checkpoints"] or n not in ex:
                    continue
                rec, prog = attempts(d / "checkpoints", n, goal="selfright" if name.startswith("selfright") else "stand")
                R["checkpoints"][str(n)] = dict(steps=n, exam=ex[n], progress=prog, replays=rec)
                changed = True
                print(f"{name} {n}: {prog}", flush=True)
            R["finished"] = (d / "policy.zip").exists()
        if changed:
            data["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")))
            tmp.replace(out)
        if all(data["runs"][n].get("finished") for n, _ in runs):
            print("finished", flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
