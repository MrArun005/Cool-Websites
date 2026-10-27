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

from getup_env import GetUpEnv, GetUpPolicy

KINDS = ("side", "back", "tipped", "standing")


def exams(log):
    out = {}
    if log.exists():
        for m in re.finditer(r"checkpoint (\d+) ready: (\{.*\})", log.read_text()):
            out[int(m.group(1))] = ast.literal_eval(m.group(2))
    return out


def attempts(ck, n, seed=4321, goal="stand"):
    p = GetUpPolicy(ck, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl")
    env = GetUpEnv(randomize=False, goal=goal)
    rec, prog = {}, {}
    for kind in KINDS:
        env.exam_start = kind
        obs, _ = env.reset(seed=seed)
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
            R["goal"] = "selfright" if name.startswith("selfright") else "stand"
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
