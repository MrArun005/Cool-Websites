"""Watch Bolt learn: test and film every checkpoint while training runs.

    python watch.py runs/bolt --log train.log --out runs/bolt/watch

Runs alongside train.py and keeps going until the final policy.zip appears.
Writes into --out:
    data.json          learning curve, plus exam results for every checkpoint
    clips/<steps>.mp4  the dog given a sequence of commands, acting the way
                       it does while practising (with its exploration noise),
                       so you see it try, wobble and fall
Step 0 is an untrained dog: random motor twitches.
"""
import argparse
import datetime as dt
import json
import re
import time
from pathlib import Path

import numpy as np

from bolt_env import COMMANDS, BoltEnv
from evaluate import Policy, evaluate

SCRIPT = [("stand", 100), ("sit", 150), ("stand", 75), ("lie", 150), ("stand", 75), ("walk", 200)]


class Untrained:
    """What a fresh network does: near-zero outputs plus exploration noise."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=True):
        return np.clip(self.rng.normal(0, 0.37, 12), -1, 1)


def read_curve(log_path):
    """[(steps, reward per episode, episode length, seconds elapsed)] from train.py's output."""
    points, row = [], {}
    if not log_path.exists():
        return points
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.match(r"\|\s+(ep_rew_mean|ep_len_mean|time_elapsed|total_timesteps)\s+\|\s+([-\d.e+]+)", line)
        if not m:
            continue
        row[m.group(1)] = float(m.group(2))
        if m.group(1) == "total_timesteps" and "ep_rew_mean" in row:
            points.append([int(row["total_timesteps"]), round(row["ep_rew_mean"], 1),
                           round(row.get("ep_len_mean", 0)), int(row.get("time_elapsed", 0))])
    return points


def film(policy, path, steps_label):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        big = small = ImageFont.load_default()
    env = BoltEnv(randomize=False, max_steps=100_000)
    env.forced_command = "stand"
    obs, _ = env.reset(seed=7)
    falls, fallen_for = 0, 0
    with imageio.get_writer(path, fps=25, codec="libx264", quality=6, macro_block_size=8) as video:
        for command, steps in SCRIPT:
            env.set_command(command)
            for k in range(steps):
                obs, _, fell, _, _ = env.step(policy.act(obs, deterministic=False))
                if fell and fallen_for == 0:
                    falls += 1
                    fallen_for = 1
                elif fallen_for:
                    fallen_for += 1
                if k % 2 == 0:  # 25 frames per second
                    frame = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(frame)
                    draw.text((20, 14), f'"{command}"', fill=(25, 25, 25), font=big)
                    draw.text((20, 48), steps_label, fill=(70, 70, 70), font=small)
                    if fallen_for:
                        draw.text((20, 74), "fell over, restarting", fill=(200, 40, 40), font=small)
                    video.append_data(np.asarray(frame))
                if fallen_for > 50:  # show the fall for a second, then stand it back up
                    obs, _ = env.reset(seed=7 + falls)
                    env.set_command(command)
                    fallen_for = 0
    env.close()
    return falls


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--log", required=True, help="file that train.py's output goes to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=6_000_000)
    ap.add_argument("--episodes", type=int, default=10, help="exam episodes per command")
    args = ap.parse_args()

    run, out, log = Path(args.run), Path(args.out), Path(args.log)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    data_path = out / "data.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else {"checkpoints": []}
    done = {c["steps"] for c in data["checkpoints"]}

    def save(status):
        curve = read_curve(log)
        data.update(status=status, target_steps=args.target, curve=curve,
                    steps=curve[-1][0] if curve else 0,
                    updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        write_json(data_path, data)

    def process(steps, policy):
        label = "untrained" if steps == 0 else f"after {steps:,} practice steps"
        clip = f"clips/{steps:010d}.mp4"
        falls = film(policy, out / clip, label)
        exam = evaluate(policy, args.episodes, seed=1000)
        data["checkpoints"].append(dict(steps=steps, exam={k: round(v, 2) for k, v in exam.items()},
                                        clip=clip, falls_in_clip=falls))
        data["checkpoints"].sort(key=lambda c: c["steps"])
        done.add(steps)
        save("training")
        print(f"checkpoint {steps} ready: {exam}", flush=True)

    if 0 not in done:
        process(0, Untrained())
    while True:
        for zip_path in sorted(run.glob("checkpoints/policy_*_steps.zip"), key=lambda p: int(re.findall(r"\d+", p.name)[0])):
            steps = int(re.findall(r"\d+", zip_path.name)[0])
            stats = zip_path.with_name(f"policy_vecnormalize_{steps}_steps.pkl")
            if steps not in done and stats.exists():
                tmp = out / "tmp_policy"
                tmp.mkdir(exist_ok=True)
                (tmp / "policy.zip").write_bytes(zip_path.read_bytes())
                (tmp / "vecnormalize.pkl").write_bytes(stats.read_bytes())
                process(steps, Policy(tmp))
        final = run / "policy.zip"
        if final.exists() and (run / "vecnormalize.pkl").exists():
            steps = max(args.target, max(done))
            if steps not in done:
                process(steps, Policy(run))
            save("finished")
            print("finished", flush=True)
            return
        save("training")
        time.sleep(20)


if __name__ == "__main__":
    main()
