"""Watch Bolt learn: test and film every checkpoint while training runs.

    python watch.py runs/bolt --log train.log --out runs/bolt/watch

Runs alongside train.py and keeps going until the final policy.zip appears.
Writes into --out:
    data.json          learning curve, plus exam results for every checkpoint
    clips/<steps>.mp4  the dog given a sequence of commands, showing what it has
                       learned (no exploration noise; pass --noise to see it the
                       way it acts while practising)
Step 0 is an untrained dog: random motor twitches.

Early stopping: every checkpoint gets a score (average pass rate over the commands).
The best one is copied to <run>/best/. If no new best appears for --patience checkpoints
in a row, the watcher writes <run>/STOP and train.py stops, so a run that has stopped
improving doesn't waste hours. Package <run>/best/ afterwards, not the last checkpoint.
"""
import shutil
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


def film(policy, path, steps_label, noise=False, script=SCRIPT, gait_generator=False):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        big = small = ImageFont.load_default()
    env = BoltEnv(randomize=False, max_steps=100_000, gait_generator=gait_generator, turning=getattr(policy, "turning", False))
    env.forced_command = "stand"
    obs, _ = env.reset(seed=7)
    falls, fallen_for = 0, 0
    with imageio.get_writer(path, fps=25, codec="libx264", quality=6, macro_block_size=8) as video:
        for command, steps, *turn in script:
            env.set_command(command)
            if env.turning:
                env.set_turn(turn[0] if turn else 0.0)
            for k in range(steps):
                obs, _, fell, _, _ = env.step(policy.act(obs, deterministic=not noise))
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
    ap.add_argument("--label", default="", help="name of this run, shown on the progress page")
    ap.add_argument("--noise", action="store_true", help="film with exploration noise, as while practising")
    ap.add_argument("--skills", nargs="+", help="the commands this brain was trained for (exam and score only these)")
    ap.add_argument("--gait-generator", nargs="?", const="crouch", default=None, choices=["crouch", "tall", "dog"], help="the walk runs on the built-in trot, around this stance (no value = crouch, as in runs 5-6)")
    ap.add_argument("--patience", type=int, default=4,
                    help="stop training after this many checkpoints in a row without a new best (0 = never)")
    ap.add_argument("--handover-brain", help="posture brain folder: walk hand-overs in the exam start from it")
    args = ap.parse_args()
    handover = Policy(args.handover_brain) if args.handover_brain else None

    run, out, log = Path(args.run), Path(args.out), Path(args.log)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    data_path = out / "data.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else {"checkpoints": []}
    done = {c["steps"] for c in data["checkpoints"]}

    def save(status):
        curve = read_curve(log)
        data.update(status=status, target_steps=args.target, curve=curve, label=args.label,
                    steps=curve[-1][0] if curve else 0,
                    updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        write_json(data_path, data)

    def score(exam):
        keys = list(args.skills or COMMANDS) + [k for k in ("turn_left", "turn_right") if k in exam]
        return float(np.mean([exam[c] for c in keys]))

    def keep_best(steps, exam, zip_path=None, stats=None):
        """Remember the best checkpoint; ask training to stop when it stops improving."""
        s_now = score(exam)
        best = data.get("best")
        if best is None or s_now > best["score"]:
            data["best"] = dict(steps=steps, score=round(s_now, 3))
            data["since_best"] = 0
            if zip_path is not None:
                (run / "best").mkdir(exist_ok=True)
                shutil.copy(zip_path, run / "best" / "policy.zip")
                shutil.copy(stats, run / "best" / "vecnormalize.pkl")
                (run / "best" / "best.json").write_text(json.dumps(dict(steps=steps, exam=exam, score=s_now), indent=2))
            print(f"new best at {steps}: score {s_now:.2f}", flush=True)
        elif steps > 0:
            data["since_best"] = data.get("since_best", 0) + 1
            if args.patience and data["since_best"] >= args.patience and not (run / "STOP").exists():
                reason = (f"no new best for {data['since_best']} checkpoints "
                          f"(best {data['best']['score']:.2f} at {data['best']['steps']:,} steps)")
                (run / "STOP").write_text(reason + "\n")
                data["stopped"] = reason
                print("early stop: " + reason, flush=True)

    def process(steps, policy, zip_path=None, stats=None):
        label = "untrained" if steps == 0 else f"after {steps:,} practice steps"
        clip = f"clips/{steps:010d}.mp4"
        script = [(c, 250) for c in args.skills] if args.skills else SCRIPT
        if getattr(policy, "turning", False):
            script = [("walk", 150, 0.0), ("walk", 150, 0.4), ("walk", 150, -0.4)]  # straight, left, right
        falls = film(policy, out / clip, label, noise=args.noise, script=script, gait_generator=args.gait_generator)
        exam = evaluate(policy, args.episodes, seed=1000, skills=args.skills, gait_generator=args.gait_generator,
                        handover=handover)
        data["checkpoints"].append(dict(steps=steps, exam={k: round(v, 2) for k, v in exam.items()},
                                        clip=clip, falls_in_clip=falls))
        data["checkpoints"].sort(key=lambda c: c["steps"])
        done.add(steps)
        keep_best(steps, exam, zip_path, stats)
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
                process(steps, Policy(tmp), zip_path, stats)
        final = run / "policy.zip"
        if final.exists() and (run / "vecnormalize.pkl").exists():
            steps = max(args.target, max(done))
            if steps not in done and not (run / "STOP").exists():  # an early-stopped run keeps its best instead
                process(steps, Policy(run))
            save("finished")
            print("finished", flush=True)
            return
        save("training")
        time.sleep(20)


if __name__ == "__main__":
    main()
