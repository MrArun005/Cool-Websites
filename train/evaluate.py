"""Test a trained policy like an exam, and optionally film it.

    python evaluate.py runs/bolt                 # success rate per command
    python evaluate.py runs/bolt --video bolt.mp4

Every episode uses randomized physics (mass, friction, motor strength) and
starts standing or lying, so the score measures the skill, not one lucky
setup. Pass marks match pet/ROADMAP.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import COMMANDS, TARGETS, BoltEnv


class Policy:
    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / "policy.zip", device="cpu")
        self.norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), DummyVecEnv([BoltEnv]))
        self.norm.training = False

    def act(self, obs, deterministic=True):
        action, _ = self.model.predict(self.norm.normalize_obs(obs), deterministic=deterministic)
        return action


def passed(command, window):
    """Did the dog really do it over the last stretch of the window?

    Checks the MEASURED pose (see poses.py) and the feet: standing needs all four feet down,
    walking needs every foot to take steps.
    """
    goal = TARGETS[command]
    h = np.mean([p["height"] for p in window])
    pitch = np.mean([p["pitch"] for p in window])
    roll = np.mean([abs(p["roll"]) for p in window])
    vx = np.mean([p["vx"] for p in window])
    feet = np.mean([p["feet"] for p in window], axis=0)
    upright = min(p["up"] for p in window) > 0.2
    if command == "walk":
        return upright and 0.3 <= vx <= 0.75 and abs(pitch) < 0.25 and roll < 0.25 and feet.min() >= 0.2
    speed = np.mean([np.hypot(p["vx"], p["vy"]) for p in window])
    ok = upright and abs(h - goal["height"]) < 0.03 and abs(pitch - goal["pitch"]) < 0.12 and roll < 0.12 and speed < 0.15
    if command == "stand":
        ok = ok and feet.min() >= 0.9
    return bool(ok)


def run_segment(env, policy, obs, command, steps):
    env.set_command(command)
    window = []
    for _ in range(steps):
        obs, _, fell, _, info = env.step(policy.act(obs))
        window.append(info)
        if fell:
            return obs, False, True
    return obs, passed(command, window[-steps // 3:]), False


def evaluate(policy, episodes, seed=0, skills=None):
    env = BoltEnv(randomize=True, max_steps=10_000)
    results = {}
    for command in (skills or COMMANDS):
        wins = 0
        for k in range(episodes):
            env.forced_command = command
            obs, _ = env.reset(seed=seed + k)
            _, ok, _ = run_segment(env, policy, obs, command, 250)
            wins += ok
        results[command] = wins / episodes

    # Obeying a sequence of commands in a row, the way you'd actually use it.
    if skills and set(skills) != set(COMMANDS):
        return results  # a single-skill brain isn't asked to do the whole sequence
    sequence = ["stand", "sit", "stand", "lie", "stand", "walk", "stand"]
    seq_wins, falls = 0, 0
    for k in range(episodes):
        env.forced_command = "stand"
        obs, _ = env.reset(seed=seed + 10_000 + k)
        all_ok = True
        for command in sequence:
            obs, ok, fell = run_segment(env, policy, obs, command, 175)
            all_ok &= ok
            if fell:
                falls += 1
                all_ok = False
                break
        seq_wins += all_ok
    results["sequence"] = seq_wins / episodes
    results["fell"] = falls / episodes
    return results


def film(policy, path):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    env = BoltEnv(randomize=False, max_steps=10_000)
    env.forced_command = "stand"
    obs, _ = env.reset(seed=3)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except OSError:
        font = ImageFont.load_default()
    script = [("stand", 100), ("sit", 150), ("stand", 100), ("lie", 150), ("stand", 100), ("walk", 250), ("stand", 100)]
    with imageio.get_writer(path, fps=50, codec="libx264", quality=7, macro_block_size=8) as video:
        for command, steps in script:
            env.set_command(command)
            for _ in range(steps):
                obs, *_ = env.step(policy.act(obs))
                frame = Image.fromarray(env.render())
                ImageDraw.Draw(frame).text((28, 22), f'command: "{command}"', fill=(30, 30, 30), font=font)
                video.append_data(np.asarray(frame))
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="folder with policy.zip and vecnormalize.pkl")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--video", help="also record an .mp4 of the dog obeying a sequence of commands")
    args = ap.parse_args()

    policy = Policy(args.run)
    results = evaluate(policy, args.episodes)
    print(f"\nSuccess over {args.episodes} randomized episodes each:")
    for k, v in results.items():
        label = {"sequence": "all 7 commands in a row", "fell": "fell over (sequence)"}.get(k, f'"{k}"')
        print(f"  {label:<26} {v:6.0%}")
    (Path(args.run) / "eval.json").write_text(json.dumps(results, indent=2))
    if args.video:
        film(policy, args.video)
        print(f"video -> {args.video}")


if __name__ == "__main__":
    main()
