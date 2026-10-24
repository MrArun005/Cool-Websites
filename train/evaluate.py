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
        # brains that can turn have a 53rd input (the turn command); build a matching env
        n_inputs = self.model.observation_space.shape[0]
        self.turning, self.speed_control = n_inputs >= 53, n_inputs >= 54
        # sharp-turn brains (sideways-stepping turns) have the same inputs, so their folder says so
        gait = run_dir / "gait.json"
        self.sharp_turns = bool(json.loads(gait.read_text()).get("sharp_turns")) if gait.exists() else False
        self.norm = VecNormalize.load(str(run_dir / "vecnormalize.pkl"),
                                      DummyVecEnv([lambda: BoltEnv(turning=self.turning, speed_control=self.speed_control)]))
        self.norm.training = False

    def act(self, obs, deterministic=True):
        # The turn command is appended as the last input, so a brain that can't turn just ignores it.
        obs = obs[..., : self.model.observation_space.shape[0]]
        action, _ = self.model.predict(self.norm.normalize_obs(obs), deterministic=deterministic)
        return action


def passed(command, window):
    """Did the dog really do it over the last stretch of the window?

    Checks the MEASURED pose (see poses.py) and the feet: standing needs all four feet down,
    walking needs every foot to take steps, at the stance's own height, with no knee, shin or
    thigh on the floor (runs 5-6 passed while dragging a knee and crawling in a crouch).
    """
    goal = TARGETS[command]
    h = np.mean([p["height"] for p in window])
    pitch = np.mean([p["pitch"] for p in window])
    roll = np.mean([abs(p["roll"]) for p in window])
    vx = np.mean([p["vx"] for p in window])
    feet = np.mean([p["feet"] for p in window], axis=0)
    upright = min(p["up"] for p in window) > 0.2
    if command == "walk":
        speed_ok = 0.3 <= vx <= 0.75
        if window[0].get("speed_cmd") is not None:  # brains with a speed command: go the asked speed
            v = window[0]["speed_cmd"]
            speed_ok = abs(vx - v) < 0.1 + 0.15 * v
        # brains that can turn must turn as asked, within ~11°/s (0 = straight); older brains aren't asked
        turned_right = True
        if window[0].get("turning"):
            yaw = np.mean([p["yaw_rate"] for p in window])
            turned_right = abs(yaw - window[0]["yaw_cmd"]) < max(0.2, 0.25 * abs(window[0]["yaw_cmd"]))
        scraping = np.mean([p.get("scrape", 0) > 0 for p in window])
        tall_enough = abs(h - window[0].get("walk_height", goal["height"])) < 0.05
        # lower legs near upright, like a real dog's (a "dog" walk with no brain: ~20°)
        upright_legs = np.median([p.get("shin_tilt", 0) for p in window]) < 30
        return bool(upright and speed_ok and abs(pitch) < 0.25 and roll < 0.25 and feet.min() >= 0.2
                    and scraping < 0.02 and tall_enough and upright_legs and turned_right)
    speed = np.mean([np.hypot(p["vx"], p["vy"]) for p in window])
    goal_h = window[0].get("goal_height", goal["height"])  # the env's own target (the dog stance stands taller)
    ok = upright and abs(h - goal_h) < 0.03 and abs(pitch - goal["pitch"]) < 0.12 and roll < 0.12 and speed < 0.15
    if command in ("stand", "sit"):  # a real dog sits on all four paws too
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


def reset_standing(env, seed):
    """Reset until the dog starts on its feet: a hand-over is from standing, not from lying."""
    obs, _ = env.reset(seed=seed)
    while env.data.qpos[2] < 0.2 * env.s:
        seed += 10_000
        obs, _ = env.reset(seed=seed)
    return obs


def evaluate(policy, episodes, seed=0, skills=None, gait_generator=False, turning=None, handover=None):
    turning = getattr(policy, "turning", False) if turning is None else turning
    speed = getattr(policy, "speed_control", False)
    sharp = getattr(policy, "sharp_turns", False)
    env = BoltEnv(randomize=True, max_steps=10_000, gait_generator=gait_generator, turning=turning, speed_control=speed,
                  sharp_turns=sharp)
    if speed:
        env.set_speed(0.4)  # the walk and turn parts of the exam walk at 0.4 m/s
    results = {}
    for command in (skills or COMMANDS):
        wins = 0
        for k in range(episodes):
            env.forced_command = command
            if command == "walk" and k % 2:
                # Hand-over: stand first (with the real posture brain when given one, otherwise
                # the motors hold the standing pose), then the walk brain takes over mid-stride.
                env.forced_command = "stand"
                obs = reset_standing(env, seed + k)
                for _ in range(50 + 7 * k % 50):
                    obs, *_ = env.step(handover.act(obs) if handover else np.zeros(12))
                env.forced_command = "walk"
            else:
                obs, _ = env.reset(seed=seed + k)
            if env.turning:
                env.set_turn(0.0)  # this part of the exam walks straight; turning is tested below
                obs = env._obs()
            _, ok, _ = run_segment(env, policy, obs, command, 250)
            wins += ok
        results[command] = wins / episodes

    if env.turning and "walk" in (skills or COMMANDS):
        # Turning: walk while asked to turn left or right at 0.4 rad/s (~23°/s); sharp-turn brains
        # also at 1.5 rad/s (~86°/s).
        turns = (("turn_left", 0.4), ("turn_right", -0.4)) + ((("sharp_left", 1.5), ("sharp_right", -1.5)) if sharp else ())
        for name, rate in turns:
            wins = 0
            for k in range(episodes):
                env.forced_command = "walk"
                obs, _ = env.reset(seed=seed + 500 + k)
                env.set_turn(rate)
                obs = env._obs()
                _, ok, _ = run_segment(env, policy, obs, "walk", 250)
                wins += ok
            results[name] = wins / episodes
        env.forced_turn_set = False

    if speed and "walk" in (skills or COMMANDS):
        # Running: straight at 1.0 m/s; then a slow walk at 0.2 m/s.
        for name, v in (("run", 1.0), ("slow", 0.2)):
            wins = 0
            for k in range(episodes):
                env.forced_command = "walk"
                obs, _ = env.reset(seed=seed + 700 + k)
                env.set_turn(0.0)
                env.set_speed(v)
                obs = env._obs()
                _, ok, _ = run_segment(env, policy, obs, "walk", 250)
                wins += ok
            results[name] = wins / episodes
        env.set_speed(0.4)
        env.forced_speed_set = False

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
