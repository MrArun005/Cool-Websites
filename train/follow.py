"""Follow a person: a steering layer on top of Bolt's trained brains, tested in the simulator.

    python follow.py --posture ../models/bolt-dog-posture-dog-2.5M --walk ../models/bolt-dog-walk-run \
                     -o follow.json

On the robot, the camera says where the person is. Here a simulated person walks a route
(stands, walks off, turns, breaks into a run, turns again, stops) and the follow controller
only gets what a camera would give: the person's position relative to Bolt. It turns toward
them, sets the speed from the distance (faster as they pull away), and lets the posture brain
stand Bolt still when they stop nearby. The walk-and-run brain does the legs.
Writes a Gait Lab clip (with the person) and prints how closely Bolt kept up.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bolt_env import BoltEnv
from evaluate import Policy, reset_standing
from record3d import capture, describe

# The person's route: (seconds, speed m/s, turn rate rad/s). Speeds up to a jog (1.1 m/s).
ROUTE = [(2.0, 0.0, 0.0), (5.0, 0.5, 0.0), (4.0, 0.5, 0.35), (5.0, 1.1, 0.0), (4.0, 0.8, -0.4),
         (3.0, 0.5, 0.0), (4.0, 0.0, 0.0)]


class Person:
    def __init__(self, x, y, heading):
        self.x, self.y, self.heading, self.v = x, y, heading, 0.0

    def step(self, v, turn, dt):
        self.heading += turn * dt
        self.v = v
        self.x += v * np.cos(self.heading) * dt
        self.y += v * np.sin(self.heading) * dt


class FollowController:
    """From the person's position relative to Bolt to commands for its brains."""
    KEEP = 1.0                     # m: how far behind the person Bolt likes to stay
    MAX_TURN, MAX_SPEED = 0.6, 1.2

    def __init__(self):
        self.standing = True

    def __call__(self, dist, bearing):
        # stand when close (with a little hysteresis so it doesn't flicker between stand and walk)
        if self.standing and dist > self.KEEP + 0.4:
            self.standing = False
        elif not self.standing and dist < self.KEEP - 0.1:
            self.standing = True
        if self.standing:
            return "stand", 0.0, 0.0
        turn = float(np.clip(1.5 * bearing, -self.MAX_TURN, self.MAX_TURN))
        speed = 0.2 + 1.0 * (dist - self.KEEP)                  # further away: faster
        speed *= max(0.3, np.cos(bearing))                      # turn first if they're off to the side
        return "walk", turn, float(np.clip(speed, 0.2, self.MAX_SPEED))


def run(posture, walk, seed=0):
    env = BoltEnv(randomize=False, max_steps=10**6, gait_generator="dog", turning=True,
                  speed_control=getattr(walk, "speed_control", False))
    env.forced_command = "stand"
    obs = reset_standing(env, seed)
    for _ in range(100):                                      # stand up first
        obs, *_ = env.step(posture.act(obs))
    d = env.data
    yaw0 = np.arctan2(*env._rotation()[[1, 0], 0])
    person = Person(d.qpos[0] + 1.3 * np.cos(yaw0), d.qpos[1] + 1.3 * np.sin(yaw0), yaw0)
    ctrl = FollowController()
    geoms = describe(env.model)
    index = {g["id"]: i for i, g in enumerate(geoms)}
    frames, dists, command = [], [], "stand"
    for seconds, v, turn in ROUTE:
        for _ in range(int(seconds * 50)):
            person.step(v, turn, env.dt)
            R = env._rotation()
            rel = R[:2, :2].T @ np.array([person.x - d.qpos[0], person.y - d.qpos[1]])  # in Bolt's frame
            dist, bearing = float(np.hypot(*rel)), float(np.arctan2(rel[1], rel[0]))
            want, yaw_cmd, speed_cmd = ctrl(dist, bearing)
            if want != command:
                env.forced_command = want
                env.set_command(want)
                command = want
            if command == "walk":
                env.set_turn(yaw_cmd)
                if env.speed_control:
                    env.set_speed(speed_cmd)
            brain = walk if command == "walk" else posture
            obs, _, fell, _, _ = env.step(brain.act(env._obs()))
            dists.append(dist)
            frames.append(capture(env, geoms, index, cmd=command, brain="walk" if command == "walk" else "posture",
                                  turn=yaw_cmd, speed=speed_cmd, fell=bool(fell),
                                  person=[round(person.x, 3), round(person.y, 3), round(person.heading, 3), round(v, 2)],
                                  dist=round(dist, 2)))
            if fell:
                return geoms, frames, dists, True
    return geoms, frames, dists, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posture", required=True)
    ap.add_argument("--walk", required=True)
    ap.add_argument("--seeds", type=int, default=5, help="how many times to run the route for the numbers")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    posture, walk = Policy(args.posture), Policy(args.walk)
    results = []
    for seed in range(args.seeds):
        geoms, frames, dists, fell = run(posture, walk, seed)
        d = np.array(dists)
        results.append(dict(seed=seed, fell=fell, mean=float(d.mean()), max=float(d.max()), lost=float(np.mean(d > 3.0))))
        print(f"seed {seed}: {'FELL' if fell else 'ok'}, distance to person: average {d.mean():.2f} m, "
              f"furthest {d.max():.2f} m, more than 3 m away {np.mean(d > 3.0):.0%} of the time", flush=True)
        if seed == 0:
            Path(args.out).write_text(json.dumps(dict(label="Follow a person", fps=50, geoms=geoms, frames=frames,
                                                      targets={}, summary=[]), separators=(",", ":")))
    print(json.dumps(dict(falls=sum(r["fell"] for r in results), mean_distance=round(float(np.mean([r["mean"] for r in results])), 2),
                          worst_distance=round(float(max(r["max"] for r in results)), 2))))


if __name__ == "__main__":
    main()
