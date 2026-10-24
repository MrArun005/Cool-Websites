"""Sanity checks for the MuJoCo model and training environment.

    python -m unittest test_env.py
"""
import unittest

import mujoco
import numpy as np

from bolt_env import ACTION_SCALE, COMMANDS, MODEL, STAND_POSE, TARGETS, BoltEnv
from evaluate import passed


class ModelTest(unittest.TestCase):
    def test_weighs_the_same_as_the_browser_dog(self):
        m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.assertAlmostEqual(sum(m.body_mass), 12.0, places=3)
        self.assertEqual(m.nu, 12)

    def test_motors_can_hold_a_standing_pose(self):
        m = mujoco.MjModel.from_xml_path(str(MODEL))
        d = mujoco.MjData(m)
        d.qpos[2] = 0.33
        d.qpos[7:] = d.ctrl[:] = STAND_POSE
        mujoco.mj_step(m, d, nstep=1500)  # 3 seconds
        self.assertGreater(d.qpos[2], 0.29)
        self.assertLess(np.abs(d.actuator_force).max(), 10)  # well under the 17 N·m limit
        touching = {m.geom(c.geom2).name for c in d.contact[:d.ncon]} | {m.geom(c.geom1).name for c in d.contact[:d.ncon]}
        self.assertEqual(touching, {"floor", "FL_foot", "FR_foot", "RL_foot", "RR_foot"})


class EnvTest(unittest.TestCase):
    def test_api_shapes_and_command_switching(self):
        env = BoltEnv(command_period=10, max_steps=60)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, env.observation_space.shape)
        seen, done, steps = set(), False, 0
        while not done:
            obs, r, fell, truncated, info = env.step(env.action_space.sample())
            seen.add(info["command"])
            self.assertTrue(np.isfinite(r) and np.all(np.isfinite(obs)))
            done, steps = fell or truncated, steps + 1
        self.assertGreater(len(seen), 1)

    def test_holding_the_right_posture_pays_best(self):
        env = BoltEnv(randomize=False)
        env.forced_command = "stand"
        env.reset(seed=0)
        rewards = {}
        for c in COMMANDS:
            env.set_command(c)
            rewards[c] = env._reward(np.zeros(12))[0]
        self.assertEqual(max(rewards, key=rewards.get), "stand")

    def test_same_seed_same_episode(self):
        a, b = BoltEnv(), BoltEnv()
        oa, _ = a.reset(seed=5)
        ob, _ = b.reset(seed=5)
        np.testing.assert_allclose(oa, ob)


class TargetTest(unittest.TestCase):
    """Training targets must be reachable: holding each measured pose must pass the exam."""

    def hold(self, command, steps=250):
        env = BoltEnv(randomize=False, max_steps=10_000)
        env.forced_command = command
        env.reset(seed=1)
        action = (np.asarray(TARGETS[command]["pose"]) - STAND_POSE) / ACTION_SCALE
        window = [env.step(action)[4] for _ in range(steps)]
        return env, window

    def test_held_poses_pass_the_exam(self):
        for command in ["stand", "sit", "lie"]:
            _, window = self.hold(command)
            self.assertTrue(passed(command, window[-80:]), command)

    def test_each_pose_pays_most_under_its_own_command(self):
        for command in ["stand", "sit", "lie"]:
            env, _ = self.hold(command)
            pay = {}
            for c in ["stand", "sit", "lie"]:
                env.set_command(c)
                pay[c] = env._reward(np.zeros(12))[0]
            self.assertEqual(max(pay, key=pay.get), command, pay)

    def test_standing_on_three_legs_pays_less(self):
        env, _ = self.hold("stand")
        four = env._reward(np.zeros(12))[0]
        env.feet_on_ground = lambda: np.array([1.0, 1.0, 1.0, 0.0])
        self.assertLess(env._reward(np.zeros(12))[0], four * 0.75)


class WalkRewardTest(unittest.TestCase):
    def test_moving_forward_pays_more_than_standing_still(self):
        env = BoltEnv(randomize=False)
        env.forced_command = "walk"
        env.reset(seed=0)
        env.total_steps = env.walk_ramp_steps  # end of the curriculum: target 0.5 m/s
        base = env.posture()
        pay = {}
        for vx in (0.0, 0.2, 0.4):
            env.posture = lambda vx=vx: dict(base, vx=vx, height=0.29, pitch=0.0, roll=0.0)
            pay[vx] = env._reward(np.zeros(12))[0]
        self.assertLess(pay[0.0], pay[0.2])
        self.assertLess(pay[0.2], pay[0.4])
        self.assertGreater(pay[0.2] - pay[0.0], 0.2)  # a clear signal to start moving

    def test_copying_the_trot_pays_more_than_standing_still(self):
        from bolt_env import trot
        totals = {}
        for mode in ("stand still", "copy trot"):
            env = BoltEnv(randomize=False, max_steps=10_000)
            env.forced_command = "walk"
            env.reset(seed=0)
            total = 0.0
            for _ in range(200):
                q = trot(2 * np.pi * 2.0 * env.t * env.dt)[0] if mode == "copy trot" else STAND_POSE
                total += env.step((q - STAND_POSE) / ACTION_SCALE)[1]
            totals[mode] = total
        self.assertGreater(totals["copy trot"], totals["stand still"] * 1.3, totals)


if __name__ == "__main__":
    unittest.main()
