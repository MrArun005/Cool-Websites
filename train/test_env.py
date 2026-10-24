"""Sanity checks for the MuJoCo model and training environment.

    python -m unittest test_env.py
"""
import unittest

import mujoco
import numpy as np

from bolt_env import COMMANDS, MODEL, STAND_POSE, BoltEnv


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


if __name__ == "__main__":
    unittest.main()
