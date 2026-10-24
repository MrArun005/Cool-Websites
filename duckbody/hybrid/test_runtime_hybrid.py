"""Unit tests for the hybrid runtime and its scripted moves (short sims, about 1 min in all).

    /home/user/cool-websites/.venv/bin/python duckbody/hybrid/test_runtime_hybrid.py
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import moves_hybrid as mv  # noqa: E402
import runtime_hybrid as rh  # noqa: E402
from harness import DEFAULT_POSE  # noqa: E402

RT = None


def rt():
    global RT
    if RT is None:
        RT = rh.HybridRuntime(seed=0, randomize=False, log=None)
    RT.reset("standing")
    return RT


class Moves(unittest.TestCase):
    def test_leg_ik_table_matches_kinematics(self):
        hs = mv.LEG_IK_H[mv.LEG_IK_H >= 0]
        x = mv.leg_ik(rh.SCENE.parent / "hybrid_allcollisions_noantenna.xml", hs)
        np.testing.assert_allclose(x, mv.LEG_IK[mv.LEG_IK_H >= 0], atol=0.02)   # table: 6-D least squares, here 3-D

    def test_legs_mirrored(self):
        q = mv.legs(DEFAULT_POSE.copy(), 0.03)
        d = q - DEFAULT_POSE
        np.testing.assert_allclose(d[[2, 3, 4]], -d[[11, 12, 13]])
        self.assertTrue(np.all(d[[2, 3, 4]] > 0))
        self.assertEqual(np.count_nonzero(d[5:9]), 0)

    def test_balance_sign(self):
        obs = np.zeros(61, np.float32)
        obs[3:6] = [math.sin(0.1), 0, -math.cos(0.1)]          # leaning forward 0.1 rad
        self.assertAlmostEqual(mv.pitch(obs), 0.1, places=5)
        self.assertGreater(mv.balance(obs), 0)

    def test_keyframes_end_on_last_key(self):
        k = mv.Keyframes([DEFAULT_POSE + 0.1, DEFAULT_POSE + 0.2], [0.2, 0.2], head=False)
        k.begin(None)
        a = [k(np.zeros(61, np.float32)) for _ in range(30)]
        np.testing.assert_allclose(a[-1], 0.2, atol=1e-5)
        self.assertAlmostEqual(k.duration, 0.4)


class Runtime(unittest.TestCase):
    def test_aliases_and_unknown(self):
        r = rt()
        self.assertIn("unknown", r.command("fly"))
        r.command("wake up")
        self.assertEqual(r.mode, "stand")

    def test_walk_command_vector_has_heading_hold(self):
        r = rt()
        r.command("walk")
        r.yaw_est = 0.2
        c = r._command_vector()
        self.assertAlmostEqual(c[0], rh.WALK_VX)
        self.assertAlmostEqual(c[2], -rh.HEADING_K * 0.2, places=5)

    def test_run_ramps_up(self):
        r = rt()
        r.command("run")
        self.assertLess(r._command_vector()[0], 0.1)
        r.plan_t = int(rh.RUN_RAMP_S * rh.HZ)
        self.assertAlmostEqual(r._command_vector()[0], rh.RUN_VX, places=5)

    def test_sit_flag_and_stand_chain(self):
        r = rt()
        r.command("sit")
        self.assertEqual(r.skill, "sit")
        self.assertEqual(r._command_vector()[0], 1.0)
        r.command("walk")                       # from sitting: stand up first, then walk
        self.assertEqual(r.skill, "sit_stand")
        self.assertEqual(r._command_vector()[0], 0.0)
        r.plan_t = int(rh.SIT_TO_STAND_S * rh.HZ) - 1
        r.tick()
        self.assertEqual(r.mode, "walk")
        self.assertEqual(r.skill, "velstand")

    def test_head_slew_limited_and_quiet_in_roll(self):
        r = rt()
        r.command("stand")
        r.look(yaw=1.0)
        r.run(0.1)
        self.assertLessEqual(r.head_cmd[2], rh.HEAD_RATE * 0.1 + 1e-6)
        r.run(1.0)
        self.assertAlmostEqual(r.head_cmd[2], 1.0, places=4)
        r.command("roll")
        r.run(0.6)
        self.assertLess(abs(r.head_cmd[2]), 0.5)

    def test_talking_moves_jaw(self):
        r = rt()
        r.command("idle")
        n = r.say("Hello there, I am a duck!")
        self.assertGreater(n, 20)
        jaws = []
        r.run(n * rh.DT, every=lambda x: jaws.append(float(x.sim.d.qpos[x.sim.m.jnt_qposadr[x.sim.m.joint("mouth").id]])))
        self.assertGreater(max(jaws), 0.25)
        r.run(0.5)
        self.assertFalse(r.speaking)
        self.assertEqual(r.jaw, 0.0)

    def test_audio_lip_sync(self):
        from mouth import demo_voice
        r = rt()
        audio, _ = demo_voice(seconds=1.0, seed=1)
        r.say_audio(audio, 22050)
        js = []
        r.run(1.0, every=lambda x: js.append(x.jaw))
        self.assertGreater(max(js), 0.2)

    def test_gets_up_after_fall(self):
        r = rt()
        r.reset("face_down")
        self.assertEqual(r.mode, "get up")
        r.run(3.0)
        self.assertLess(r.sim.tilt_deg(), 25)
        self.assertGreater(r.sim.trunk_z(), 0.10)

    def test_walk_then_squat_settles_first(self):
        r = rt()
        r.command("walk")
        r.run(1.0)
        r.command("squat")
        self.assertEqual(r.mode, "settle")
        self.assertEqual(r._command_vector()[0], 0.0)
        r.run(1.0)
        self.assertEqual(r.mode, "squat")

    def test_squat_lowers_and_stands(self):
        r = rt()
        r.run(0.5)
        z0 = r.sim.trunk_z()
        r.command("squat")
        r.run(2.0)
        self.assertLess(r.sim.trunk_z(), z0 - 0.03)
        self.assertLess(r.sim.tilt_deg(), 20)
        r.command("stand")
        self.assertEqual(r.skill, "squat_up")
        r.run(2.0)
        self.assertEqual(r.skill, "velstand")
        self.assertGreater(r.sim.trunk_z(), 0.10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
