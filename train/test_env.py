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

    def test_getting_closer_to_the_trot_pays_gradually(self):
        """No cliff: halfway to the reference trot must already earn a visible share."""
        from bolt_env import trot
        env = BoltEnv(randomize=False, max_steps=10_000)
        env.forced_command = "walk"
        env.reset(seed=0)
        q_ref = trot(0.0)[0]
        crouch = q_ref + 0.6  # far from the trot, like run 4's crouch (~0.6 rad away)
        half = q_ref + 0.3
        imit = lambda q: np.exp(-np.mean((q - q_ref) ** 2) / 0.3)
        self.assertGreater(imit(half), 0.5)
        self.assertGreater(imit(crouch), 0.2)
        self.assertGreater(imit(half) - imit(crouch), 0.2)


class GaitGeneratorTest(unittest.TestCase):
    def test_do_nothing_brain_already_trots_in_rhythm(self):
        env = BoltEnv(randomize=False, max_steps=10_000, skills=["walk"], gait_generator=True)
        env.forced_command = "walk"
        env.reset(seed=3)
        hips, x0 = [], None
        for t in range(300):
            _, _, fell, _, info = env.step(np.zeros(12))
            self.assertFalse(fell)
            if t == 100:
                x0 = env.posture()
            if t >= 100:
                hips.append(env.data.qpos[7:][[1, 4, 7, 10]])
        h = np.array(hips)
        self.assertGreater(np.corrcoef(h[:, 0], h[:, 3])[0, 1], 0.8)   # diagonal pair together
        self.assertLess(np.corrcoef(h[:, 0], h[:, 1])[0, 1], -0.5)     # left and right alternate

    def test_generator_only_affects_walking(self):
        env = BoltEnv(randomize=False, gait_generator=True)
        env.forced_command = "sit"
        env.reset(seed=0)
        np.testing.assert_allclose(env.base_pose(), STAND_POSE)
        np.testing.assert_allclose(env.action_scale(), ACTION_SCALE)



class TallWalkTest(unittest.TestCase):
    """Run 7 fixes: tall stance, no knees on the floor, hand-over from the standing brain."""

    def test_do_nothing_brain_trots_tall_without_scraping(self):
        for seed in range(3):
            env = BoltEnv(randomize=True, max_steps=10_000, skills=["walk"], gait_generator="tall")
            env.forced_command = "walk"
            env.reset(seed=seed)
            hs = []
            for t in range(300):
                _, _, fell, _, info = env.step(np.zeros(12))
                self.assertFalse(fell)
                if t >= 25:  # after landing from the slightly tipped start
                    self.assertEqual(info["scrape"], 0)
                if t > 100:
                    hs.append(info["height"])
            self.assertAlmostEqual(np.mean(hs), info["walk_height"], delta=0.02)

    def test_knee_on_the_floor_is_detected(self):
        env = BoltEnv(randomize=False)
        env.reset(seed=0)
        self.assertEqual(env.leg_scrapes(), 0)  # standing on paws
        env.data.qpos[2] = 0.12                  # body dropped: knees and thighs into the floor
        mujoco.mj_forward(env.model, env.data)
        self.assertGreater(env.leg_scrapes(), 0)

    def test_knee_on_the_floor_pays_less(self):
        env = BoltEnv(randomize=False, gait_generator="tall")
        env.forced_command = "walk"
        env.reset(seed=0)
        env.total_steps = env.walk_ramp_steps
        base = env.posture()
        env.posture = lambda: dict(base, vx=0.5, vy=0.0, height=0.4, pitch=0.0, roll=0.0, yaw_rate=0.0)
        clean = env._reward(np.zeros(12))[0]      # walking well, all paws clean
        env.leg_scrapes = lambda: 1
        scraped = env._reward(np.zeros(12))[0]
        self.assertLess(scraped, clean - 0.1, (clean, scraped))

    def test_exam_fails_a_walk_that_drags_a_knee(self):
        good = dict(height=0.4, walk_height=0.4, pitch=0.0, roll=0.0, vx=0.5, vy=0.0, up=1.0,
                    feet=np.array([0.5, 0.5, 0.5, 0.5]), scrape=0)
        self.assertTrue(passed("walk", [good] * 80))
        self.assertFalse(passed("walk", [good] * 70 + [dict(good, scrape=1)] * 10))
        self.assertFalse(passed("walk", [dict(good, height=0.3)] * 80))  # crawling in a crouch

    def test_walk_eases_in_from_standing(self):
        env = BoltEnv(randomize=False, gait_generator="tall")
        env.forced_command = "stand"
        env.reset(seed=0)
        for _ in range(30):
            env.step(np.zeros(12))
        env.set_command("walk")
        np.testing.assert_allclose(env.base_pose(), STAND_POSE)  # no jump at the hand-over
        for _ in range(30):
            env.step(np.zeros(12))
        self.assertGreater(np.abs(env.base_pose() - STAND_POSE).max(), 0.3)  # fully into the tall trot



class DogGaitTest(unittest.TestCase):
    """The paw-path trot: legs under the body, lower legs near upright, like a real dog."""

    def test_leg_ik_puts_the_paw_where_asked(self):
        from reference_gait import leg_ik
        m = mujoco.MjModel.from_xml_path(str(MODEL))
        d = mujoco.MjData(m)
        d.qpos[2], d.qpos[3] = 1.0, 1.0
        for x, z in [(0.0, -0.385), (0.06, -0.38), (-0.06, -0.38), (0.0, -0.325)]:
            for leg, front in (("FL", True), ("RR", False)):
                i = ["FL", "FR", "RL", "RR"].index(leg)
                d.qpos[7:] = 0
                d.qpos[7 + 3 * i + 1], d.qpos[7 + 3 * i + 2] = leg_ik(x, z, front)
                mujoco.mj_kinematics(m, d)
                rel = d.geom_xpos[m.geom(f"{leg}_foot").id] - d.xpos[m.body(f"{leg}_hip").id]
                np.testing.assert_allclose(rel[[0, 2]], [x, z], atol=1e-6)

    def test_do_nothing_brain_walks_with_upright_lower_legs(self):
        env = BoltEnv(randomize=False, max_steps=10_000, skills=["walk"], gait_generator="dog")
        env.forced_command = "walk"
        env.reset(seed=0)
        tilt, hs, vx = [], [], []
        for t in range(300):
            _, _, fell, _, info = env.step(np.zeros(12))
            self.assertFalse(fell)
            if t > 100:
                tilt.append(info["shin_tilt"]); hs.append(info["height"]); vx.append(info["vx"])
        self.assertLess(np.median(tilt), 27)            # the old tall trot: ~33°
        self.assertAlmostEqual(np.mean(hs), info["walk_height"], delta=0.02)
        self.assertGreater(np.mean(vx), 0.3)            # walks on its own, before any training

    def test_dog_stance_stands_tall_and_pays_best_for_it(self):
        from bolt_env import STANCES
        pays = {}
        for name, held in (("dog stance", STANCES["dog"]), ("old crouch", STAND_POSE)):
            env = BoltEnv(randomize=False, max_steps=10_000, gait_generator="dog")
            env.forced_command = "stand"
            env.reset(seed=0)
            total, window = 0.0, []
            for t in range(250):
                _, r, fell, _, info = env.step((held - env.base_pose()) / ACTION_SCALE)  # hold the pose, no brain
                self.assertFalse(fell)
                total += r
                window.append(info)
            pays[name] = total
            if name == "dog stance":
                self.assertTrue(passed("stand", window[-80:]))   # tall, level, four paws down
            else:
                self.assertFalse(passed("stand", window[-80:]))  # the crouch is no longer a "stand" here
        self.assertGreater(pays["dog stance"], pays["old crouch"] * 1.3, pays)

    def test_rising_toward_the_tall_stand_pays_gradually(self):
        env = BoltEnv(randomize=False, gait_generator="dog")
        env.forced_command = "stand"
        env.reset(seed=0)
        base = env.posture()
        pay = {}
        for h in (0.32, 0.346, 0.40, 0.436):
            env.posture = lambda h=h: dict(base, height=h, pitch=0.0, roll=0.0, vx=0.0, vy=0.0)
            pay[h] = env._reward(np.zeros(12))[0]
        self.assertLess(pay[0.32], pay[0.346])
        self.assertLess(pay[0.346], pay[0.40])
        self.assertLess(pay[0.40], pay[0.436])
        self.assertGreater(pay[0.40] - pay[0.346], 0.02)  # each step up is worth taking

    def test_dog_stand_is_centred_on_the_dog_stance(self):
        from bolt_env import STANCES
        env = BoltEnv(randomize=False, gait_generator="dog")
        env.forced_command = "stand"
        env.reset(seed=0)
        np.testing.assert_allclose(env.base_pose(), STANCES["dog"])  # a zero output already stands tall
        env.set_command("sit")
        np.testing.assert_allclose(env.base_pose(), STAND_POSE)      # sit and lie keep their centre

    def test_sit_needs_all_four_paws_down(self):
        good = dict(height=0.208, goal_height=0.208, pitch=0.54, roll=0.0, vx=0.0, vy=0.0, up=0.85,
                    feet=np.array([1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(passed("sit", [good] * 80))
        self.assertFalse(passed("sit", [dict(good, feet=np.array([0.4, 1.0, 1.0, 1.0]))] * 80))  # front-left paw hovering

    def test_walk_eases_in_from_the_tall_stand_without_a_dip(self):
        from bolt_env import STANCES
        env = BoltEnv(randomize=False, gait_generator="dog")
        env.forced_command = "stand"
        env.reset(seed=0)
        for _ in range(60):
            env.step(np.zeros(12))                       # standing tall
        env.set_command("walk")
        np.testing.assert_allclose(env.base_pose(), STANCES["dog"], atol=1e-6)  # starts where it stood
        lowest = min(env.step(np.zeros(12))[4]["height"] for _ in range(40))
        self.assertGreater(lowest, 0.40)                 # never sags toward the old 31 cm crouch

    def test_exam_hand_over_starts_from_a_standing_dog(self):
        from evaluate import reset_standing
        env = BoltEnv(randomize=True, max_steps=10_000, gait_generator="dog")
        env.forced_command = "stand"
        heights = []
        for seed in range(1000, 1030):
            reset_standing(env, seed)
            heights.append(env.data.qpos[2])
        self.assertGreater(min(heights), 0.3)  # some of these seeds start lying without it

    def test_turn_command_turns_the_dog_without_a_brain(self):
        rates = {}
        for rate in (0.4, -0.4, 0.0):
            env = BoltEnv(randomize=False, max_steps=10_000, skills=["walk"], gait_generator="dog", turning=True)
            env.forced_command = "walk"
            obs, _ = env.reset(seed=1)
            env.set_turn(rate)
            self.assertEqual(env._obs().shape, (53,))
            ys = []
            for t in range(300):
                _, _, fell, _, info = env.step(np.zeros(12))
                self.assertFalse(fell)
                if t > 100:
                    ys.append(info["yaw_rate"])
            rates[rate] = np.mean(ys)
        self.assertGreater(rates[0.4], 0.25)
        self.assertLess(rates[-0.4], -0.25)
        self.assertLess(abs(rates[0.0]), 0.05)

    def test_exam_checks_the_turn_only_for_turning_brains(self):
        good = dict(height=0.435, walk_height=0.435, pitch=0.0, roll=0.0, vx=0.45, vy=0.0, up=1.0,
                    feet=np.array([0.5] * 4), scrape=0, shin_tilt=22.0, yaw_rate=0.0, yaw_cmd=0.4, turning=True)
        self.assertFalse(passed("walk", [good] * 80))                       # asked left, went straight
        self.assertTrue(passed("walk", [dict(good, yaw_rate=0.35)] * 80))  # turned close to the asked rate
        self.assertTrue(passed("walk", [dict(good, turning=False)] * 80))  # older brains aren't asked to turn

    def test_walk_practice_can_start_from_the_real_standing_brain(self):
        env = BoltEnv(randomize=True, max_steps=10_000, skills=["walk"], gait_generator="dog",
                      handover_brain=str(MODEL.parent.parent / "models" / "bolt-dog-posture-dog-2.5M"))
        handed = 0
        for seed in range(8):
            env.reset(seed=seed)
            self.assertEqual(COMMANDS[env.command], "walk")
            if env.t > 0:                      # the posture brain stood the dog up first
                handed += 1
                self.assertGreater(env.data.qpos[2], 0.38)  # standing tall when the walk takes over
                self.assertEqual(env.walk_since, env.t)
        self.assertGreater(handed, 0)

    def test_speed_command_sets_the_pace_without_a_brain(self):
        got = {}
        for speed in (0.4, 1.0):
            env = BoltEnv(randomize=False, max_steps=10_000, skills=["walk"], gait_generator="dog",
                          turning=True, speed_control=True)
            env.forced_command = "walk"
            env.reset(seed=1)
            env.set_turn(0.0)
            env.set_speed(speed)
            self.assertEqual(env._obs().shape, (54,))
            vs = []
            for t in range(300):
                _, _, fell, _, info = env.step(np.zeros(12))
                self.assertFalse(fell)
                if t > 150:
                    vs.append(info["vx"])
            got[speed] = np.mean(vs)
        self.assertAlmostEqual(got[1.0], 1.0, delta=0.12)
        self.assertGreater(got[1.0], got[0.4] + 0.4)  # running really is faster

    def test_exam_checks_the_asked_speed(self):
        good = dict(height=0.435, walk_height=0.435, pitch=0.0, roll=0.0, vx=1.0, vy=0.0, up=1.0, feet=np.array([0.5] * 4),
                    scrape=0, shin_tilt=23.0, yaw_rate=0.0, yaw_cmd=0.0, turning=True, speed_cmd=1.0)
        self.assertTrue(passed("walk", [good] * 80))
        self.assertFalse(passed("walk", [dict(good, vx=0.5)] * 80))  # asked to run, walked

    def test_felt_speed_is_relative_to_the_asked_speed(self):
        env = BoltEnv(randomize=False, gait_generator="dog", turning=True, speed_control=True)
        env.forced_command = "walk"
        env.reset(seed=0)
        env.set_speed(env.SPEED_REF)
        at_ref = env._obs()
        env.set_speed(1.0)
        fast = env._obs()
        self.assertAlmostEqual(at_ref[6] - fast[6], 1.0 - env.SPEED_REF, places=5)  # forward speed input shifted
        np.testing.assert_allclose(np.delete(at_ref, [6, 53]), np.delete(fast, [6, 53]), atol=1e-6)

    def test_other_stances_keep_the_old_stand(self):
        self.assertEqual(BoltEnv(gait_generator="tall").targets["stand"], TARGETS["stand"])

    def test_exam_fails_sloping_lower_legs(self):
        good = dict(height=0.435, walk_height=0.435, pitch=0.0, roll=0.0, vx=0.5, vy=0.0, up=1.0,
                    feet=np.array([0.5, 0.5, 0.5, 0.5]), scrape=0, shin_tilt=22.0)
        self.assertTrue(passed("walk", [good] * 80))
        self.assertFalse(passed("walk", [dict(good, shin_tilt=38.0)] * 80))  # like run 7

    def test_sharp_turns_step_sideways(self):
        from reference_gait import paw_trot
        straight = paw_trot(1.0)[0]
        np.testing.assert_allclose(paw_trot(1.0, spin=0.0)[0], straight)  # no spin: the same gait as before
        self.assertTrue(np.all(straight[0::3] == 0))                       # and the sideways joints stay still
        q = paw_trot(1.0, spin=0.3)[0]
        self.assertGreater(np.abs(q[0::3]).max(), 0.02)                     # spinning rolls the legs out
        self.assertLess(np.abs(q[0::3]).max(), 0.5)                         # within the joints' range

    def test_sharp_turns_turn_fast_without_a_brain(self):
        env = BoltEnv(randomize=False, max_steps=10_000, skills=["walk"], gait_generator="dog",
                      turning=True, speed_control=True, sharp_turns=True)
        env.forced_command = "walk"
        env.reset(seed=1)
        env.set_speed(0.2)
        env.set_turn(1.5)
        yaw = []
        for t in range(200):
            _, _, fell, _, info = env.step(np.zeros(12))
            self.assertFalse(fell)
            if t > 50:
                yaw.append(info["yaw_rate"])
        self.assertGreater(np.mean(yaw), 0.8)  # stride-length turning managed ~0.5 rad/s at most

    def test_sharp_turn_practice_gets_gentler_when_running(self):
        env = BoltEnv(gait_generator="dog", turning=True, speed_control=True, sharp_turns=True)
        self.assertEqual(env.max_turn(0.4), env.SHARP_TURN)
        self.assertEqual(env.max_turn(1.2), 1.0)
        self.assertEqual(BoltEnv(gait_generator="dog", turning=True, speed_control=True).max_turn(0.4), BoltEnv.MAX_TURN)

    def test_exam_allows_a_little_more_error_on_sharp_turns(self):
        good = dict(height=0.435, walk_height=0.435, pitch=0.0, roll=0.0, vx=0.4, vy=0.0, up=1.0, feet=np.array([0.5] * 4),
                    scrape=0, shin_tilt=23.0, yaw_rate=1.25, yaw_cmd=1.5, turning=True, speed_cmd=0.4)
        self.assertTrue(passed("walk", [good] * 80))
        self.assertFalse(passed("walk", [dict(good, yaw_rate=0.6)] * 80))  # the old top turn rate fails


if __name__ == "__main__":
    unittest.main()
