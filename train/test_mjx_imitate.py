"""Parity of mjx_imitate.MjxImitate with imitate_env.ImitateEnv, and of the warm-started Brax brain with the SB3 one.

The CPU stack (torch, SB3) and JAX live in different virtualenvs, so this runs in two halves:
    <cpu python> test_mjx_imitate.py dump  <sb3 zip> <warm.pkl> <cases.pkl>   # ImitateEnv states, obs, rewards, actions
    <jax python> test_mjx_imitate.py check <warm.pkl> <cases.pkl>             # the same through MjxImitate and Brax

Compared: the 75 observations and the five reward terms for identical states (qpos, qvel, clip, time, last action),
the reference-state start (same clip, moment and heading), the clip switch (same new clip), one control step of
physics, and the deterministic action of the warm-started Brax policy against ImitatePolicy.act.
Landing starts (landing_rate > 0): the same start state from the same landing state and heading, then 40 steps
from identical states (forced switch to trot after the bridge, the height-check grace) in both envs.

Moving-start posture commands (moving_rate > 0): the same start state, clip moment, try length and time limit from the
same draw, and the same pose decisions and pose pay along tries. Motor brake on: one control step against ImitateEnv
with motor_brake=True (and how far the brake-off MJX step is from it, to show the brake acts).

Transition term (transition_weight > 0): the same roll/pitch excess on identical states, and the same window after a
forced canter -> trot switch.

Real-robot gaps (latency, motor lag, sensor noise): one control step with a 26 ms latency and a 20 ms motor lag from
identical states and queues, and the noise's bounds.

Rough terrain and pushes: the same field, xml and ground heights; the same start on the field, inputs and one control
step on it (paws and shins on the bumps); the same shove.

Peng's global root terms (root_global > 0): the two terms, the new reference root and its start at a try's first step
on identical states and reference roots, tries with a forced clip switch; alone with
    <cpu python> test_mjx_imitate.py dump-root-global <cases.pkl>   then   <jax python> test_mjx_imitate.py check-root-global <cases.pkl>

The Go2 body with steering (MjxImitate(body="go2"), ImitateEnv(body="go2"), the motor brake on), each with steering off
and on (steer_max 1, a fixed command of +-0.8 rad/s per try): the 75 inputs, the reward terms (steer included), the total
reward and the end-of-try decision on identical states; the reference-state start; one control step of physics; whole
tries replayed open-loop in MJX with the CPU env's actions (obs, reward and end of try per step); the steering draws; and
the warm-started Go2 brain against ImitatePolicy.act:
    <cpu python> test_mjx_imitate.py dump-go2 <run dir with policy.zip + vecnormalize.pkl> <warm.pkl> <cases.pkl>
    <jax python> test_mjx_imitate.py check-go2 <warm.pkl> <cases.pkl>

Unit tests (CPU stack; landing and moving starts keep the 75-input layout, rate 0 leaves the env as it was):
    <cpu python> -m unittest test_mjx_imitate
"""
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BRAIN_360M = HERE.parent / "models" / "bolt-dog-imitate-gpu-360M" / "params.pkl"


class LandingStartTest(unittest.TestCase):
    """A landing start (imitate_env landing_rate) is an ordinary try for the brain: same inputs, same order, same size."""

    def test_landing_start_obs_layout(self):
        from imitate_env import ALL_CLIPS, DOG, LANDING_FILE, ImitateEnv
        from mjx_imitate import MjxImitatePolicy
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, landing_rate=1.0)
        plain = ImitateEnv(randomize=False, clips=ALL_CLIPS)
        self.assertEqual(env.observation_space.shape, plain.observation_space.shape)
        self.assertEqual(env.observation_space.shape, (75,))
        air = np.flatnonzero(np.load(LANDING_FILE)["airborne"])
        self.assertGreater(len(air), 0)
        env.exam_landing = int(air[0])   # a mid-air landing state
        obs, _ = env.reset(seed=0)
        d = env.data
        self.assertEqual(obs.shape, (75,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(env.feet_on_ground().sum(), 0)   # it starts in the air
        np.testing.assert_allclose(obs[9], d.qpos[2], atol=1e-6)                      # body height
        np.testing.assert_allclose(obs[10:22], d.qpos[7:19] - DOG, atol=1e-6)          # joint angles - stance
        np.testing.assert_allclose(obs[22:34], d.qvel[6:18] * 0.1, atol=1e-5)          # joint velocities
        np.testing.assert_array_equal(obs[34:46], 0.0)                                 # no previous action
        onehot = np.zeros(len(ALL_CLIPS))
        onehot[ALL_CLIPS.index("canter")] = 1
        np.testing.assert_array_equal(obs[48:57], onehot)                              # tracking the canter
        np.testing.assert_allclose(obs[57:69], env.ref("joints", 1) - DOG, atol=1e-6)  # reference joints
        self.assertEqual(obs[74], 0.0)                                                 # a loop: no clip end
        pol = MjxImitatePolicy(BRAIN_360M)   # the 360M brain (9 slots) runs on it unchanged
        self.assertEqual(pol.slots, 9)
        a = pol.act(obs)
        self.assertEqual(a.shape, (12,))
        obs2, r, fell, done, info = env.step(a)
        self.assertEqual(obs2.shape, (75,))
        self.assertTrue(np.all(np.isfinite(obs2)) and np.isfinite(r))
        np.testing.assert_allclose(obs2[34:46], a, atol=1e-6)                          # previous action
        self.assertEqual(info["clip"], "canter")
        self.assertFalse(fell)   # in the air, ~10 cm above the canter's height: the grace keeps the try going

    def test_landing_bridge_switch(self):
        from imitate_env import ALL_CLIPS, ImitateEnv
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, landing_rate=1.0, landing_bridge=0.1)
        env.exam_landing = 0
        env.reset(seed=1)
        clips = []
        for _ in range(6):
            _, _, fell, _, info = env.step(np.zeros(12))
            clips.append(info["clip"])
        self.assertEqual(clips, ["canter"] * 4 + ["trot"] * 2)   # 0.1 s = 5 steps of canter, then trot

    def test_rate_zero_is_the_old_env(self):
        """landing_rate 0 (the default) draws nothing extra: the same tries, step for step."""
        from imitate_env import ALL_CLIPS, ImitateEnv
        a, b = ImitateEnv(seed=3, clips=ALL_CLIPS, switch_rate=0.3), ImitateEnv(seed=3, clips=ALL_CLIPS, switch_rate=0.3, landing_rate=0.0)
        rng = np.random.default_rng(0)
        for ep in range(6):
            oa, _ = a.reset(seed=ep if ep % 2 else None)
            ob, _ = b.reset(seed=ep if ep % 2 else None)
            np.testing.assert_array_equal(oa, ob)
            for _ in range(30):
                act = rng.uniform(-1, 1, 12)
                ra, rb = a.step(act), b.step(act)
                np.testing.assert_array_equal(ra[0], rb[0])
                self.assertEqual(ra[1:4], rb[1:4])
                if ra[2] or ra[3]:
                    break


class MovingStartTest(unittest.TestCase):
    """A moving-start posture command (imitate_env moving_rate): an ordinary try for the brain, paid by the shared pose test."""

    def test_moving_start_obs_layout_and_pay(self):
        from imitate_env import ALL_CLIPS, DOG, ImitateEnv, closest_moment
        from mjx_imitate import MjxImitatePolicy
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, moving_rate=1.0)
        env.exam_moving = ("sit", "walk", 0.2, 1.2)
        obs, _ = env.reset(seed=0)
        d = env.data
        self.assertEqual(obs.shape, (75,))
        self.assertEqual(env.clip.name, "sit")
        # the walk's body velocity at that moment, scaled so the clip's pace becomes 1.2 m/s
        self.assertAlmostEqual(np.linalg.norm(d.qvel[0:3]), 1.2 * np.linalg.norm(env.clips["walk"].at(0.2, "v_body")) / env.gait_speed("walk"), places=6)
        self.assertEqual(env.tau0, closest_moment(env.clips["sit"], d.qpos[7:19], d.qvel[6:18]))
        self.assertGreaterEqual(env.limit, env.pose_until + 1)
        self.assertEqual(env.pose_until, 200)   # 4 s at 50 Hz (the switch test's sit time)
        onehot = np.zeros(len(ALL_CLIPS))
        onehot[ALL_CLIPS.index("sit")] = 1
        np.testing.assert_array_equal(obs[48:57], onehot)
        np.testing.assert_allclose(obs[57:69], env.ref("joints", 1) - DOG, atol=1e-6)
        pol = MjxImitatePolicy(BRAIN_360M)
        hits = 0
        for _ in range(env.limit):
            obs, r, fell, done, info = env.step(pol.act(obs))
            self.assertEqual(obs.shape, (75,))
            track = sum(env.weights[k] * info[f"r_{k}"] for k in env.weights)
            self.assertAlmostEqual(r, track + env.pose_weight * (info["pose_hit"] and info["pose_ok"]), places=6)
            self.assertEqual(info["pose_ok"], env.pose_now("sit"))
            hits += info["pose_hit"]
            if fell or done:
                break
        self.assertGreater(hits, 0)   # the 360M brain sits from this 1.2 m/s walk start (168 paid steps, 28 Oct)

    def test_same_pose_definition_as_switch_test(self):
        from pose_reached import POSE_OK, posture_ok
        self.assertEqual(POSE_OK["sit"], dict(height_max=0.27, pitch_min=0.3))   # test_skills' numbers before the move
        self.assertEqual(POSE_OK["lie"], dict(height_max=0.19, up_min=0.9))
        self.assertTrue(posture_ok("sit", dict(height=0.23, pitch=0.6, up=0.8)))
        self.assertFalse(posture_ok("sit", dict(height=0.23, pitch=-0.5, up=0.8)))   # nose down: the walk->sit failure
        self.assertTrue(posture_ok("lie", dict(height=0.17, pitch=0.0, up=0.99)))


class TransitionTermTest(unittest.TestCase):
    """The transition term charges roll/pitch beyond the threshold only in the window after a mid-try switch."""

    def test_window_and_charge(self):
        from imitate_env import ALL_CLIPS, ImitateEnv
        from mjx_imitate import MjxImitatePolicy
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, transition_weight=0.5, transition_window=0.4, transition_threshold=0.05)
        env.exam_start = ("canter", 0.1)
        obs, _ = env.reset(seed=2)
        pol = MjxImitatePolicy(BRAIN_360M)
        charged, outside = 0, 0
        for t in range(60):
            if t == 10:
                env._switch_to(env.clips["trot"])   # what a drawn switch does (canter -> trot)
            obs, r, fell, done, info = env.step(pol.act(obs))
            self.assertEqual(obs.shape, (75,))
            track = sum(env.weights[k] * info[f"r_{k}"] for k in env.weights)
            in_window = info["since_switch"] is not None and info["since_switch"] < 20   # 0.4 s
            self.assertAlmostEqual(r, track - (0.5 * env.transition_excess() if in_window else 0.0), places=6)
            charged += in_window and info["transition"] > 0
            outside += (not in_window) and info["transition"] != 0
            if fell or done:
                break
        self.assertEqual(outside, 0)
        self.assertGreater(charged, 0)   # a 0.05 rad threshold is crossed by an ordinary trot


class RealGapTest(unittest.TestCase):
    """Latency, motor lag and sensor noise (imitate_env REAL), as mastermind/env.py's stress test."""

    def _record(self, env, actions):
        import imitate_env
        seen, real = [], imitate_env.physics

        def physics(model, data, n):
            for _ in range(n):
                seen.append(data.ctrl.copy())
                real(model, data, 1)
        imitate_env.physics = physics
        try:
            for a in actions:
                env.step(a)
        finally:
            imitate_env.physics = real
        return seen

    def test_latency_queue(self):
        from imitate_env import ALL_CLIPS, ImitateEnv
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, latency_ms=(0, 40))
        env.exam_latency_ms, env.exam_start = 26.0, ("trot", 0.1)   # 13 substeps: 1 control step + 3 substeps
        env.reset(seed=0)
        self.assertEqual(env.delay_sub, 13)
        c0 = env.data.ctrl.copy()
        rng = np.random.default_rng(0)
        acts = rng.uniform(-1, 1, (3, 12))
        sent = []
        import imitate_env
        real = imitate_env.physics
        seen = []

        def physics(model, data, n):
            for _ in range(n):
                seen.append(data.ctrl.copy())
                real(model, data, 1)
        imitate_env.physics = physics
        try:
            for a in acts:
                base = env.ref("joints", 1)
                sent.append(np.clip(base + imitate_env.ACTION_SCALE * a, env.lo, env.hi))
                obs, *_ = env.step(a)
                np.testing.assert_allclose(obs[34:46], a, atol=1e-6)   # the brain sees what it sent
        finally:
            imitate_env.physics = real
        want = [c0] * 10 + [c0] * 3 + [sent[0]] * 7 + [sent[0]] * 3 + [sent[1]] * 7
        for i, (s_, w) in enumerate(zip(seen, want)):
            np.testing.assert_allclose(s_, w, atol=1e-12, err_msg=f"substep {i}")

    def test_motor_lag(self):
        from imitate_env import ALL_CLIPS, ImitateEnv
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, motor_lag_ms=20.0)
        env.exam_start = ("trot", 0.1)
        env.reset(seed=0)
        y = env.data.ctrl.copy()
        a = np.full(12, 0.5)
        import imitate_env
        target = np.clip(env.ref("joints", 1) + imitate_env.ACTION_SCALE * a, env.lo, env.hi)
        seen = self._record(env, [a])
        g = 1 - np.exp(-0.002 / 0.020)
        for s_ in seen:
            y = y + g * (target - y)
            np.testing.assert_allclose(s_, y, atol=1e-12)

    def test_obs_noise_bounds(self):
        from imitate_env import ALL_CLIPS, NOISE, ImitateEnv
        a, b = ImitateEnv(randomize=False, clips=ALL_CLIPS), ImitateEnv(randomize=False, clips=ALL_CLIPS, obs_noise=1.0)
        for e in (a, b):
            e.exam_start = ("walk", 0.2)
        oa, _ = a.reset(seed=0)
        ob, _ = b.reset(seed=0)
        lim = np.concatenate([np.full(3, NOISE["grav"]), np.full(3, NOISE["gyro"] * 0.25), np.full(3, NOISE["linvel"]), [NOISE["height"]],
                              np.full(12, NOISE["qpos"]), np.full(12, NOISE["qvel"] * 0.1)])
        diff = np.abs(ob - oa)
        self.assertTrue(np.all(diff[:34] <= lim + 1e-6) and np.any(diff[:34] > 0))
        np.testing.assert_array_equal(ob[34:], oa[34:])


class TerrainPushTest(unittest.TestCase):
    """Rough terrain (bolt_env TERRAIN) and pushes (bolt_env.apply_shove) in ImitateEnv."""

    def test_terrain_start_and_inputs(self):
        from bolt_env import TERRAIN
        from imitate_env import ALL_CLIPS, DOG, ImitateEnv
        from mjx_imitate import MjxImitatePolicy
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS, terrain_cm=5.0)
        env.exam_start = ("trot", 0.2)
        obs, _ = env.reset(seed=0)
        d = env.data
        self.assertEqual(obs.shape, (75,))
        self.assertLess(np.abs(d.qpos[:2] - TERRAIN["at"]).max(), env.terrain_spread + 1e-9)   # on the field
        gz = env.ground_z()
        self.assertGreater(gz, 0.0)
        np.testing.assert_allclose(obs[9], d.qpos[2] - gz, atol=1e-6)        # height from the ground under the body
        np.testing.assert_allclose(obs[10:22], d.qpos[7:19] - DOG, atol=1e-6)
        pol = MjxImitatePolicy(BRAIN_360M)
        touched = set()
        for _ in range(50):
            obs, r, fell, done, info = env.step(pol.act(obs))
            self.assertEqual(obs.shape, (75,))
            touched |= {env.model.geom(c.geom1 if c.geom2 == env.terrain["geom"] else c.geom2).name
                        for c in d.contact[:d.ncon] if env.terrain["geom"] in (c.geom1, c.geom2)}
        self.assertTrue(touched and all(n.endswith("_foot") or n == "" for n in touched), touched)   # paws (shins unnamed)
        env.exam_start = ("sit", 0.0)
        env.reset(seed=0)
        self.assertEqual(float(d.qpos[0]), 0.0)   # other clips start on the flat floor

    def test_push(self):
        from bolt_env import apply_shove
        from imitate_env import ALL_CLIPS, ImitateEnv
        a = ImitateEnv(randomize=False, clips=ALL_CLIPS)
        b = ImitateEnv(randomize=False, clips=ALL_CLIPS, push_interval=0.1, push_prob=1.0, push_speed=1.0, push_spin=2.0)
        for e in (a, b):
            e.exam_start = ("trot", 0.1)
            e.reset(seed=0)
        for t in range(5):   # no push before step 5 (0.1 s)
            ra, rb = a.step(np.zeros(12)), b.step(np.zeros(12))
            np.testing.assert_array_equal(ra[0], rb[0])
        qv, R = b.data.qvel.copy(), b._rotation().copy()
        seen = []
        import imitate_env
        real = imitate_env.physics
        imitate_env.physics = lambda m, d, n: (seen.append(d.qvel.copy()), real(m, d, n))
        try:
            b.step(np.zeros(12))
        finally:
            imitate_env.physics = real
        pushed = seen[0]
        ok = [np.allclose(pushed, apply_shove(qv, R, 1.0, 2.0, s)) for s in (1.0, -1.0)]
        self.assertTrue(any(ok), "the push is the switch test's shove")


def dump_terrain(pol, rng):
    """Terrain tries through ImitateEnv (5 cm): start states and 30 steps each, the ground under the body (check_terrain)."""
    import mujoco
    from bolt_env import terrain_field, terrain_xml, MODEL
    from imitate_env import ALL_CLIPS, ImitateEnv
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, terrain_cm=5.0)
    starts, steps = [], []
    for k in range(8):
        name = ("walk", "trot")[k % 2]
        env.exam_start = (name, float(rng.uniform(0, env.clips[name].length)))
        obs, _ = env.reset(seed=1900 + k)
        R = env._rotation()
        u, _, vt = np.linalg.svd(env.ref("tilt"))
        M_ = R @ (u @ vt).T
        starts.append(dict(clip=ALL_CLIPS.index(name), tau0=env.tau0, yaw=float(np.arctan2(M_[1, 0], M_[0, 0])), qpos=env.data.qpos.copy(),
                           qvel=env.data.qvel.copy(), obs=obs, limit=env.limit))
        for t in range(30):
            pre = dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, last_action=env.last_action.copy(), qpos=env.data.qpos.copy(),
                       qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy())
            a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
            obs, r, fell, done, info = env.step(a)
            d2 = mujoco.MjData(env.model)
            d2.qpos[:], d2.qvel[:], d2.ctrl[:] = env.data.qpos, env.data.qvel, env.data.ctrl
            mujoco.mj_forward(env.model, d2)
            d, env.data = env.data, d2
            f_obs, (f_terms, f_err), f_bof = env._obs(), env.tracking(), env.body_on_floor()
            n_ter = sum(1 for c in d2.contact[:d2.ncon] if env.terrain["geom"] in (c.geom1, c.geom2))
            env.data = d
            steps.append(dict(pre=pre, action=a, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), f_obs=f_obs, f_terms=f_terms,
                              f_bof=f_bof, n_ter=n_ter, gz=env.ground_z()))
            if fell or done:
                break
    xs = rng.uniform(15, 65, 500), rng.uniform(-25, 25, 500)
    from bolt_env import terrain_height
    return dict(starts=starts, steps=steps, field=terrain_field(), xml=terrain_xml(MODEL.read_text(), 5.0), probe=xs,
                heights=terrain_height(terrain_field(), 5.0, *xs))


def check_terrain(M, C):
    """Terrain: the copies in mjx_imitate against bolt_env's, then starts, inputs, rewards and one step on the field."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    same_field = np.array_equal(M.terrain_field(), C["field"])
    same_xml = M.terrain_xml((M.HERE / "bolt.xml").read_text(), 5.0) == C["xml"]
    env = M.MjxImitate(randomize=False, switch_rate=0.0, terrain_cm=5.0)
    hz = np.asarray(jax.vmap(lambda x, y: M.terrain_height(env.terrain["field"], 5.0, x, y))(jp.array(C["probe"][0], jp.float32), jp.array(C["probe"][1], jp.float32)))
    print(f"[terrain copies] field identical {same_field}, xml identical {same_xml}, ground height at 500 points |diff| max {np.abs(hz - C['heights']).max():.1e} m")
    names = env.names

    def info_of(s, t=0, la=None):
        i = dict(kp=jp.float32(1), mass=jp.float32(1), fric=jp.float32(1), clip=jp.int32(names.index(M.ALL_CLIPS[s["clip"]])), tau0=jp.float32(s["tau0"]),
                 t=jp.int32(t), last_action=jp.array(np.zeros(12) if la is None else la, jp.float32), trunc=jp.float32(0), limit=jp.int32(s.get("limit", 0)))
        return i

    start_data = jax.jit(env.start_data)
    dq = []
    for s in C["starts"]:
        info = env.terrain_info(info_of(s), jp.bool_(True), jp.array(s["qpos"][:2], jp.float32))
        d = start_data(info, jp.float32(s["yaw"]))
        q = np.array(d.qpos)
        if np.dot(q[3:7], s["qpos"][3:7]) < 0:
            q[3:7] *= -1
        o = np.asarray(env.obs(d, info))
        dq.append(max(np.abs(q - s["qpos"]).max(), np.abs(np.asarray(d.qvel) - s["qvel"]).max(), np.abs(o - s["obs"]).max()))
    print(f"[terrain start] {len(dq)} walk/trot starts on the field: max |diff| qpos/qvel/obs {max(dq):.2e} (median {np.median(dq):.1e})")

    @jax.jit
    def senses(qpos, qvel, info):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        terms, err = env.tracking(d, info)
        return env.obs(d, info), terms, env.body_on_floor(d)

    @jax.jit
    def one_step(qpos, qvel, ctrl, info, a):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        d = env.physics(env.mx, d.replace(ctrl=jp.clip(env.ref(info, "joints", 1) + env.scale * a, env.lo, env.hi)))
        c = env.contact(d)
        tg = env.terrain["geom"]
        return d.qpos, jp.sum((c.dist < 0) & ((c.geom[:, 0] == tg) | (c.geom[:, 1] == tg)))

    do, dt, dp, nter_cpu, nter_mjx = [], [], [], 0, 0
    for s in C["steps"]:
        p = s["pre"]
        info = info_of(dict(p), t=p["t"] + 1, la=s["action"])
        o, terms, bof = senses(jp.array(s["qpos"], jp.float32), jp.array(s["qvel"], jp.float32), info)
        do.append(np.abs(np.asarray(o) - s["f_obs"]).max())
        dt.append(max(abs(float(terms[k]) - s["f_terms"][k]) for k in M.WEIGHTS))
        q, nt = one_step(jp.array(p["qpos"], jp.float32), jp.array(p["qvel"], jp.float32), jp.array(p["ctrl"], jp.float32),
                         info_of(dict(p), t=p["t"]), jp.array(s["action"], jp.float32))
        dp.append(np.abs(np.asarray(q)[7:19] - s["qpos"][7:19]).max())
        nter_cpu += s["n_ter"] > 0
        nter_mjx += int(nt) > 0
    pc = lambda x: ", ".join(f"p{k} {np.percentile(x, k):.1e}" for k in (50, 90)) + f", max {max(x):.1e}"
    print(f"[terrain steps] {len(do)} steps: obs |diff| {pc(do)}; reward terms |diff| max {max(dt):.1e}; one step joint angles |diff| {pc(dp)} rad; "
          f"paws/shins on the bumps in {nter_cpu} (CPU) / {nter_mjx} (MJX) states")
    envp = M.MjxImitate(randomize=True, terrain_cm=5.0, push_interval=0.1, push_prob=1.0)
    st = jax.jit(envp.reset)(jax.random.PRNGKey(2))
    for _ in range(6):
        st = jax.jit(envp.step)(st, jp.zeros(12))
    print(f"[terrain+push reset+steps] obs {st.obs.shape}, finite {bool(jp.all(jp.isfinite(st.obs)))}")


def dump_shove(rng):
    """bolt_env.apply_shove on random states (check_shove)."""
    from bolt_env import apply_shove
    cases = []
    for k in range(50):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        qv, sp, spin, side = rng.normal(size=18), rng.uniform(0, 3), rng.uniform(0, 6), (1.0, -1.0)[k % 2]
        cases.append(dict(R=R, qvel=qv, speed=sp, spin=spin, side=side, out=apply_shove(qv, R, sp, spin, side)))
    return cases


def check_shove(M, C):
    import jax.numpy as jp
    d = max(np.abs(np.asarray(M.apply_shove(jp.array(c["qvel"], jp.float32), jp.array(c["R"], jp.float32), c["speed"], c["spin"], c["side"])) - c["out"]).max()
            for c in C)
    print(f"[shove] {len(C)} shoves: mjx_imitate.apply_shove vs bolt_env.apply_shove |diff| max {d:.1e}")


def dump_real(pol, rng):
    """One control step each with a 26 ms latency and a 20 ms motor lag, from 60 states (check_real)."""
    from imitate_env import ALL_CLIPS, ImitateEnv
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, latency_ms=(0, 40), motor_lag_ms=20.0)
    steps = []
    for k in range(12):
        env.exam_latency_ms, env.exam_start = 26.0, (ALL_CLIPS[k % 6], float(rng.uniform(0, 0.4)))
        obs, _ = env.reset(seed=1700 + k)
        for t in range(5):
            a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
            pre = dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                       queue=np.array(env._queue), lagy=env._lag_y.copy(), dsub=env.delay_sub)
            obs, r, fell, done, info = env.step(a)
            steps.append(dict(pre=pre, action=a, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), queue=np.array(env._queue), lagy=env._lag_y.copy()))
            if fell or done:
                break
    return dict(steps=steps, latency=(0, 40), lag=20.0)


def dump_transition(pol, rng):
    """Canter tries with a forced switch to trot at step 15, transition term on (check_transition)."""
    import mujoco
    from imitate_env import ALL_CLIPS, ImitateEnv
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, transition_weight=0.5, transition_window=1.0, transition_threshold=0.1)
    steps = []
    for k in range(6):
        env.exam_start = ("canter", float(rng.uniform(0, env.clips["canter"].length)))
        obs, _ = env.reset(seed=1500 + k)
        for t in range(70):
            if t == 15:
                env._switch_to(env.clips["trot"])
            a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
            obs, r, fell, done, info = env.step(a)
            d2 = mujoco.MjData(env.model)
            d2.qpos[:], d2.qvel[:], d2.ctrl[:] = env.data.qpos, env.data.qvel, env.data.ctrl
            mujoco.mj_forward(env.model, d2)
            d, env.data = env.data, d2
            f_ex = env.transition_excess()
            env.data = d
            steps.append(dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, swt=env.switch_t if env.switch_t is not None else -10 ** 6,
                              qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), f_ex=f_ex,
                              f_on=env.switch_t is not None and env.t - env.switch_t < env.transition_steps))
            if fell or done:
                break
    return dict(steps=steps, weight=0.5, window=1.0, threshold=0.1)


def dump_moving(pol, rng):
    """Moving starts through ImitateEnv: start states and up to 60 steps each, with the pose decisions (check_moving)."""
    import mujoco
    from imitate_env import ALL_CLIPS, ImitateEnv
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, moving_rate=1.0)
    starts, steps = [], []
    k = 0
    for post in ("sit", "lie", "getup"):
        for gait in ("walk", "trot"):
            for v in (0.6, 1.9):
                tau_g = float(rng.uniform(0, env.clips[gait].length))
                env.exam_moving = (post, gait, tau_g, v)
                obs, _ = env.reset(seed=900 + k)
                k += 1
                R = env._rotation()
                st = dict(post=ALL_CLIPS.index(post), gait=ALL_CLIPS.index(gait), tau_g=tau_g, vscale=v / env.gait_speed(gait),
                          yaw=None, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), obs=obs, tau0=env.tau0, limit=env.limit,
                          until=env.pose_until)
                # the heading _rsi drew: the body's yaw minus the clip frame's own (the gait reference has yaw 0)
                Rt = env.clips[gait].at(tau_g, "tilt")
                u, _, vt = np.linalg.svd(Rt)
                M = R @ (u @ vt).T
                st["yaw"] = float(np.arctan2(M[1, 0], M[0, 0]))
                starts.append(st)
                for t in range(60):
                    pre = dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, hit=env.pose_hit)
                    a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
                    obs, r, fell, done, info = env.step(a)
                    d2 = mujoco.MjData(env.model)   # the pose decision on up-to-date kinematics, both sides
                    d2.qpos[:], d2.qvel[:], d2.ctrl[:] = env.data.qpos, env.data.qvel, env.data.ctrl
                    mujoco.mj_forward(env.model, d2)
                    d, env.data = env.data, d2
                    f_pose = env.clip.name == env.pose_clip and env.pose_now({"sit": "sit", "lie": "lie", "getup": "stand"}[post])
                    env.data = d
                    steps.append(dict(pre=pre, start=len(starts) - 1, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                                      f_pose=bool(f_pose), f_hit=bool(pre["hit"] or (f_pose and env.t <= env.pose_until)), t=env.t))
                    if fell or done:
                        break
    return dict(starts=starts, steps=steps)


def dump_brake(pol, rng):
    """One control step each from 120 states with the motor brake on (ImitateEnv motor_brake=True): Bolt held 1 m up
    (no contacts, so float32 and float64 physics stay together and the brake is what is measured), its joints spinning
    at 0-40 rad/s, a random action (check_brake)."""
    import mujoco
    from bolt_env import ACTION_SCALE, physics
    from imitate_env import ALL_CLIPS, ImitateEnv
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, motor_brake=True)
    off = ImitateEnv(randomize=False, clips=ALL_CLIPS, motor_brake=False)
    assert env.motor_brake[0] and not off.motor_brake[0], (env.motor_brake, off.motor_brake)
    steps = []
    for k in range(120):
        env.reset(seed=1200 + k)
        d = env.data
        d.qpos[2] += 1.0
        d.qvel[:] = 0.0
        d.qvel[6:18] = rng.choice([-1, 1], 12) * rng.uniform(0, 40, 12)
        mujoco.mj_forward(env.model, d)
        pre = dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy())
        a = rng.uniform(-1, 1, 12)
        d.ctrl[:] = np.clip(env.ref("joints", 1) + ACTION_SCALE * a, env.lo, env.hi)
        d1 = mujoco.MjData(env.model)   # one physics substep (2 ms), as bolt_env.physics does it, brake included
        d1.qpos[:], d1.qvel[:], d1.ctrl[:] = d.qpos, d.qvel, d.ctrl
        physics(env.model, d1, 1)
        d0 = mujoco.MjData(off.model)   # the same substep without the brake
        d0.qpos[:], d0.qvel[:], d0.ctrl[:] = d.qpos, d.qvel, d.ctrl
        physics(off.model, d0, 1)
        euler = {}
        for e, model in (("on", env.model), ("off", off.model)):   # and both with the explicit Euler integrator
            integ = model.opt.integrator
            model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
            de = mujoco.MjData(model)
            de.qpos[:], de.qvel[:], de.ctrl[:] = d.qpos, d.qvel, d.ctrl
            physics(model, de, 1)
            model.opt.integrator = integ
            euler[e] = de.qvel.copy()
        steps.append(dict(pre=dict(pre, ctrl=d.ctrl.copy()), action=a, qvel1=d1.qvel.copy(), qvel0=d0.qvel.copy(), brake1=d1.qfrc_applied[6:18].copy(),
                          qvel1e=euler["on"], qvel0e=euler["off"],
                          fast=float(np.abs(pre["qvel"][6:18]).max())))
    return dict(steps=steps, settings=env.motor_brake)


def dump_landing(pol, rng):
    """Landing starts through ImitateEnv: start states and 40 steps each (for check_landing)."""
    import mujoco
    from imitate_env import ALL_CLIPS, LANDING_FILE, ImitateEnv
    z = np.load(LANDING_FILE)
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS, landing_rate=1.0)
    picks = np.unique(np.concatenate([np.flatnonzero(z["airborne"])[::max(1, int(z["airborne"].sum()) // 6)][:6],
                                      np.flatnonzero(~z["airborne"])[::max(1, int((~z["airborne"]).sum()) // 4)][:4]]))
    starts, steps = [], []
    for k, i in enumerate(picks):
        env.exam_landing = int(i)
        obs, _ = env.reset(seed=700 + k)
        R = env._rotation()
        starts.append(dict(idx=int(i), yaw=float(np.arctan2(R[1, 0], R[0, 0])), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), obs=obs,
                           limit=env.limit, clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, grace=env.grace_steps, lswitch=env.land_switch_at))
        for t in range(40):
            pre = dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, last_action=env.last_action.copy(), qpos=env.data.qpos.copy(),
                       qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), limit=env.limit, idx=int(i), grace=env.grace_steps, lswitch=env.land_switch_at)
            a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
            obs, r, fell, done, info = env.step(a)
            d2 = mujoco.MjData(env.model)   # up-to-date kinematics, as dump() does
            d2.qpos[:], d2.qvel[:], d2.ctrl[:] = env.data.qpos, env.data.qvel, env.data.ctrl
            mujoco.mj_forward(env.model, d2)
            d, env.data = env.data, d2
            _, f_err = env.tracking()
            env.data = d
            steps.append(dict(pre=pre, action=a, clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, limit=env.limit, fell=fell, done=done,
                              f_err=f_err, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), t=env.t))
            if fell or done:
                break
    return dict(starts=starts, steps=steps)


def check_moving(M, C):
    """The moving-start parts of check(): MjxImitate(moving_rate=1) against dump_moving's ImitateEnv tries."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    env = M.MjxImitate(randomize=False, switch_rate=0.0, moving_rate=1.0)
    names = env.names
    base = dict(kp=jp.float32(1), mass=jp.float32(1), fric=jp.float32(1), clip=jp.int32(0), tau0=jp.float32(0), t=jp.int32(0),
                last_action=jp.zeros(12), trunc=jp.float32(0), limit=jp.int32(0))
    idx = lambda c: jp.int32(names.index(M.ALL_CLIPS[c]))
    info_of = jax.jit(lambda s: env.moving_info(base, jp.bool_(True), s["post"], s["gait"], s["tau_g"], s["vscale"]))
    start_data = jax.jit(env.start_data)
    dq, bad, infos = [], 0, []
    for s in C["starts"]:
        info = info_of(dict(post=idx(s["post"]), gait=idx(s["gait"]), tau_g=jp.float32(s["tau_g"]), vscale=jp.float32(s["vscale"])))
        infos.append(info)
        bad += abs(float(info["tau0"]) - s["tau0"]) > 1e-5 or int(info["limit"]) != s["limit"] or int(info["puntil"]) != s["until"]
        d = start_data(info, jp.float32(s["yaw"]))
        q = np.array(d.qpos)
        if np.dot(q[3:7], s["qpos"][3:7]) < 0:
            q[3:7] *= -1
        o = np.asarray(env.obs(d, info))
        dq.append(max(np.abs(q - s["qpos"]).max(), np.abs(np.asarray(d.qvel) - s["qvel"]).max(), np.abs(o - s["obs"]).max()))
    print(f"[moving start] {len(dq)} moving starts (sit/lie/getup from walk/trot at 0.6 and 1.9 m/s): max |diff| qpos/qvel/obs "
          f"{max(dq):.2e} (median {np.median(dq):.1e}); clip moment/try length/time limit differ in {bad}")

    @jax.jit
    def pose(qpos, qvel, info):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        return env.pose_term(d, info)

    dp, dh, n_pose, n_pay = 0, 0, 0, 0
    for s in C["steps"]:
        p = s["pre"]
        info = dict(infos[s["start"]], clip=jp.int32(names.index(M.ALL_CLIPS[p["clip"]])), tau0=jp.float32(p["tau0"]),
                    t=jp.int32(s["t"]), phit=jp.int32(p["hit"]))
        bonus, hit, ok = pose(jp.array(s["qpos"], jp.float32), jp.array(s["qvel"], jp.float32), info)
        dp += bool(ok) != s["f_pose"]
        dh += bool(hit) != s["f_hit"]
        n_pose += s["f_pose"]
        n_pay += s["f_pose"] and s["f_hit"]
        assert abs(float(bonus) - env.pose_weight * (bool(hit) and bool(ok))) < 1e-6
    print(f"[moving steps] {len(C['steps'])} steps (CPU: in the pose in {n_pose}, paid in {n_pay}): pose decision differs in {dp}, "
          f"reached-in-time differs in {dh}")
    st = jax.jit(env.reset)(jax.random.PRNGKey(5))
    st2 = jax.jit(env.step)(st, jp.zeros(12))
    print(f"[moving reset+step] obs {st.obs.shape} -> {st2.obs.shape}, clip {names[int(st.info['clip'])]} from {names[int(st.info['mg'])]}, "
          f"finite {bool(jp.all(jp.isfinite(st2.obs)))}")
    env0 = M.MjxImitate(randomize=True)
    assert env0.M is None and env0.brake is None and "mg" not in env0.reset(jax.random.PRNGKey(0)).info


def check_transition(M, C):
    """The transition term on identical states: MjxImitate.transition_excess and its window against ImitateEnv's."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    env = M.MjxImitate(randomize=False, switch_rate=0.0, transition_weight=C["weight"], transition_window=C["window"],
                       transition_threshold=C["threshold"])

    @jax.jit
    def ex(qpos, qvel, info):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        return env.transition_excess(d, info), info["t"] - info["swt"] < env.transition_steps

    de, dw, n_on, n_pay = [], 0, 0, 0
    for s in C["steps"]:
        info = dict(clip=jp.int32(env.names.index(M.ALL_CLIPS[s["clip"]])), tau0=jp.float32(s["tau0"]), t=jp.int32(s["t"]), swt=jp.int32(s["swt"]))
        e, on = ex(jp.array(s["qpos"], jp.float32), jp.array(s["qvel"], jp.float32), info)
        de.append(abs(float(e) - s["f_ex"]))
        dw += bool(on) != s["f_on"]
        n_on += s["f_on"]
        n_pay += s["f_on"] and s["f_ex"] > 0
    print(f"[transition] {len(de)} steps ({n_on} in the window after a canter -> trot switch, {n_pay} charged): excess |diff| max {max(de):.1e} rad; "
          f"window differs in {dw}")
    st = jax.jit(env.reset)(jax.random.PRNGKey(4))
    st2 = jax.jit(env.step)(st, jp.zeros(12))
    print(f"[transition reset+step] obs {st.obs.shape} -> {st2.obs.shape}, finite {bool(jp.all(jp.isfinite(st2.obs)))}, last switch {int(st2.info['swt'])}")


def check_real(M, C):
    """Latency + motor lag: one control step of MjxImitate.physics_real against ImitateEnv._delayed_physics from identical
    states, queues and lag states; and MJX's sensor noise within its bounds."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    env = M.MjxImitate(randomize=False, switch_rate=0.0, latency_ms=C["latency"], motor_lag_ms=C["lag"])

    @jax.jit
    def one(qpos, qvel, info, a):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        d = d.replace(ctrl=jp.clip(env.ref(info, "joints", 1) + env.scale * a, env.lo, env.hi))
        d, info = env.physics_real(env.mx, d, info)
        return d.qpos, info["qhist"], info["lagy"]

    dq, dqueue, dlag = [], [], []
    for s in C["steps"]:
        p = s["pre"]
        assert p["queue"].shape[0] == env.queue_len, (p["queue"].shape, env.queue_len)
        info = dict(clip=jp.int32(env.names.index(M.ALL_CLIPS[p["clip"]])), tau0=jp.float32(p["tau0"]), t=jp.int32(p["t"]),
                    qhist=jp.array(p["queue"], jp.float32), lagy=jp.array(p["lagy"], jp.float32), dsub=jp.int32(p["dsub"]))
        q, hist, y = one(jp.array(p["qpos"], jp.float32), jp.array(p["qvel"], jp.float32), info, jp.array(s["action"], jp.float32))
        dq.append(np.abs(np.asarray(q)[7:19] - s["qpos"][7:19]).max())
        dqueue.append(np.abs(np.asarray(hist) - s["queue"]).max())
        dlag.append(np.abs(np.asarray(y) - s["lagy"]).max())
    pc = lambda x: ", ".join(f"p{k} {np.percentile(x, k):.1e}" for k in (50, 90)) + f", max {max(x):.1e}"
    print(f"[latency+lag] {len(dq)} control steps at 26 ms latency, 20 ms motor lag: joint angles |diff| {pc(dq)} rad; "
          f"target queue |diff| max {max(dqueue):.1e}, lag state |diff| max {max(dlag):.1e} rad")
    envn = M.MjxImitate(randomize=False, switch_rate=0.0, obs_noise=1.0)
    st = jax.jit(envn.reset)(jax.random.PRNGKey(1))
    o_noisy = np.asarray(envn.obs(st.pipeline_state, st.info, jax.random.PRNGKey(2)))
    o_clean = np.asarray(envn.obs(st.pipeline_state, st.info))
    lim = np.concatenate([np.full(3, M.NOISE["grav"]), np.full(3, M.NOISE["gyro"] * 0.25), np.full(3, M.NOISE["linvel"]), [M.NOISE["height"]],
                          np.full(12, M.NOISE["qpos"]), np.full(12, M.NOISE["qvel"] * 0.1)])
    diff = np.abs(o_noisy - o_clean)
    print(f"[obs noise] x1: inputs 0-33 within the levels {bool(np.all(diff[:34] <= lim + 1e-6))} (largest share of its level "
          f"{(diff[:34] / lim).max():.2f}), inputs 34-74 untouched {bool(np.all(diff[34:] == 0))}")
    envl = M.MjxImitate(randomize=True, latency_ms=(0, 40), obs_noise=1.0, motor_lag_ms=20.0)
    st = jax.jit(envl.reset)(jax.random.PRNGKey(6))
    st2 = jax.jit(envl.step)(st, jp.zeros(12))
    print(f"[real reset+step] obs {st.obs.shape} -> {st2.obs.shape}, latency {int(st.info['dsub']) * 2} ms, finite {bool(jp.all(jp.isfinite(st2.obs)))}")


def check_brake(M, C):
    """One physics substep with the motor brake on, MJX against bolt_env.physics with ImitateEnv(motor_brake=True): what
    the brake does to the joints' speed (brake-on minus brake-off substep, each side against its own), and the joint
    speeds themselves (MJX's float32 implicit-damping substep is ~1 rad/s off MuJoCo's at 25-40 rad/s, brake or not).
    Then the same with the explicit Euler integrator on both sides, where the brake's effect must match exactly."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    on, speed, gain = C["settings"]
    import mujoco
    out = {}
    for brake in (True, False):
        env = M.MjxImitate(randomize=False, switch_rate=0.0, motor_brake=brake, brake_speed=speed, brake_gain=gain)
        me = mujoco.MjModel.from_xml_string(M.collisions((M.HERE / "bolt.xml").read_text()))
        me.opt.iterations, me.opt.ls_iterations, me.opt.integrator = env.m.opt.iterations, env.m.opt.ls_iterations, mujoco.mjtIntegrator.mjINT_EULER
        for tag, mx in (("", env.mx), ("e", mjx.put_model(me))):
            sub = jax.jit(lambda qpos, qvel, ctrl: env.physics(mx, mjx.forward(mx, mjx.make_data(mx).replace(qpos=qpos, qvel=qvel, ctrl=ctrl)), 1))
            out[(brake, tag)] = np.array([np.asarray(sub(jp.array(s["pre"]["qpos"], jp.float32), jp.array(s["pre"]["qvel"], jp.float32),
                                                         jp.array(s["pre"]["ctrl"], jp.float32)).qvel)[6:18] for s in C["steps"]])
    out[True], out[False] = out[(True, "")], out[(False, "")]
    cpu_on, cpu_off = np.array([s["qvel1"][6:18] for s in C["steps"]]), np.array([s["qvel0"][6:18] for s in C["steps"]])
    effect_cpu, effect_mjx = cpu_on - cpu_off, out[True] - out[False]
    braking = np.array([np.abs(s["brake1"]).max() for s in C["steps"]])
    rel = np.abs(effect_mjx - effect_cpu).max(1) / np.maximum(np.abs(effect_cpu).max(1), 1e-9)
    pc = lambda x: f"p50 {np.percentile(x, 50):.1e}, p90 {np.percentile(x, 90):.1e}, max {x.max():.1e}"
    print(f"[brake] on above {speed:g} rad/s, {gain:g} N·m per rad/s: {len(braking)} substeps from joints at 0-40 rad/s, 1 m up (CPU brake "
          f"torque up to {braking.max():.1f} N·m). The brake's effect on joint speed in one substep: CPU up to {np.abs(effect_cpu).max():.2f} rad/s "
          f"(median of the largest {np.median(np.abs(effect_cpu).max(1)):.2f}); MJX minus CPU effect |diff| {pc(np.abs(effect_mjx - effect_cpu).max(1))} rad/s, "
          f"relative {pc(rel)}. Joint speeds themselves, MJX vs CPU: brake on {pc(np.abs(out[True] - cpu_on).max(1))}, brake off "
          f"{pc(np.abs(out[False] - cpu_off).max(1))} rad/s")
    e_cpu = np.array([s["qvel1e"][6:18] - s["qvel0e"][6:18] for s in C["steps"]])
    e_mjx = out[(True, "e")] - out[(False, "e")]
    print(f"[brake, Euler both sides] the brake's effect: CPU up to {np.abs(e_cpu).max():.2f} rad/s; MJX minus CPU |diff| "
          f"{pc(np.abs(e_mjx - e_cpu).max(1))} rad/s; joint speeds MJX vs CPU, brake on: {pc(np.abs(out[(True, 'e')] - np.array([s['qvel1e'][6:18] for s in C['steps']])).max(1))}")


def check_landing(M, C):
    """The landing parts of check(): MjxImitate(landing_rate=1) against dump_landing's ImitateEnv tries."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    env = M.MjxImitate(randomize=False, switch_rate=0.0, landing_rate=1.0)
    names = env.names

    def info_of(s, t=0, last_action=None):
        info = dict(kp=jp.float32(1), mass=jp.float32(1), fric=jp.float32(1), clip=jp.int32(names.index(M.ALL_CLIPS[s["clip"]])),
                    tau0=jp.float32(s["tau0"]), t=jp.int32(t), last_action=jp.array(np.zeros(12) if last_action is None else last_action, jp.float32),
                    trunc=jp.float32(0), limit=jp.int32(s["limit"]))
        info = env.landing_info(info, jp.int32(s["idx"]))   # the landing keys (land, grace, lswitch) as at the try's start
        return dict(info, clip=jp.int32(names.index(M.ALL_CLIPS[s["clip"]])), tau0=jp.float32(s["tau0"]), limit=jp.int32(s["limit"])) if t else info

    start_data = jax.jit(env.start_data)
    dq, bad_info = [], 0
    for s in C["starts"]:
        info = info_of(s)
        bad_info += int(info["clip"]) != names.index(M.ALL_CLIPS[s["clip"]]) or abs(float(info["tau0"]) - s["tau0"]) > 1e-5 or \
            int(info["limit"]) != s["limit"] or int(info["grace"]) != s["grace"] or int(info["lswitch"]) != s["lswitch"]
        d = start_data(info, jp.float32(s["yaw"]))
        q = np.array(d.qpos)
        if np.dot(q[3:7], s["qpos"][3:7]) < 0:
            q[3:7] *= -1
        o = np.asarray(env.obs(d, info))
        dq.append(max(np.abs(q - s["qpos"]).max(), np.abs(np.asarray(d.qvel) - s["qvel"]).max(), np.abs(o - s["obs"]).max()))
    print(f"[landing start] {len(dq)} landing starts: max |diff| qpos/qvel/obs {max(dq):.2e} (median {np.median(dq):.1e}); "
          f"clip/moment/limit/grace/switch step differ in {bad_info}")

    @jax.jit
    def one_step(qpos, qvel, ctrl, info, a):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        st = env.State(pipeline_state=d, obs=jp.zeros(75), reward=jp.float32(0), done=jp.float32(0),
                       metrics={f"r_{k}": jp.float32(0) for k in M.WEIGHTS} | dict(fell=jp.float32(0), canter=jp.float32(0), canter_steps=jp.float32(0)),
                       info=info | dict(rng=jax.random.PRNGKey(0)))
        n = env.step(st, a)
        return n.obs, n.done, n.info, n.metrics["fell"]

    sw, dn, near, n_sw, dp = 0, 0, 0, 0, []
    for s in C["steps"]:
        p = s["pre"]
        info = info_of(p, t=p["t"], last_action=p["last_action"])
        o, done, info2, fell = one_step(jp.array(p["qpos"], jp.float32), jp.array(p["qvel"], jp.float32), jp.array(p["ctrl"], jp.float32), info,
                                        jp.array(s["action"], jp.float32))
        self_switch = s["clip"] != p["clip"]
        n_sw += self_switch
        if not bool(done):
            sw += int(info2["clip"]) != names.index(M.ALL_CLIPS[s["clip"]]) or abs(float(info2["tau0"]) - s["tau0"]) > 1e-4 or int(info2["limit"]) != s["limit"]
            dp.append(np.abs(np.asarray(o)[10:22] - (s["qpos"][7:19] - M.DOG)).max())
        if bool(done) != (s["fell"] or s["done"]):
            # a float32 step may land on the other side of a termination threshold (15 cm / 0.8 rad)
            e = s["f_err"]
            near += min(abs(abs(e["height_err"]) - 0.15), abs(e["tilt"] - 0.8)) < 0.02
            dn += 1
    print(f"[landing steps] {len(C['steps'])} steps ({n_sw} bridge switches to trot): clip/moment/end step differ in {sw}; "
          f"end of try differs in {dn} ({near} within 2 cm / 0.02 rad of a limit); joint angles |diff| max {max(dp):.1e} rad")
    # rate 0 builds no landing keys and draws nothing extra
    env0 = M.MjxImitate(randomize=True)
    assert env0.L is None and "land" not in env0.reset(jax.random.PRNGKey(0)).info
    # a landing start through reset, one step: the 75 inputs, nothing else
    st = jax.jit(env.reset)(jax.random.PRNGKey(3))
    st2 = jax.jit(env.step)(st, jp.zeros(12))
    print(f"[landing reset+step] obs {st.obs.shape} -> {st2.obs.shape}, landing state {int(st.info['land'])}, finite {bool(jp.all(jp.isfinite(st2.obs)))}")


def dump_root_global(pol, rng):
    """Tries with root_global=1 (Peng's global root terms), a forced clip switch mid-try in some (check_root_global):
    per step the reference root before and after, and the two terms recomputed on up-to-date kinematics."""
    import mujoco
    from imitate_env import ALL_CLIPS, ImitateEnv
    clips = [c for c in ALL_CLIPS if c != "getup"]
    env = ImitateEnv(randomize=False, clips=clips, slots=8, root_global=1.0)
    steps = []
    for k in range(8):
        name = ("trot", "pace", "walk", "canter", "turn_left", "turn_right", "trot", "walk")[k]
        env.exam_start = (name, float(rng.uniform(0, env.clips[name].length)))
        obs, _ = env.reset(seed=1900 + k)
        for t in range(40):
            if k % 2 and t == 20:   # a forced switch: the reference root carries on (ImitateEnv does not restart it)
                env._switch_to(env.clips["trot" if name != "trot" else "walk"])
            ref = getattr(env, "_ref_root", None)
            pre = None if ref is None or ref[2] != env._try_id else (ref[0].copy(), float(ref[1]))
            a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
            obs, r, fell, done, info = env.step(a)
            post = (env._ref_root[0].copy(), float(env._ref_root[1]))
            d2 = mujoco.MjData(env.model)
            d2.qpos[:], d2.qvel[:], d2.ctrl[:] = env.data.qpos, env.data.qvel, env.data.ctrl
            mujoco.mj_forward(env.model, d2)
            d, env.data = env.data, d2
            saved = env._ref_root
            env._ref_root = None if pre is None else [pre[0].copy(), pre[1], env._try_id]
            f_root, f_vel = env._global_root_terms()
            f_post = (env._ref_root[0].copy(), float(env._ref_root[1]))
            env._ref_root, env.data = saved, d
            steps.append(dict(clip=ALL_CLIPS.index(env.clip.name), tau0=env.tau0, t=env.t, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                              pre=pre, post=post, f_post=f_post, f_root=float(f_root), f_vel=float(f_vel),
                              r_root=info["r_root"], r_root_vel=info["r_root_vel"]))
            if fell or done:
                break
    return dict(steps=steps, clips=clips)


def check_root_global(M, C):
    """Peng's global root terms on identical states and reference roots: MjxImitate.global_root_terms against
    ImitateEnv._global_root_terms (the terms, the new reference root, and its start at a try's first step)."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    env = M.MjxImitate(randomize=False, switch_rate=0.0, clips=C["clips"], root_global=1.0)

    @jax.jit
    def terms(qpos, qvel, info):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        g, v, info = env.global_root_terms(d, info)
        return g, v, info["gp"], info["gyaw"]

    dr, dv, dp, dy, n_first, n_sw = [], [], [], [], 0, 0
    prev = None
    for s in C["steps"]:
        first = s["pre"] is None
        n_first += first
        n_sw += prev is not None and not first and s["clip"] != prev
        prev = s["clip"]
        gp0, gy0 = (np.zeros(2), 0.0) if first else s["pre"]
        info = dict(clip=jp.int32(env.names.index(M.ALL_CLIPS[s["clip"]])), tau0=jp.float32(s["tau0"]), t=jp.int32(s["t"]),
                    gp=jp.array(gp0, jp.float32), gyaw=jp.float32(gy0), ginit=jp.int32(0 if first else 1))
        g, v, gp, gy = terms(jp.array(s["qpos"], jp.float32), jp.array(s["qvel"], jp.float32), info)
        dr.append(abs(float(g) - s["f_root"]))
        dv.append(abs(float(v) - s["f_vel"]))
        dp.append(np.abs(np.asarray(gp) - s["f_post"][0]).max())
        dy.append(abs(float(gy) - s["f_post"][1]))
    print(f"[root global] {len(dr)} steps ({n_first} try starts, {n_sw} switches): root |diff| max {max(dr):.1e}, root_vel |diff| max {max(dv):.1e}; "
          f"reference root position |diff| max {max(dp):.1e} m, heading {max(dy):.1e} rad")
    st = jax.jit(env.reset)(jax.random.PRNGKey(3))
    step = jax.jit(env.step)
    for _ in range(3):
        st = step(st, jp.zeros(12))
    print(f"[root global reset+step] finite obs {bool(jp.all(jp.isfinite(st.obs)))}, reward {float(st.reward):.3f}, "
          f"reference root set {int(st.info['ginit'])}, r_root {float(st.metrics['r_root']):.3f}")
    return max(dr), max(dv), max(dp), max(dy)


def dump(zip_path, warm_path, out):
    import mujoco
    from imitate_env import ALL_CLIPS, ImitateEnv, ImitatePolicy
    from mjx_imitate import MjxImitatePolicy
    z = Path(zip_path)
    pol = ImitatePolicy(z.parent, z.name, z.name.replace("policy_", "policy_vecnormalize_", 1).replace(".zip", ".pkl"))
    npol = MjxImitatePolicy(warm_path)
    rng = np.random.default_rng(0)
    env = ImitateEnv(randomize=False, clips=ALL_CLIPS)
    starts, steps, switches = [], [], []
    for ci, name in enumerate(ALL_CLIPS):
        c = env.clips[name]
        for k in range(3):
            tau0 = float(rng.uniform(0, c.length if c.loop else c.length + 0.3))
            env.exam_start = (name, tau0)
            obs, _ = env.reset(seed=100 * ci + k)
            tilt = env.ref("tilt")
            u, _, vt = np.linalg.svd(tilt)
            M = env._rotation() @ (u @ vt).T
            starts.append(dict(clip=ci, tau0=tau0, yaw=float(np.arctan2(M[1, 0], M[0, 0])), qpos=env.data.qpos.copy(),
                               qvel=env.data.qvel.copy(), obs=obs, limit=env.limit))
            for t in range(40):
                pre = dict(clip=ci, tau0=env.tau0, t=env.t, last_action=env.last_action.copy(), qpos=env.data.qpos.copy(),
                           qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy())
                a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
                obs, r, fell, done, info = env.step(a)
                d = env.data
                # the same senses with up-to-date kinematics (after mj_step, xmat and geom_xpos are from before the last
                # substep, in MuJoCo and MJX alike); the JAX side recomputes them with mjx.forward
                d2 = mujoco.MjData(env.model)
                d2.qpos[:], d2.qvel[:], d2.ctrl[:] = d.qpos, d.qvel, d.ctrl
                mujoco.mj_forward(env.model, d2)
                env.data = d2
                f_obs, (f_terms, f_err), f_bof = env._obs(), env.tracking(), env.body_on_floor()
                env.data = d
                legleg = sum(1 for cc in d.contact[:d.ncon] if cc.geom1 != env.floor and cc.geom2 != env.floor)
                steps.append(dict(pre=pre, action=a, obs=obs, f_obs=f_obs, f_terms=f_terms, f_err=f_err, f_bof=f_bof, reward=r, terms={k: info[f"r_{k}"] for k in ("pose", "vel", "paw", "root", "root_vel")},
                                  height_err=info["height_err"], tilt=info["tilt"], body_on_floor=env.body_on_floor(), fell=fell,
                                  qpos=d.qpos.copy(), qvel=d.qvel.copy(), legleg=legleg, t=env.t))
                if t == 20 and (c.loop or env.finished() > 0.5):   # a forced switch from here
                    env2 = ImitateEnv(randomize=False, clips=ALL_CLIPS, switch_rate=1e9)
                    env2.data.qpos[:] = d.qpos
                    env2.clip, env2.tau0, env2.t, env2.rng = env.clip, env.tau0, env.t, np.random.default_rng(t + ci)
                    env2._maybe_switch()
                    switches.append(dict(clip=ci, tau0=env.tau0, t=env.t, q=d.qpos[7:19].copy(), new=ALL_CLIPS.index(env2.clip.name),
                                         new_tau0=env2.tau0, new_limit=env2.limit))
                if fell or done:
                    break
    # 20 observations from training-like tries (randomized), for the policy check
    envr = ImitateEnv(randomize=True, clips=ALL_CLIPS)
    obs_set = []
    for k in range(20):
        o, _ = envr.reset(seed=500 + k)
        for _ in range(int(rng.integers(0, 30))):
            o, _, f, dn, _ = envr.step(pol.act(o))
            if f or dn:
                break
        obs_set.append(o)
    obs_set = np.array(obs_set)
    acts = np.array([pol.act(o) for o in obs_set])
    nacts = np.array([npol.act(o) for o in obs_set])
    print(f"numpy brain vs ImitatePolicy.act: max |diff| {np.abs(acts - nacts).max():.2e}")
    Path(out).write_bytes(pickle.dumps(dict(starts=starts, steps=steps, switches=switches, obs=obs_set, acts=acts, landing=dump_landing(pol, rng),
                                            moving=dump_moving(pol, rng), brake=dump_brake(pol, rng),
                                            transition=dump_transition(pol, rng), real=dump_real(pol, rng),
                                            terrain=dump_terrain(pol, rng), shove=dump_shove(rng),
                                            root_global=dump_root_global(pol8(), rng))))
    print(f"{len(starts)} starts, {len(steps)} steps, {len(switches)} switches -> {out}")


def check(warm_path, cases):
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from brax.training.agents.ppo import networks as ppo_networks
    import mjx_imitate as M
    C = pickle.loads(Path(cases).read_bytes())
    env = M.MjxImitate(randomize=False, switch_rate=0.0)

    def info_of(s):
        return dict(kp=jp.float32(1), mass=jp.float32(1), fric=jp.float32(1), clip=jp.int32(s["clip"]), tau0=jp.float32(s["tau0"]),
                    t=jp.int32(s["t"]), last_action=jp.array(s["last_action"], jp.float32), trunc=jp.float32(0), limit=jp.int32(0))

    @jax.jit
    def senses(qpos, qvel, info):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
        terms, err = env.tracking(d, info)
        return env.obs(d, info), terms, err, env.body_on_floor(d)

    @jax.jit
    def one_step(qpos, qvel, ctrl, info, a):
        d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
        st = env.State(pipeline_state=d, obs=jp.zeros(75), reward=jp.float32(0), done=jp.float32(0),
                       metrics={f"r_{k}": jp.float32(0) for k in M.WEIGHTS} | dict(fell=jp.float32(0), canter=jp.float32(0), canter_steps=jp.float32(0)),
                       info=info | dict(rng=jax.random.PRNGKey(0), limit=jp.int32(10 ** 6)))
        n = env.step(st, a)
        return n.pipeline_state.qpos, n.obs

    start_data = jax.jit(env.start_data)
    # 1. observations and reward terms for identical states
    do, dt, de, flo, n_body = [], {k: [] for k in M.WEIGHTS}, [], 0, 0
    for s in C["steps"]:
        info = info_of(dict(s["pre"], t=s["t"], last_action=s["action"]))
        o, terms, err, bof = senses(jp.array(s["qpos"], jp.float32), jp.array(s["qvel"], jp.float32), info)
        do.append(np.abs(np.asarray(o) - s["f_obs"]))
        for k in M.WEIGHTS:
            dt[k].append(abs(float(terms[k]) - s["f_terms"][k]))
        de.append(abs(float(err["tilt"]) - s["f_err"]["tilt"]) + abs(float(err["height_err"]) - s["f_err"]["height_err"]))
        flo += bool(bof) != bool(s["f_bof"])
        n_body += bool(s["f_bof"])
    do = np.array(do)
    worst = np.argsort(do.max(0))[::-1][:3]
    print(f"[obs] {len(do)} states: max |diff| {do.max():.2e} (worst inputs {list(worst)}: {[f'{do.max(0)[i]:.1e}' for i in worst]}), mean {do.mean():.1e}")
    print("[reward terms] max |diff| " + ", ".join(f"{k} {max(v):.1e}" for k, v in dt.items()) + f"; errors (tilt+height) {max(de):.1e}")
    print(f"[body on floor] disagree in {flo} of {len(do)} states (CPU: body on floor in {n_body})")
    # 2. reference-state start
    dq = []
    for s in C["starts"]:
        info = info_of(dict(s, t=0, last_action=np.zeros(12)))
        d = start_data(info, jp.float32(s["yaw"]))
        q = np.array(d.qpos)
        qq = s["qpos"].copy()
        if np.dot(q[3:7], qq[3:7]) < 0:
            q[3:7] *= -1
        dq.append(max(np.abs(q - qq).max(), np.abs(np.asarray(d.qvel) - s["qvel"]).max()))
        o = np.asarray(env.obs(d, info))
        dq[-1] = max(dq[-1], np.abs(o - s["obs"]).max())
    print(f"[start] {len(dq)} reference-state starts: max |diff| qpos/qvel/obs {max(dq):.2e} (median {np.median(dq):.1e})")
    # 3. clip switches
    bad = 0
    for s in C["switches"]:
        info = info_of(dict(s, last_action=np.zeros(12)))
        tau0, limit = env.switch_to(info, jp.array(s["q"], jp.float32), jp.int32(s["new"]))
        bad += abs(float(tau0) - s["new_tau0"]) > 1e-4 or int(limit) != s["new_limit"]
    print(f"[switch] {len(C['switches'])} forced switches: {bad} differ in join time or end step")
    # 4. one control step of physics (float32 MJX vs float64 MuJoCo; MJX has the floor and thigh-torso contacts only)
    dp, do1, wi = [], [], []
    for s in C["steps"][::3]:
        p = s["pre"]
        q, o = one_step(jp.array(p["qpos"], jp.float32), jp.array(p["qvel"], jp.float32), jp.array(p["ctrl"], jp.float32), info_of(p),
                        jp.array(s["action"], jp.float32))
        dp.append(np.abs(np.asarray(q)[7:19] - s["qpos"][7:19]).max())
        e = np.abs(np.asarray(o) - s["obs"])
        do1.append(e.max())
        wi.append(int(e.argmax()))
    pc = lambda x: ", ".join(f"p{k} {np.percentile(x, k):.1e}" for k in (50, 90, 99)) + f", max {max(x):.1e}"
    print(f"[one step] {len(dp)} steps: joint angles |diff| {pc(dp)} rad; obs |diff| {pc(do1)} (worst inputs {np.bincount(wi, minlength=75).argsort()[::-1][:3].tolist()}); "
          f"CPU self-contacts (thigh-torso) in {sum(s['legleg'] > 0 for s in C['steps'])} of {len(C['steps'])} states")
    if "landing" in C:
        check_landing(M, C["landing"])
    if "moving" in C:
        check_moving(M, C["moving"])
    if "brake" in C:
        check_brake(M, C["brake"])
    if "transition" in C:
        check_transition(M, C["transition"])
    if "real" in C:
        check_real(M, C["real"])
    if "terrain" in C:
        check_terrain(M, C["terrain"])
    if "shove" in C:
        check_shove(M, C["shove"])
    if "root_global" in C:
        check_root_global(M, C["root_global"])
    # 5. warm-started Brax policy vs ImitatePolicy.act
    W = pickle.loads(Path(warm_path).read_bytes())
    nets = M.make_networks(W["hidden"], float(np.mean(W["log_std"])))(75, 12)
    params = M.to_brax(W)
    act = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    a = np.array([np.clip(np.asarray(act(jp.array(o), jax.random.PRNGKey(0))[0]), -1, 1) for o in C["obs"]])
    print(f"[warm start] Brax deterministic action vs ImitatePolicy.act on {len(a)} obs: max |diff| {np.abs(a - C['acts']).max():.2e}")
    v = np.asarray(nets.value_network.apply(params[0], params[2], jp.array(C["obs"])))
    print(f"[warm start] critic on them: {v.min():.2f} .. {v.max():.2f} (SB3 normalized-return units)")


class Go2TablesTest(unittest.TestCase):
    """mjx_imitate's copies for the Go2 path equal imitate_env's (CPU stack only; mjx_imitate imports without JAX)."""

    def test_go2_clip_tables(self):
        import mjx_imitate as M
        from imitate_env import ALL_CLIPS, Clip
        T = M.clip_tables(ALL_CLIPS, prefix="go2_")
        for i, name in enumerate(ALL_CLIPS):
            c = Clip(name, "go2_")
            self.assertEqual((T["n"][i], T["fps"][i], T["loop"][i]), (c.n, c.fps, c.loop))
            for k in M.KEYS:
                np.testing.assert_allclose(T[k][i][:c.n], getattr(c, k), atol=1e-12, err_msg=f"{name} {k}")

    def test_steer_constants(self):
        import imitate_env
        import mjx_imitate as M
        self.assertEqual(M.STEER, imitate_env.STEER)
        self.assertEqual(M.STEER_CLIPS, imitate_env.STEER_CLIPS)


def dump_go2(run_dir, warm_path, out, per_clip=2, steps=30, seed=0):
    """The Go2 body (ImitateEnv(body="go2"), motor brake on), steering off and on: per clip `per_clip` tries of up to `steps`
    steps of the brain in run_dir (plus noise), each with its start state, the actions, the env's outputs, and the senses
    recomputed on up-to-date kinematics (as dump); then 20 randomized observations for the warm-start check."""
    import os
    import mujoco
    os.environ["BOLT_BODY"] = "go2"   # ImitatePolicy's normalizer env, as train_imitate.py --body go2
    from imitate_env import ALL_CLIPS, ImitateEnv, ImitatePolicy
    from mjx_imitate import MjxImitatePolicy
    pol, npol = ImitatePolicy(run_dir), MjxImitatePolicy(warm_path)
    rng = np.random.default_rng(seed)
    cases = {}
    for steer_max in (0.0, 1.0):
        env = ImitateEnv(body="go2", randomize=False, clips=ALL_CLIPS, motor_brake=True, steer_max=steer_max)
        assert env.motor_brake[0] and env.body == "go2"
        tries = []
        for ci, name in enumerate(ALL_CLIPS):
            c = env.clips[name]
            for k in range(per_clip):
                tau0 = float(rng.uniform(0, c.length if c.loop else c.length + 0.3))
                env.exam_start = (name, tau0)
                env.exam_steer = (0.8 if k % 2 else -0.8) if steer_max else None
                obs, _ = env.reset(seed=3000 + 100 * ci + k)
                u, _, vt = np.linalg.svd(env.ref("tilt"))
                Rm = env._rotation() @ (u @ vt).T
                start = dict(clip=ci, tau0=tau0, steer=env.steer, yaw=float(np.arctan2(Rm[1, 0], Rm[0, 0])), qpos=env.data.qpos.copy(),
                             qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy(), obs=obs, limit=env.limit)
                rows = []
                for t in range(steps):
                    pre = dict(clip=ci, tau0=env.tau0, t=env.t, limit=env.limit, steer=env.steer, last_action=env.last_action.copy(), qpos=env.data.qpos.copy(),
                               qvel=env.data.qvel.copy(), ctrl=env.data.ctrl.copy())
                    a = np.clip(pol.act(obs) + rng.normal(0, 0.1, 12), -1, 1)
                    obs, r, fell, done, info = env.step(a)
                    d = env.data
                    d2 = mujoco.MjData(env.model)   # the senses on up-to-date kinematics (as dump)
                    d2.qpos[:], d2.qvel[:], d2.ctrl[:] = d.qpos, d.qvel, d.ctrl
                    mujoco.mj_forward(env.model, d2)
                    env.data = d2
                    f_obs, (f_terms, f_err), f_bof = env._obs(), env.tracking(), env.body_on_floor()
                    env.data = d
                    f_fell = abs(f_err["height_err"]) > env.lost_h or f_err["tilt"] > 0.8 or (f_bof and not env.ref("height") < env.floor_ok_h)
                    selfc = sum(1 for cc in d.contact[:d.ncon] if env.floor not in (cc.geom1, cc.geom2))
                    rows.append(dict(pre=pre, action=a, obs=obs, reward=r, fell=fell, done=done, terms={k: info[f"r_{k}"] for k in f_terms},
                                     f_obs=f_obs, f_terms=f_terms, f_err=f_err, f_bof=f_bof, f_fell=bool(f_fell), f_reward=sum(env.weights.get(k, env.steer_weight) * v for k, v in f_terms.items()),
                                     qpos=d.qpos.copy(), qvel=d.qvel.copy(), selfc=selfc, t=env.t))
                    if fell or done:
                        break
                tries.append(dict(start=start, rows=rows))
        cases[steer_max] = tries
    # the end-of-try decision near its limits: states of those tries moved down/up (-20..+5 cm) and tilted (0..1.2 rad)
    edges = []
    for t in cases[0.0]:
        for r in t["rows"][::3]:
            env.clip, env.tau0, env.t = env.clips[ALL_CLIPS[r["pre"]["clip"]]], r["pre"]["tau0"], r["t"]
            d = env.data
            d.qpos[:], d.qvel[:] = r["qpos"], r["qvel"]
            d.qpos[2] += rng.uniform(-0.2, 0.05)
            ax = rng.normal(size=3)
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, ax / np.linalg.norm(ax), rng.uniform(0, 1.2))
            mujoco.mju_mulQuat(d.qpos[3:7], q, r["qpos"][3:7].copy())
            mujoco.mj_forward(env.model, d)
            terms, err = env.tracking()
            bof = env.body_on_floor()
            fell = abs(err["height_err"]) > env.lost_h or err["tilt"] > 0.8 or (bof and not env.ref("height") < env.floor_ok_h)
            edges.append(dict(clip=r["pre"]["clip"], tau0=env.tau0, t=env.t, qpos=d.qpos.copy(), qvel=d.qvel.copy(), f_fell=bool(fell),
                              f_bof=bool(bof), f_err=err))
    envr = ImitateEnv(body="go2", randomize=True, clips=ALL_CLIPS, motor_brake=True)
    obs_set = []
    for k in range(20):
        o, _ = envr.reset(seed=3500 + k)
        for _ in range(int(rng.integers(0, 30))):
            o, _, f, dn, _ = envr.step(pol.act(o))
            if f or dn:
                break
        obs_set.append(o)
    obs_set = np.array(obs_set)
    acts, nacts = np.array([pol.act(o) for o in obs_set]), np.array([npol.act(o) for o in obs_set])
    print(f"[go2] numpy brain vs ImitatePolicy.act: max |diff| {np.abs(acts - nacts).max():.2e}")
    n = sum(len(t["rows"]) for v in cases.values() for t in v)
    Path(out).write_bytes(pickle.dumps(dict(cases=cases, edges=edges, obs=obs_set, acts=acts, clips=ALL_CLIPS)))
    print(f"[go2] {sum(len(v) for v in cases.values())} tries, {n} steps -> {out}")


def check_go2(warm_path, cases):
    """MjxImitate(body="go2") against the dump_go2 cases, steering off and on (module docstring)."""
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from brax.training.agents.ppo import networks as ppo_networks
    import mjx_imitate as M
    C = pickle.loads(Path(cases).read_bytes())
    out = {}
    for steer_max, tries in C["cases"].items():
        env = M.MjxImitate(body="go2", randomize=False, switch_rate=0.0, motor_brake=True, steer_max=steer_max)
        env100 = M.MjxImitate(body="go2", randomize=False, switch_rate=0.0, motor_brake=True, steer_max=steer_max, iterations=100, ls_iterations=50)
        tag = f"go2, steer {'off' if not steer_max else f'max {steer_max:g}'}"

        def info_of(s, e=env):
            i = dict(kp=jp.float32(1), mass=jp.float32(1), fric=jp.float32(1), clip=jp.int32(s["clip"]), tau0=jp.float32(s["tau0"]),
                     t=jp.int32(s["t"]), last_action=jp.array(s["last_action"], jp.float32), trunc=jp.float32(0), limit=jp.int32(s.get("limit", 10 ** 6)))
            if e.steer_max:
                i["steer"] = jp.float32(s["steer"])
            return i

        @jax.jit
        def senses(qpos, qvel, info):
            d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
            terms, err = env.tracking(d, info)
            fell = (jp.abs(err["height_err"]) > env.lost_h) | (err["tilt"] > 0.8) | (env.body_on_floor(d) & ~(env.ref(info, "height") < env.floor_ok_h))
            reward = sum(env.weights.get(k, env.steer_weight) * v for k, v in terms.items())
            return env.obs(d, info), terms, err, env.body_on_floor(d), fell, reward

        def state_of(e, d, info):
            metrics = {f"r_{k}": jp.float32(0) for k in M.WEIGHTS} | dict(fell=jp.float32(0), canter=jp.float32(0), canter_steps=jp.float32(0))
            if e.steer_max:
                metrics["r_steer"] = jp.float32(0)
            return e.State(pipeline_state=d, obs=jp.zeros(75), reward=jp.float32(0), done=jp.float32(0), metrics=metrics,
                           info=info | dict(rng=jax.random.PRNGKey(0)))

        steps = {}
        for name, e in (("", env), ("100", env100)):
            @jax.jit
            def one_step(qpos, qvel, ctrl, info, a, e=e):
                d = mjx.forward(e.mx, mjx.make_data(e.mx).replace(qpos=qpos, qvel=qvel, ctrl=ctrl))
                n = e.step(state_of(e, d, info), a)
                return n.pipeline_state.qpos, n.obs, n.reward, n.done
            steps[name] = one_step
        rows = [r for t in tries for r in t["rows"]]
        # 1. senses, reward terms, total reward and end of try on identical states
        do, dt, dr, dfell, nfell, ddone_env = [], {}, [], 0, 0, 0
        for r in rows:
            info = info_of(dict(r["pre"], t=r["t"], last_action=r["action"]))
            o, terms, err, bof, fell, rew = senses(jp.array(r["qpos"], jp.float32), jp.array(r["qvel"], jp.float32), info)
            do.append(np.abs(np.asarray(o) - r["f_obs"]))
            assert set(terms) == set(r["f_terms"]), (set(terms), set(r["f_terms"]))
            for k in terms:
                dt.setdefault(k, []).append(abs(float(terms[k]) - r["f_terms"][k]))
            dr.append(abs(float(rew) - r["f_reward"]))
            dfell += bool(fell) != r["f_fell"]
            nfell += r["f_fell"]
            ddone_env += bool(fell) != r["fell"]
        do = np.array(do)
        worst = np.argsort(do.max(0))[::-1][:3]
        print(f"[{tag}: senses] {len(do)} states: obs max |diff| {do.max():.2e} (worst inputs {worst.tolist()}: {[f'{do.max(0)[i]:.1e}' for i in worst]}); "
              "reward terms max |diff| " + ", ".join(f"{k} {max(v):.1e}" for k, v in dt.items()) + f"; reward {max(dr):.1e}; "
              f"end of try disagrees in {dfell} (CPU ends {nfell}; vs the CPU step's own flag, on stale kinematics: {ddone_env})")
        # 2. reference-state starts
        dq = []
        for t in tries:
            s = t["start"]
            info = info_of(dict(s, t=0, last_action=np.zeros(12)))
            d = jax.jit(env.start_data)(info, jp.float32(s["yaw"]))
            q = np.array(d.qpos)
            if np.dot(q[3:7], s["qpos"][3:7]) < 0:
                q[3:7] *= -1
            dq.append(max(np.abs(q - s["qpos"]).max(), np.abs(np.asarray(d.qvel) - s["qvel"]).max(), np.abs(np.asarray(env.obs(d, info)) - s["obs"]).max()))
        print(f"[{tag}: start] {len(dq)} reference-state starts: max |diff| qpos/qvel/obs {max(dq):.2e} (median {np.median(dq):.1e})")
        # 3. one control step of physics from identical states (MJX solver 5/8 iterations, and MuJoCo's 100/50)
        res = {}
        for name, one_step in steps.items():
            dp, do1, drw, dd = [], [], [], 0
            for r in rows:
                p = r["pre"]
                q, o, rw, dn = one_step(jp.array(p["qpos"], jp.float32), jp.array(p["qvel"], jp.float32), jp.array(p["ctrl"], jp.float32),
                                        info_of(p), jp.array(r["action"], jp.float32))
                dp.append(np.abs(np.asarray(q)[7:19] - r["qpos"][7:19]).max())
                do1.append(np.abs(np.asarray(o) - r["obs"]).max() if not (r["fell"] or r["done"]) else 0.0)   # (MJX restarts a finished try)
                drw.append(abs(float(rw) - r["reward"]))
                dd += bool(dn) != (r["fell"] or r["done"])
            res[name] = (dp, do1, drw, dd)
        pc = lambda x: ", ".join(f"p{k} {np.percentile(x, k):.1e}" for k in (50, 90, 99)) + f", max {max(x):.1e}"
        for name, (dp, do1, drw, dd) in res.items():
            print(f"[{tag}: one step{', solver 100/50' if name else ''}] {len(dp)} steps: joint angles |diff| {pc(dp)} rad; obs |diff| {pc(do1)}; "
                  f"reward |diff| {pc(drw)}; end of try disagrees in {dd}")
        print(f"[{tag}] CPU self-contacts (absent in MJX) in {sum(r['selfc'] > 0 for r in rows)} of {len(rows)} states")
        # 4. whole tries replayed open-loop in MJX with the CPU actions: obs, reward, end of try per step
        step = jax.jit(env.step)
        by_k, rsum, ends = {}, [], [0, 0]
        for t in tries:
            s = t["start"]
            info = info_of(dict(s, t=0, last_action=np.zeros(12), limit=s["limit"]))
            d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=jp.array(s["qpos"], jp.float32), qvel=jp.array(s["qvel"], jp.float32),
                                                                  ctrl=jp.array(s["ctrl"], jp.float32)))
            st = state_of(env, d, info)
            rc, rm = 0.0, 0.0
            for k, r in enumerate(t["rows"]):
                st = step(st, jp.array(r["action"], jp.float32))
                rc, rm = rc + r["reward"], rm + float(st.reward)
                end_cpu, end_mjx = r["fell"] or r["done"], bool(st.done)
                if end_cpu or end_mjx:
                    ends[0] += 1
                    ends[1] += end_cpu == end_mjx
                    break
                by_k.setdefault(k + 1, []).append((np.abs(np.asarray(st.obs) - r["obs"]).max(), abs(float(st.reward) - r["reward"])))
            rsum.append(abs(rc - rm) / max(len(t["rows"]), 1))
        hz = [h for h in (1, 2, 5, 10, 20, 30) if h in by_k]
        print(f"[{tag}: open-loop tries] {len(tries)} tries: " + "; ".join(
            f"step {h}: obs |diff| median {np.median([x[0] for x in by_k[h]]):.1e} max {max(x[0] for x in by_k[h]):.1e}, reward |diff| max {max(x[1] for x in by_k[h]):.1e} (n {len(by_k[h])})" for h in hz)
            + f"; reward per step over the try |diff| max {max(rsum):.3f}; ends {ends[0]}: the same step in {ends[1]}")
        out[steer_max] = dict(obs=do.max(), terms={k: max(v) for k, v in dt.items()}, reward=max(dr), fell=dfell)
        # 5. steering draws (training): share with a command, range, redraw rate
        if steer_max:
            ks = jax.random.split(jax.random.PRNGKey(5), 20000)
            v = np.asarray(jax.vmap(env.draw_steer)(ks))
            print(f"[{tag}: draws] {len(v)} draws: share with a command {np.mean(v != 0):.3f} (want {env.steer_share:g}), range {v.min():.3f} .. {v.max():.3f}, "
                  f"mean {v.mean():+.3f}")
            st = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(6), 64))
            step_v = jax.jit(jax.vmap(env.step))
            s0, changes, n = np.asarray(st.info["steer"]), 0, 0
            for _ in range(50):
                st = step_v(st, jp.zeros((64, 12)))
                s1 = np.asarray(st.info["steer"])
                live = np.asarray(st.done) == 0
                changes += int(np.sum((s1 != s0) & live))
                n += int(live.sum())
                s0 = s1
            print(f"[{tag}: redraws] {changes} command changes in {n} live env-steps: {changes / max(n, 1) / env.dt:.2f}/s "
                  f"(want ~{env.steer_change * env.steer_share + env.steer_change * (1 - env.steer_share) * env.steer_share:.2f}/s: a redraw changes "
                  "the command unless 0 -> 0)")
    # 6. the end-of-try decision near its limits (moved and tilted states; MJX's without self-contacts)
    if "edges" in C:
        env = M.MjxImitate(body="go2", randomize=False, switch_rate=0.0)

        @jax.jit
        def ends(qpos, qvel, info):
            d = mjx.forward(env.mx, mjx.make_data(env.mx).replace(qpos=qpos, qvel=qvel))
            _, err = env.tracking(d, info)
            bof = env.body_on_floor(d)
            return (jp.abs(err["height_err"]) > env.lost_h) | (err["tilt"] > 0.8) | (bof & ~(env.ref(info, "height") < env.floor_ok_h)), bof, err

        dis, dbof, n_end, n_bof, near = 0, 0, 0, 0, 0
        for e in C["edges"]:
            info = dict(clip=jp.int32(e["clip"]), tau0=jp.float32(e["tau0"]), t=jp.int32(e["t"]))
            fell, bof, err = ends(jp.array(e["qpos"], jp.float32), jp.array(e["qvel"], jp.float32), info)
            bad = bool(fell) != e["f_fell"]
            dis += bad
            dbof += bool(bof) != e["f_bof"]
            n_end += e["f_fell"]
            n_bof += e["f_bof"]
            near += bad and (abs(abs(e["f_err"]["height_err"]) - env.lost_h) < 2e-3 or abs(e["f_err"]["tilt"] - 0.8) < 2e-3)
        print(f"[go2: end of try near its limits] {len(C['edges'])} moved/tilted states (CPU ends {n_end}, body on floor {n_bof}): "
              f"end-of-try decision differs in {dis} ({near} within 2 mm / 2 mrad of a limit), body-on-floor in {dbof}")
    # 7. the warm-started Go2 brain in Brax against ImitatePolicy.act
    W = pickle.loads(Path(warm_path).read_bytes())
    nets = M.make_networks(W["hidden"], float(np.mean(W["log_std"])))(75, 12)
    params = M.to_brax(W)
    act = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    a = np.array([np.clip(np.asarray(act(jp.array(o), jax.random.PRNGKey(0))[0]), -1, 1) for o in C["obs"]])
    print(f"[go2 warm start] body {W.get('body', 'bolt')}: Brax deterministic action vs ImitatePolicy.act on {len(a)} obs: max |diff| {np.abs(a - C['acts']).max():.2e}")
    return out


def pol8():
    """The 8-slot brain the heading pilot started from (run 18g 35.5M), for the root-global tries."""
    from imitate_env import ImitatePolicy
    return ImitatePolicy(HERE.parent / "models" / "bolt-dog-canter-switching-18g-35.5M", "policy.zip", "vecnormalize.pkl")


if __name__ == "__main__":
    if sys.argv[1] == "dump-go2":               # <cpu python> test_mjx_imitate.py dump-go2 <run dir> <warm.pkl> <cases.pkl>
        dump_go2(*sys.argv[2:5])
    elif sys.argv[1] == "check-go2":            # <jax python> test_mjx_imitate.py check-go2 <warm.pkl> <cases.pkl>
        check_go2(*sys.argv[2:4])
    elif sys.argv[1] == "dump-root-global":       # <cpu python> test_mjx_imitate.py dump-root-global <cases.pkl>
        Path(sys.argv[2]).write_bytes(pickle.dumps(dict(root_global=dump_root_global(pol8(), np.random.default_rng(0)))))
    elif sys.argv[1] == "check-root-global":    # <jax python> test_mjx_imitate.py check-root-global <cases.pkl>
        import mjx_imitate
        check_root_global(mjx_imitate, pickle.loads(Path(sys.argv[2]).read_bytes())["root_global"])
    else:
        dump(*sys.argv[2:5]) if sys.argv[1] == "dump" else check(*sys.argv[2:4])
