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
                                            moving=dump_moving(pol, rng), brake=dump_brake(pol, rng))))
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
    # 5. warm-started Brax policy vs ImitatePolicy.act
    W = pickle.loads(Path(warm_path).read_bytes())
    nets = M.make_networks(W["hidden"], float(np.mean(W["log_std"])))(75, 12)
    params = M.to_brax(W)
    act = jax.jit(ppo_networks.make_inference_fn(nets)(params, deterministic=True))
    a = np.array([np.clip(np.asarray(act(jp.array(o), jax.random.PRNGKey(0))[0]), -1, 1) for o in C["obs"]])
    print(f"[warm start] Brax deterministic action vs ImitatePolicy.act on {len(a)} obs: max |diff| {np.abs(a - C['acts']).max():.2e}")
    v = np.asarray(nets.value_network.apply(params[0], params[2], jp.array(C["obs"])))
    print(f"[warm start] critic on them: {v.min():.2f} .. {v.max():.2f} (SB3 normalized-return units)")


if __name__ == "__main__":
    dump(*sys.argv[2:5]) if sys.argv[1] == "dump" else check(*sys.argv[2:4])
