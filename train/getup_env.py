"""Teach Bolt to get up after a fall: from lying on its side, on its back or half-tipped over, roll over
and stand in the dog stance, by itself.

In the games a fall used to end with Bolt teleported back onto its feet. A real robot dog has to get up
on its own, so this brain learns it from scratch, with the full motor range (like the jump brain).

    python train_getup.py --name getup1 --steps 15000000

Each try: Bolt is dropped from 0.5 m in a random fallen pose (any roll, legs anywhere) and lands; then
it gets 4 s. It is paid for being upright, then for standing at its height in the dog stance, and for
standing still once up; jerky or wasteful motor use costs a little. A tenth of the tries start already
standing, slightly tipped, so it also learns not to fall over while standing.

Run 17 learned to lie still: a charge on joint speed (0.001 x speed^2, all 12 joints) cost 1-5 per step
while flailing to roll over, far more than being upright paid. A search over motor sequences (no
learning) found a way to roll over from the side with this body, so it is possible. Run 17b: no
joint-speed charge, a fifth of the effort charge, a bonus for each new best uprightness (progress), and
more exploration at the start.

Run 17c (goal="selfright"), the way real robot dogs recover (ANYmal: Lee, Hwangbo & Hutter 2019, a
self-righting skill, then a standing skill, then walking, picked by a selector): this brain only rolls
Bolt onto its belly with its legs tucked and lies still. Bolt's posture brain already stands up from
there (20/20 from the lying pose, 17/20 with the legs 0.6 rad off it). Nothing pays for height.
By 6M it rolled from its back past its side (up to 0.53) but always settled back at a 0.16 tilt. Two
limits: the brain could only command the sideways hip joints to +-0.4 rad (ACTION_SCALE, made for
walking), and the joints stopped at +-0.5 rad (real robot dogs: Unitree Go2 about +-0.84). Run 17d:
Bolt's sideways hip range is +-0.9 rad (the other brains never command more than +-0.4 there), and the
self-righting brain's actions cover each joint's whole range around the dog stance.

Other bodies (body="go2" or BOLT_BODY=go2; python train_getup.py --body go2): the same task on a robot of
robot_bodies.BODIES, its stance at the centre of the actions and inputs, every Bolt-sized number of the starts, reward
and exams taken from getup_sizes (below; Bolt's numbers are unchanged). The full recovery (exam_full: self-right, then
Bolt's posture brain tucks and stands) is Bolt's only: there is no posture brain for another body.
"""
import os

import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bolt_env
from bolt_env import ACTION_SCALE, LIE_POSE, STANCES, BoltEnv, physics

DOG = STANCES["dog"]
STAND_HEIGHT = 0.43


EXAM_LATENCY_MS = 0.0   # fixed control delay (ms) in exams (fair_exam.py --latency-ms)

# ---------- body sizes (body="go2", BOLT_BODY=go2; robot_bodies.BODIES) ----------
# Every length and kick of the starts, the reward and the exams was tuned on Bolt (12.0 kg, stands at 0.436 m in the dog
# stance). On another body they come from getup_sizes(body); for Bolt every value is exactly the old one (sizes()).
# Factors, as jump_env.jump_sizes (rounded to 2 places, so a brain's task never shifts with a model rebuild):
#   kz  vertical: the body's stand height / Bolt's (robot_bodies stand_height / 0.436). Go2: 0.304 / 0.436 = 0.70
#       - the standing-tall target 0.43 m and its width 0.08 m (Go2 0.301 / 0.056: as jump_env, the stance's geometric
#         height, which a Go2 brain reaches by pushing, not the height its servos sag to), standing starts from 0.45 m
#         (Go2 0.315)
#   "standing" (exams, the stillness pay, stood_at) = upright (up > 0.9) and the body above 0.38 m on Bolt: 87% of the
#       0.437 m its stiff motors hold the stance at (measured, 2 s with no brain; 0.436-0.438 m at kp 75-125%, torso
#       mass x1.3). The Go2's kp 40 servos sag: 0.261 m nominal, 0.244 at kp 75%, 0.231 at kp 75% and torso x1.3 (the
#       hard physics' weakest), 0.269 at kp 125%. 0.38 x kz = 0.266 would fail a Go2 holding its stance, so its
#       threshold is GETUP_STANDING_Z: 0.22 m, below the weakest held height and 85% of the nominal one. Lying it is
#       at 0.06-0.15 m (back, belly), so 0.22 still tells lying from standing.
#   kv  kick speeds (m/s: the shove, the second kick, sliding/drifting starts): Bolt's x 0.80 on the Go2.
#   kw  kick spins (rad/s: the shove's roll, the second kick's, tumbling/sliding starts): Bolt's x 1.0 on the Go2.
#       A kick is a velocity change, so mass does not enter it directly. Go2 (15.2 kg, torso 6.9 kg) is heavier than
#       Bolt (12.0 kg, torso 9.0 kg): the same boot (impulse) would give it 12.0 / 15.2 = 0.79 x the speed, and Froude
#       scaling for a body 0.70 x as tall gives sqrt(0.70) = 0.84, so speeds x 0.80. Spins: the same angular impulse
#       would spin the Go2 ~3x faster (its trunk's roll inertia 0.107 kg m^2 vs Bolt's 0.370), Froude x 1.2; neither
#       is a measurement of a real kick, so spins stay Bolt's. Checked, not tuned (60 shoved starts each, the
#       motors holding the stance for 2 s after the kick): every kick tips both bodies past 60 degrees, Bolt in a
#       median 10 steps (0.2 s), the Go2 in 8 (kw 0.4-0.7 would give 9-10). They come to rest differently: Bolt on
#       its side 72%, back 28%; the Go2 on its back ~70% whatever the kick (kv 0.6-1.0, kw 0.4-1.2 all gave 63-95%):
#       holding its stance it does not stay on its side, a body property, so its shoved starts are mostly back starts.
# Unscaled (ASSUMPTIONS): the fallen starts' drop height 0.5 m and the bad-landing drops 0.55-1.0 m (the Go2's legs are
# as long as Bolt's, 2 x 0.213 vs 2 x 0.2 m, and with a leg stuck out it reaches 0.62 m from the trunk centre, Bolt
# 0.68: a kz-scaled 0.35 m drop would start legs in the floor); fall roll/pitch angles; the 60-degree "fallen" check;
# the uprightness thresholds (0.9 standing, R[2, 2] < 0.5 tipped); the "still" speeds (0.2-0.5 m/s, rad/s); joint-angle
# terms (the stance and lying-pose closeness, in rad: the Go2's own stance and robot_bodies' "lie" pose replace Bolt's);
# the action scale (bolt_env ACTION_SCALE, as jump_env; clipped to the Go2's joint ranges); the physics randomization
# (relative). The effort cost 1e-5 x force^2 is per joint x (33.5 / that joint's peak torque)^2 (Bolt: 33.5 N·m on
# every joint), so full effort costs the same: Go2 hip/thigh 2.0e-5 (23.7 N·m), calf 5.4e-6 (45.43 N·m).
BOLT_STAND, BOLT_TORQUE = 0.436, 33.5
GETUP_KICK = {"go2": (0.80, 1.0)}   # (kv, kw) per body: see above
GETUP_STANDING_Z = {"go2": 0.22}    # m: "standing" height per body (Bolt: 0.38 x kz); see above


def sizes(kz=1.0, kv=1.0, kw=1.0, torque=None, standing_z=None):
    """The body-dependent numbers of the get-up (Bolt: every factor 1, every value exactly Bolt's as before)."""
    return dict(
        kz=kz, kv=kv, kw=kw,
        stand_z=STAND_HEIGHT * kz, stand_w=0.08 * kz, start_stand_z=0.45 * kz,
        standing_z=0.38 * kz if standing_z is None else standing_z,
        start_fallen_z=0.5, drop_z=(0.55, 1.0),
        shove=(1.5 * kv, 4.5 * kv), shove_spin=(3.0 * kw, 12.0 * kw),          # the shoved / ice starts
        kick2=(0.5 * kv, 2.0 * kv), kick2_spin=(1.0 * kw, 5.0 * kw),           # the second kick while getting up
        drift=1.5 * kv, tumble=3.0 * kw,                                       # bad landings: drifting, tumbling
        slide=1.5 * kv, slide_spin=2.0 * kw,                                   # lying starts still sliding
        effort=None if torque is None else 1e-5 * (BOLT_TORQUE / np.asarray(torque, dtype=float)) ** 2)


BOLT_SIZES = sizes()


def getup_sizes(body=None):
    """sizes() for a body (None: BOLT_BODY, else bolt). Go2: see the block above."""
    body = body or os.environ.get("BOLT_BODY", "bolt")
    if body == "bolt":
        return BOLT_SIZES
    import robot_bodies
    r = robot_bodies.get(body)
    kv, kw = GETUP_KICK.get(body, (1.0, 1.0))
    return sizes(round(r.stand_height / BOLT_STAND, 2), kv, kw, torque=r.motor["torque"], standing_z=GETUP_STANDING_Z.get(body))


class GetUpEnv(BoltEnv):
    OBS = 46

    def __init__(self, randomize=True, seed=None, max_steps=200, goal="stand", hard=False, motor_brake=None,
                 latency_ms=(0.0, 0.0), motor_lag_ms=(0.0, 0.0), drop=(0.0, 0.0),
                 latency_jitter_ms=0.0, latency_spike_p=0.0, delay_input=False, body=None):
        # body: "bolt" (default) or a robot of robot_bodies.BODIES ("go2"); None = the BOLT_BODY env var, else "bolt".
        # Heights, kicks and the effort cost come from the body's sizes (getup_sizes); the stance and lying pose are its own.
        # delay_input (30 Oct, as jump_env): one more input, the delay Bolt measures (bolt_env measured_delay_ms / 40 ms)
        self.delay_input = delay_input
        self.OBS = 46 + bool(delay_input)
        super().__init__(randomize=randomize, max_steps=max_steps, seed=seed, gait_generator="dog", skills=["stand"],
                         motor_brake=motor_brake, body=body)
        self.G = getup_sizes(self.body)
        self.dog = DOG if self.body == "bolt" else self.robot.stance          # the action's centre and inputs 7-18
        self.lie_pose = LIE_POSE if self.body == "bolt" else self.robot.lie   # selfright's belly-down, legs-tucked pose
        self.effort = self.G["effort"]   # None: Bolt's single 1e-5 for every joint
        # hard=True (run "recover", from 17e): tougher falls and physics, see HARD_KINDS and _hard_physics
        self.hard = hard
        self.shove_at, self.ice = None, False
        self.goal = goal   # "stand": get all the way up; "selfright": belly down, legs tucked, still
        # self-righting reaches each joint's whole range around the dog stance (walking's ACTION_SCALE
        # only reached +-0.4 rad sideways)
        self.scale = np.maximum(self.hi - self.dog, self.dog - self.lo) if goal == "selfright" else ACTION_SCALE
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.exam_start = None   # "side", "back", "tipped" or "standing": exams set it; training draws it
        # latency_ms=(lo, hi): the motors get the targets sent that long ago, drawn per try (as imitate_env and
        # jump_env; 29 Oct: at 20 ms MasterMind's recovery stood up 16/30). Exams: EXAM_LATENCY_MS (fair_exam --latency-ms)
        self.latency_ms = tuple(latency_ms)
        # motor_lag_ms, drop = (lo, hi): motor lag and lost control packets per try; exams use bolt_env.EXAM_GAPS
        self.motor_lag_range, self.drop_range = tuple(motor_lag_ms), tuple(drop)
        # latency_jitter_ms: per try the delay wobbles +- up to U(0, this) ms per step; latency_spike_p: per try a
        # spike chance in U(0, this) per step (bolt_env.start_delay)
        self.jitter_max, self.spike_max = float(latency_jitter_ms), float(latency_spike_p)

    def _obs(self):
        d, R = self.data, self._rotation()
        obs = np.concatenate([
            R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3], [d.qpos[2]],
            d.qpos[7:19] - self.dog, d.qvel[6:18] * 0.1, self.last_action,
        ])
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)
        if self.delay_input:   # after the sensor noise (it has its own clock error), so the random stream is as before
            obs = np.append(obs, self.measured_delay_ms() / 40.0)
        return obs.astype(np.float32)

    def _fallen_pose(self, kind):
        """Roll and pitch (radians) for a start: on its side, on its back, half-tipped, or standing."""
        r = self.rng
        if kind == "side":
            return r.choice([-1, 1]) * r.uniform(1.2, 1.9), r.uniform(-0.4, 0.4)
        if kind == "back":
            return r.choice([-1, 1]) * r.uniform(2.4, np.pi), r.uniform(-0.4, 0.4)
        if kind == "tipped":
            return r.choice([-1, 1]) * r.uniform(0.5, 1.2), r.uniform(-0.6, 0.6)
        return r.uniform(-0.15, 0.15), r.uniform(-0.15, 0.15)

    def _hard_physics(self, kind):
        """Run "recover": wider physics than the other brains (motor stiffness 75-125%, every body's mass
        90-110% and the torso's 80-130%, friction 40-140%), and a fifth of the tries on ice (all friction
        12-30% of normal: MuJoCo takes the larger friction of a contact pair, so the paws must be slippery too)."""
        m, r = self.model, self.rng
        kp = self.nominal_kp * r.uniform(0.75, 1.25)
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
        m.body_mass[:] = self.nominal_mass * r.uniform(0.9, 1.1, self.nominal_mass.shape)
        m.body_mass[self.torso] = self.nominal_mass[self.torso] * r.uniform(0.8, 1.3)
        self.ice = kind == "ice" or (self.exam_start is None and r.random() < 0.2)
        m.geom_friction[:, 0] = self.nominal_friction * (r.uniform(0.12, 0.3) if self.ice else r.uniform(0.4, 1.4))

    def _shove(self, speed, spin):
        """A kick: body velocity sideways (and a little forward/back) and a roll (and some pitch/yaw) rate."""
        r, d = self.rng, self.data
        R = self._rotation()
        s = r.choice([-1, 1])
        d.qvel[0:3] += R @ np.array([r.uniform(-0.4, 0.4) * speed, s * speed, r.uniform(0.0, 0.3) * speed])
        d.qvel[3:6] += np.array([-s * spin, r.uniform(-0.3, 0.3) * spin, r.uniform(-0.3, 0.3) * spin])

    def _hard_start(self, kind):
        """shoved / ice: standing in the dog stance, kicked (1.5-4.5 m/s, rolling 3-12 rad/s), the motors
        holding the stance until it has fallen (body tipped past 60 degrees, at most 1.5 s), as when MasterMind's
        fall detector hands over. dropped: a bad landing, from 0.55-1.0 m in any roll, pitched up to 45
        degrees, tumbling and drifting, under control from the start. Lying starts slide and spin half the time."""
        m, d, r, G = self.model, self.data, self.rng, self.G
        if kind in ("shoved", "ice"):
            d.qpos[2], d.qpos[7:19] = G["start_stand_z"], np.clip(self.dog + r.normal(0, 0.03, 12), self.lo, self.hi)
            d.ctrl[:] = d.qpos[7:19]
            mujoco.mj_forward(m, d)
            for _ in range(25):
                physics(m, d, self.frame_skip)
            self._shove(r.uniform(*G["shove"]), r.uniform(*G["shove_spin"]))
            for _ in range(75):
                physics(m, d, self.frame_skip)
                if self._rotation()[2, 2] < 0.5:
                    break
            return
        if kind == "dropped":
            d.qpos[2] = r.uniform(*G["drop_z"])
            mujoco.mj_forward(m, d)
            d.qvel[0:2] = r.uniform(-G["drift"], G["drift"], 2)
            d.qvel[3:6] = r.normal(0, G["tumble"], 3)
            return
        for _ in range(50):   # lying kinds: as before, 1 s to drop and come to rest
            physics(m, d, self.frame_skip)
        if r.random() < 0.5:   # still sliding and rolling when the brain takes over
            d.qvel[0:2] += r.uniform(-G["slide"], G["slide"], 2)
            d.qvel[3:6] += r.normal(0, G["slide_spin"], 3)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.hard:
            kind = self.exam_start or self.rng.choice(HARD_KINDS, p=HARD_P)
            self._hard_physics(kind)
            self._place(kind)
            self._hard_start(kind)
            # a second kick while getting up, in a third of the training tries (never in exams)
            self.shove_at = int(self.rng.integers(25, 150)) if self.exam_start is None and self.rng.random() < 0.35 else None
            return self._begin(kind)
        if self.randomize:   # the same physics randomization as the other brains
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)
        kind = self.exam_start or self.rng.choice(["side", "back", "tipped", "standing"], p=[0.35, 0.3, 0.25, 0.1])
        self._place(kind)
        for _ in range(50):   # 1 s to drop and come to rest, the motors holding that pose
            physics(m, d, self.frame_skip)
        return self._begin(kind)

    def _place(self, kind):
        """Bolt in the start pose of this kind of fall (not yet simulated)."""
        m, d = self.model, self.data
        if kind in ("shoved", "ice"):
            roll, pitch = 0.0, 0.0
        elif kind == "dropped":
            roll, pitch = self.rng.uniform(-np.pi, np.pi), self.rng.uniform(-0.8, 0.8)
        else:
            roll, pitch = self._fallen_pose(kind)
        yaw = self.rng.uniform(-np.pi, np.pi)
        # roll about the body's x, then pitch about y, then yaw about the vertical
        qr = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
        qp = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
        qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qp, qr)
        mujoco.mju_mulQuat(q, qy, q.copy())
        d.qpos[3:7] = q
        if kind == "standing":
            legs = self.dog + self.rng.normal(0, 0.03, 12)
            d.qpos[2] = self.G["start_stand_z"]
        else:   # legs anywhere: halfway between the dog stance and a random pose within the joint limits
            legs = 0.5 * self.dog + 0.5 * self.rng.uniform(self.lo, self.hi)
            d.qpos[2] = self.G["start_fallen_z"]
        d.qpos[7:19] = d.ctrl[:] = np.clip(legs, self.lo, self.hi)
        mujoco.mj_forward(m, d)

    def _begin(self, kind):
        self.kind = kind
        self.t, self.last_action = 0, np.zeros(12)
        lat = EXAM_LATENCY_MS if self.exam_start is not None else (
            self.rng.uniform(*self.latency_ms) if self.latency_ms[1] > 0 else 0.0)
        if self.exam_start is not None:
            lag, drop = bolt_env.EXAM_GAPS["lag_ms"], bolt_env.EXAM_GAPS["drop"]
        else:
            lag, drop = (self.rng.uniform(*r) if r[1] > 0 else 0.0 for r in (self.motor_lag_range, self.drop_range))
        if self.exam_start is not None:
            jit, sp = bolt_env.EXAM_GAPS["jitter_ms"], bolt_env.EXAM_GAPS["spike_p"]
        else:
            jit, sp = self.rng.uniform(0, self.jitter_max), self.rng.uniform(0, self.spike_max)
        self.start_delay(lat, lag, drop, jitter_ms=jit, spike_p=sp, spike_ms=bolt_env.SPIKE_MS if sp else 0.0)
        self.stood_at = None
        self.best_up = self._rotation()[2, 2]
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        G = self.G
        self.data.ctrl[:] = np.clip(self.dog + self.scale * a, self.lo, self.hi)   # the full motor range
        if self.shove_at is not None and self.t == self.shove_at:   # knocked again while getting up
            self._shove(self.rng.uniform(*G["kick2"]), self.rng.uniform(*G["kick2_spin"]))
        self._physics_step()
        self.t += 1
        self.env_steps += 1
        d, R = self.data, self._rotation()
        p = self.posture()
        up = R[2, 2]                                           # 1 upright, -1 on its back
        upright = max(up, 0.0)
        tall = np.exp(-((p["height"] - G["stand_z"]) / G["stand_w"]) ** 2)
        stance = np.exp(-np.sum((d.qpos[7:19] - self.dog) ** 2) / 1.0)
        speed = np.hypot(p["vx"], p["vy"])
        reward = 1.0 * (up + 1) / 2                           # turning over
        reward += 5.0 * max(0.0, up - self.best_up)            # progress: each new best uprightness
        self.best_up = max(self.best_up, up)
        if self.goal == "selfright":   # belly down, legs tucked under, still: ready for the posture brain
            fold = np.exp(-np.sum((d.qpos[7:19] - self.lie_pose) ** 2) / 2.0)
            reward += 1.0 * fold * upright ** 2
            still = np.exp(-(np.linalg.norm(d.qvel[0:3]) / 0.2) ** 2) * np.exp(-(np.linalg.norm(d.qvel[3:6]) / 0.5) ** 2)
            reward += 0.5 * still * upright ** 2
            standing = up > 0.9 and np.linalg.norm(d.qvel[0:3]) < 0.2   # "righted"
        else:
            reward += 1.0 * tall * upright                         # standing up to its height
            reward += 0.5 * stance * upright ** 2                  # in the dog stance
            standing = up > 0.9 and p["height"] > G["standing_z"]
        if standing and self.goal != "selfright":
            reward += 0.5 * np.exp(-(speed / 0.3) ** 2) * np.exp(-(p["yaw_rate"] / 0.5) ** 2)   # and still
        if standing and self.stood_at is None:
            self.stood_at = self.t
        effort = 1e-5 * np.sum(d.actuator_force ** 2) if self.effort is None else np.sum(self.effort * d.actuator_force ** 2)
        reward -= 0.01 * np.sum((a - self.last_action) ** 2) + effort
        self.last_action = a
        done = self.t >= self.max_steps
        info = dict(p, up=up, standing=standing, stood_at=self.stood_at, kind=self.kind)
        return self._obs(), float(reward), False, done, info


# run "recover": the kinds of fall in training (hard=True) and their shares; "ice" is a shove on ice
HARD_KINDS = ["side", "back", "tipped", "standing", "shoved", "dropped", "ice"]
HARD_P = [0.15, 0.15, 0.12, 0.05, 0.23, 0.2, 0.1]
EXAM_KINDS = ("side", "back", "tipped", "shoved", "dropped", "ice")


class GetUpPolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl", body=None):
        """body: the robot the brain drives (None: a saved brain's own -- the folder's best.json / model.json "body", bolt
        when it records none, and a models/ folder (MODEL.md) without either is Bolt's --, else BOLT_BODY, else bolt).
        Bolt and Go2 brains have the same inputs, so the body cannot be told from the network."""
        import json
        from pathlib import Path
        run_dir = Path(run_dir)
        saved = [run_dir / f for f in ("best.json", "model.json") if (run_dir / f).exists()]
        if body is None and saved:   # a saved brain (run's best/ or models/ folder): its own body; none recorded = Bolt
            body = json.loads(saved[0].read_text()).get("body") or "bolt"
        elif body is None and (run_dir / "MODEL.md").exists():   # models/ folders of the Bolt era carry only MODEL.md
            body = "bolt"
        self.body = body or os.environ.get("BOLT_BODY", "bolt")
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.delay_input = self.model.observation_space.shape[0] == 47   # 30 Oct on: the measured delay as an input
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: GetUpEnv(
            delay_input=self.delay_input, body=self.body)]))
        self.norm.training = False

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, episodes=10, seed=7000, kinds=("side", "back", "tipped", "standing"), goal="stand", hard=False,
         motor_brake=None):
    """From each kind of fall: standing (upright, at height) at the end of the 4 s, and the median time it
    took to first stand up. The policy's body (GetUpPolicy.body, else BOLT_BODY) is the exam's: "standing" is upright
    and above that body's standing height (getup_sizes; Bolt 38 cm)."""
    out = {}
    env = GetUpEnv(randomize=True, goal=goal, hard=hard, motor_brake=motor_brake,   # motor_brake None: process default
                   delay_input=getattr(policy, "delay_input", False), body=getattr(policy, "body", None))
    for kind in kinds:
        wins, times = 0, []
        for k in range(episodes):
            env.exam_start = kind
            obs, _ = env.reset(seed=seed + k)
            info = {}
            for _ in range(env.max_steps):
                obs, _, _, done, info = env.step(policy.act(obs))
                if done:
                    break
            ok = bool(info["standing"])
            wins += ok
            if ok and info["stood_at"] is not None:
                times.append(info["stood_at"] * env.dt)
        out[f"up_{kind}"] = wins / episodes
        out[f"secs_{kind}"] = round(float(np.median(times)), 2) if times else None
    return out


_POSTURE = None


def full_recovery(env, righter, obs):
    """The whole recovery MasterMind and the games use (learnwatch.recovery, skills.Bolt.recover): self-right
    until belly-down and still for 0.3 s (at most 4 s), tuck with the posture brain's "lie" (until every joint
    is within 0.6 rad of the lying pose, at most 2 s), then its "stand" for 3 s. Returns (stood at the end:
    upright and body above 38 cm, seconds from the hand-over to first standing like that or None).
    Bolt only: the tuck and stand are Bolt's posture brain (BoltEnv's command tasks), there is none for other bodies."""
    global _POSTURE
    assert env.body == "bolt", "the full recovery uses Bolt's posture brain; other bodies have none"
    from pathlib import Path
    from bolt_env import LIE_POSE
    from evaluate import Policy
    if _POSTURE is None:
        _POSTURE = Policy(Path(__file__).resolve().parent.parent / "models" / "bolt-dog-posture-getup-3M")
    env.max_steps = 10 ** 6
    t, first, calm = 0, None, 0

    def check():
        nonlocal first
        p = env.posture()
        if first is None and p["up"] > 0.9 and p["height"] > 0.38:
            first = t * env.dt
    for _ in range(200):
        obs, _, _, _, info = env.step(righter.act(obs))
        t += 1
        calm = calm + 1 if info["standing"] else 0
        if calm >= 15:
            break
    for cmd, steps in (("lie", 100), ("stand", 150)):
        env.forced_command = cmd   # forced, as skills.Bolt does: BoltEnv.step never draws a new command
        env.set_command(cmd)
        env.last_action = np.zeros(12)
        for _ in range(steps):
            BoltEnv.step(env, _POSTURE.act(BoltEnv._obs(env)))
            t += 1
            if cmd == "stand":
                check()
            if cmd == "lie" and np.abs(env.data.qpos[7:19] - LIE_POSE).max() < 0.6:
                break
    p = env.posture()
    ok = bool(p["up"] > 0.9 and p["height"] > 0.38)
    return ok, (first if ok else None)


def exam_full(righter, episodes=10, seed=7000, kinds=EXAM_KINDS, hard=True, motor_brake=None):
    """Fair full-recovery exam: from each kind of fall (hard=True: run "recover"'s tougher falls and physics,
    the same fixed seeds for every brain, no second kick), stood_<kind>: share of tries standing at the end;
    tsec_<kind>: median seconds from the hand-over to first standing (self-right + tuck + stand)."""
    out = {}
    env = GetUpEnv(randomize=True, goal="selfright", hard=hard, motor_brake=motor_brake,   # motor_brake None: process default
                   delay_input=getattr(righter, "delay_input", False), body=getattr(righter, "body", None))
    for kind in kinds:
        wins, times = 0, []
        for k in range(episodes):
            env.exam_start = kind
            env.max_steps = 200
            obs, _ = env.reset(seed=seed + k)
            ok, sec = full_recovery(env, righter, obs)
            wins += ok
            if sec is not None:
                times.append(sec)
        out[f"stood_{kind}"] = wins / episodes
        out[f"tsec_{kind}"] = round(float(np.median(times)), 2) if times else None
    return out
