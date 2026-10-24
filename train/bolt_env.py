"""Training ground: teach Bolt to obey commands, the way you'd train a puppy.

Every 1/50 s the policy (a small neural network) sees what the dog feels
(tilt, spin, joint angles, which command it was given) and picks 12 motor
targets. It gets a reward for doing what was asked: the right height and
posture for "sit" or "lie", moving forward for "walk"; and a penalty for
jerky, wasteful movement or falling over. Reinforcement learning (PPO)
nudges the network toward whatever earned more reward. Millions of tries
later, it has learned the tricks.

To make sure it learns the trick and not this one exact simulation, every
episode randomizes the body mass, floor friction and motor strength.
"""
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

MODEL = Path(__file__).with_name("bolt.xml")

COMMANDS = ["stand", "sit", "lie", "walk"]


def pose(front_hip, front_knee, rear_hip, rear_knee):
    """Joint targets for all 12 motors (abd, hip, knee per leg). Front legs bend opposite to rear legs."""
    front, rear = [0.0, front_hip, -front_knee], [0.0, -rear_hip, rear_knee]
    return np.array(front * 2 + rear * 2)


_A = 0.8128  # hip angle for the standing pose: feet straight under the hips
STAND_POSE = pose(_A, 2 * _A, _A, 2 * _A)
SIT_POSE = pose(0.55, 1.1, 1.75, 2.7)    # rear legs folded, rump down, front legs braced
LIE_POSE = pose(1.35, 2.7, 1.35, 2.7)    # all four legs folded under the body

# What each command should look like. Heights and pitches were MEASURED by holding each pose
# in simulation (poses.py), not guessed: body-centre height (m), nose-up pitch (rad), forward
# speed (m/s), and the joint pose to aim for.
TARGETS = {
    "stand": dict(height=0.315, pitch=0.0, speed=0.0, pose=STAND_POSE.round(4).tolist()),
    "sit": dict(height=0.208, pitch=0.54, speed=0.0, pose=SIT_POSE.round(4).tolist()),
    "lie": dict(height=0.131, pitch=0.0, speed=0.0, pose=LIE_POSE.round(4).tolist()),
    "walk": dict(height=0.29, pitch=0.0, speed=0.5, pose=None),
}
ACTION_SCALE = np.tile([0.4, 1.0, 1.2], 4)  # radians per unit action: abd, hip, knee

# What the built-in trot is built around (gait_generator="crouch" or "tall"). "crouch" is the
# standing pose, a Z-shaped crouch with knees ~14 cm up: the walk looked like crawling and the
# brain pushed knees onto the floor (runs 5-6). "tall" is a dog-like stance, front legs straighter
# than rear; held open-loop it carries the body at ~40 cm with knees ~19 cm up, trots faster
# with less motor force, and scrapes nothing (measured, see docs/LESSONS.md).
# "dog" plans the trot by the paw's path (reference_gait.paw_trot), the way real dogs and cats
# walk: paws stay under the body, lower legs near upright, swing on a low arc. With no brain it
# tilts the lower legs ~20° instead of ~33°, trots at 0.47 m/s instead of 0.27, and needs a
# third less motor force than "tall" (measured). Run 7's lower legs sloped to 55°.
STANCES = {"crouch": STAND_POSE, "tall": pose(0.35, 0.7, 0.55, 1.1), "dog": pose(0.2747, 0.5493, 0.2747, 0.5493)}
WALK_HEIGHT = {None: TARGETS["walk"]["height"], "crouch": TARGETS["walk"]["height"], "tall": 0.40, "dog": 0.435}
PAW_GAIT = dict(step=0.125, lift=0.06, height=0.385)  # m: stride length, paw lift, paw depth below the hip


def trot(phase, stance="crouch", turn=0.0):
    from reference_gait import paw_trot, trot as reference_trot  # imported late: reference_gait imports this module
    if stance == "dog":
        return paw_trot(phase, **PAW_GAIT, turn=turn)
    return reference_trot(phase, hip_swing=0.15, knee_lift=0.3, base=STANCES[stance])
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class BoltEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, randomize=True, command_period=200, max_steps=600, render_mode=None, seed=None, scale=1.0,
                 skills=None, gait_generator=False, turning=False):
        # gait_generator: while walking, the reference trot (reference_gait.py) drives the legs and the
        # brain only adds corrections on top (residual control, as in "policies modulating trajectory
        # generators"). The legs are coordinated from the first step instead of having to discover it.
        # True means "crouch", the stance runs 5-6 were trained on; new brains use "tall".
        self.stance = "crouch" if gait_generator is True else (gait_generator or None)
        self.gait_generator = bool(self.stance)
        self.walk_height = WALK_HEIGHT[self.stance]
        # With the "dog" stance the dog also STANDS that way, so standing and walking match and
        # a walk can start without first rising 12 cm (run 8 fell 2 in 10 hand-overs from the
        # old crouch). Held by the motors it settles at 43.6 cm, level, all four paws down, with
        # under half the motor force of the crouch (measured with poses.settle).
        self.targets = {k: dict(v) for k, v in TARGETS.items()}
        if self.stance == "dog":
            self.targets["stand"].update(height=0.436, pose=STANCES["dog"].round(4).tolist())
        # skills: which commands this brain practises, e.g. ["walk"] for a walk-only brain. The
        # observation layout always keeps all commands, so every brain can be swapped in instantly.
        self.skills = [COMMANDS.index(c) for c in (skills or COMMANDS)]
        # scale < 1 builds a smaller, dynamically similar Bolt (see scale_robot.py). Everything the
        # policy observes and is scored on is converted back to full-size units, so the same brain
        # can drive any size.
        self.s, self.ts = scale, np.sqrt(scale)  # length and time scale
        if scale == 1.0:
            self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        else:
            from scale_robot import scaled_xml
            self.model = mujoco.MjModel.from_xml_string(scaled_xml(scale))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10  # 0.002 s physics step × 10 = 50 Hz control
        self.dt = self.model.opt.timestep * self.frame_skip
        self.randomize = randomize
        self.command_period = command_period
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.renderer = None
        self.forced_command = None
        self.rng = np.random.default_rng(seed)
        self.total_steps = 0  # this env's steps across all episodes (drives the walk curriculum)
        self.t, self.command, self.walk_since, self.phase0 = 0, 0, 0, 0.0
        self.forced_turn_set = False
        self.walk_from = STAND_POSE.copy()  # motor targets the walk eases in from

        m = self.model
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.nominal_kp = m.actuator_gainprm[:, 0].copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_friction = m.geom_friction[:, 0].copy()
        self.torso = m.body("torso").id
        self.foot_ids = [m.geom(f).id for f in FEET]
        self.floor = m.geom("floor").id
        # which leg each geom belongs to (0-3), or -1 for the body (torso, head, snout)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in m.geom_bodyid]
        self.geom_leg = np.array([["FL", "FR", "RL", "RR"].index(n[:2]) if n[:2] in ("FL", "FR", "RL", "RR") else -1
                                  for n in names])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        # turning=True: the walk also gets a turn command (rad/s, + is left) as a 53rd input.
        # Only the dog gait can turn: its paw paths take longer strides on the outside of the turn.
        self.turning = turning
        self.yaw_cmd = 0.0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (53 if turning else 52,), np.float32)

    # ---------- helpers ----------
    RESIDUAL_SCALE = 0.35  # with the gait generator on, the brain's corrections are this fraction of normal

    def walking_on_generator(self):
        return self.gait_generator and COMMANDS[self.command] == "walk"

    def gait_phase(self):
        return 2 * np.pi * 2.0 * self.t * self.dt / self.ts + self.phase0  # 2 Hz gait clock in full-size time

    def base_pose(self):
        """What the motors do when the brain outputs zero."""
        if not self.walking_on_generator():
            # The dog-stance stand is centred on that stance. Centred on the crouch, holding it
            # needed motor commands at the very edge of their range (run 9b's stand pinned its
            # outputs at ±1.0 and stalled 7 cm short). Sit and lie keep the crouch centre they
            # were learned with.
            if self.stance == "dog" and COMMANDS[self.command] == "stand":
                return STANCES["dog"]
            return STAND_POSE
        q = trot(self.gait_phase(), self.stance, self.gait_turn())[0]
        if self.stance in ("tall", "dog"):
            # Ease into the trot over half a second from walk_from (see _switch); runs 5-6 jumped
            # straight in and the 2M brain fell 9 times in 10 when taking over from standing.
            q = self.walk_from + min(1.0, (self.t - self.walk_since) / 25) * (q - self.walk_from)
        return q

    TURN_GAIN = 0.075  # m of extra stride on the outside per rad/s of turn (measured: 0.04 m -> ~0.56 rad/s)
    MAX_TURN = 0.6     # rad/s (~35°/s): strides stay within what the paw-path gait handled without a brain

    def gait_turn(self):
        return float(np.clip(self.TURN_GAIN * self.yaw_cmd, -0.05, 0.05)) if self.turning else 0.0

    def _new_turn(self):
        """A turn command like an owner would give: often straight, otherwise left or right."""
        return 0.0 if self.rng.random() < 0.3 else float(self.rng.uniform(-self.MAX_TURN, self.MAX_TURN))

    def set_turn(self, rate):
        """Ask for a turn rate in rad/s (+ is left); stays until changed."""
        self.yaw_cmd = float(rate)
        self.forced_turn_set = True

    def action_scale(self):
        return ACTION_SCALE * (self.RESIDUAL_SCALE if self.walking_on_generator() else 1.0)

    walk_ramp_steps = 2_000_000  # total env steps over which the walk target rises 0.2 -> 0.5 m/s

    def walk_speed(self):
        """Walk target speed: starts easy and rises as training goes on (a curriculum)."""
        done = min(1.0, self.total_steps / self.walk_ramp_steps)
        return 0.2 + (TARGETS["walk"]["speed"] - 0.2) * done

    def set_command(self, name):
        """Give the dog a command, e.g. env.set_command("sit")."""
        self._switch(COMMANDS.index(name))

    def _switch(self, command):
        if command != self.command and COMMANDS[command] == "walk":
            self.walk_since = self.t
            # Ease in from this stance's standing pose, whichever brain was standing before: the
            # pose the walk practised and is examined on. Easing from the standing brain's last
            # motor targets (which push past the legs to hold the weight) fell 8 times in 10, and
            # from the sagging actual leg angles 9 in 10; from the dog stance, 0 in 10.
            self.walk_from = (STANCES["dog"] if self.stance == "dog" else STAND_POSE).copy() if self.t > 0 else STAND_POSE.copy()
        self.command = command

    def shin_tilt(self):
        """Mean lower-leg tilt from upright, in degrees (real dogs keep it near upright)."""
        d, m, tilts = self.data, self.model, []
        for leg, foot in zip(("FL", "FR", "RL", "RR"), self.foot_ids):
            v = d.xpos[m.body(f"{leg}_shin").id] - d.geom_xpos[foot]
            tilts.append(np.degrees(np.arctan2(np.hypot(v[0], v[1]), v[2])))
        return float(np.mean(tilts))

    def leg_scrapes(self):
        """How many legs have a knee, shin or thigh on the floor (body on the floor counts as one).

        The paw is a ball that sinks a few mm into the soft floor, so the shin's tip right above it
        touches too; only contact more than 3 cm from the paw counts.
        """
        d, legs = self.data, set()
        for c in d.contact[:d.ncon]:
            if self.floor not in (c.geom1, c.geom2):
                continue
            g = c.geom2 if c.geom1 == self.floor else c.geom1
            leg = self.geom_leg[g]
            if leg < 0:
                legs.add("body")
            elif g != self.foot_ids[leg] and np.linalg.norm(c.pos - d.geom_xpos[self.foot_ids[leg]]) > 0.03 * self.s:
                legs.add(leg)
        return len(legs)

    def _rotation(self):
        return self.data.xmat[self.torso].reshape(3, 3)

    def posture(self):
        """Height, nose-up pitch, roll, uprightness, body-frame velocity, yaw rate."""
        R = self._rotation()
        pitch = np.arcsin(np.clip(R[2, 0], -1, 1))   # nose-up is positive
        roll = np.arcsin(np.clip(R[2, 1], -1, 1))
        v_body = R.T @ self.data.qvel[0:3]
        return dict(height=self.data.qpos[2] / self.s, pitch=pitch, roll=roll, up=R[2, 2],
                    vx=v_body[0] / self.ts, vy=v_body[1] / self.ts, yaw_rate=self.data.qvel[5] * self.ts)

    def feet_on_ground(self):
        return np.array([self.data.geom_xpos[g][2] < 0.03 * self.s for g in self.foot_ids], dtype=float)

    def _obs(self):
        d, R = self.data, self._rotation()
        gravity = R.T @ np.array([0, 0, -1.0])
        v_body = R.T @ d.qvel[0:3] / self.ts
        clock = self.gait_phase()  # 2 Hz gait clock (full-size time)
        cmd = np.zeros(len(COMMANDS)); cmd[self.command] = 1
        obs = np.concatenate([
            gravity, d.qvel[3:6] * self.ts * 0.25, v_body, [d.qpos[2] / self.s],
            d.qpos[7:] - STAND_POSE, d.qvel[6:] * self.ts * 0.1, self.last_action,
            cmd, [np.sin(clock), np.cos(clock)],
        ] + ([[self.yaw_cmd]] if self.turning else []))
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)  # sensors are noisy
        return obs.astype(np.float32)

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)

        start_lying = self.randomize and self.rng.random() < 0.25
        pose = LIE_POSE if start_lying else STAND_POSE
        d.qpos[2] = (0.14 if start_lying else 0.33) * self.s
        # With the dog stance the dog normally stands tall, so half the episodes start that way:
        # run 8's walk only ever practised starting from the crouch and fell 3-5 times in 10
        # when it took over from the tall stand.
        if self.stance == "dog" and not start_lying and self.randomize and self.rng.random() < 0.5:
            pose = STANCES["dog"]
            d.qpos[2] = 0.45 * self.s
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.phase0 = 0.0
        if self.stance in ("tall", "dog") and self.randomize:
            # start anywhere in the stride, slightly tipped, like a hand-over from another brain
            self.phase0 = self.rng.uniform(0, 2 * np.pi)
            tip = self.rng.normal(0, 0.04, 2)
            q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            for axis, angle in ((1, tip[0]), (2, tip[1])):  # small roll and pitch
                r = np.zeros(4); r[0] = np.cos(angle / 2); r[axis] = np.sin(angle / 2)
                out = np.zeros(4); mujoco.mju_mulQuat(out, q, r); q = out
            d.qpos[3:7] = q
        d.qpos[7:] = np.clip(pose + self.rng.normal(0, 0.05, 12), self.lo, self.hi)
        d.ctrl[:] = d.qpos[7:]
        mujoco.mj_forward(m, d)

        self.t = 0
        self.last_action = np.zeros(12)
        self.command = COMMANDS.index(self.forced_command) if self.forced_command else self.rng.choice(self.skills)
        self.walk_since = 0
        self.walk_from = STAND_POSE.copy()  # episodes start from the standing pose
        self.yaw_cmd = self._new_turn() if self.turning and not self.forced_turn_set else self.yaw_cmd
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(self.base_pose() + self.action_scale() * action, self.lo, self.hi)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.t += 1
        self.total_steps += 8  # 8 parallel envs share the curriculum clock (train.py default)

        reward, info = self._reward(action)
        self.last_action = action
        fell = info["up"] < 0.2
        if fell:
            reward -= 5.0  # falling ends the attempt; a huge penalty made "never try anything" win
        if self.forced_command is None and self.t % self.command_period == 0:
            self._switch(self.rng.choice(self.skills))  # new command, like a real owner would give
        if self.turning and not self.forced_turn_set and self.t % 150 == 0:
            self.yaw_cmd = self._new_turn()  # a new turn every 3 s
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), bool(fell), bool(truncated), info

    def _reward(self, action):
        """Pay for doing what was asked, multiplied together so one bad part spoils the score.

        Additive rewards let the dog settle for a compromise (stand on three legs, half-sit);
        a product of terms doesn't. A small additive share keeps a learning signal when the
        dog is still far from the goal.
        """
        name = COMMANDS[self.command]
        goal, p = self.targets[name], self.posture()
        scrapes, tilt = self.leg_scrapes(), self.shin_tilt()
        feet = self.feet_on_ground()
        upright = np.exp(-(p["roll"] / 0.15) ** 2)

        if name == "walk":
            # Imitate the reference trot (reference_gait.py): match its joint angles and its
            # foot contacts at this moment of the stride. Following an example is far faster to
            # learn than discovering walking from scratch (Disney BDX, Open Duck Mini).
            q_ref, stance_ref = trot(self.gait_phase(), self.stance or "crouch", self.gait_turn())  # same clock as the obs
            gait = np.mean(feet == stance_ref)
            # Width matters: at 0.05 a dog halfway to the trot scored ~0, so there was no
            # "warmer, warmer" signal. At 0.3, halfway earns about a third.
            imitation = np.exp(-np.mean((self.data.qpos[7:] - q_ref) ** 2) / 0.3)
            # Wide enough that standing still still scores a little: a too-strict speed term makes
            # the whole product zero, and then nothing rewards starting to move.
            target = self.walk_speed()
            terms = [
                np.exp(-((p["vx"] - target) / 0.35) ** 2) * np.exp(-(p["vy"] / 0.25) ** 2),
                np.exp(-((p["height"] - self.walk_height) / 0.06) ** 2),
                np.exp(-(p["pitch"] / 0.25) ** 2) * upright,
                np.exp(-((p["yaw_rate"] - self.yaw_cmd) / (0.3 if self.turning else 0.6)) ** 2),  # turn as asked (0 = straight)
                np.exp(-scrapes),  # each leg with a knee, shin or thigh on the floor costs ~2/3
                # "dog" gait only: lower legs sloping past 30° cost more and more
                np.clip(1 - max(0.0, tilt - 30) / 30, 0.2, 1.0) if self.stance == "dog" else 1.0,
                0.3 + 0.7 * gait,
                0.2 + 0.8 * imitation,
            ]
            progress = np.clip(p["vx"] / target, -1.0, 1.0)  # every bit of forward speed pays
            progress += imitation  # following the example pays on its own, even before it moves
        else:
            pose_err = np.mean((self.data.qpos[7:] - np.asarray(goal["pose"])) ** 2)
            speed = np.hypot(p["vx"], p["vy"])
            # The dog-stance stand is 12 cm taller than the old crouch the posture brain knew. At
            # width 0.06, standing 9 cm short paid ~10%, so run 9b rose only 3 cm in 500k steps.
            # Width 0.12 pays ~57% there: a "warmer, warmer" signal. The exam still wants ±3 cm.
            h_width = 0.12 if name == "stand" and self.stance == "dog" else 0.06
            terms = [
                np.exp(-((p["height"] - goal["height"]) / h_width) ** 2),
                np.exp(-((p["pitch"] - goal["pitch"]) / 0.25) ** 2) * upright,
                # Sit is matched tighter: at 0.4 the brain sat lopsided (hips 25° apart left/right)
                # with one front paw hovering and still scored well.
                np.exp(-pose_err / (0.15 if name == "sit" else 0.4)),
                np.exp(-(speed / 0.15) ** 2),
            ]
            if name in ("stand", "sit"):
                terms.append((feet.sum() / 4) ** 4)  # three paws down scores 0.32, four score 1  # three feet down scores 0.32, four scores 1
            progress = 0.0
        reward = 2.0 * np.prod(terms) + 0.3 * np.mean(terms) + 0.5 * progress

        torque = self.data.actuator_force / self.s ** 4  # full-size units
        reward -= 0.02 * np.sum((action - self.last_action) ** 2)  # smooth, not twitchy
        reward -= 2e-4 * np.sum(torque ** 2)                        # don't waste energy
        reward -= 1e-4 * np.sum((self.data.qvel[6:] * self.ts) ** 2)
        return reward * 0.5, dict(command=name, feet=feet, scrape=scrapes, shin_tilt=tilt, walk_height=self.walk_height, yaw_cmd=self.yaw_cmd, turning=self.turning,
                                  goal_height=goal["height"], **p)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, 540, 960)
            self.camera = mujoco.MjvCamera()
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = self.torso
            self.camera.distance, self.camera.elevation, self.camera.azimuth = 1.8 * self.s, -18, 120
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
