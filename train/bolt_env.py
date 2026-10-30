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
import os

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

# The motors, as real ones: Bolt weighs what a Unitree A1 does (12 kg), whose joint motors give 33.5 N·m,
# 21 rad/s and 420 W peak each, and so do Bolt's (until run 19 they were capped at 17 N·m, half the real
# motor, which held back running and jumping; BOLT_MOTOR_TORQUE=17 reproduces the old runs). A motor
# driving a joint can give at most 420 W (so less torque the faster it spins) and nothing past 21 rad/s
# (fading to zero by 25 rad/s); braking against the motion keeps the full torque. Before this, the jump and self-righting brains spun
# joints at up to 57 rad/s and 900 W, which no motor this size can. Sideways hips swing ±48° (Go2).
MOTOR = dict(torque=float(os.environ.get("BOLT_MOTOR_TORQUE", 33.5)), power=420.0, speed=21.0, cutoff=25.0)

# Motor braking above the rated speed (--motor-brake, owner decision 28 Oct). The motor model above only stops
# DRIVING a joint past 21 rad/s; nothing slows a joint that a landing or a swinging leg back-drives faster, so
# simulated joints reached 30-57 rad/s. A real brushless motor driven past its no-load speed brakes itself: its
# back-EMF (k_e * w) exceeds the driver's supply voltage, current flows backwards and the motor makes a torque
# against the motion, k_t (k_e * w - V) / R. That is the motor's own torque-speed line continued past no-load
# speed, linear in the excess speed. Here: above brake_speed (rad/s, default the drive cutoff, 25: where this model's
# driving torque has already faded to zero, so drive fade and brake never overlap) each joint gets
#     torque = -sign(w) * min(brake_gain * (|w| - brake_speed), torque)
# brake_gain in N·m per rad/s of excess speed; default BRAKE_GAIN = 3.5 N·m·s/rad, the estimate from the A1 motor's
# constants (k_t * k_e / R at the joint) rather than the gentler 1.6 (torque / rated speed, 33.5 / 21): owner decision
# 28 Oct to take the stricter, more realistic body so results are not flattered. Capped at the motor's peak
# torque (the driver's current limit). It acts on the joint whatever the brain commands (the servo's own torque,
# clipped by motor_limits, is added on top), as back-EMF braking does.
# Process default: OFF (BOLT_MOTOR_BRAKE unset or 0), so every brain trained before 28 Oct is examined in the
# body it was trained in and old exam numbers stay bit-for-bit the same. The training scripts turn it ON
# (set_motor_brake, their --motor-brake flag, default on). Env vars, as BOLT_MOTOR_TORQUE: BOLT_MOTOR_BRAKE=1,
# BOLT_MOTOR_BRAKE_SPEED (rad/s), BOLT_MOTOR_BRAKE_GAIN (N·m per rad/s); one env: BoltEnv(motor_brake=True/False).
BRAKE_GAIN = 3.5   # N·m per rad/s of excess speed (see above); override with BOLT_MOTOR_BRAKE_GAIN or --brake-gain
MOTOR.update(brake=os.environ.get("BOLT_MOTOR_BRAKE", "0").lower() in ("1", "true", "on", "yes"),
             brake_speed=float(os.environ.get("BOLT_MOTOR_BRAKE_SPEED", MOTOR["cutoff"])),
             brake_gain=float(os.environ.get("BOLT_MOTOR_BRAKE_GAIN", BRAKE_GAIN)))
_BRAKE = {}   # id(MjModel) -> (on, speed, gain) for envs given motor_brake explicitly; others use MOTOR's


def set_motor_brake(on, speed=None, gain=None):
    """Process-wide brake setting (and the env vars, so subprocess environments created later inherit it).
    speed / gain None keep the current values. Returns MOTOR's brake settings."""
    MOTOR["brake"] = bool(on)
    if speed is not None:
        MOTOR["brake_speed"] = float(speed)
    if gain is not None:
        MOTOR["brake_gain"] = float(gain)
    os.environ["BOLT_MOTOR_BRAKE"] = "1" if on else "0"
    os.environ["BOLT_MOTOR_BRAKE_SPEED"] = repr(MOTOR["brake_speed"])
    os.environ["BOLT_MOTOR_BRAKE_GAIN"] = repr(MOTOR["brake_gain"])
    return brake_settings()


def brake_settings(model=None):
    """(on, speed rad/s, gain N·m·s/rad) in effect for this model (an env's explicit setting, else MOTOR's)."""
    if model is not None and id(model) in _BRAKE:
        return _BRAKE[id(model)]
    return MOTOR["brake"], MOTOR["brake_speed"], MOTOR["brake_gain"]


def add_brake_args(ap, default):
    """--motor-brake/--no-motor-brake, --brake-speed, --brake-gain on an argparse parser (default: on or off)."""
    import argparse
    ap.add_argument("--motor-brake", action=argparse.BooleanOptionalAction, default=default,
                    help=f"motor braking above --brake-speed (bolt_env.MOTOR, back-EMF model); default {'on' if default else 'off'}"
                         + ("" if default else " (old brains were trained without it)"))
    ap.add_argument("--brake-speed", type=float, default=MOTOR["brake_speed"],
                    help=f"rad/s where braking starts (default {MOTOR['brake_speed']:g}, the drive cutoff)")
    ap.add_argument("--brake-gain", type=float, default=MOTOR["brake_gain"],
                    help=f"braking N·m per rad/s above --brake-speed (default {MOTOR['brake_gain']:.3g}, motor-constant estimate)")


def brake_torque(qvel, speed, gain):
    """Per joint braking torque (N·m) at joint speeds qvel (rad/s): zero up to `speed`, then against the motion."""
    excess = np.maximum(np.abs(qvel) - speed, 0.0)
    return -np.sign(qvel) * np.minimum(gain * excess, MOTOR["torque"])


def motor_limits(qvel):
    """Per joint (lo, hi) torque limits for the current joint speeds (rad/s)."""
    w = np.abs(qvel)
    drive = np.minimum(MOTOR["torque"], MOTOR["power"] / np.maximum(w, 1e-6))
    drive *= np.clip((MOTOR["cutoff"] - w) / (MOTOR["cutoff"] - MOTOR["speed"]), 0.0, 1.0)
    lo = np.where(qvel < 0, -drive, -MOTOR["torque"])
    hi = np.where(qvel >= 0, drive, MOTOR["torque"])
    return lo, hi


def physics(model, data, n):
    """n physics steps with the real-motor limits applied at every one (joints are qvel[6:18]); with the brake
    on (brake_settings), the braking torque goes in data.qfrc_applied on the joints, also at every step."""
    on, speed, gain = brake_settings(model)
    for _ in range(n):
        lo, hi = motor_limits(data.qvel[6:18])
        model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1] = lo, hi
        if on:
            data.qfrc_applied[6:18] = brake_torque(data.qvel[6:18], speed, gain)
        mujoco.mj_step(model, data)

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


def trot(phase, stance="crouch", turn=0.0, step=None, spin=0.0, lift=None):
    from reference_gait import paw_trot, trot as reference_trot  # imported late: reference_gait imports this module
    if stance == "dog":
        gait = dict(PAW_GAIT, **({"step": step} if step is not None else {}), **({"lift": lift} if lift is not None else {}))
        return paw_trot(phase, **gait, turn=turn, spin=spin)
    return reference_trot(phase, hip_swing=0.15, knee_lift=0.3, base=STANCES[stance])
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Hurdles: a bar across the path (a box sunk into the floor to show any height). Bolt steps over it
# instead of leaping: its belly is ~36 cm up, so only the legs have to clear a bar up to ~25 cm, and
# near a bar the trot lifts each paw to the bar's height plus 7 cm. With the 12M walk brain unchanged,
# that cleared 15 and 25 cm bars 48 times in 48 at 0.3-1.0 m/s on randomised physics. (A hand-made
# bounding leap, run 14, reared up and tipped over backwards in every variant tried.)
HURDLE_HALF = 0.3           # m: the bar is a 60 cm box, sunk into the floor to show any height up to 60 cm
# The bar is a mocap body (moved through data.mocap_pos). Moving a plain world geom at run time is not
# reliable in MuJoCo: a ball dropped on a moved 30 cm box fell straight through it, and the dog's legs
# went through bars without a touch being counted (the "48 in 48 clean" step-overs were not real).
HURDLE_XML = (f'<body name="hurdle" mocap="true" pos="100 0 -1"><geom name="hurdle" type="box" '
              f'size="0.025 0.6 {HURDLE_HALF}" rgba="0.89 0.38 0.04 1"/></body>')
HURDLE_MARGIN = 0.07        # m of paw clearance over the bar

# Rough terrain (step 6, --terrain-cm): a height field, uniform random heights per 10 cm cell up to `cm` (legged_gym's
# "random uniform" rough terrain), 40 x 40 m, beside the flat floor: its middle is TERRAIN["at"], 40 m from the origin,
# so a try started at the origin stays on flat floor and a try started on the field stays on it for a 5 s try
# (trot 1.9 m/s). It is added to the world only when cm > 0 (terrain_xml), as the last world geom so every other geom
# keeps its id; with cm = 0 the model is bolt.xml exactly. The field is fixed per seed (terrain_field), so the CPU and
# GPU envs share it; heights are the field's 0..1 values times cm.
TERRAIN = dict(cm=0.0, half=20.0, cell=0.1, at=(40.0, 0.0), seed=0, base=0.05)


def terrain_field(seed=TERRAIN["seed"], half=TERRAIN["half"], cell=TERRAIN["cell"]):
    """The field's heights, 0..1 (rows along y, columns along x, as MuJoCo's hfield_data)."""
    n = int(round(2 * half / cell)) + 1
    return np.random.default_rng(seed).uniform(0.0, 1.0, (n, n))


def terrain_xml(xml, cm, half=TERRAIN["half"], cell=TERRAIN["cell"], at=TERRAIN["at"], base=TERRAIN["base"]):
    """bolt.xml with the height field (cm high) added; the heights are set from terrain_field after loading."""
    n = int(round(2 * half / cell)) + 1
    asset = f'<hfield name="terrain" nrow="{n}" ncol="{n}" size="{half} {half} {cm / 100.0} {base}"/>'
    xml = xml.replace("<asset>", "<asset>\n    " + asset, 1)
    # only the paws and shins touch the bumps (contype bit 4); the body, head and thighs meet the flat floor under the
    # field. Box-on-hfield contacts made MJX 6x slower (paws and shins: 1.9x); on 5-10 cm bumps an upright body is 30+ cm
    # above them, and a fall is still caught (height and tilt limits, the floor under the field)
    geom = (f'<geom name="terrain" type="hfield" hfield="terrain" pos="{at[0]} {at[1]} 0" contype="0" conaffinity="4" '
            'rgba="0.55 0.5 0.42 1"/>')
    for cls, typ in (("foot", "sphere"), ("shin", "capsule")):   # CPU: 1 | 4 (mjx_imitate.collisions makes it 2 | 4)
        xml = xml.replace(f'<default class="{cls}"><geom type="{typ}"', f'<default class="{cls}"><geom type="{typ}" contype="5"', 1)
    i = xml.rindex("</worldbody>")
    return xml[:i] + "  " + geom + "\n  " + xml[i:]


def terrain_height(field, cm, x, y, half=TERRAIN["half"], cell=TERRAIN["cell"], at=TERRAIN["at"]):
    """Ground height (m) at (x, y): bilinear in the field's cells, 0 off the field. Works on numpy and JAX arrays
    (xp: the array module of field)."""
    xp = np if isinstance(field, np.ndarray) else __import__("jax.numpy", fromlist=["numpy"])
    n = field.shape[0]
    u, v = (x - at[0] + half) / cell, (y - at[1] + half) / cell     # column, row
    on = (u >= 0) & (u <= n - 1) & (v >= 0) & (v <= n - 1)
    u, v = xp.clip(u, 0, n - 1 - 1e-6), xp.clip(v, 0, n - 1 - 1e-6)
    c, r = xp.floor(u).astype(xp.int32), xp.floor(v).astype(xp.int32)
    fu, fv = u - c, v - r
    h = (field[r, c] * (1 - fu) * (1 - fv) + field[r, c + 1] * fu * (1 - fv) + field[r + 1, c] * (1 - fu) * fv
         + field[r + 1, c + 1] * fu * fv)
    return xp.where(on, h * cm / 100.0, 0.0)


def apply_shove(qvel, R, speed, spin, side):
    """A sideways shove (a kick to the flank): body velocity changed by `speed` m/s along the body's y axis and the
    roll rate by `spin` rad/s away from the push; side +1/-1. skills.Bolt.shove (the switch test's) and the pushes
    of imitate_env/mjx_imitate all use this. Returns the new qvel (numpy or JAX)."""
    xp = np if isinstance(qvel, np.ndarray) else __import__("jax.numpy", fromlist=["numpy"])
    dv = R @ xp.array([0.0, 1.0, 0.0]) * (side * speed)
    dw = xp.array([-1.0, 0.0, 0.0]) * (side * spin)   # body frame: roll away from the push
    if xp is np:
        qvel = qvel.copy()
        qvel[0:3] += dv
        qvel[3:6] += dw
        return qvel
    return qvel.at[0:3].add(dv).at[3:6].add(dw)
HURDLE_ZONE = (-0.7, 1.0)   # m: high steps from 1.0 m before the bar until the body is 0.7 m past it

# motor lag (ms) and lost-packet share in jump and get-up exams (fair_exam.py --motor-lag-ms/--drop); 0 = off
EXAM_GAPS = dict(lag_ms=0.0, drop=0.0, jitter_ms=0.0, spike_p=0.0)
SPIKE_MS = 40.0   # largest delay spike (ms) when spikes are on


def add_exam_gap_args(ap):
    """--exam-latency-ms/--exam-lag-ms/--exam-drop: the real-body gaps every jump and get-up exam of this process uses
    (training with delay should pick its best checkpoint with delay; 29 Oct: no-delay exams froze "best" at the start)."""
    ap.add_argument("--exam-latency-ms", type=float, default=0.0, help="control delay (ms) in the exams")
    ap.add_argument("--exam-lag-ms", type=float, default=0.0, help="motor lag (ms) in the exams")
    ap.add_argument("--exam-drop", type=float, default=0.0, help="share of control packets lost in the exams")
    ap.add_argument("--exam-jitter-ms", type=float, default=0.0, help="delay jitter (+- ms per step) in the exams")
    ap.add_argument("--exam-spike-p", type=float, default=0.0, help="chance per step of a delay spike in the exams")


def set_exam_gaps(latency_ms=0.0, lag_ms=0.0, drop=0.0, jitter_ms=0.0, spike_p=0.0):
    """Set the exam gaps of jump_env and getup_env in this process (call after the training envs are made)."""
    import getup_env
    import jump_env
    jump_env.EXAM_LATENCY_MS = getup_env.EXAM_LATENCY_MS = float(latency_ms)
    EXAM_GAPS.update(lag_ms=float(lag_ms), drop=float(drop), jitter_ms=float(jitter_ms), spike_p=float(spike_p))
    return dict(latency_ms=float(latency_ms), lag_ms=float(lag_ms), drop=float(drop), jitter_ms=float(jitter_ms),
                spike_p=float(spike_p))


class BoltEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, randomize=True, command_period=200, max_steps=600, render_mode=None, seed=None, scale=1.0,
                 skills=None, gait_generator=False, turning=False, handover_brain=None, speed_control=False,
                 sharp_turns=False, getting_up=False, hurdles=False, motor_brake=None, terrain_cm=TERRAIN["cm"],
                 terrain_seed=TERRAIN["seed"]):
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
        if scale == 1.0 and hurdles == "real":   # a real agility hurdle: solid stands, a loose bar (hurdle.py)
            import hurdle
            self.model = mujoco.MjModel.from_xml_string(hurdle.world_xml(MODEL.read_text(), n=1))
        elif scale == 1.0 and hurdles:  # the same robot, plus a hurdle (parked far away until placed)
            xml = MODEL.read_text().replace('<geom name="floor"', HURDLE_XML + '\n    <geom name="floor"', 1)
            self.model = mujoco.MjModel.from_xml_string(xml)
        elif scale == 1.0 and terrain_cm > 0:   # rough terrain beside the floor (TERRAIN, terrain_xml)
            self.model = mujoco.MjModel.from_xml_string(terrain_xml(MODEL.read_text(), terrain_cm))
        elif scale == 1.0:
            self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        else:
            from scale_robot import scaled_xml
            self.model = mujoco.MjModel.from_xml_string(scaled_xml(scale))
        self.data = mujoco.MjData(self.model)
        # motor_brake: True/False for this env only (speed and gain from MOTOR); None = the process default
        if motor_brake is not None:
            import weakref
            key = id(self.model)
            _BRAKE[key] = (bool(motor_brake), MOTOR["brake_speed"], MOTOR["brake_gain"])
            weakref.finalize(self, _BRAKE.pop, key, None)
        self.motor_brake = brake_settings(self.model)
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
        self.forced_speed_set = False
        self.env_steps = 0  # steps this env has taken since it was created (drives the speed ramp)
        self.walk_from = STAND_POSE.copy()  # motor targets the walk eases in from

        m = self.model
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.nominal_kp = m.actuator_gainprm[:, 0].copy()
        self.nominal_mass = m.body_mass.copy()
        self.nominal_friction = m.geom_friction[:, 0].copy()
        self.torso = m.body("torso").id
        self.foot_ids = [m.geom(f).id for f in FEET]
        self.floor = m.geom("floor").id
        # rough terrain: the field (heights 0..1), its height (cm) and geom; None on flat ground
        assert not (terrain_cm > 0 and (hurdles or scale != 1.0)), "terrain is for the full-size robot without hurdles"
        self.terrain = None
        if terrain_cm > 0:
            field = terrain_field(terrain_seed)
            m.hfield_data[:] = field.ravel()
            self.terrain = dict(field=field, cm=float(terrain_cm), geom=m.geom("terrain").id)
        self.ground = {self.floor} | ({self.terrain["geom"]} if self.terrain else set())   # what the paws stand on
        # which leg each geom belongs to (0-3), or -1 for the body (torso, head, snout)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in m.geom_bodyid]
        self.geom_leg = np.array([["FL", "FR", "RL", "RR"].index(n[:2]) if n[:2] in ("FL", "FR", "RL", "RR") else -1
                                  for n in names])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        # turning=True: the walk also gets a turn command (rad/s, + is left) as a 53rd input.
        # Only the dog gait can turn: its paw paths take longer strides on the outside of the turn.
        self.turning = turning
        self.yaw_cmd = 0.0
        # handover_brain: folder of a posture brain. Half the randomized walk episodes then begin
        # with that brain standing the dog up, then hand over, so the walk practises taking over
        # from the real standing brain (run 10 fell 4 in 10 there while passing an exam that
        # handed over from a perfect motor-held stand).
        self.handover_brain, self._handover = handover_brain, None
        # speed_control=True (with turning): a speed command (m/s) as a 54th input, from a slow walk to
        # a run. The gait steps faster and takes longer strides to match: with no brain it reaches
        # 1.29 m/s at 3 steps/s and 20 cm strides, well inside the motors' force limit (measured).
        assert turning or not speed_control, "speed control comes after the turn command in the input layout"
        self.speed_control, self.speed_cmd, self.phase = speed_control, 0.4, 0.0
        # sharp_turns=True (dog gait with speed control): turn by stepping sideways (paw_trot's spin),
        # up to 2 rad/s at a walk instead of 0.6. Same inputs as before, so it is marked in the brain's
        # folder (gait.json) rather than by the input count.
        assert not sharp_turns or (speed_control and self.stance == "dog"), "sharp turns are for the dog gait with speed control"
        self.sharp_turns = sharp_turns
        # getting_up=True (posture brains): practise standing up from sit and lie much more often, and
        # without spinning. The 2.5M posture brain fell 4 times in 12 standing straight up from a sit,
        # twisting at up to 3 rad/s on the way up.
        self.getting_up = getting_up
        # hurdles=True (dog gait): a hurdle in the world, which the trot steps over (see HURDLE_XML)
        assert not hurdles or self.stance == "dog", "hurdles are for the dog gait"
        self.hurdles = hurdles
        self.hurdle_at, self.hurdle_dir, self.hurdle_h = np.array([100.0, 0.0]), np.array([1.0, 0.0]), 0.0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (52 + turning + speed_control,), np.float32)

    # ---------- helpers ----------
    RESIDUAL_SCALE = 0.35  # with the gait generator on, the brain's corrections are this fraction of normal

    def walking_on_generator(self):
        return self.gait_generator and COMMANDS[self.command] == "walk"

    def gait_phase(self):
        if self.speed_control:
            return self.phase  # integrated: the step rate changes with the speed command
        return 2 * np.pi * 2.0 * self.t * self.dt / self.ts + self.phase0  # 2 Hz gait clock in full-size time

    MAX_SPEED = 1.2  # m/s, a trot-run: 3 steps/s with 20 cm strides
    SPEED_REF = 0.4  # m/s, the pace the walk brain learned before it had a speed command

    def gait_freq(self):
        """Steps per second for each leg pair: 2 at a walk, up to 3 when running."""
        if not self.speed_control:
            return 2.0
        return 2.0 + float(np.clip((self.speed_cmd - 0.5) / 0.7, 0.0, 1.0))

    def gait_stride(self):
        """Stride length (m) that gives the commanded speed at this step rate (measured: v ≈ 2.15·f·stride)."""
        if not self.speed_control:
            return None
        return float(np.clip(self.speed_cmd / (2.15 * self.gait_freq()), 0.0, 0.2))

    def place_hurdle(self, ahead, height):
        """Put a hurdle `ahead` m in front of the dog, across its path, `height` m tall."""
        R = self._rotation()
        fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
        self.hurdle_dir, self.hurdle_h = fwd, height
        self.hurdle_at = self.data.qpos[:2] + fwd * ahead
        k = self.model.body_mocapid[self.model.body("hurdle").id]
        self.data.mocap_pos[k] = [self.hurdle_at[0], self.hurdle_at[1], height - HURDLE_HALF]  # sunk to show `height`
        yaw = np.arctan2(fwd[1], fwd[0])
        self.data.mocap_quat[k] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    def hurdle_ahead(self):
        """How far the hurdle is ahead of the body's centre, along the path (negative once past it)."""
        return float(np.dot(self.hurdle_at - self.data.qpos[:2], self.hurdle_dir))

    def hurdle_hits(self):
        """Contacts between the dog and the hurdle (a knocked bar, in agility)."""
        g = self.model.geom("hurdle").id
        return sum(g in (c.geom1, c.geom2) for c in self.data.contact[:self.data.ncon])

    def gait_lift(self):
        """Paw lift for the trot: higher while stepping over a hurdle (None: the usual lift)."""
        if self.hurdles and HURDLE_ZONE[0] < self.hurdle_ahead() < HURDLE_ZONE[1]:
            return self.hurdle_h + HURDLE_MARGIN
        return None

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
        q = trot(self.gait_phase(), self.stance, self.gait_turn(), self.gait_stride(), self.gait_spin(), self.gait_lift())[0]
        if self.stance in ("tall", "dog"):
            # Ease into the trot over half a second from walk_from (see _switch); runs 5-6 jumped
            # straight in and the 2M brain fell 9 times in 10 when taking over from standing.
            q = self.walk_from + min(1.0, (self.t - self.walk_since) / 25) * (q - self.walk_from)
        return q

    TURN_GAIN = 0.075  # m of extra stride on the outside per rad/s of turn (measured: 0.04 m -> ~0.56 rad/s)
    MAX_TURN = 0.6     # rad/s (~35°/s): strides stay within what the paw-path gait handled without a brain

    SHARP_TURN = 2.0  # rad/s (~115°/s): the fastest turn asked of sharp_turns brains, at a walk

    def max_turn(self, speed=None):
        """Fastest turn practised at this speed: sharp turns get gentler as the dog runs faster."""
        if not self.sharp_turns:
            return self.MAX_TURN
        v = self.speed_cmd if speed is None else speed
        return float(np.clip(self.SHARP_TURN - 1.25 * max(0.0, v - 0.4), 1.0, self.SHARP_TURN))

    def gait_spin(self):
        """Radians to turn during one stance (sharp_turns): the asked turn rate times the stance time."""
        if not self.sharp_turns:
            return 0.0
        return float(np.clip(self.yaw_cmd, -self.SHARP_TURN, self.SHARP_TURN)) * 0.5 / self.gait_freq()

    def gait_turn(self):
        if not self.turning or self.sharp_turns:  # sharp turns steer with spin instead
            return 0.0
        stride = self.gait_stride()
        limit = 0.05 if stride is None else min(0.05, max(0.0, 0.22 - stride))  # the outside stride stays <= 22 cm
        return float(np.clip(self.TURN_GAIN * self.yaw_cmd, -limit, limit))

    def _new_speed(self):
        """A speed command: often a normal walk, otherwise anything from slow to running.

        The fastest speed practised rises from 0.7 to 1.2 m/s over the first 1.5M steps of a
        training run (8 envs), so it learns to balance at each speed before going faster.
        """
        top = 0.7 + (self.MAX_SPEED - 0.7) * min(1.0, self.env_steps * 8 / 1.5e6)
        return 0.4 if self.rng.random() < 0.2 else float(self.rng.uniform(0.2, top))

    def set_speed(self, v):
        """Ask for a forward speed in m/s; stays until changed."""
        self.speed_cmd = float(v)
        self.forced_speed_set = True

    def _new_turn(self):
        """A turn command like an owner would give: often straight, otherwise left or right.

        Sharp-turn brains practise up to max_turn(), and the sharpest turn practised rises from 0.6
        to 2 rad/s over the first 1M steps of a run (8 envs), like the speed ramp.
        """
        top = self.MAX_TURN
        if self.sharp_turns:
            top = min(self.max_turn(), self.MAX_TURN + (self.SHARP_TURN - self.MAX_TURN) * min(1.0, self.env_steps * 8 / 1e6))
        return 0.0 if self.rng.random() < 0.3 else float(self.rng.uniform(-top, top))

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
        if self.speed_control and COMMANDS[self.command] == "walk":
            # The walk brain learned a reflex to hold ~0.4 m/s and brakes anything faster (run 11
            # ignored speed commands for 1M steps). So it feels its speed relative to the asked
            # speed: holding its usual pace now means holding the asked pace.
            v_body = v_body - np.array([self.speed_cmd - self.SPEED_REF, 0.0, 0.0])
        clock = self.gait_phase()  # 2 Hz gait clock (full-size time)
        cmd = np.zeros(len(COMMANDS)); cmd[self.command] = 1
        obs = np.concatenate([
            gravity, d.qvel[3:6] * self.ts * 0.25, v_body, [d.qpos[2] / self.s],
            d.qpos[7:19] - STAND_POSE, d.qvel[6:18] * self.ts * 0.1, self.last_action,
            cmd, [np.sin(clock), np.cos(clock)],
        ] + ([[self.yaw_cmd]] if self.turning else []) + ([[self.speed_cmd]] if self.speed_control else []))
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
        d.qpos[7:19] = np.clip(pose + self.rng.normal(0, 0.05, 12), self.lo, self.hi)
        d.ctrl[:] = d.qpos[7:19]
        mujoco.mj_forward(m, d)

        self.t = 0
        self.last_action = np.zeros(12)
        self.command = COMMANDS.index(self.forced_command) if self.forced_command else self.rng.choice(self.skills)
        self.walk_since = 0
        self.walk_from = STAND_POSE.copy()  # episodes start from the standing pose
        if self.speed_control and not self.forced_speed_set:
            self.speed_cmd = self._new_speed()
        self.yaw_cmd = self._new_turn() if self.turning and not self.forced_turn_set else self.yaw_cmd
        self.phase = self.phase0
        if self.hurdles:  # park the hurdle until someone places it
            self.hurdle_at = np.array([100.0, 0.0])
            self.data.mocap_pos[self.model.body_mocapid[self.model.body("hurdle").id]] = [100.0, 0.0, -1.0]
        if (self.handover_brain and self.randomize and COMMANDS[self.command] == "walk" and not start_lying
                and self.rng.random() < 0.5):
            self._stand_then_hand_over(int(self.rng.integers(60, 150)))
        return self._obs(), {}

    def _stand_then_hand_over(self, steps):
        """Let the posture brain stand the dog for a while, then switch to walking."""
        if self._handover is None:
            from evaluate import Policy  # late: evaluate imports this module
            self._handover = Policy(self.handover_brain)
        walk, forced = self.command, self.forced_command
        self.command, self.forced_command = COMMANDS.index("stand"), "stand"
        for _ in range(steps):
            self.step(self._handover.act(self._obs()))
        self.forced_command = forced
        self._switch(walk)

    # ---- control delay (29 Oct, owner: "train delay into everything"): a real robot's motors get the targets sent
    # delay_sub physics steps ago. Envs set delay_sub per try (start_delay); 0 = no delay, the old behaviour.
    # lag_ms: first-order motor lag of the targets (mastermind/env.py's stress test); drop: share of control packets
    # lost, the motors keeping the last target they got (29 Oct: 20 ms lag cost MasterMind 18 points, 10% loss 7).
    delay_sub, _queue, lag_ms, drop, _lag_y, _last_exec = 0, None, 0.0, 0.0, None, None
    # owner 30 Oct: delay is not one fixed number. jitter_ms: each control step's delay is the try's delay +- up to
    # jitter_ms (uniform); spike: with chance spike_p per step a delay spike of up to spike_ms lasting 2-10 steps (Wi-Fi
    # or a busy bus). The motors always act on the NEWEST target that has arrived (a late packet never rolls them back).
    jitter_ms, spike_p, spike_ms, _spike_left, _k, _lastk = 0.0, 0.0, 0.0, 0, 0, None

    def start_delay(self, ms, lag_ms=0.0, drop=0.0, jitter_ms=0.0, spike_p=0.0, spike_ms=0.0):
        """Control delay of `ms` for this try (whole 2 ms physics steps); the robot holds its first target meanwhile.
        Optional motor lag (ms) and packet loss (0..1) for the try too, and a delay that moves step to step (jitter_ms,
        spikes)."""
        ts = self.model.opt.timestep
        self.delay_sub = int(round(ms / 1000.0 / ts))
        self.jitter_ms, self.spike_p, self.spike_ms, self._spike_left = float(jitter_ms), float(spike_p), float(spike_ms), 0
        longest = self.delay_sub + int(round((self.jitter_ms + self.spike_ms) / 1000.0 / ts))
        dynamic = self.delay_sub or self.jitter_ms or self.spike_p
        self._queue = [self.data.ctrl.copy() for _ in range(longest // self.frame_skip + 3)] if dynamic else None
        self._k, self._lastk = 0, None
        self.lag_ms, self.drop, self._last_exec = float(lag_ms), float(drop), None
        self._lag_y = self.data.ctrl.copy() if lag_ms else None

    def _delay_now(self):
        """This control step's delay in physics substeps (the try's delay, jitter and any spike)."""
        ts, d = self.model.opt.timestep, float(self.delay_sub)
        if self.jitter_ms:
            d += self.rng.uniform(-1, 1) * self.jitter_ms / 1000.0 / ts
        if self.spike_p:
            if self._spike_left == 0 and self.rng.random() < self.spike_p:
                self._spike_left, self._spike = int(self.rng.integers(2, 11)), self.rng.uniform(0.3, 1.0) * self.spike_ms
            if self._spike_left:
                self._spike_left -= 1
                d += self._spike / 1000.0 / ts
        return int(np.clip(round(d), 0, (len(self._queue) - 2) * self.frame_skip))

    def _physics_step(self):
        """One control step of physics: the target just set in ctrl, or with start_delay's latency the newest one that
        has arrived (sent `delay` physics substeps ago or earlier, never older than one already used); ctrl keeps what
        was sent."""
        if self._queue is None and not (self.lag_ms or self.drop):
            return physics(self.model, self.data, self.frame_skip)
        m, d = self.model, self.data
        sent = d.ctrl.copy()
        got = sent
        if self.drop and self._last_exec is not None and self.rng.random() < self.drop:
            got = self._last_exec            # the packet was lost: the motors keep the last target they got
        self._last_exec = got
        targets = [got] * self.frame_skip
        if self._queue is not None:
            self._queue = self._queue[1:] + [got]          # _queue[-1] was sent at control step k (now)
            dsub, n, fs = self._delay_now(), len(self._queue), self.frame_skip
            for i in range(fs):
                k = (self._k * fs + i - dsub) // fs          # newest control step whose target has arrived by now
                if self._lastk is not None:
                    k = max(k, self._lastk)                  # a late packet never rolls the motors back
                k = min(k, self._k)
                self._lastk = k
                targets[i] = self._queue[max(0, n - 1 - (self._k - k))]
            self._k += 1
        g = 1.0 - np.exp(-m.opt.timestep / (self.lag_ms / 1000.0)) if self.lag_ms else 1.0
        for i in range(self.frame_skip):
            want = targets[i]
            if self.lag_ms:
                self._lag_y = self._lag_y + g * (want - self._lag_y)
                want = self._lag_y
            d.ctrl[:] = want
            physics(m, d, 1)
        d.ctrl[:] = sent

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(self.base_pose() + self.action_scale() * action, self.lo, self.hi)
        self._physics_step()
        self.t += 1
        self.env_steps += 1
        if self.speed_control:
            self.phase += 2 * np.pi * self.gait_freq() * self.dt / self.ts
        self.total_steps += 8  # 8 parallel envs share the curriculum clock (train.py default)

        reward, info = self._reward(action)
        self.last_action = action
        fell = info["up"] < 0.2
        if fell:
            reward -= 5.0  # falling ends the attempt; a huge penalty made "never try anything" win
        if self.forced_command is None and self.t % self.command_period == 0:
            if self.getting_up and COMMANDS[self.command] in ("sit", "lie") and self.rng.random() < 0.6:
                self._switch(COMMANDS.index("stand"))     # get up (practised more: see getting_up)
            else:
                self._switch(self.rng.choice(self.skills))  # new command, like a real owner would give
        if self.speed_control and not self.forced_speed_set and self.t % 150 == 0:
            self.speed_cmd = self._new_speed()  # a new speed every 3 s
        if self.turning and not self.forced_turn_set and self.t % 150 == 0:
            self.yaw_cmd = self._new_turn()  # and a new turn (sharp turns depend on the speed)
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

        hits = self.hurdle_hits() if self.hurdles else 0
        if name == "walk":            # Imitate the reference trot (reference_gait.py): match its joint angles and its
            # foot contacts at this moment of the stride. Following an example is far faster to
            # learn than discovering walking from scratch (Disney BDX, Open Duck Mini).
            q_ref, stance_ref = trot(self.gait_phase(), self.stance or "crouch", self.gait_turn(), self.gait_stride(), self.gait_spin(), self.gait_lift())  # same clock as the obs
            gait = np.mean(feet == stance_ref)
            # Width matters: at 0.05 a dog halfway to the trot scored ~0, so there was no
            # "warmer, warmer" signal. At 0.3, halfway earns about a third.
            imitation = np.exp(-np.mean((self.data.qpos[7:19] - q_ref) ** 2) / 0.3)
            # Wide enough that standing still still scores a little: a too-strict speed term makes
            # the whole product zero, and then nothing rewards starting to move.
            target = self.speed_cmd if self.speed_control else self.walk_speed()
            v_width = 0.15 + 0.2 * target if self.speed_control else 0.35  # faster asks allow a bit more error
            terms = [
                np.exp(-((p["vx"] - target) / v_width) ** 2) * np.exp(-(p["vy"] / 0.25) ** 2),
                np.exp(-((p["height"] - self.walk_height) / 0.06) ** 2),
                np.exp(-(p["pitch"] / 0.25) ** 2) * upright,
                np.exp(-((p["yaw_rate"] - self.yaw_cmd) / ((0.3 + 0.15 * abs(self.yaw_cmd)) if self.sharp_turns else 0.3 if self.turning else 0.6)) ** 2),  # turn as asked (0 = straight)
                np.exp(-scrapes),  # each leg with a knee, shin or thigh on the floor costs ~2/3
                # "dog" gait only: lower legs sloping past 30° cost more and more
                np.clip(1 - max(0.0, tilt - 30) / 30, 0.2, 1.0) if self.stance == "dog" else 1.0,
                0.3 + 0.7 * gait,
                0.2 + 0.8 * imitation,
                np.exp(-3.0 * hits),  # walking into a hurdle
            ]
            progress = np.clip(p["vx"] / target, -1.0, 1.0)  # every bit of forward speed pays
            progress += imitation  # following the example pays on its own, even before it moves
        else:
            pose_err = np.mean((self.data.qpos[7:19] - np.asarray(goal["pose"])) ** 2)
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
                terms.append((feet.sum() / 4) ** 4)  # three paws down scores 0.32, four score 1
            if self.getting_up:
                terms.append(np.exp(-(p["yaw_rate"] / 0.6) ** 2))  # no spinning while changing posture
            progress = 0.0
        reward = 2.0 * np.prod(terms) + 0.3 * np.mean(terms) + 0.5 * progress

        torque = self.data.actuator_force / self.s ** 4  # full-size units
        reward -= 0.02 * np.sum((action - self.last_action) ** 2)  # smooth, not twitchy
        reward -= 2e-4 * np.sum(torque ** 2)                        # don't waste energy
        reward -= 1e-4 * np.sum((self.data.qvel[6:18] * self.ts) ** 2)
        return reward * 0.5, dict(command=name, feet=feet, scrape=scrapes, shin_tilt=tilt, walk_height=self.walk_height, yaw_cmd=self.yaw_cmd, turning=self.turning,
                                  speed_cmd=self.speed_cmd if self.speed_control else None,
                                  goal_height=goal["height"], hurdle_hits=hits, **p)

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
