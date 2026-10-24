"""Run Pollen Robotics' official Microduck policies (ONNX) in MuJoCo, on their own body.

No training. This reproduces, in one small file, what their CPU rehearsal script
(microduck_rl/scripts/infer_policy.py) and their robot daemon (microduck/duck-control/src/obs.rs)
do every 20 ms:

  obs (61) = [ gyro(3)            trunk angular velocity in the trunk frame (imu_ang_vel sensor; the
                                  imu site has identity orientation in the trunk)
               gravity(3)         world (0,0,-1) rotated into the trunk frame (upright = [0,0,-1])
               joint_pos(14)      qpos - DEFAULT_POSE           (policy joint order, mouth excluded)
               joint_vel(14)      qvel
               last_action(14)    the policy's previous raw output
               command(13)        twist[vx,vy,wz], head[neck_pitch,head_pitch,head_yaw,head_roll],
                                  body[x,y,z,roll,pitch,yaw]  (all zero except what a policy uses) ]
  target = DEFAULT_POSE + action * action_scale   (action_scale 1.0; ONNX metadata agrees)
  control 50 Hz = 4 physics steps of 5 ms.

Actuators are the BAM "m6" XL330 model the policies were trained against (Rhoban/bam at the commit
microduck_rl pins), driven exactly like infer_policy.py's load_mujoco_with_bam: position actuators
become torque motors, the firmware P-loop (kp_fw=200) + DC motor + load-dependent friction run in
Python each physics step. A 3-6 physics-step command delay is applied like training.

Paths (env vars override): MICRODUCK_RL (clone of pollen-robotics/microduck_rl), BAM_SRC (clone of
Rhoban/bam), MICRODUCK_POLICIES (folder with the official ONNX files). setup.sh fills .deps/.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPS = HERE / ".deps"
RL = Path(os.environ.get("MICRODUCK_RL", DEPS / "microduck_rl"))
BAM_SRC = Path(os.environ.get("BAM_SRC", DEPS / "bam"))
POLICIES = Path(os.environ.get("MICRODUCK_POLICIES", HERE / "policies"))
ROBOT_DIR = RL / "src/mjlab_microduck/robot/microduck"

if str(BAM_SRC) not in sys.path:
    sys.path.insert(0, str(BAM_SRC))

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

# --- constants, each copied from the source named beside it -------------------------------------
# duck-control/src/model.rs DEFAULT_POSITION minus the mouth == infer_policy.py DEFAULT_POSE
# == microduck_constants.HOME_FRAME == ONNX metadata default_joint_pos.
JOINT_NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_yaw", "head_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530,
                         0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530], dtype=np.float32)
# microduck_velstand_env_cfg.SIT_JOINT_OVERRIDES (== sitstand SIT keyframe), servo indices.
SIT_OVERRIDES = {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0, 10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0}
# infer_policy.py / microduck_constants._BAM_ACTUATOR_KWARGS
BAM_KP_FW = 200.0
BAM_VIN = 7.4               # infer_policy default (training samples 6.5-8.2)
BAM_VIN_DROP_GAIN = 0.1     # infer_policy default (training samples 0-0.2)
BAM_VIN_MIN = 6.0
BAM_STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
BAM_STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
DELAY_LAG = (3, 6)          # physics steps, _BAM_ACTUATOR_KWARGS delay_min_lag/max_lag
TIMESTEP = 0.005            # body_server.py TIMESTEP, infer_policy.py
DECIMATION = 4              # 50 Hz
HOME_TRUNK_Z = 0.125        # body_server.py HOME_TRUNK_Z
# velstand_env_cfg "recovery complete": tilt < 25 deg and trunk z > 0.09
UP_TILT_DEG, UP_Z = 25.0, 0.09
NOISE = np.r_[[0.03] * 3, [0.01] * 3, [0.001] * 14, [0.25] * 14, [0.0] * 14].astype(np.float32)
STAND_Z = 0.10              # our stricter "at standing height" (measured standing trunk z ~0.115)

SCENE_FULL = "scene_allcollisions.xml"   # every part collides (velstand was trained on this body)
SCENE_GROUND = "scene.xml"               # curated ground-contact set (sitstand / roulade / standup)


def policy_path(name: str) -> Path:
    p = POLICIES / (name if name.endswith(".onnx") else name + ".onnx")
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run setup.sh (downloads pollen-robotics/microduck-policies)")
    return p


class Policy:
    def __init__(self, name: str):
        self.name = Path(name).stem
        self.sess = ort.InferenceSession(str(policy_path(name)), providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.action_scale = float(meta.get("action_scale", 1.0))
        names = tuple(meta["joint_names"].split(","))
        assert names == JOINT_NAMES, names
        dp = np.array([float(v) for v in meta["default_joint_pos"].split(",")], dtype=np.float32)
        assert np.allclose(dp, DEFAULT_POSE, atol=1e-3), dp
        assert self.sess.get_inputs()[0].shape == [1, 61]
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.sess.run([self.out], {self.inp: obs.reshape(1, -1).astype(np.float32)})[0][0]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _axis_quat(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


class MicroduckSim:
    def __init__(self, scene: str = SCENE_FULL, vin: float = BAM_VIN, delay: bool = True, obs_noise: bool = True,
                 seed: int = 0):
        from bam.model import load_model
        from bam.mujoco import MujocoController

        self.rng = np.random.default_rng(seed)
        bam = load_model(motor_name="xl330", model="m6")
        bam.actuator.kp = BAM_KP_FW
        bam.actuator.vin = vin
        bam.actuator.max_current = None
        force_limit = bam.actuator.vin * bam.kt.value / bam.R.value

        spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / scene))
        names = []
        for act in spec.actuators:
            tgt = act.target if isinstance(act.target, str) else act.target.name
            if tgt.startswith("passive_"):
                continue
            act.set_to_motor()
            act.forcelimited = True
            act.forcerange = (-force_limit, force_limit)
            act.ctrllimited = False
            act.gear = [1.0, 0, 0, 0, 0, 0]
            names.append(act.name)
            j = spec.joint(tgt)
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = BAM_STIFF_SOLREF_FRICTION
            j.solimp_friction = BAM_STIFF_SOLIMP_FRICTION
        self.m = spec.compile()
        self.m.opt.timestep = TIMESTEP
        self.d = mujoco.MjData(self.m)
        self.bam = MujocoController(bam, names, self.m, self.d, vin_drop_gain=BAM_VIN_DROP_GAIN, vin_min=BAM_VIN_MIN)

        jids = [self.m.actuator_trnid[self.m.actuator(n).id, 0] for n in names]
        jn = tuple(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in jids)
        assert jn == JOINT_NAMES, jn  # actuator order == policy order (checked, not assumed)
        self.qadr = self.m.jnt_qposadr[jids]
        self.vadr = self.m.jnt_dofadr[jids]
        self.jrange = self.m.jnt_range[jids]
        self.trunk = self.m.body("trunk_base").id
        s = self.m.sensor("imu_ang_vel")
        self.gyro_adr = slice(s.adr[0], s.adr[0] + 3)
        self.delay = delay
        self.obs_noise = obs_noise
        self.reset_state()

    # --- state ---------------------------------------------------------------------------------
    def reset_state(self):
        mujoco.mj_resetData(self.m, self.d)
        self.last_action = np.zeros(14, np.float32)
        self.command = np.zeros(13, np.float32)
        self.lag = int(self.rng.integers(DELAY_LAG[0], DELAY_LAG[1] + 1)) if self.delay else 0
        self.target_hist = []
        self.bam.last_ts = self.d.time

    def place(self, kind: str, joint_noise: float = 0.05, random_joints: bool = False):
        """Starts mirroring microduck_rl's reset events (set_random_prone_orientation etc.)."""
        self.reset_state()
        r = self.rng
        yaw = r.uniform(-math.pi, math.pi)
        cy, sy, s = math.cos(yaw / 2), math.sin(yaw / 2), 2 ** -0.5
        q = DEFAULT_POSE.astype(float).copy()
        if kind == "standing":
            quat, z = np.array([cy, 0, 0, sy]), HOME_TRUNK_Z
        elif kind == "sitting":
            for i, v in SIT_OVERRIDES.items():
                q[i] = v
            joint_noise = max(joint_noise, 0.15)   # sitting_joint_noise_std 0.15
            tilt = r.uniform(-1, 1, 2) * math.radians(10)
            quat = _quat_mul(np.array([cy, 0, 0, sy]), _quat_mul(_axis_quat([1, 0, 0], tilt[0]), _axis_quat([0, 1, 0], tilt[1])))
            z = r.uniform(0.065, 0.08)
        elif kind == "face_down":
            quat, z = np.array([s * cy, -s * sy, s * cy, s * sy]), r.uniform(0.05, 0.09)
        elif kind == "face_up":
            quat, z = np.array([s * cy, s * sy, -s * cy, s * sy]), r.uniform(0.05, 0.09)
        else:
            raise ValueError(kind)
        if random_joints:   # randomize_servo_joints_uniform, range_frac 0.9, +6 cm clearance
            lo, hi = self.jrange[:, 0], self.jrange[:, 1]
            mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
            q = r.uniform(mid - half, mid + half)
            z += 0.06
        else:
            q = q + r.normal(0, joint_noise, 14)
        q = np.clip(q, self.jrange[:, 0], self.jrange[:, 1])
        self.d.qpos[:3] = [0, 0, z]
        self.d.qpos[3:7] = quat / np.linalg.norm(quat)
        self.d.qpos[self.qadr] = q
        self.bam.q_target[:] = q
        mujoco.mj_forward(self.m, self.d)

    # --- sensing, exactly duck-control obs.rs / infer_policy.get_observations -------------------
    def gravity(self):
        R = self.d.xmat[self.trunk].reshape(3, 3)
        return -R[2, :]          # R^T (0,0,-1)

    def tilt_deg(self):
        return math.degrees(math.acos(float(np.clip(-self.gravity()[2], -1, 1))))

    def trunk_z(self):
        return float(self.d.xpos[self.trunk][2])

    def obs(self):
        o = np.concatenate([
            self.d.sensordata[self.gyro_adr],
            self.gravity(),
            self.d.qpos[self.qadr] - DEFAULT_POSE,
            self.d.qvel[self.vadr],
            self.last_action,
            self.command,
        ]).astype(np.float32)
        if self.obs_noise:   # microduck_velocity_env_cfg actor noise: Unoise gyro .03, grav .01, qpos .001, qvel .25
            o[:48] += self.rng.uniform(-1, 1, 48).astype(np.float32) * NOISE
        return o

    # --- acting ---------------------------------------------------------------------------------
    def step(self, policy: Policy):
        action = policy(self.obs())
        self.last_action = action.astype(np.float32)
        target = DEFAULT_POSE + action * policy.action_scale
        for _ in range(DECIMATION):
            self.target_hist.append(target)
            if len(self.target_hist) > self.lag + 1:
                self.target_hist.pop(0)
            self.bam.q_target[:] = self.target_hist[0]
            self.bam.update()
            mujoco.mj_step(self.m, self.d)
        return action

    def frame(self):
        return np.concatenate([self.d.xpos[1:], self.d.xquat[1:]], 1).round(5).ravel().tolist()

    def body_names(self):
        return [self.m.body(b).name for b in range(1, self.m.nbody)]
