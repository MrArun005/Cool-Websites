"""Real-robot bodies for Bolt's envs: one table entry per robot (BODIES), so a new robot is added as data, not code.

    BOLT_BODY=go2 python train_imitate.py ...        # or  python train_imitate.py --body go2 ...
    ImitateEnv(body="go2")                            # one env; BoltEnv(body="go2") loads the same model
    JumpEnv(body="go2"), python train_jump.py --body go2   # the hurdle jump on it (jump_env.jump_sizes: lengths scaled)
    python robot_bodies.py go2                        # print what the entry resolves to (kinematics, motors, stance)

"bolt" (bolt.xml, bolt_env.py's own constants) is not in the table: it is the default and its code path is unchanged.
Supported robots: Unitree Go2 (first target, README "Target robot"). DEEP Robotics Lite3 is planned second; its entry
needs the fields below (its model files and licence are not in the repo yet).

What an entry holds (all of it data):
  xml          MJCF file (relative to train/), physics as published; visual meshes may be stripped (robots/bodies.py)
  licence      the model's licence, kept next to the file
  torso, feet  the trunk body's and the four paw geoms' names in that file (FL, FR, RL, RR)
  joints       the 12 joint names in OUR order: FL, FR, RL, RR x (abduction, thigh, knee) = qpos[7:19] = ctrl
  sdk_order    our index for each motor in the vendor SDK's order (for the robot-side loop; not used in simulation)
  stance       (abduction, thigh, knee) of the standing pose, front and rear (imitate_env's DOG on this body)
  stand_height body-centre height (m) in that stance, paws on the floor
  torque, speed  peak torque (N·m) and rated joint speed (rad/s) per (abduction, thigh, knee), cited
  power        peak power per joint (W), and cutoff_ratio: the drive fades to 0 torque by speed x cutoff_ratio
               (bolt_env's A1 model: 21 rated, out by 25); brake_gain: back-EMF braking (N·m·s/rad) per joint type,
               or None to scale bolt_env's A1 estimate by torque / speed
  kp, kv       the position servo the brain drives (the robot's own joint PD: target, kp, kd, no feed-forward)
  joint_damping  None keeps the file's joint damping; a number replaces it (env var BOLT_JOINT_DAMPING overrides both)
  clip_prefix  the clips retargeted onto this body: dogmotion/clips/<prefix><clip>.npz (dogmotion/retarget_body.py)
  floor_ok, lost  imitate_env's body-size limits (Bolt: 0.30 and 0.15 m), scaled by stand height
  pose_ok      pose_reached's sit/lie/stand thresholds on this body's own clip end poses
  lie, sit     folded legs for retargeting sit and lie: (abduction, thigh, knee) front and rear
The leg kinematics (abduction joint positions, hip offset, thigh and shin lengths, paw radius) are MEASURED from the
model (Body.kin), not typed in. The leg solver assumes Unitree's convention, which load() checks: the zero pose has
the legs straight down, abduction about +x, thigh and knee about +y, every knee bending backward (negative knee
range). A robot with another convention needs an entry field for joint signs/offsets (a code change, small).
"""
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
LEGS = ("FL", "FR", "RL", "RR")
A1_BRAKE = 3.5 / (33.5 / 21.0)   # bolt_env.BRAKE_GAIN over the A1's torque / rated speed (see bolt_env.MOTOR)

BODIES = {
    "go2": dict(
        name="Unitree Go2", xml="robots/unitree_go2/go2_nomesh.xml",
        licence="BSD-3-Clause, (c) Unitree Robotics (robots/unitree_go2/LICENSE); MuJoCo Menagerie",
        torso="base", feet=("FL", "FR", "RL", "RR"),
        joints=[f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "calf")],
        sdk_order=[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],   # unitree_sdk2 LowCmd motors: FR, FL, RR, RL x (hip, thigh, calf)
        # paws straight under the thigh joints, thigh 0.85, calf -1.7: body 2 x 0.213 cos 0.85 + 0.022 = 0.304 m up, the
        # ~0.30 m of Unitree's stand (Menagerie's "home", 0.9 / -1.8, is 0.27 m). The calf cannot open past -0.84 rad, so the
        # longest leg is 0.389 m; the stance keeps ~10 cm of it for strides.
        stance=dict(front=(0.0, 0.85, -1.7), rear=(0.0, 0.85, -1.7)), stand_height=0.304,
        # Peak torque: go2.xml ctrlrange (23.7 N·m hip/thigh, 45.43 N·m calf: the calf drive has an extra ~1.9x reduction).
        # Speed: the velocity limits of Unitree's go2_description URDF (unitree_ros), 30.1 rad/s hip/thigh, 15.7 calf.
        # APPROXIMATE (assumed, Unitree publishes no torque-speed curve): the URDF speed taken as the rated speed, drive out
        # by 25/21 of it (35.8 / 18.7 rad/s) as Bolt's A1 model; 700 W peak (23.7 x 30.1 = 45.4 x 15.7 = ~713 W, so it hardly
        # binds); braking gain scaled from the A1 estimate by torque / speed: 1.7 hip/thigh, 6.3 calf (gear ratio squared).
        torque=(23.7, 23.7, 45.43), speed=(30.1, 30.1, 15.7), power=700.0, cutoff_ratio=25.0 / 21.0, brake_gain=None,
        # Published Go2 joint PD: kp 20-40, kd 0.5-1 (legged_gym / unitree_rl_gym's Go2 config 20 / 0.5). 40 / 1.0, the top of
        # that range: held in the stance with no brain the body sags to 0.21 m at kp 20, 0.24 at 30, 0.26 at 40 (measured, 2 s).
        kp=40.0, kv=1.0,
        # Menagerie's joint damping 2 N·m·s/rad kept, as the port plan asks. It is a generic Menagerie value (the A1 file has
        # it too), not a Go2 measurement, and large: at the trot clip's 10.5 rad/s thigh speed it resists with 21 N·m.
        joint_damping=None,
        clip_prefix="go2_",
        floor_ok=0.21, lost=0.105,   # Bolt's 0.30 / 0.15 m x 0.304 / 0.436 (stand heights)
        # the sit clip ends at 24 cm nose up 0.6 rad, lie at 15 cm level, getup back at 29 cm (retarget_body.py), with
        # Bolt's margins: sit < 27 cm and nose up > 0.3 rad, lie < 17 cm, stand > 27 cm (Bolt's 38 of 41 cm)
        pose_ok={"sit": dict(height_max=0.27, pitch_min=0.3), "lie": dict(height_max=0.17, up_min=0.9),
                 "stand": dict(height_min=0.27, up_min=0.9)},
        lie=dict(front=(0.0, 1.15, -2.65), rear=(0.0, 1.15, -2.65)),   # calves tucked to their limit under the thighs
        sit=dict(front=(0.0, 0.35, -0.9), rear=(0.0, 1.9, -2.65)),     # front legs nearly straight, rear folded under
    ),
    # "lite3": dict(name="DEEP Robotics Lite3", xml=..., ...)   # planned; see the module docstring for the fields
}


def legs(p):
    """12 joint values from an entry's {front: (abd, thigh, knee), rear: (...)}: the same on both sides."""
    return np.array([*p["front"], *p["front"], *p["rear"], *p["rear"]], dtype=float)


class Body:
    """One table entry, resolved: stance, motor model (bolt_env's MOTOR keys, per joint arrays), model xml, kinematics."""

    def __init__(self, key):
        assert key in BODIES, f"unknown body {key!r}: bolt or one of {sorted(BODIES)}"
        e = self.entry = BODIES[key]
        self.key, self.joints, self.clip_prefix = key, list(e["joints"]), e["clip_prefix"]
        self.stance, self.lie, self.sit = legs(e["stance"]), legs(e["lie"]), legs(e["sit"])
        self.floor_ok, self.lost, self.pose_ok, self.stand_height = e["floor_ok"], e["lost"], e["pose_ok"], e["stand_height"]
        T, W = np.tile(e["torque"], 4).astype(float), np.tile(e["speed"], 4).astype(float)
        gain = np.round(A1_BRAKE * T / W, 1) if e["brake_gain"] is None else np.tile(e["brake_gain"], 4).astype(float)
        self.motor = dict(torque=T, speed=W, cutoff=W * e["cutoff_ratio"], power=np.full(12, float(e["power"])),
                          brake_speed=W * e["cutoff_ratio"], brake_gain=gain)
        damping = os.environ.get("BOLT_JOINT_DAMPING")
        self.joint_damping = float(damping) if damping else e["joint_damping"]
        self._xml, self.kin = None, None

    def xml(self):
        """The model as Bolt's envs load it: the published physics; position servos (kp, kv; forcerange per joint, rewritten
        every physics step by bolt_env.physics from self.motor) in place of whatever actuators the file has; timestep
        0.002 s (50 Hz control = 10 steps) and implicitfast (stable with stiff servos); the trunk named "torso", the paws
        "FL_foot".. (bolt_env looks them up by name); Bolt's floor, light and sky; collision geoms shown."""
        if self._xml is not None:
            return self._xml
        e = self.entry
        s = mujoco.MjSpec.from_file(str(HERE / e["xml"]))
        s.option.timestep, s.option.integrator = 0.002, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for k in list(s.keys):
            s.delete(k)
        for a in list(s.actuators):
            s.delete(a)
        s.body(e["torso"]).name = "torso"
        for leg, g in zip(LEGS, e["feet"]):
            s.geom(g).name = f"{leg}_foot"
            s.geom(f"{leg}_foot").rgba = [0.21, 0.23, 0.27, 1]
        if self.joint_damping is not None:
            for j in self.joints:
                s.joint(j).damping = [self.joint_damping] + [0.0] * (len(s.joint(j).damping) - 1)
        for i, j in enumerate(self.joints):
            jn = s.joint(j)
            a = s.add_actuator(name=j.replace("_joint", ""), target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_position(kp=e["kp"], kv=e["kv"])
            a.ctrlrange, a.ctrllimited = list(jn.range), 1
            t = float(self.motor["torque"][i])
            a.forcerange, a.forcelimited = [-t, t], 1
        for g in s.geoms:   # the collision geoms are what there is to see (no meshes): visible, Go2-grey
            if g.group == 3:
                g.group = 0
                if g.rgba[3] == 1 and tuple(g.rgba[:3]) == (0.5, 0.5, 0.5):
                    g.rgba = [0.62, 0.64, 0.68, 1]
        v = s.visual
        v.headlight.ambient, v.headlight.diffuse, v.headlight.specular = [0.3] * 3, [0.35] * 3, [0.1] * 3
        v.global_.offwidth, v.global_.offheight = 960, 540
        s.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=[0.82, 0.86, 0.9], rgb2=[0.98, 0.98, 0.97], width=256, height=256)
        s.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                      rgb1=[0.78, 0.77, 0.73], rgb2=[0.7, 0.69, 0.65], width=512, height=512)
        mat = s.add_material(name="grid", texrepeat=[12, 12], reflectance=0.05)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        s.worldbody.add_light(pos=[1, 1, 3], dir=[-0.3, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                              castshadow=True, diffuse=[0.6, 0.6, 0.6])
        s.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], material="grid",
                             friction=[1.0, 0.02, 0.01])
        self._xml = s.to_xml()
        return self._xml

    def model(self):
        m = mujoco.MjModel.from_xml_string(self.xml())
        if self.kin is None:
            self.kin = measure(m, self)
        return m


def measure(m, body):
    """Leg kinematics from the model at its zero pose (body frame = world, trunk at the origin): abduction joint
    positions, the hip offset (abduction axis to thigh plane), thigh length, the paw centre in the knee frame, paw
    radius; and the convention checks of the module docstring."""
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0   # the trunk at the origin, level; every joint at 0
    mujoco.mj_kinematics(m, d)
    jid = [m.joint(j).id for j in body.joints]
    k = {}
    for i, leg in enumerate(LEGS):
        a, t, kn = jid[3 * i:3 * i + 3]
        for j, ax in ((a, [1, 0, 0]), (t, [0, 1, 0]), (kn, [0, 1, 0])):
            assert np.allclose(m.jnt_axis[j], ax), f"{body.key}: joint {m.joint(j).name} axis {m.jnt_axis[j]} (want {ax})"
        assert m.jnt_range[kn][1] <= 0, f"{body.key}: knee {m.joint(kn).name} must bend backward (negative range)"
        foot = m.geom(f"{leg}_foot").id
        pa, pt, pk, pf = d.xanchor[a], d.xanchor[t], d.xanchor[kn], d.geom_xpos[foot]
        assert abs(pk[0] - pt[0]) < 1e-6 and pk[2] < pt[2], f"{body.key}: {leg} thigh not straight down at the zero pose"
        k[leg] = dict(abd=pa.copy(), offset=float(pt[1] - pa[1]), thigh=float(pt[2] - pk[2]), paw=pf - pk,
                      r=float(m.geom_size[foot][0]))
    fl = k["FL"]
    for leg in LEGS:   # one leg geometry, mirrored
        sx, sy = (1 if leg[0] == "F" else -1), (1 if leg[1] == "L" else -1)
        assert np.allclose(np.abs(k[leg]["abd"][:2]), np.abs(fl["abd"][:2]), atol=1e-4) and np.sign(k[leg]["abd"][0]) == sx \
            and np.sign(k[leg]["abd"][1]) == sy, f"{body.key}: {leg} abduction joint not the mirror of FL's"
        assert abs(abs(k[leg]["offset"]) - abs(fl["offset"])) < 1e-4 and abs(k[leg]["thigh"] - fl["thigh"]) < 1e-4
    paw = fl["paw"].copy()
    paw[1] = 0.0
    return dict(abd_x=float(fl["abd"][0]), abd_y=float(fl["abd"][1]), abd_z=float(fl["abd"][2]), hip_offset=fl["offset"],
                thigh=fl["thigh"], calf_paw=paw, paw_r=fl["r"])


_CACHE = {}


def get(key):
    """The resolved Body (model built once per process to measure the kinematics)."""
    if key not in _CACHE:
        b = Body(key)
        b.model()
        _CACHE[key] = b
    return _CACHE[key]


def leg_ik(kin, p, left, abd=None):
    """(abduction, thigh, knee) putting the paw centre at p = (x fwd, y left, z up) from the leg's abduction joint (m),
    in the body frame; and how far out of reach the paw was (m, 0 if reachable). Knees backward on every leg.
    abd: a given (e.g. clamped) abduction angle; thigh and knee are then solved in the leg plane it rolls to."""
    x, y, z = p
    d = kin["hip_offset"] if left else -kin["hip_offset"]
    if abd is None:
        zp = -np.sqrt(max(y * y + z * z - d * d, 1e-9))          # paw depth in the rolled leg plane
        abd = np.arctan2(z, y) - np.arctan2(zp, d)
        abd = float((abd + np.pi) % (2 * np.pi) - np.pi)
    zp = -np.sin(abd) * y + np.cos(abd) * z                        # p rolled back by abd: the plane's depth axis
    thigh, knee, short = plane_ik(kin, x, zp)
    return np.array([abd, thigh, knee]), short


def plane_ik(kin, x, z):
    """(thigh, knee, out of reach m) for the paw at (x fwd, z up) from the thigh joint, in the leg plane."""
    # the shin's paw vector is a straight link of length lc at angle delta: solve the triangle with the effective shin
    L, cp = kin["thigh"], kin["calf_paw"]
    lc, delta = np.hypot(cp[0], cp[2]), np.arctan2(-cp[0], -cp[2])
    dist = np.hypot(x, z)
    short = max(0.0, dist - (L + lc - 1e-6))
    dist = min(max(dist, abs(L - lc) + 1e-6), L + lc - 1e-6)
    bend = np.pi - np.arccos(np.clip((L ** 2 + lc ** 2 - dist ** 2) / (2 * L * lc), -1, 1))   # knee fold from straight
    beta = np.arctan2(-x, -z)                                   # thigh joint to paw, from straight down (+ = behind)
    alpha = np.arcsin(np.clip(lc * np.sin(bend) / dist, -1, 1))
    return float(beta + alpha), float(-bend - delta), float(short)


def paw_positions(kin, q):
    """Paw centres (4, 3) in the body frame for joint angles q (forward kinematics; checks leg_ik)."""
    ry = lambda th: np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])  # noqa: E731
    out = np.zeros((4, 3))
    for i, leg in enumerate(LEGS):
        a, t, c = q[3 * i:3 * i + 3]
        sx, sy = (1 if leg[0] == "F" else -1), (1 if leg[1] == "L" else -1)
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        paw = ry(t) @ (np.array([0, 0, -kin["thigh"]]) + ry(c) @ kin["calf_paw"])
        out[i] = np.array([sx * kin["abd_x"], sy * kin["abd_y"], kin["abd_z"]]) + rx @ (np.array([0, sy * kin["hip_offset"], 0.0]) + paw)
    return out


if __name__ == "__main__":
    b = get(sys.argv[1] if len(sys.argv) > 1 else "go2")
    print(b.entry["name"], "-", b.entry["licence"])
    print("kinematics:", {k: np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(v, 4) for k, v in b.kin.items()})
    print("stance:", b.stance.tolist(), "paws:", np.round(paw_positions(b.kin, b.stance), 4).tolist())
    print("motor:", {k: v[:3].round(2).tolist() for k, v in b.motor.items()})
