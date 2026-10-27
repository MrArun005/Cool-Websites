"""Fast running and turning, the way the fastest learned robot dogs do it (run 20).

Velocity tracking with no gait generator, after legged_gym (Rudin et al. 2021) and "Rapid Locomotion via
RL" (Margolis et al. 2022, Mini Cheetah 3.9 m/s real): the brain gets a command (forward and sideways
speed, turn rate) and is paid for tracking it; the gait is whatever works. The fastest command practised
widens as Bolt masters the current one (a speed curriculum), up to 4 m/s and 5 rad/s. Random shoves
(1 m/s, as legged_gym) and randomized friction, payload and motor strength make it robust.

Same real body as everything else: bolt.xml, the A1 motor model (bolt_env.physics), ±48° hip abduction.

Inputs (49): gravity in the body frame (3), body angular velocity (3), body linear velocity (3),
command vx, vy, yaw rate (3), joint angles minus the dog stance (12), joint speeds x0.05 (12), last
action (12), body height (1).
"""
import gymnasium as gym
import mujoco
import numpy as np

from bolt_env import MODEL, STANCES, physics

DOG = STANCES["dog"]
SCALE = np.tile([0.2, 0.5, 0.6], 4)   # rad per unit action: abd, hip, knee
MAX_V, MAX_W = 4.0, 5.0               # curriculum ceilings: Mini Cheetah ran 3.9 m/s, turned 5.7 rad/s

# Running like a dog (run 22): a style reward for being close to the nearest frame of a reference gait, joint
# angles and (scaled) joint speeds, with no clock (a simple stand-in for adversarial motion priors, Escontrela
# 2022). At 2 m/s and up: Bolt's own gallop, the run-out of the run 16f jump brain after landing (3.1 m/s,
# real A1 motors); from 0.6 to 2 m/s: the real dog's trot (motion capture, retargeted).
def _style_set(name):
    from pathlib import Path
    c = np.load(Path(__file__).with_name("dogmotion") / "clips" / f"{name}.npz")
    return np.concatenate([c["joints"], 0.05 * c["joint_vel"]], 1).astype(np.float32)


class RunEnv(gym.Env):
    OBS = 49

    def __init__(self, randomize=True, seed=None, max_steps=1000, pushes=True, turn_weight=0.75, turn_focus=0.0, style_weight=0.0):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.frame_skip, self.dt = 10, m.opt.timestep * 10
        self.randomize, self.max_steps, self.pushes = randomize, max_steps, pushes
        # turn_weight: pay for tracking the turn rate; turn_focus: share of commands that are fast turns or spins (run 21)
        self.turn_weight, self.turn_focus = turn_weight, turn_focus
        self.style_weight = style_weight
        self.styles = dict(gallop=_style_set("bolt_gallop"), trot=_style_set("trot")) if style_weight else None
        self.rng = np.random.default_rng(seed)
        self.torso = m.body("torso").id
        self.foot_ids = [m.geom(f"{l}_foot").id for l in ("FL", "FR", "RL", "RR")]
        legs = {m.body(f"{l}_{p}").id for l in ("FL", "FR", "RL", "RR") for p in ("thigh", "shin")}
        self.knee_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in legs and g not in self.foot_ids}
        self.body_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.torso}
        self.floor = m.geom("floor").id
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.nominal = dict(kp=m.actuator_gainprm[:, 0].copy(), mass=m.body_mass[self.torso], fric=m.geom_friction[:, 0].copy())
        self.top_v, self.top_w = 1.0, 1.0   # current curriculum ranges
        self.track_log = []                 # tracking quality of recent episodes asked to go near the top speed
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.exam_command = None            # (vx, vy, w): exams fix the command

    # ---------- helpers ----------
    def R(self):
        return self.data.xmat[self.torso].reshape(3, 3)

    def v_body(self):
        return self.R().T @ self.data.qvel[0:3]

    def _command(self):
        if self.exam_command is not None:
            return np.array(self.exam_command, float)
        r = self.rng.random()
        vx = self.rng.uniform(-1.0, self.top_v) if r > 0.1 else 0.0
        vx = self.rng.uniform(0.6, 1.0) * self.top_v if r > 0.6 else vx   # often near the current top speed
        vy = self.rng.uniform(-0.5, 0.5) if self.rng.random() < 0.3 else 0.0
        w = self.rng.uniform(-self.top_w, self.top_w) if self.rng.random() < 0.6 else 0.0
        if self.rng.random() < self.turn_focus:   # a fast turn or a spin on the spot
            vx, vy = self.rng.uniform(0.0, 1.0), 0.0
            w = self.rng.choice([-1, 1]) * self.rng.uniform(0.5, 1.0) * self.top_w
        return np.array([vx, vy, w])

    def _obs(self):
        d, R = self.data, self.R()
        return np.concatenate([R.T @ [0, 0, -1], R.T @ d.qvel[3:6] * 0.25, self.v_body(), self.cmd * [1, 1, 0.5],
                               d.qpos[7:19] - DOG, d.qvel[6:18] * 0.05, self.last_action, [d.qpos[2]]]).astype(np.float32)

    def contacts(self):
        """(knee/shin/thigh touches the floor, body touches the floor, paws on the floor)."""
        knee = body = False
        feet = np.zeros(4, bool)
        for c in self.data.contact[:self.data.ncon]:
            g = c.geom2 if c.geom1 == self.floor else c.geom1 if c.geom2 == self.floor else None
            if g is None:
                continue
            if g in self.foot_ids:
                feet[self.foot_ids.index(g)] = True
            elif g in self.knee_geoms:
                knee = True
            elif g in self.body_geoms:
                body = True
        return knee, body, feet

    # ---------- episode ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:   # legged_gym-style randomization
            kp = self.nominal["kp"] * self.rng.uniform(0.9, 1.1)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[self.torso] = self.nominal["mass"] + self.rng.uniform(-1.0, 2.0)
            m.geom_friction[:, 0] = self.nominal["fric"] * self.rng.uniform(0.5, 1.25)
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[2] = 0.45
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:19] = d.ctrl[:] = np.clip(DOG + self.rng.normal(0, 0.03, 12), self.lo, self.hi)
        mujoco.mj_forward(m, d)
        for _ in range(20):
            physics(m, d, self.frame_skip)
        self.t, self.last_action, self.air = 0, np.zeros(12), np.zeros(4)
        self.cmd = self._command()
        self.track_sum, self.track_n = 0.0, 0
        return self._obs(), {}

    def _curriculum(self):
        """Widen the speed range once Bolt tracks commands near the current top well (Margolis 2022)."""
        if self.exam_command is not None or self.track_n < 200:
            return
        if self.cmd[0] > 0.7 * self.top_v:
            self.track_log = (self.track_log + [self.track_sum / self.track_n])[-20:]
            if len(self.track_log) == 20 and np.mean(self.track_log) > 0.75:
                self.top_v, self.track_log = min(MAX_V, self.top_v + 0.5), []
                self.top_w = min(MAX_W, self.top_w + 0.5)

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(action, -1, 1)
        d.ctrl[:] = np.clip(DOG + SCALE * a, self.lo, self.hi)
        if self.pushes and self.exam_command is None and self.t > 0 and self.t % 400 == 0:   # a shove every 8 s
            d.qvel[0:2] += self.rng.uniform(-1.0, 1.0, 2)
        physics(m, d, self.frame_skip)
        self.t += 1
        if self.exam_command is None and self.t % 500 == 0:   # a new command every 10 s
            self.cmd = self._command()

        R, vb, w = self.R(), self.v_body(), d.qvel[3:6]
        knee, body, feet = self.contacts()
        # paw air time, paid on touchdown when it took a real step (> 0.25 s): rewards proper strides
        first = feet & (self.air > 0)
        moving = np.linalg.norm(self.cmd) > 0.1
        air_pay = float(np.sum((self.air - 0.25) * first)) if moving else 0.0
        self.air = np.where(feet, 0.0, self.air + self.dt)

        e_v = np.sum((self.cmd[:2] - vb[:2]) ** 2)
        track_v = np.exp(-e_v / max(0.25, 0.1 * self.cmd[0] ** 2))   # tolerance grows with speed, as in fast-running work
        # tolerance grows with the asked turn rate, as for speed: with a fixed 0.25 a 3 rad/s turn
        # scored ~0 for every try, so there was nothing to learn from (run 20 turned 0.2 rad/s of 3)
        track_w = np.exp(-(self.cmd[2] - w[2]) ** 2 / max(0.25, 0.1 * self.cmd[2] ** 2))
        torque = d.actuator_force
        style = self.style() if self.styles is not None else 0.0
        r = (1.5 * track_v + self.turn_weight * track_w
             - 2.0 * vb[2] ** 2 - 0.05 * np.sum(w[:2] ** 2) - 1.0 * np.sum((R.T @ [0, 0, -1])[:2] ** 2)
             - 2e-4 * np.sum(torque ** 2) - 0.01 * np.sum((a - self.last_action) ** 2)
             - 1.0 * knee + 1.0 * air_pay + self.style_weight * style)
        self.last_action = a
        self.track_sum += track_v
        self.track_n += 1
        fell = bool(body or R[2, 2] < 0.3 or d.qpos[2] < 0.15)
        trunc = self.t >= self.max_steps
        if fell:
            r -= 10.0
        if fell or trunc:
            self._curriculum()
        info = dict(style=float(style), track_v=float(track_v), vx=float(vb[0]), w=float(w[2]), top_v=self.top_v, fell=fell)
        return self._obs(), float(r), fell, trunc, info


    def style(self):
        """0..1: how close the legs are to the nearest frame of the reference gait for the asked speed."""
        v = self.cmd[0]
        if v < 0.6:
            return 0.0
        ref = self.styles["gallop" if v >= 2.0 else "trot"]
        f = np.concatenate([self.data.qpos[7:19], 0.05 * self.data.qvel[6:18]])
        return float(np.exp(-np.min(np.mean((ref - f) ** 2, 1)) / 0.1))   # run 20's own gait scores ~0.2, the gallop ~0.9


class RunPolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl"):
        from pathlib import Path
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: RunEnv()]))
        self.norm.training = False

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, episodes=3, seed=7000):
    """Speed kept at each asked speed (achieved / asked over the last 3 s of a 5 s run), turn rate kept,
    falls, and push survival (a 1.5 m/s sideways shove while trotting at 1 m/s)."""
    env = RunEnv(randomize=True, pushes=False, max_steps=250, style_weight=1.0)   # weight only switches the style measure on
    out = {}
    for name, cmd in (("v1", (1, 0, 0)), ("v2", (2, 0, 0)), ("v3", (3, 0, 0)), ("v4", (4, 0, 0)), ("w3", (0.5, 0, 3.0)), ("w5", (0.5, 0, 5.0))):
        kept, falls, style = [], 0, []
        for k in range(episodes):
            env.exam_command = cmd
            obs, _ = env.reset(seed=seed + k)
            got = []
            for t in range(250):
                obs, _, fell, trunc, info = env.step(policy.act(obs))
                if t >= 100:
                    got.append(info["w"] if name[0] == "w" else info["vx"])
                    style.append(info["style"])
                if fell:
                    break
            falls += fell
            kept.append(0.0 if fell else np.mean(got) / (cmd[2] if name[0] == "w" else cmd[0]))
        out[f"speed_{name}" if name[0] == "v" else f"turn_{name}"] = round(float(np.mean(kept)), 2)
        out[f"fall_{name}"] = round(falls / episodes, 2)
        if name in ("v2", "v3"):   # how dog-like the gait is: 1 = exactly the reference trot / gallop
            out[f"style_{name}"] = round(float(np.mean(style)) if style else 0.0, 2)
    survived = 0
    for k in range(episodes * 2):
        env.exam_command = (1, 0, 0)
        obs, _ = env.reset(seed=seed + 100 + k)
        fell = False
        for t in range(250):
            if t == 100:
                env.data.qvel[1] += 1.5 * (1 if k % 2 else -1)
            obs, _, fell, _, _ = env.step(policy.act(obs))
            if fell:
                break
        survived += not fell
    out["push_1.5"] = round(survived / (episodes * 2), 2)
    return out


def score(r):
    """Fastest speed held with no falls (m/s) + turn rate held, for picking the best brain."""
    top = max([v for v, k in ((1, "v1"), (2, "v2"), (3, "v3"), (4, "v4")) if r[f"fall_{k}"] == 0 and r[f"speed_{k}"] > 0.8] or [0])
    return round(top + 0.3 * min(r["turn_w3"], 1.0) * (r["fall_w3"] == 0) + 0.5 * r["push_1.5"]
                 + 2.0 * (r.get("style_v2", 0) + r.get("style_v3", 0)) / 2, 3)   # a dog's gait counts as much as 2 m/s of speed
