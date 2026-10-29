"""Teach Bolt to really jump: clear a bar too high to step over, from standing, and land on its feet.

Bolt's belly is ~36 cm off the ground, so it steps over bars up to ~25 cm without leaving the ground
(bolt_env HURDLE_XML). A 40 cm bar can only be cleared by jumping: the whole body has to go up and
over. A hand-made leap (run 14) reared up and tipped over backwards in all 64 versions tried, so this
brain learns the jump from scratch, with the full motor range (not small corrections to a plan).

    python train_jump.py --name jump1 --steps 6000000

Each try: Bolt stands 45-65 cm in front of a bar, gets 2 s, and is paid for moving its hindmost paw
past the bar, a big bonus once all four are past (less if it touched the bar), then for landing and
standing tall. Every moment still in front of the bar costs a little; touching the bar costs; a fall
ends the try. The bar height adapts, like a coach's: it rises 0.5 cm while clean jumps are common (over
60%), and drops when they are rare, from 12 cm up to 48 cm.

Run 15b (paid for standing upright before the jump, bars from 20 cm) learned to stand still and hop in
place: staying put paid steadily, trying risked touch and fall penalties. Run 15c's first 1M steps
learned to launch (all paws up to 22 cm, 0.4 s in the air) but dived over and crashed every time: the
clear bonus beat the fall penalty. So most of the bonus now waits for a landing: still upright 0.5 s
after clearing. By 2.5M it cleared 25 cm bars and landed, but touched the bar every time (sliding over
its top), so touching now costs 5 at once and 1 per step, and only clean jumps earn the big bonuses.
At 6.5M it jumped 15 cm bars clean 11 times in 12 and 20 cm 9 in 12; by 7M it had collapsed (0 clean even
at 10 cm) right after the coach raised the practice bars from 13 to 20 cm in 1M steps. Run 15d resumes
from 6.5M with a third of the learning rate and a coach half as quick to raise the bar: it dipped the
same way at 7M and stayed down. Watching the jumps showed why: both still cleared 15 cm bars cleanly, but
leapt harder and crashed half a second after landing, because a crash after clearing (-10) cost less than
the clear bonus plus the progress pay. Run 15e: a crash after clearing costs 25, and speed before the bar
is paid only up to 1 m/s (was 2). Run 15e still broke within 500k steps (the third run from 6.5M to do so):
each update saw only 4096 steps (~40 jumps), too noisy a signal. Run 15f: 4x the data per update, from
15c's 8M brain (landing again, practising 25 cm bars).

Run 16 (real=True): a real agility hurdle (hurdle.py: solid wing stands, a loose bar in cups that drops
if knocked) instead of a solid board. In the games, run 15f's brains landed twisted 40-120 degrees off
the way they jumped, so run 16 starts each try up to 11 degrees askew and up to 10 cm off the centre
line, feels both (two more inputs: sin of the heading error, distance off the line), and is paid for
landing facing along the jump. A dropped bar costs 5 more, and only a dropped bar spoils a jump (the
agility rule); a touch still costs. It still cleared every bar but by 11M fell after every landing: the
landing bonus needed Bolt facing within 20 degrees, a cliff it rarely reached. Run 16b instead charges a
little for spinning all through the jump (where the twist starts) and scales the landing bonus by how
straight Bolt lands. It fell after 8 landings in 12 by 11M too: the per-step charge for facing sideways
after landing added up to about the -25 for a fall, and a fall ends the episode and the charges, so a
twisted Bolt was better off falling. Run 16c: no charge after landing, standing still pays whichever way
Bolt faces; only the spin charge and the landing bonus care about the heading.
"""
import gymnasium as gym
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bolt_env import ACTION_SCALE, STANCES, BoltEnv, physics

DOG = STANCES["dog"]


EXAM_LATENCY_MS = 0.0   # fixed control delay (ms) for exams (fair_exam.py --latency-ms)


class JumpEnv(BoltEnv):
    OBS = 49   # 51 with real=True

    def __init__(self, randomize=True, seed=None, max_steps=100, real=False, approach=(0.45, 0.65), approach_speed=0.0,
                 approach_speed_min=1.0, run_frac=0.5, gait_start=0.0, bar_max=0.50, motor_brake=None, under_margin=0.0,
                 handover_file=None, handover_rate=0.0, latency_ms=(0.0, 0.0)):
        # under_margin (m): the body's centre must be at least bar height + this when it crosses the bar's line, or
        # Bolt went UNDER the bar (28 Oct: bars of 60-75 cm are higher than Bolt's back, and "all paws past the bar"
        # counted running under a 70 cm bar as a clean jump). 0 = centre above the bar top; a trunk that crosses
        # lower than that either hits the bar or passes beneath it. Going under ends the try as a failure.
        self.under_margin = under_margin
        # handover_file: real trot -> jump hand-over states recorded in MasterMind's world (research/record_handover.py);
        # handover_rate of the training tries start from one of them (body state as recorded, the bar where it was,
        # a practice height), instead of the synthetic mid-stride pose. Exams: exam_handover = a state index.
        self.handover = dict(np.load(handover_file)) if handover_file else None
        self.handover_rate, self.exam_handover = handover_rate, None
        # latency_ms=(lo, hi): the motors get the targets sent that long ago, drawn per try (imitate_env's real-robot
        # gap; 29 Oct: at 20 ms MasterMind's teacher chain jumps 0-1/30, the jump brain was never trained with it).
        # exam_latency_ms: a fixed latency for exams.
        self.latency_ms, self.exam_latency_ms, self.delay_sub, self._queue = tuple(latency_ms), EXAM_LATENCY_MS or None, 0, None
        super().__init__(randomize=randomize, max_steps=max_steps, seed=seed, gait_generator="dog",
                         hurdles="real" if real else True, skills=["stand"], motor_brake=motor_brake)
        self.real = real
        self.OBS = 51 if real else 49
        if real:
            import hurdle
            self.bar_geom = self.model.geom("bar0").id
            self.stand_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.model.body("hurdle0").id}
            self.bar_q = self.model.jnt_qposadr[self.model.joint("bar0").id]
            self.bar_v = self.model.jnt_dofadr[self.model.joint("bar0").id]
            self.exam_askew = None   # (heading error rad, metres off the line) for exams
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.OBS,), np.float32)
        self.exam_bar = None  # (distance, height): exams set the bar; training draws it
        # approach: how far ahead the bar starts (m). Up to run 16h 0.45-0.65 m, a standing jump; run 16i trains
        # from 1.5-3 m so Bolt runs up and picks its own take-off, as dogs clear high bars (more steps per try)
        self.approach = approach
        # approach_speed: half the tries start already running at the bar (1 m/s up to this, e.g. 3 = a canter),
        # the way a dog arrives at a jump on a course (run 16i). 0: always from standing.
        self.approach_speed = approach_speed
        # run "jump-trot" (from 16i): running tries start at approach_speed_min..approach_speed (1.2-2.4: trot to
        # canter, as MasterMind hands over), run_frac of all tries start running (16i: 0.5), and gait_start of the
        # running tries start mid-stride in a trot/canter pose taken from the dog clips (legs and joint speeds of
        # a random moment of the gait) instead of the settled dog stance: in MasterMind the jump brain takes over
        # from the trot brain mid-stride. bar_max: the highest practice bar (16i: 0.50). Defaults = old runs.
        self.approach_speed_min, self.run_frac, self.gait_start, self.bar_max = approach_speed_min, run_frac, gait_start, bar_max
        self.exam_gait = False   # exams: start mid-stride (with exam_speed)
        self._clips = None
        self.exam_speed = None   # exams: start speed (m/s) or None for standing
        # adaptive bar height, like a coach: it goes up (0.5 cm) while clean jumps are common, down when rare
        self.bar_level, self.clean_rate = 0.12, 0.0

    def _bar_height(self):
        return float(np.clip(self.rng.uniform(self.bar_level - 0.05, self.bar_level + 0.03), 0.06, self.bar_max))

    def _grade(self, clean):
        """After each training try: raise the bar when clean jumps are common, lower it when they are rare."""
        self.clean_rate = 0.9 * self.clean_rate + 0.1 * float(clean)
        if self.clean_rate > 0.6:
            self.bar_level = min(self.bar_max - 0.02, self.bar_level + 0.005)
        elif self.clean_rate < 0.15:
            self.bar_level = max(0.10, self.bar_level - 0.005)

    def paws_past(self):
        """How far each paw is past the bar, along the jump (negative: still in front of it)."""
        return np.array([np.dot(self.data.geom_xpos[f][:2] - self.hurdle_at, self.hurdle_dir) for f in self.foot_ids])

    def heading_error(self):
        """Angle from Bolt's heading to the jump direction (radians, positive: the bar's line is to the left)."""
        R = self._rotation()
        return float(np.arctan2(R[0, 0] * self.hurdle_dir[1] - R[1, 0] * self.hurdle_dir[0],
                                R[0, 0] * self.hurdle_dir[0] + R[1, 0] * self.hurdle_dir[1]))

    def off_line(self):
        """How far Bolt's body is to the left of the line through the middle of the bar (m)."""
        rel = self.data.qpos[:2] - self.hurdle_at
        return float(-rel[0] * self.hurdle_dir[1] + rel[1] * self.hurdle_dir[0])

    def place_real(self, ahead, height, askew, side):
        """The real hurdle `ahead` m in front, its line turned `askew` rad from Bolt's heading, Bolt `side` m
        left of that line; the bar resting in its cups."""
        import hurdle
        R = self._rotation()
        fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
        c, s = np.cos(askew), np.sin(askew)
        u = np.array([c * fwd[0] - s * fwd[1], s * fwd[0] + c * fwd[1]])
        n = np.array([-u[1], u[0]])
        self.hurdle_dir, self.hurdle_h = u, height
        self.hurdle_at = self.data.qpos[:2] + u * ahead - n * side
        pos, q, bar, bq = hurdle.rest_pose(self.hurdle_at, np.arctan2(u[1], u[0]), height)
        k = self.model.body_mocapid[self.model.body("hurdle0").id]
        self.data.mocap_pos[k], self.data.mocap_quat[k] = pos, q
        self.data.qpos[self.bar_q:self.bar_q + 3], self.data.qpos[self.bar_q + 3:self.bar_q + 7] = bar, bq
        self.data.qvel[self.bar_v:self.bar_v + 6] = 0
        self.bar_rest = bar.copy()
        mujoco.mj_forward(self.model, self.data)

    def hurdle_hits(self):
        if not self.real:
            return super().hurdle_hits()
        # Bolt touching a stand or the bar (not the bar in its cups, nor the bar on the floor)
        hurdle_geoms, floor = self.stand_geoms | {self.bar_geom}, self.model.geom("floor").id
        return sum((c.geom1 in hurdle_geoms) != (c.geom2 in hurdle_geoms) and floor not in (c.geom1, c.geom2)
                   for c in self.data.contact[:self.data.ncon])

    def bar_dropped(self):
        """The bar knocked out of its cups (moved 3 cm or more): the only fault in agility."""
        return self.real and float(np.linalg.norm(self.data.qpos[self.bar_q:self.bar_q + 3] - self.bar_rest)) > 0.03

    def _obs(self):
        d, R = self.data, self._rotation()
        obs = np.concatenate([
            R.T @ np.array([0, 0, -1.0]), d.qvel[3:6] * 0.25, R.T @ d.qvel[0:3], [d.qpos[2]],
            d.qpos[7:19] - DOG, d.qvel[6:18] * 0.1, self.last_action,
            [self.hurdle_ahead(), self.hurdle_h, self.t / self.max_steps],
        ] + ([[np.sin(self.heading_error()), self.off_line()]] if self.real else []))
        if self.randomize:
            obs += self.rng.normal(0, 0.01, obs.shape)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = self.np_random
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.randomize:  # the same physics randomization as the other brains
            kp = self.nominal_kp * self.rng.uniform(0.85, 1.15)
            m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 1] = kp, -kp
            m.body_mass[:] = self.nominal_mass
            m.body_mass[self.torso] = self.nominal_mass[self.torso] * self.rng.uniform(0.85, 1.15)
            m.geom_friction[:, 0] = self.nominal_friction * self.rng.uniform(0.6, 1.3)
        yaw = self.rng.uniform(-np.pi, np.pi)
        d.qpos[2] = 0.45
        d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qpos[7:19] = d.ctrl[:] = np.clip(DOG + self.rng.normal(0, 0.02, 12), self.lo, self.hi)
        mujoco.mj_forward(m, d)
        for _ in range(25):  # settle into the dog stance on the motors
            physics(m, d, self.frame_skip)
        exam = self.exam_bar is not None
        ho = self.exam_handover if exam else (
            int(self.rng.integers(len(self.handover["ahead"]))) if self.handover is not None and self.rng.random() < self.handover_rate else None)
        if ho is not None:
            return self._handover_start(ho)
        v0 = self.exam_speed if exam else (
            self.rng.uniform(self.approach_speed_min, self.approach_speed) if self.approach_speed and self.rng.random() < self.run_frac else 0.0)
        if v0 and ((exam and self.exam_gait) or (not exam and self.gait_start and self.rng.random() < self.gait_start)):
            self._gait_pose(v0)
        dist, h = self.exam_bar or (self.rng.uniform(*self.approach), self._bar_height())
        if self.real:   # like a game: up to 11 degrees askew and 10 cm off the line
            askew, side = self.exam_askew or (self.rng.uniform(-0.2, 0.2), self.rng.uniform(-0.1, 0.1))
            self.place_real(dist, h, askew, side)
        else:
            self.place_hurdle(dist, h)
        if v0:   # arriving at a run: the body moves at v0 along its heading
            R = self._rotation()
            d.qvel[0:2] = v0 * R[:2, 0] / np.linalg.norm(R[:2, 0])
        self.t, self.last_action = 0, np.zeros(12)
        self.start_try()
        self._start_latency()
        return self._obs(), {}

    def _handover_start(self, k):
        """Start from recorded hand-over k: Bolt's body as MasterMind handed it over, the bar where it stood then."""
        H, d = self.handover, self.data
        d.qpos[2:19], d.qvel[0:18], d.ctrl[:] = H["qpos"][k], H["qvel"][k], H["ctrl"][k]
        mujoco.mj_forward(self.model, d)
        h = self.exam_bar[1] if self.exam_bar is not None else self._bar_height()
        self.place_real(float(H["ahead"][k]), h, float(H["heading_err"][k]), float(H["off_line"][k]))
        self.t, self.last_action = 0, np.clip((d.ctrl - DOG) / ACTION_SCALE, -1, 1)
        self.start_try()
        self._start_latency()
        return self._obs(), {}

    def _start_latency(self):
        lat = self.exam_latency_ms if self.exam_latency_ms is not None else (
            self.rng.uniform(*self.latency_ms) if self.latency_ms[1] > 0 else 0.0)
        self.delay_sub = int(round(lat / 1000.0 / self.model.opt.timestep))
        if self.delay_sub:
            self._queue = [self.data.ctrl.copy() for _ in range(self.delay_sub // self.frame_skip + 2)]

    def _delayed_physics(self):
        """One control step with latency (imitate_env._delayed_physics, no motor lag)."""
        m, d = self.model, self.data
        sent = d.ctrl.copy()
        self._queue = self._queue[1:] + [sent]
        back, sub = divmod(self.delay_sub, self.frame_skip)
        target, older = self._queue[-1 - back], self._queue[-2 - back]
        for i in range(self.frame_skip):
            d.ctrl[:] = older if i < sub else target
            physics(m, d, 1)
        d.ctrl[:] = sent

    def start_try(self):
        """The per-try bookkeeping, for reset and for brains handed the jump mid-run (skills.Bolt, MasterMind)."""
        self.touched = self.cleared = self.landed = self.dropped = self.under = False
        self.clear_t = self.cross_z = None
        self.prev_ahead = self.hurdle_ahead()
        self.best_past = self.paws_past().min()

    def _gait_pose(self, v0):
        """Mid-stride: the legs (and joint speeds) and the body's tilt and height at a random moment of the
        dog's trot (below 2 m/s) or canter clip, keeping the heading; joint speeds scaled to the cadence of v0."""
        from imitate_env import Clip, _rotz
        if self._clips is None:
            self._clips = {n: Clip(n) for n in ("trot", "canter")}
        c = self._clips["trot" if v0 < 2.0 else "canter"]
        k = int(self.rng.integers(c.n))
        m, d = self.model, self.data
        R = self._rotation()
        yaw = np.arctan2(R[1, 0], R[0, 0])
        Rw = _rotz(yaw) @ c.tilt[k]
        u, _, vt = np.linalg.svd(Rw)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (u @ vt).reshape(-1))
        d.qpos[2], d.qpos[3:7] = c.height[k] + 0.005, q
        d.qpos[7:19] = d.ctrl[:] = np.clip(c.joints[k], self.lo, self.hi)
        clip_speed = max(0.5, float(np.linalg.norm(c.v_body[:, :2], axis=1).mean()))
        d.qvel[:] = 0
        d.qvel[3:6] = c.w_body[k] * 0.5
        d.qvel[6:18] = c.joint_vel[k] * float(np.clip(v0 / clip_speed, 0.6, 1.2))
        mujoco.mj_forward(m, d)
        pen = max([0.0] + [-x.dist for x in d.contact[:d.ncon]])   # never start with a paw in the floor
        if pen > 0:
            d.qpos[2] += pen + 0.002
            mujoco.mj_forward(m, d)

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        self.data.ctrl[:] = np.clip(DOG + ACTION_SCALE * a, self.lo, self.hi)  # the full motor range
        if self.delay_sub:
            self._delayed_physics()
        else:
            physics(self.model, self.data, self.frame_skip)
        self.t += 1
        self.env_steps += 1
        d, R = self.data, self._rotation()
        p = self.posture()
        hits = self.hurdle_hits()
        first_touch = hits > 0 and not self.touched
        self.touched |= hits > 0
        past = self.paws_past()
        upright = np.exp(-(p["roll"] / 0.4) ** 2) * np.exp(-(p["pitch"] / 0.6) ** 2)
        reward = -1.0 * (hits > 0) - 5.0 * first_touch   # touching the bar is a knocked bar
        if self.real and not self.dropped and self.bar_dropped():
            self.dropped = True
            reward -= 5.0   # knocked the bar out of its cups: a fault
        e = self.heading_error() if self.real else 0.0
        facing = np.exp(-(e / 0.35) ** 2)   # 1 facing along the jump, ~0.5 at 17 degrees off
        just_cleared = False
        ahead_now = self.hurdle_ahead()
        if self.cross_z is None and self.prev_ahead > 0 >= ahead_now:   # the body's centre crosses the bar's line
            self.cross_z = float(d.qpos[2])
            if self.cross_z < self.hurdle_h + self.under_margin:
                self.under = True
                reward -= 5.0   # under the bar: a refusal in agility, as bad as a knocked bar
        self.prev_ahead = ahead_now
        if self.under:
            pass   # the try ends this step
        elif not self.cleared:
            reward -= 0.1  # every moment still in front of the bar costs a little
            reward += 15.0 * max(0.0, past.min() - self.best_past)  # the hindmost paw moving past the bar
            self.best_past = max(self.best_past, past.min())
            reward += 0.5 * np.clip(p["vx"], -0.5, 1.0) * upright
            # Judge the bar like a dog (run 16i). Take off from the right spot for THIS height: agility dogs leave the
            # ground about 1-1.5x the bar height before it (further for higher bars); rising there pays, still
            # being on the ground when too close to take off costs; and when the body passes over the bar its
            # centre should be at least bar height + 30 cm (the legs hang below it). Before this, run 16i's 30M brain
            # peaked at 0.72 m but ran into a 40 cm bar: nothing paid for timing the jump to the bar.
            ahead, h = self.hurdle_ahead(), self.hurdle_h
            grounded = self.feet_on_ground().sum() > 0
            if 0.3 + 0.8 * h < ahead < 0.5 + 1.5 * h and not grounded:
                reward += 0.5 * max(0.0, d.qvel[2])
            if 0.0 < ahead < 0.25 + 0.5 * h and grounded:
                reward -= 0.3
            if abs(ahead) < 0.1:
                reward += 2.0 * np.clip(d.qpos[2] - (h + 0.30), -0.2, 0.2)
            # tuck: a paw passing over the bar (within 12 cm of it) should be above it, with 3 cm to spare.
            # At 4-5M it cleared and landed 11 times in 12 at 10-12 cm, but the trailing rear legs clipped
            # the bar, so clean jumps stayed rare and the practice bar stuck at 10-12 cm.
            over = np.abs(past) < 0.12
            if over.any():
                z = np.array([self.data.geom_xpos[f][2] for f in self.foot_ids])
                reward += 3.0 * np.sum(np.clip(z[over] - (self.hurdle_h + 0.03), -0.1, 0.1))
            if past.min() > 0.08:  # all four paws past the bar: a little now, the rest for landing
                self.cleared = just_cleared = True
                self.clear_t = self.t
                reward += 5.0 if not self.touched else 1.0
        else:  # landed: stand tall and still
            if not self.landed and self.t - self.clear_t >= 25 and R[2, 2] > 0.85 and p["height"] > 0.3:
                self.landed = True   # still on its feet half a second after clearing; worth more facing forward
                reward += (1.0 if self.dropped else 12.0 if not self.touched else 6.0) * (0.5 + 0.5 * facing)
            speed = np.hypot(p["vx"], p["vy"])
            reward += 1.0 * np.exp(-((p["height"] - 0.43) / 0.08) ** 2) * upright * np.exp(-(speed / 0.5) ** 2)
        if self.real:   # spinning about the vertical, all through the jump, is where a twisted landing starts
            reward -= 0.02 * d.qvel[5] ** 2
        reward -= 0.01 * np.sum((a - self.last_action) ** 2) + 5e-5 * np.sum(d.actuator_force ** 2)
        self.last_action = a
        fell = R[2, 2] < 0.2
        if fell:
            reward -= 25.0 if self.cleared else 5.0   # diving over and crashing must not pay (15c/15d: -10 did)
        if (fell or self.under or self.t >= self.max_steps) and self.exam_bar is None:
            self._grade(self.landed and not self.under and not (self.dropped if self.real else self.touched))
        info = dict(p, cleared=self.cleared, touched=self.touched, just_cleared=just_cleared, hits=hits,
                    bar=self.hurdle_h, past=float(past.min()), level=self.bar_level, landed=self.landed,
                    dropped=self.dropped, heading_error=e, under=self.under, cross_z=self.cross_z)
        return self._obs(), float(reward), bool(fell or self.under), self.t >= self.max_steps, info


class JumpPolicy:
    def __init__(self, run_dir, zip_name="policy.zip", stats_name="vecnormalize.pkl"):
        from pathlib import Path
        run_dir = Path(run_dir)
        self.model = PPO.load(run_dir / zip_name, device="cpu")
        self.real = self.model.observation_space.shape[0] == 51   # run 16 on: the real hurdle
        self.norm = VecNormalize.load(str(run_dir / stats_name), DummyVecEnv([lambda: JumpEnv(real=self.real)]))
        self.norm.training = False

    def act(self, obs):
        return self.model.predict(self.norm.normalize_obs(obs), deterministic=True)[0]


def exam(policy, episodes=10, seed=5000, heights=(0.15, 0.20, 0.25, 0.30), dist=0.55, max_steps=100, speed=None,
         strict_too=True, gait=False, motor_brake=None):
    """Jump the bar from `dist` m away (55 cm up to run 16h; 1.5 m with a run-up from run 16i), optionally
    arriving at `speed` m/s. Two scores per height:
    jump_<cm>  fair, agility rules: straight approach, fixed physics; a jump counts if the bar stays in its
               cups (a brush is no fault, as in dog agility) and Bolt lands on its feet without falling.
    strict_<cm> the old test: randomized physics, half the tries 6 degrees askew and 5 cm off the line, no touch
               at all, standing tall at the end and facing within 20 degrees of the jump.
    strict_too=False: the fair score only. gait=True (with speed): start mid-stride in the dog's trot/canter pose.
    motor_brake: None = the process default (bolt_env.MOTOR["brake"]: off unless a training script or
    BOLT_MOTOR_BRAKE turned it on), True/False for this exam."""
    out = {}
    real = getattr(policy, "real", False)
    for strict in ((False, True) if strict_too else (False,)):
        env = JumpEnv(randomize=strict, real=real, max_steps=max_steps, motor_brake=motor_brake)
        env.exam_speed, env.exam_gait = speed, gait
        for h in heights:
            wins = 0
            for k in range(episodes):
                env.exam_bar = (dist, h)
                if real:
                    env.exam_askew = (0.0, 0.0) if (k % 2 == 0 or not strict) else ((0.1, 0.05) if k % 4 == 1 else (-0.1, -0.05))
                obs, _ = env.reset(seed=seed + k)
                info, fell = {}, False
                for _ in range(env.max_steps):
                    obs, _, fell, trunc, info = env.step(policy.act(obs))
                    if fell or trunc:
                        break
                if strict:
                    wins += bool(info["cleared"] and not info["touched"] and not fell and info["up"] > 0.9
                                 and info["height"] > 0.35 and abs(info.get("heading_error", 0.0)) < 0.35)
                else:
                    wins += bool(info["cleared"] and not info.get("dropped", info["touched"]) and not fell and info["up"] > 0.7)
            out[f"{'strict' if strict else 'jump'}_{round(h * 100)}"] = wins / episodes
    return out
