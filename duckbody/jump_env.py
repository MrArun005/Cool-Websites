"""Jump (hop) skill for the mid duck: repeated small hops that land softly on both feet.

    python train.py --skills --env jump_env:JumpEnv --model model/duck_mid.xml --task jump --name mid_jump

Same observation / action layout as SkillEnv (only the jump reward and practice starts change).

What the body can do (jump_probe.py, scripted crouch → extend through the policy's own action range):
the trunk rises up to ~3.5 cm above standing height, but most of that is the legs straightening;
the soles clear the floor by ~5–8 mm with 80–110 ms of flight. With the XL330's real torque–speed
curve (back-EMF, ~10.8 rad/s no-load at 5 V, which the model does not have) that drops to ~2 mm and
~60 ms. So a "hop" here is a small bounce, and the landing is what can hurt the robot.

Servos (v2): JumpEnv adds the servo's torque–speed limit (back-EMF): joint damping τ_stall/ω0 with
ω0 ~ U(9, 11) rad/s per episode (XL330 no-load 10.8 rad/s at 5 V; nominal in jump_check.py), so the
brain cannot use power the real servo lacks. Set JumpEnv.back_emf = False for v1 physics.

Reward (per 20 ms step), products of terms (LESSONS §4) and one landing penalty:
    reward = upright × no_spin × stay_near_start × phase − shock
  phase, both soles off the floor:  1 + clip(½·rise/(3 cm×L) + ½·sole clearance/6 mm)   (1 … 2)
  phase, on the floor:              0.1 ready (½ on one foot; 0.05 in a heap below the crouch)
                                    + 0.2 × clip(upward CoM speed / 0.4 m/s)   (push-off shaping)
  no_spin uses the whole body's (CoM) horizontal speed: the trunk origin swings ±0.4 m/s as it pitches
  during a push, which is not drift.
  shock = 1.0·x + 2.0·x² + 5·max(0, F/W − 3.5),  x = max(0, F/W − 2.0),  F/W = 1 + a_com_z / g over the step
          (whole-body ground force from the centre-of-mass acceleration, × body weight; the 2 ms sole
          contact forces are solver noise with 1 iteration and spike to 30–100× W on any touchdown)
  v1 was 0.5·x + 0.7·x² from 2.5 W: the brain bounced at ~4 Hz with 71% of landings above 2.5 W and
  peaks of 3.8 W. v2 makes a 3.5 W landing cost about a whole hop's flight credit.
A scripted hop that tucks and lands stiff (no back-EMF) peaks at F/W ≈ 4.4 and scores far below
standing still. jump_check.py verifies the signs: of two landings, the softer one scores higher.

Single hop on command (no layout change): run this brain until the foot switches report the first
touchdown after a flight, then switch to the stand brain (jump_eval.py --single 0: one hop, landing
≤ 2.2 W, stands).

Practice starts (training only; exams start standing): 20% crouched, ready to push; 20% in the
air just above the floor, to practise landings.
"""
import mujoco
import numpy as np

from poses import leg_q
from skills_env import SkillEnv

SHOCK_FREE = 2.0          # × body weight over a 20 ms step that a landing may use for free
SHOCK_LIN, SHOCK_SQ = 1.0, 2.0   # v1 (2.5, 0.5, 0.7): 71% of landings > 2.5 W, peaks 3.8 W
SHOCK_HARD, SHOCK_HARD_W = 3.5, 5.0   # extra, per body weight beyond 3.5 W: never land that hard
OMEGA_FREE = 10.8         # rad/s, XL330 no-load speed at 5 V (back-EMF)
OMEGA_RANGE = (9.0, 11.0) # drawn per episode in training (battery sag, servo spread)
CLEAR_FULL = 0.006        # m of sole clearance for full flight credit
PUSH_V = 0.4              # m/s upward CoM speed for full push-off credit
READY = 0.1


class JumpEnv(SkillEnv):
    back_emf = True

    def _back_emf(self):
        """Servo torque–speed limit: a DC motor's available torque falls linearly with speed,
        τ ≤ τ_stall·(1 − ω/ω0). With the position loop saturating at τ_stall that is exactly a joint
        damping of τ_stall/ω0 outside the actuator clip (MuJoCo applies dof damping as a passive force,
        after forcerange). The model's own damping (0.053) is kept where it is already larger."""
        m, r = self.model, self.rng
        w0 = r.uniform(*OMEGA_RANGE) if self.randomize else OMEGA_FREE
        base_damp = m.dof_damping[self.dadr] if self.randomize else self.nominal["damping"][self.dadr]
        m.dof_damping[self.dadr] = np.maximum(base_damp, np.abs(m.actuator_forcerange[:, 1]) / w0)
        self.omega0 = w0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        m, d, r = self.model, self.data, self.rng
        if self.back_emf:
            self._back_emf()
        elif not self.randomize:
            m.dof_damping[:] = self.nominal["damping"]
        if self.task == "jump" and self.practice:
            u = r.random()
            if u < 0.4:
                q = d.qpos.copy()
                if u < 0.2:                                     # crouched, ready to push
                    k = r.uniform(1.45, 1.9)
                    leg_q(m, q, -0.46 * k, k, -0.54 * k)
                    lift, vz = 0.0, 0.0
                else:                                           # in the air, about to land
                    k = r.uniform(0.6, 1.4)
                    hr = r.uniform(0.45, 0.6)
                    leg_q(m, q, -hr * k, k, -(k - hr * k) + r.uniform(-0.4, 0.0))
                    lift, vz = r.uniform(0.002, 0.012), r.uniform(-0.3, 0.3)
                q[self.qadr] = np.clip(q[self.qadr] + r.normal(0, 0.02, m.nu), self.lo, self.hi)
                mujoco.mj_resetData(m, d)
                d.qpos[:] = q
                mujoco.mj_forward(m, d)
                d.qpos[2] -= min(self._sole_bottom(g) for g in self.sole_ids) - lift
                d.qvel[2] = vz
                d.ctrl[:] = q[self.qadr]
                mujoco.mj_forward(m, d)
                self.pending = [d.ctrl.copy()] * 2
                obs = self._obs()
        mujoco.mj_subtreeVel(m, d)
        self.com_vz = float(d.subtree_linvel[1][2])
        self.start_xy = d.qpos[:2].copy()
        self.peak_fw, self.shock_sum, self.air_steps, self.hops = 1.0, 0.0, 0, 0
        return obs, info

    def _reward(self, action):
        if self.command != "jump":
            # Another brain in charge (e.g. stand after a single hop): its reward, plus the landing
            # measurements jump_eval.py reads (hops and best_hop keep counting).
            reward, info = super()._reward(action)
            m, d = self.model, self.data
            mujoco.mj_subtreeVel(m, d)
            vz = float(d.subtree_linvel[1][2])
            fw = 1.0 + (vz - self.com_vz) / self.dt / 9.81
            self.com_vz = vz
            self.peak_fw = max(self.peak_fw, fw)
            airborne = self.feet_on_ground().sum() == 0
            self.air_steps = self.air_steps + 1 if airborne else 0
            if self.air_steps == 2:
                self.hops += 1
            clear = min(self._sole_bottom(g) for g in self.sole_ids)
            info.update(fw=fw, shock=0.0, clear=clear, airborne=bool(airborne))
            return reward, info
        m, d = self.model, self.data
        p = self.posture()
        L = self.L
        feet = self.feet_on_ground()
        airborne = feet.sum() == 0

        mujoco.mj_subtreeVel(m, d)
        vz = float(d.subtree_linvel[1][2])
        fw = 1.0 + (vz - self.com_vz) / self.dt / 9.81          # mean ground force / body weight this step
        self.com_vz = vz
        x = max(0.0, fw - SHOCK_FREE)
        shock = SHOCK_LIN * x + SHOCK_SQ * x * x + SHOCK_HARD_W * max(0.0, fw - SHOCK_HARD)
        self.peak_fw = max(self.peak_fw, fw)
        self.shock_sum += shock

        rise = p["height"] - self.stand_height
        clear = min(self._sole_bottom(g) for g in self.sole_ids)        # lowest point of either sole
        if airborne:
            self.best_hop = max(self.best_hop, rise)                   # SkillEnv's exam measure
            self.air_steps += 1
            if self.air_steps == 2:
                self.hops += 1
            frac = 0.5 * np.clip(rise / (0.03 * L), 0, 1) + 0.5 * np.clip(clear / CLEAR_FULL, 0, 1)
            phase = 1.0 + float(frac)
        else:
            self.air_steps = 0
            ready = READY * (0.5 if feet.sum() == 1 else 1.0)
            if p["height"] < self.crouch_height - 0.03 * L:
                ready = 0.05
            phase = ready + 0.2 * float(np.clip(vz / PUSH_V, 0, 1))
        up = 1.0 if p["up"] >= 0.9 else float(np.exp(-((p["up"] - 0.9) / 0.15) ** 2))
        # Drift of the whole body (CoM), not of the trunk origin: the trunk pitches during a push.
        v_xy = d.subtree_linvel[1][:2]
        spin = np.exp(-(p["yaw_rate"] / 0.8) ** 2) * np.exp(-float(v_xy @ v_xy) / (0.25 * L) ** 2)
        drift = np.linalg.norm(d.qpos[:2] - self.start_xy)
        near = 0.5 + 0.5 * np.exp(-(drift / (0.15 * L)) ** 2)
        reward = up * spin * near * phase - shock
        info = dict(h_ref=self.stand_height, up_ref=1.0, fw=fw, shock=shock, clear=clear, airborne=bool(airborne), **p)
        return float(reward), info
