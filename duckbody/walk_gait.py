"""A reference walk for the mid duck to imitate (the Bolt dog's lesson: a gait to follow is far
faster to learn than walking from nothing; Disney BDX / Open Duck Mini do the same).

    python walk_gait.py                      # play it open-loop through SkillEnv, report what happens
    python walk_gait.py --vx 0.17 --seconds 6

The gait is planned for the feet and turned into joint angles by inverse kinematics on the model
itself (no guessed leg lengths). Per leg, in the trunk frame:
  stance  the foot slides backward (and sideways / rotates for vy / yaw) at the commanded speed, flat
  swing   it lifts (sin bump) and swings forward again, still flat
Left and right are half a cycle apart, at the env's gait clock (sin, cos of 2π·f·t, f = 1.8 Hz walk,
2.6 Hz run), so the reference phase is exactly what the brain sees. A small sideways sway of both
feet moves the body over the stance foot. Left foot in stance for phase in [0, π).
"""
import argparse

import mujoco
import numpy as np

from env import HERE

LEGS = ("left", "right")
JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


class Gait:
    def __init__(self, model_path=None, lift=0.015, sway=0.015, x_off=-0.012, swing=0.35, lead=0.2):
        self.m = mujoco.MjModel.from_xml_path(str(model_path or HERE / "model" / "duck_mid.xml"))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.lift, self.sway, self.x_off, self.swing, self.lead = lift, sway, x_off, swing, lead
        self.home_q = m.key("home").qpos.copy()
        self.home_ctrl = m.key("home").ctrl.copy()
        self.trunk = m.body("trunk_assembly").id
        self.sites = [m.site(f"{s}_foot").id for s in LEGS]
        self.jid = [[m.joint(f"{s}_{j}").id for j in JOINTS] for s in LEGS]
        self.qadr = [np.array([m.jnt_qposadr[j] for j in js]) for js in self.jid]
        self.dadr = [np.array([m.jnt_dofadr[j] for j in js]) for js in self.jid]
        self.act = [np.array([m.actuator(f"{s}_{j}").id for j in JOINTS]) for s in LEGS]
        self.lo = [m.jnt_range[js, 0] for js in self.jid]
        self.hi = [m.jnt_range[js, 1] for js in self.jid]
        d.qpos[:] = self.home_q
        d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        mujoco.mj_kinematics(m, d)
        self.foot0 = [d.site_xpos[s].copy() for s in self.sites]                 # trunk frame (base at origin)
        self.rot0 = [d.site_xmat[s].reshape(3, 3).copy() for s in self.sites]
        self.cache = {}

    def feet(self, phase, cmd, freq):
        """Foot positions and yaw (trunk frame) at a gait phase, for command (vx, vy, yaw rate).

        Each leg: stance for (1 - swing) of the cycle, sliding back by the commanded speed × stance
        time; swing for the rest, lifting and returning. Left swings at the end of the cycle, right
        half a cycle earlier; the body sways over the single-support foot (lead: phase advance).
        """
        vx, vy, wz = cmd
        sw = self.swing
        t_st = (1 - sw) / freq                              # stance time of one foot
        u0 = (phase / (2 * np.pi)) % 1.0
        # Body over the left foot in the middle of the left single support (right foot swinging).
        c = 0.5 - sw / 2 - self.lead
        y_sway = -self.sway * np.cos(2 * np.pi * (u0 - c))
        out = []
        for i in range(2):
            u = (u0 + 0.5 * i) % 1.0                        # left: stance in [0, 1-sw), right shifted by 1/2
            if u < 1 - sw:                                   # stance: from +1/2 to -1/2 of the stride
                s, z = 0.5 - u / (1 - sw), 0.0
            else:                                            # swing: smooth back to the front
                w = (u - (1 - sw)) / sw
                s, z = -0.5 + (0.5 - 0.5 * np.cos(np.pi * w)), self.lift * np.sin(np.pi * w)
            pos = self.foot0[i].copy()
            pos[0] += s * vx * t_st + self.x_off
            pos[1] += s * vy * t_st + y_sway
            pos[2] += z
            out.append((pos, s * wz * t_st, z > 0))
        return out

    def ik(self, i, pos, yaw, q_init):
        """Joint angles of leg i putting its sole site at pos with the home sole tilt, turned by yaw."""
        m, d = self.m, self.d
        c, sn = np.cos(yaw), np.sin(yaw)
        Rt = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]]) @ self.rot0[i]
        q = q_init.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(30):
            d.qpos[:] = self.home_q
            d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
            d.qpos[self.qadr[i]] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.sites[i]].reshape(3, 3)
            ep = pos - d.site_xpos[self.sites[i]]
            Re = Rt @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            if np.linalg.norm(ep) < 1e-5 and np.linalg.norm(er) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.sites[i])
            w = np.array([1, 1, 1, 0.05, 0.05, 0.02])        # orientation counts as ~5 cm/rad (roll least)
            J = np.vstack([jacp[:, self.dadr[i]], jacr[:, self.dadr[i]]]) * w[:, None]
            e = np.concatenate([ep, er]) * w
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
            q = np.clip(q + dq, self.lo[i], self.hi[i])
        return q

    def table(self, cmd, freq, n=72):
        """Actuator targets [n, nu] and planned stance [n, 2] over one cycle (cached per command)."""
        key = (tuple(np.round(cmd, 4)), freq, n)
        if key in self.cache:
            return self.cache[key]
        ctrl = np.tile(self.home_ctrl, (n, 1))
        stance = np.zeros((n, 2))
        q = [self.home_q[a].copy() for a in self.qadr]
        for k in range(n):
            for i, (pos, yaw, swing) in enumerate(self.feet(2 * np.pi * k / n, cmd, freq)):
                q[i] = self.ik(i, pos, yaw, q[i])
                ctrl[k, self.act[i]] = q[i]
                stance[k, i] = float(not swing)
        if len(self.cache) > 512:
            self.cache = {k: v for k, v in self.cache.items() if k[0] == "grid"}
        self.cache[key] = (ctrl, stance)
        return ctrl, stance

    GRID = dict(vx=np.linspace(-0.1, 0.4, 11), vy=np.linspace(-0.08, 0.08, 5), wz=np.linspace(-0.8, 0.8, 9))

    def fast_table(self, cmd, freq):
        """table() for any command from per-axis tables (built once): base + the change each axis makes.
        The legs move little, so the effects add up (checked against the exact IK in __main__)."""
        key = ("grid", freq)
        if key not in self.cache:
            base = self.table((0.0, 0.0, 0.0), freq)
            axes = {}
            for j, ax in enumerate(("vx", "vy", "wz")):
                rows = []
                for v in self.GRID[ax]:
                    c = np.zeros(3)
                    c[j] = v
                    rows.append(self.table(c, freq)[0] - base[0])
                axes[ax] = np.array(rows)
            self.cache[key] = (base, axes)
        (ctrl0, _), axes = self.cache[key]
        ctrl = ctrl0.copy()
        for j, ax in enumerate(("vx", "vy", "wz")):
            g = self.GRID[ax]
            x = np.clip(cmd[j], g[0], g[-1])
            k = min(int(np.searchsorted(g, x, side="right")) - 1, len(g) - 2)
            f = (x - g[k]) / (g[k + 1] - g[k])
            ctrl += (1 - f) * axes[ax][k] + f * axes[ax][k + 1]
        # Planned contacts depend only on the phase.
        return ctrl, self.cache[key][0][1]

    def at(self, phase, cmd, freq):
        ctrl, stance = self.table(cmd, freq)
        n = len(ctrl)
        x = (phase % (2 * np.pi)) / (2 * np.pi) * n
        k = int(x) % n
        f = x - int(x)
        return (1 - f) * ctrl[k] + f * ctrl[(k + 1) % n], stance[int(round(x)) % n]


def play(vx=0.17, vy=0.0, wz=0.0, seconds=6.0, task="walk", seed=0, randomize=False, ramp=0.5, **kw):
    """Feed the reference to SkillEnv as actions (no brain) and report what the duck does."""
    from skills_env import GAIT_HZ, SkillEnv
    env = SkillEnv(task=task, randomize=randomize, seed=seed, model_path=str(HERE / "model" / "duck_mid.xml"),
                   max_seconds=seconds, practice=False)
    env.reset(seed=seed)
    env.vel_cmd = np.array([vx, vy, wz])
    g = Gait(**kw)
    f = GAIT_HZ[task]
    d = env.data
    x0, min_up, steps, done, t = d.qpos[:3].copy(), 1.0, 0, False, 0
    R0 = d.xmat[env.trunk].reshape(3, 3).copy()
    contact_match, rew, airborne = [], 0.0, []
    while not done:
        phase = 2 * np.pi * f * env.t * env.dt
        q, stance = g.at(phase, env.vel_cmd, f)
        a = min(1.0, env.t * env.dt / ramp) if ramp else 1.0
        q = env.home + a * (q - env.home)
        act = np.clip((q[env.pol] - env.home[env.pol]) / env.action_scale, -1, 1)
        _, r, term, trunc, info = env.step(act)
        rew += r
        contact_match.append(np.mean(env.feet_on_ground() == stance))
        airborne.append(1 - env.feet_on_ground())
        min_up = min(min_up, info["up"])
        done = term or trunc
    dp = R0.T @ (d.qpos[:3] - x0)
    R = d.xmat[env.trunk].reshape(3, 3)
    return dict(forward=float(dp[0]), side=float(dp[1]), speed=float(dp[0] / (env.t * env.dt)), fell=bool(term),
                min_up=float(min_up), height=float(d.qpos[2]), contact_match=float(np.mean(contact_match)), lifted=float(np.mean(airborne)),
                heading=float(np.degrees(np.arctan2((R0.T @ R)[1, 0], (R0.T @ R)[0, 0]))), reward=rew,
                seconds=env.t * env.dt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.08)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--task", default="walk")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--randomize", action="store_true")
    a = ap.parse_args()
    from walk_env import GAIT
    for s in range(a.seeds):
        r = play(a.vx, a.vy, a.wz, a.seconds, a.task, seed=s, randomize=a.randomize, **GAIT[a.task])
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
