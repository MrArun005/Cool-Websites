"""A hand-made closed-loop walker: the reference gait (walk_gait.py) plus simple balance feedback.

The reference played open-loop steps and moves, but tips over within 2-4 s: nothing keeps the body
over the feet. This adds the feedback a person would add, all in foot space, all from what the
brain also sees (IMU tilt and spin) plus the body velocity:

  dx  both feet forward when the body pitches / falls forward (foot placement), and back when it
      goes faster than commanded
  dy  both feet sideways the same way for roll
  wz  the turn the gait is asked for is raised when the body turns slower than commanded

dx / dy act through precomputed per-phase tables (joint change per metre of foot shift), so a step
costs a table lookup, not IK. It is the teacher for the behaviour-cloning warm start (walk_bc.py):
a brain that starts out stepping like this only has to learn the rest.

    python walk_expert.py                  # survival / speed / turning over random commands
"""
import numpy as np

from skills_env import GAIT_HZ

DEFAULT = dict(kx=0.0, kxd=0.0, kv=0.0, ky=0.0, kyd=0.0, kw=0.0, lead=1, dmax=0.03)
# Gains found by random search (scratch tune.py: survival and command tracking on randomized bodies).
GAINS = {"walk": dict(kx=0.055, kxd=0.0025, kv=0.0531, ky=0.0099, kyd=-0.0106, kw=1.7447)}
GAINS["run"] = dict(GAINS["walk"])


class Expert:
    _deltas = {}

    def __init__(self, env, **gains):
        self.env = env
        self.g = dict(DEFAULT, **gains)
        for t in ("walk", "run"):
            key = (id(env.ref[t]), t)
            if key not in Expert._deltas:
                Expert._deltas[key] = self._delta_tables(env.ref[t], GAIT_HZ[t])
        self.reset()

    def reset(self):
        self.wz_i = 0.0

    @staticmethod
    def _delta_tables(gait, freq, h=0.01):
        """Joint change per metre of shifting both feet in x and in y (trunk frame), per phase."""
        out = []
        for attr in ("x_off", "y_off"):
            tabs = []
            for sgn in (1, -1):
                old = getattr(gait, attr, 0.0)
                setattr(gait, attr, old + sgn * h)
                gait.cache = {k: v for k, v in gait.cache.items() if k[0] == "grid"}
                tabs.append(gait.table((0.0, 0.0, 0.0), freq, n=72)[0].copy())
                setattr(gait, attr, old)
            gait.cache = {k: v for k, v in gait.cache.items() if k[0] == "grid"}
            out.append((tabs[0] - tabs[1]) / (2 * h))
        return out

    def state(self):
        e = self.env
        g = e.gravity()                       # gravity in IMU frame: x>0 nose down? (sign set by gains)
        w = e._sensor("gyro")
        p = e.posture()
        return g, w, p

    def action(self):
        e, G = self.env, self.g
        cmd = e.command
        f = GAIT_HZ[cmd]
        g, w, p = self.state()
        v = e.vel_cmd
        dx = G["kx"] * g[0] + G["kxd"] * w[1] + G["kv"] * (p["vx"] - v[0])
        dy = G["ky"] * g[1] + G["kyd"] * w[0] + G["kv"] * (p["vy"] - v[1])
        dx, dy = np.clip([dx, dy], -G["dmax"], G["dmax"])
        self.wz_i = 0.98 * self.wz_i + 0.02 * (v[2] - p["yaw_rate"])
        wz = v[2] + G["kw"] * self.wz_i
        ctrl, stance = e.ref[cmd].fast_table(np.array([v[0], v[1], wz]), f)
        Dx, Dy = Expert._deltas[(id(e.ref[cmd]), cmd)]
        n = len(ctrl)
        x = (f * (e.t + G["lead"]) * e.dt % 1.0) * n
        k, fr = int(x) % n, x - int(x)
        lerp = lambda T: (1 - fr) * T[k] + fr * T[(k + 1) % n]
        q = lerp(ctrl) + dx * lerp(Dx) + dy * lerp(Dy)
        return np.clip((q[e.pol] - e.home[e.pol]) / e.action_scale, -1, 1)


def patch_gait():
    """Give walk_gait.Gait a y_off (both feet sideways), used by the delta tables."""
    import walk_gait
    G = walk_gait.Gait
    if getattr(G, "_y_patched", False):
        return
    feet = G.feet

    def feet_y(self, phase, cmd, freq):
        out = feet(self, phase, cmd, freq)
        dy = getattr(self, "y_off", 0.0)
        if dy:
            out = [(pos + np.array([0, dy, 0]), yaw, sw) for pos, yaw, sw in out]
        return out
    G.feet = feet_y
    G._y_patched = True


patch_gait()


def rollout(gains, task="walk", seed=0, seconds=6.0, cmd=None, randomize=True, ramp=0.5, env=None):
    from walk_env import WalkEnv
    from walk_gait import HERE
    env = env or WalkEnv(task=task, randomize=randomize, seed=seed, model_path=str(HERE / "model" / "duck_mid.xml"),
                         max_seconds=seconds, practice=False)
    env.max_seconds = seconds
    env.reset(seed=seed)
    if cmd is not None:
        env.vel_cmd = np.array(cmd, dtype=float)
        env.table = env.ref[task].fast_table(env.vel_cmd, GAIT_HZ[task])
    ex = Expert(env, **gains)
    d = env.data
    R0 = d.xmat[env.trunk].reshape(3, 3).copy()
    x0 = d.qpos[:3].copy()
    done, rew, lifts, prev = False, 0.0, np.zeros(2), env.feet_on_ground()
    yaw0 = np.arctan2(R0[1, 0], R0[0, 0])
    while not done:
        a = ex.action()
        s = min(1.0, env.t * env.dt / ramp) if ramp else 1.0
        a = s * a
        _, r, term, trunc, info = env.step(a)
        rew += r
        f = env.feet_on_ground()
        lifts += (prev == 1) & (f == 0)
        prev = f
        done = term or trunc
    T = env.t * env.dt
    dp = R0.T @ (d.qpos[:3] - x0)
    R = d.xmat[env.trunk].reshape(3, 3)
    dyaw = np.arctan2(R[1, 0], R[0, 0]) - yaw0
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    return dict(cmd=[round(float(c), 3) for c in env.vel_cmd], T=T, fell=bool(term), vx=dp[0] / T, vy=dp[1] / T,
                wz=dyaw / T, lifts=lifts.tolist(), reward=rew / max(env.t, 1))


if __name__ == "__main__":
    import sys
    import json
    gains = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    for s in range(6):
        r = rollout(gains, seed=s, seconds=6.0)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
