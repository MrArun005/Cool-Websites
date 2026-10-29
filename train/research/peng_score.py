"""Score a dog-motion brain on Peng et al. 2020's exact imitation reward (Eq. 4-9), not our relaxed root terms.

research/published_benchmarks.md, section 1: our reward has Peng's weights (0.5 pose / 0.05 joint velocity / 0.2 paws /
0.15 root pose / 0.1 root velocity) and the same pose, velocity and paw terms, but our two root terms only check body
height + tilt and body-frame velocity. Peng's Eq. 8 penalizes the GLOBAL root position and the FULL root rotation,
Eq. 9 the global root linear and angular velocity, so drifting off the reference's path or heading costs reward.

Here the reference root is integrated from where the try starts: its position follows the clip's body velocity turned by
the reference heading, its heading follows the clip's yaw rate, its height and tilt are the clip's. Per step:
    r = 0.5 exp(-5 |dq|^2) + 0.05 exp(-0.1 |dq_dot|^2) + 0.2 exp(-40 |dx_paw|^2)
        + 0.15 exp(-20 |dx_root|^2 - 10 dtheta_root^2) + 0.1 exp(-2 |dv_root|^2 - 0.2 |dw_root|^2)
(dtheta_root: the angle of the rotation between reference and robot body). Normalized return = sum of r over the
episode / episode length (Peng: return over the maximum possible), an episode that ends early (the env's loss of the
motion, or a fall) scores 0 for the steps left. Canonical (fixed) physics, as Peng's "canonical dynamics" table.

    cd train
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python research/peng_score.py ../models/bolt-dog-imitate-gpu-360M/params.pkl \
        --clips pace trot --seconds 10 --seeds 1 2 3 --episodes 5 --out research/results/peng_score_360M.json

Peng 2020, Table V (sim, canonical dynamics): Dog Pace 0.820 (robust) / 0.839 (no randomization), Dog Trot 0.727 /
0.752. Their robot (Laikago), simulator (PyBullet) and episode length differ from ours: same formula, not same test.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from imitate_env import ALL_CLIPS, ImitateEnv, _rotz  # noqa: E402


def load(brain):
    if str(brain).endswith(".pkl"):
        from mjx_imitate import MjxImitatePolicy
        return MjxImitatePolicy(brain)
    from imitate_env import ImitatePolicy
    b = Path(brain)
    return ImitatePolicy(b, "policy.zip", "vecnormalize.pkl")


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.arccos(np.clip(c, -1, 1)))


def episode(env, pol, clip, tau, seed, steps):
    env.exam_start = (clip, tau)
    obs, _ = env.reset(seed=seed)
    env.limit = steps
    d = env.data
    R = env._rotation()
    yaw_ref = float(np.arctan2(R[1, 0], R[0, 0]))
    p_ref = d.qpos[:2].copy()
    total, alive = 0.0, 0
    parts = dict(pose=0.0, vel=0.0, paw=0.0, root=0.0, root_vel=0.0, drift_m=0.0)
    for t in range(steps):
        obs, _, lost, done, _ = env.step(pol.act(obs))
        # the reference root after this step: integrate the clip's body velocity and yaw rate
        v_b, w_b = env.ref("v_body"), env.ref("w_body")
        u, _, vt = np.linalg.svd(env.ref("tilt"))   # the clip's body rotation without yaw (re-orthonormalized)
        tiltR = u @ vt
        R_ref = _rotz(yaw_ref) @ tiltR
        v_ref = R_ref @ v_b
        w_ref = R_ref @ w_b
        p_ref = p_ref + v_ref[:2] * env.dt
        yaw_ref += float(w_ref[2]) * env.dt
        R_ref = _rotz(yaw_ref) @ tiltR
        h_ref = env.ref("height")
        R = env._rotation()
        terms, _ = env.tracking()   # pose, vel, paw: the same as Peng's Eq. 4-6
        dx = np.append(p_ref - d.qpos[:2], h_ref - d.qpos[2])
        r_root = np.exp(-20 * np.sum(dx ** 2) - 10 * rot_angle(R_ref, R) ** 2)
        dv = v_ref - d.qvel[0:3]
        dw = w_ref - R @ d.qvel[3:6]
        r_rv = np.exp(-2 * np.sum(dv ** 2) - 0.2 * np.sum(dw ** 2))
        total += 0.5 * terms["pose"] + 0.05 * terms["vel"] + 0.2 * terms["paw"] + 0.15 * r_root + 0.1 * r_rv
        for k, v in (("pose", terms["pose"]), ("vel", terms["vel"]), ("paw", terms["paw"]), ("root", r_root), ("root_vel", r_rv)):
            parts[k] += v / steps
        parts["drift_m"] = float(np.linalg.norm(dx[:2]))
        alive += 1
        if lost or env._rotation()[2, 2] < 0.5:
            break
    return total / steps, alive / steps, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("--clips", nargs="+", default=["pace", "trot"])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    pol = load(a.brain)
    res = dict(brain=a.brain, seconds=a.seconds, seeds=a.seeds, episodes=a.episodes, clips={})
    for clip in a.clips:
        env = ImitateEnv(randomize=False, clips=ALL_CLIPS)
        steps = int(round(a.seconds / env.dt))
        c = env.clips[clip]
        scores, kept, P = [], [], []
        for s in a.seeds:
            for k in range(a.episodes):
                sc, al, pt = episode(env, pol, clip, (k / a.episodes) * c.length, 1000 * s + k, steps)
                scores.append(sc)
                P.append(pt)
                kept.append(al)
        res["clips"][clip] = dict(peng_normalized_return=round(float(np.mean(scores)), 3), sd=round(float(np.std(scores)), 3),
                                  full_episodes=round(float(np.mean(np.array(kept) >= 1.0)), 2),
                                  terms={k: round(float(np.mean([p[k] for p in P])), 3) for k in P[0]})
        print(clip, res["clips"][clip], flush=True)
    res["command"] = "python " + " ".join(sys.argv)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
