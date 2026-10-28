"""Landing states of the run-up jump brain, for the imitation env's landing starts (imitate_env landing_rate).

    python research/gen_landing_starts.py --tries 96 --out research/results/landing_starts.npz

The jump -> (canter) -> trot switch of skills_report.json, played by the skill manager (skills.Bolt, unchanged)
with the run-up jump brain (models/bolt-dog-jump-runup-16i-35.5M) in charge of the jump and the dog-motion brain
(default: the 360M GPU brain) for the trot before it and the canter -> trot after it: trot 2 s, a hurdle --height
(uniform) placed --ahead (uniform) m on the heading, trot up to it, jump. From the step the last paw passes the bar
(JumpEnv "cleared") until the jump brain hands back ("landed", 0.5 s later, upright) every step's body state is kept:
that is the jump brain's landing, from mid-air through touch-down, the states the dog-motion brain has to take over
from. Only jumps that landed are kept. Each state is stored in its own heading frame (x, y = 0, yaw = 0; world
linear velocity turned with it), so the env can start it at any heading, with the clip the skill manager hands over
to (canter, the bridge) at its closest-pose moment (skills.closest_moment), as Bolt does at the hand-over.

Writes an .npz: qpos (N, 19), qvel (N, 18), clip (N,) index into imitate_env.ALL_CLIPS, tau0 (N,) s, airborne (N,)
no paw on the floor, since_clear (N,) steps since the bar was cleared, height_bar (N,) m, try_id (N,), fell_after (N,)
whether that try fell in the canter bridge or the trot after it with the approach brain (the baseline the landing
starts should improve), and meta (json string: brains, args).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
MODELS = TRAIN.parent / "models"
sys.path.insert(0, str(TRAIN))

import torch  # noqa: E402
torch.set_num_threads(1)

import skills  # noqa: E402  (read-only use: the skill manager as test_skills runs it)
from imitate_env import ALL_CLIPS, ImitateEnv, ImitatePolicy  # noqa: E402
from jump_env import JumpPolicy  # noqa: E402
from mjx_imitate import MjxImitatePolicy  # noqa: E402

JUMP = MODELS / "bolt-dog-jump-runup-16i-35.5M"
DOG = MODELS / "bolt-dog-imitate-gpu-360M" / "params.pkl"


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def heading_frame(qpos, qvel):
    """The state with x, y = 0 and yaw = 0 (linear velocity turned with the body; angular velocity is body frame)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, qpos[3:7])
    R = R.reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    q, v = qpos.copy(), qvel.copy()
    q[0:2] = 0.0
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (_rotz(-yaw) @ R).reshape(-1))
    q[3:7] = quat
    v[0:3] = _rotz(-yaw) @ qvel[0:3]
    return q, v


class Recorder(skills.Bolt):
    """skills.Bolt that keeps every body state the jump brain passes through between clearing and handing back."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows, self.recording = [], False

    def _after_step(self, skill):
        w = self.world
        if self.recording and skill == "jump" and w.cleared:
            q, v = heading_frame(self.data.qpos[:19].copy(), self.data.qvel[:18].copy())
            self.rows.append(dict(qpos=q, qvel=v, airborne=bool(w.feet_on_ground().sum() == 0), since_clear=int(w.t - w.clear_t),
                                  joints=self.data.qpos[7:19].copy(), joint_vel=self.data.qvel[6:18].copy()))
        return super()._after_step(skill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=96)
    ap.add_argument("--seed", type=int, default=50000, help="test_skills uses 1000+; these seeds are kept apart")
    ap.add_argument("--height", type=float, nargs=2, default=[0.20, 0.40], help="bar height range (m), as skills_report")
    ap.add_argument("--ahead", type=float, nargs=2, default=[1.5, 2.0], help="hurdle distance range (m), as test_skills")
    ap.add_argument("--approach", default="trot", help="gait before the jump and after the bridge")
    ap.add_argument("--approach-s", type=float, default=2.0, help="s of the gait before the hurdle is placed")
    ap.add_argument("--then", type=float, default=2.0, help="s of the gait after the canter bridge (for fell_after)")
    ap.add_argument("--jump", default=str(JUMP), help="jump brain folder (policy.zip + vecnormalize.pkl)")
    ap.add_argument("--dog", default=str(DOG), help="dog-motion brain: a params.pkl (numpy) or an SB3 folder")
    ap.add_argument("--out", default=str(HERE / "results" / "landing_starts.npz"))
    a = ap.parse_args()
    t0 = time.time()
    dog_path = Path(a.dog)
    if dog_path.suffix == ".pkl":
        # skills.Bolt builds an SB3 ImitatePolicy from a folder; build it from any SB3 brain, then swap in the numpy one
        bolt = Recorder(seed=a.seed, imitate=(MODELS / "bolt-dog-imitate-18b-30M", "policy.zip", "vecnormalize.pkl"))
        bolt.dog = MjxImitatePolicy(dog_path)
        bolt.im = ImitateEnv(randomize=True, clips=skills.CLIPS, slots=bolt.dog.slots)
        skills._attach(bolt.im, bolt.world)
    else:
        bolt = Recorder(seed=a.seed, imitate=(dog_path, "policy.zip", "vecnormalize.pkl"))
        assert isinstance(bolt.dog, ImitatePolicy)
    bolt.jumper = JumpPolicy(a.jump)
    assert bolt.jumper.real, "the skill manager's world is the real hurdle: the jump brain must be a real=True brain"
    canter = bolt.im.clips["canter"]
    keep, tries = [], []
    for k in range(a.tries):
        seed = a.seed + 37 * k
        rng = np.random.default_rng(seed)
        bolt.start(a.approach, seed=seed)
        bolt.do(a.approach, a.approach_s)
        h, ahead = float(rng.uniform(*a.height)), float(rng.uniform(*a.ahead))
        bolt.rows, bolt.recording = [], True
        r = bolt.do("jump", height=h, ahead=ahead, then=0.0, bridge=0.0)   # stop at the hand-over: the after-part below
        bolt.recording = False
        rows = bolt.rows
        fell_after = None
        if r["landed"] and not r["fell"]:
            fell_after = not (bolt.do("canter", skills.BRIDGE) and bolt.do(a.approach, a.then))
            for row in rows:
                row.update(tau0=skills.closest_moment(canter, row["joints"], row["joint_vel"]), height_bar=h, try_id=k, fell_after=fell_after)
            keep += rows
        tries.append(dict(k=k, seed=seed, height=round(h, 3), ahead=round(ahead, 3), clean=r["clean"], landed=r["landed"], fell=r["fell"],
                          fell_after=fell_after, states=len(rows) if r["landed"] and not r["fell"] else 0))
        print(f"try {k}: h={h:.2f} clean={r['clean']} landed={r['landed']} fell={r['fell']} fell_after={fell_after} "
              f"states={tries[-1]['states']} (total {len(keep)}, {time.time() - t0:.0f} s)", flush=True)
    assert keep, "no landing kept"
    out = dict(qpos=np.array([r["qpos"] for r in keep], np.float64), qvel=np.array([r["qvel"] for r in keep], np.float64),
               clip=np.full(len(keep), ALL_CLIPS.index("canter"), np.int32), tau0=np.array([r["tau0"] for r in keep], np.float64),
               airborne=np.array([r["airborne"] for r in keep]), since_clear=np.array([r["since_clear"] for r in keep], np.int32),
               height_bar=np.array([r["height_bar"] for r in keep], np.float32), try_id=np.array([r["try_id"] for r in keep], np.int32),
               fell_after=np.array([r["fell_after"] for r in keep]))
    meta = dict(args=vars(a), jump=str(Path(a.jump).resolve().relative_to(TRAIN.parent)), dog=str(dog_path.resolve().relative_to(TRAIN.parent)),
                tries=tries, kept=len(keep), landed=sum(t["landed"] and not t["fell"] for t in tries), seconds=round(time.time() - t0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(f"saved {len(keep)} states from {meta['landed']}/{a.tries} landed jumps "
          f"({out['airborne'].mean():.0%} airborne, fell after with the approach brain in "
          f"{np.mean([t['fell_after'] for t in tries if t['fell_after'] is not None]):.0%} of landed tries) -> {a.out}")


if __name__ == "__main__":
    main()
