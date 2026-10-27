"""Plan, then distill: teach the dog-motion brain what the PPS planner does, so the improvement runs without
the planner (one forward pass, ONNX in the browser, a real robot with no simulator on board).

    python research/pps_distill.py collect --tries 30 --seed 20000 --out research/results/distill_r0.npz
    python research/pps_distill.py collect --tries 30 --seed 21000 --student research/results/student_r0 --out ...r1.npz  (DAgger round)
    python research/pps_distill.py fit --data r0.npz [r1.npz] --out research/results/student_r0
    python research/exam_canter.py --brain research/results/student_r0 --mode plain ...

collect: canter tries (randomized physics, random start phase, seeds disjoint from the exam's 9000+) with
the PPS planner on top of the brain being trained (the original brain in round 0, the student in DAgger
rounds), recording at every step the observation, the ORIGINAL brain's raw action mean, and the planned
perturbation e (the label is original mean + e: what the planner made the brain do). Plus plain-brain
samples on the other clips, labelled with the original brain itself (an anchor, so the other skills are
kept).
fit: fine-tune a copy of the brain's actor (policy MLP + action head; critic and VecNormalize statistics
untouched) on those labels with MSE, a few epochs.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from imitate_env import ImitateEnv, ImitatePolicy   # noqa: E402
from pps import BatchBrain, Planner                   # noqa: E402

BRAIN = HERE.parent.parent / "models" / "bolt-dog-imitate-18b-17.5M"
ANCHOR_CLIPS = ["walk", "trot", "pace", "turn_left", "turn_right", "sit", "lie"]


def raw_mean(policy, obs):
    """The brain's unclipped action mean for a batch of (already sliced) observations."""
    x = torch.as_tensor(policy.norm.normalize_obs(obs.astype(np.float32)), dtype=torch.float32)
    with torch.no_grad():
        return policy.model.policy.get_distribution(x).distribution.mean.numpy()


def slice_obs(policy, obs):
    return np.concatenate([obs[:48 + policy.slots], obs[57:]]) if len(obs) > 66 + policy.slots else obs


def collect(args):
    orig = ImitatePolicy(BRAIN)
    driver = ImitatePolicy(args.student) if args.student else orig
    rng = np.random.default_rng(args.seed)
    O, E = [], []
    env = ImitateEnv(randomize=True, clips=["canter"])
    planner = Planner(driver, ["canter"], K=16, H=30, M=5, sigma=0.2, seed=args.seed)
    c = env.clips["canter"]
    for k in range(args.tries):
        env.exam_start = ("canter", float(rng.uniform(0, c.length)))
        obs, _ = env.reset(seed=args.seed + k)
        env.limit = 200
        planner.plan[:] = 0.0
        queue = []
        for t in range(200):
            if not queue:
                queue = list(planner.new_plan(env))
            e = queue.pop(0)
            o = slice_obs(orig, obs)
            # label: what the planner made the DRIVER do, expressed as a raw action
            O.append(o)
            E.append(raw_mean(driver, o[None])[0] + e)
            obs, r, fell, done, _ = env.step(np.clip(driver.act(obs) + e, -1, 1))
            if fell:
                break
        print(f"try {k}: {t + 1} steps, fell={fell}, planner {planner.stats}", flush=True)
    # anchor samples: the original brain on the other clips (its own states, its own actions)
    A = []
    env2 = ImitateEnv(randomize=True, clips=ANCHOR_CLIPS)
    for k in range(args.anchor_tries):
        obs, _ = env2.reset(seed=args.seed + 5000 + k)
        for t in range(env2.limit):
            A.append(slice_obs(orig, obs))
            obs, r, fell, done, _ = env2.step(orig.act(obs))
            if fell or done:
                break
    A = np.array(A, dtype=np.float32)
    np.savez_compressed(args.out, obs=np.array(O, dtype=np.float32), label=np.array(E, dtype=np.float32),
                        anchor_obs=A, anchor_label=raw_mean(orig, A).astype(np.float32))
    print(f"saved {len(O)} planner samples, {len(A)} anchor samples -> {args.out}")


def fit(args):
    src = Path(args.init or BRAIN)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pol = ImitatePolicy(src)
    O, Y, W = [], [], []
    for f in args.data:
        z = np.load(f)
        O += [z["obs"], z["anchor_obs"]]
        Y += [z["label"], z["anchor_label"]]
        W += [np.full(len(z["obs"]), args.w_plan), np.full(len(z["anchor_obs"]), 1.0)]
    O, Y, W = np.concatenate(O), np.concatenate(Y), np.concatenate(W)
    X = torch.as_tensor(pol.norm.normalize_obs(O), dtype=torch.float32)
    Y, W = torch.as_tensor(Y), torch.as_tensor(W, dtype=torch.float32)
    p = pol.model.policy
    params = list(p.mlp_extractor.policy_net.parameters()) + list(p.action_net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    g = torch.Generator().manual_seed(0)
    for ep in range(args.epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            mean = p.action_net(p.mlp_extractor.forward_actor(p.extract_features(X[b], p.pi_features_extractor)))
            loss = (W[b, None] * (mean - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        print(f"epoch {ep}: loss {tot / len(X):.5f}", flush=True)
    pol.model.save(out / "policy.zip")
    shutil.copy(src / "vecnormalize.pkl", out / "vecnormalize.pkl")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit"])
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--anchor-tries", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--student")
    ap.add_argument("--data", nargs="*")
    ap.add_argument("--init")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-plan", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else fit(args)


if __name__ == "__main__":
    main()
