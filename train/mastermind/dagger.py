"""Distil every expert into one student by DAgger (Ross et al. 2011), on the unified Master env.

Round 0 rolls out the teachers (with a little motor noise, DART-style, so the data also covers small mistakes);
every later round rolls out the STUDENT (each step, with probability beta the teacher's action is executed
instead), labels every visited state with the teacher of the active command (the new command's teacher from
the very first step after a switch), adds it to the data set and retrains the student on everything so far.

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/dagger.py --rounds 6 --steps 150000 --workers 2

A handler drives every try: commands at random times along the transitions the dog has (gait <-> gait, gait ->
sit -> lie -> stand -> gait, gait -> jump over a 15-45 cm bar 1.5-2 m ahead -> back to the gait, shoves that
topple Bolt -> recovery -> walk on).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

OUT = HERE.parent / "runs" / "mastermind" / "out"   # big files (data set, checkpoints): gitignored
GAIT_W = {"walk": 3, "trot": 3, "pace": 1, "canter": 2, "turn_left": 1, "turn_right": 1}
HANDLER = {"v3": dict(jump_h=(0.15, 0.55), jump_ahead=(1.5, 2.2),
                      gait_w={"walk": 3, "trot": 3, "pace": 1, "canter": 3, "turn_left": 1, "turn_right": 1})}


def pick(rng, weights):
    k = list(weights)
    p = np.array([weights[x] for x in k], float)
    return k[rng.choice(len(k), p=p / p.sum())]


class Handler:
    """Says what to do next, like a person with a dog on a course. jump_h / jump_ahead: the bar heights and distances
    (v2: 15-45 cm, 1.5-2 m; v3: 15-55 cm, 1.5-2.2 m)."""

    def __init__(self, rng, jump_h=(0.15, 0.45), jump_ahead=(1.5, 2.0), gait_w=None):
        self.rng = rng
        self.jump_h, self.jump_ahead = jump_h, jump_ahead
        self.gait_w = gait_w or GAIT_W

    def first(self):
        r = self.rng
        if r.random() < 0.15:
            return "sit", 0.0     # standing still at the start of the sit clip
        return pick(r, self.gait_w), None

    def next(self, cmd):
        """(command, kwargs, seconds) after `cmd` has run its time."""
        r = self.rng
        if cmd in self.gait_w:
            k = pick(r, dict(gait=50, jump=22, sit=12, stand=4, shove=8 if cmd in ("walk", "trot") else 2))
            if k == "gait":
                g = pick(r, {g: w for g, w in self.gait_w.items() if g != cmd})
                return g, {}, r.uniform(1.5, 4.0)
            if k == "jump":
                return "jump", dict(height=float(r.uniform(*self.jump_h)), ahead=float(r.uniform(*self.jump_ahead))), r.uniform(1.5, 3.0)
            if k == "sit":
                return "sit", {}, r.uniform(3.0, 5.0)
            if k == "stand":
                return "stand", {}, r.uniform(2.0, 3.0)
            return "shove", {}, r.uniform(1.5, 3.0)
        if cmd == "sit":
            k = pick(r, dict(lie=65, stand=25, walk=10))
            return k, {}, (r.uniform(4.0, 6.0) if k == "lie" else r.uniform(2.0, 3.5))
        if cmd == "lie":
            return "stand", {}, r.uniform(2.5, 4.0)
        # stand (also after a recovery)
        k = pick(r, dict(walk=45, trot=35, sit=20))
        return k, {}, r.uniform(2.0, 4.0)


def collect(args):
    """One worker: `steps` labelled states. Returns dict(obs, label, info)."""
    (wid, seed, steps, student_path, beta, noise, imitate, teachers, over) = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from env import Master
    from student import Student
    rng = np.random.default_rng(seed)
    m = Master(seed=seed, imitate=imitate, teachers=teachers, **over)
    hkw = HANDLER.get(teachers, {})
    m.patient = True
    brain = Student(student_path, m) if student_path else None
    X, Y = [], []
    stats = dict(tries=0, falls=0, jumps=0, clean=0, shoves=0, recovered=0)
    ep_seed = seed * 1000
    while len(X) < steps:
        ep_seed += 1
        h = Handler(rng, **hkw)
        clip, tau = h.first()
        m.start(clip, tau=tau, seed=ep_seed)
        stats["tries"] += 1
        cmd, left = clip, int(rng.uniform(1.0, 3.0) / m.dt)
        ep_noise = noise * rng.uniform(0.0, 1.0)
        for _ in range(int(rng.uniform(15, 25) / m.dt)):
            if m.recovering() and m.rec["n"] > 400:   # stuck (patient chain): a new try
                break
            if left <= 0 and not m.recovering() and m.jump is None:
                nxt, kw, secs = h.next(m.cmd)
                if nxt == "shove":
                    m.shove()
                    stats["shoves"] += 1
                else:
                    m.command(nxt, **kw)
                    stats["jumps"] += nxt == "jump"
                left = int(secs / m.dt)
            x = m.obs()
            y = m.label()
            X.append(x)
            Y.append(y.astype(np.float32))
            if brain is None or rng.random() < beta:
                ctrl = y
            else:
                ctrl = brain.act(x)
            if ep_noise:
                ctrl = ctrl + rng.normal(0, ep_noise, 12)
            r = m.step(ctrl)
            stats["falls"] += r["fell"]
            if r.get("jump_done"):
                stats["clean"] += bool(m.last_jump.get("clean"))
            left -= 1
            if m.b.events and m.b.events[-1]["kind"] == "recovered" and m.b.events[-1]["t"] == m.t:
                stats["recovered"] += 1
            if len(X) >= steps:
                break
    return dict(obs=np.array(X, np.float32), label=np.array(Y, np.float32), stats=stats)


def sample_weights(X):
    """More weight where the data is thin and mistakes are costly: the jump brain's second (x4) and the first
    second after every command (x2)."""
    return (1.0 + 3.0 * (X[:, 82] > 0) + 1.0 * (X[:, 83] < 1 / 3)).astype(np.float32)


def train(net, X, Y, full, dog, epochs, lr, log, batch=2048):
    import torch
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    Wt = torch.as_tensor(sample_weights(X))
    full_t, dog_t = torch.as_tensor(full, dtype=torch.float32), torch.as_tensor(dog, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xt)
    perm = np.random.default_rng(0).permutation(n)
    val, tr = perm[: n // 50], perm[n // 50:]
    steps_per = max(1, len(tr) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per, eta_min=lr * 0.05)
    for ep in range(epochs):
        net.train()
        idx = np.random.permutation(tr)
        tot = 0.0
        for k in range(steps_per):
            b = idx[k * batch:(k + 1) * batch]
            ctrl = dog_t + full_t * net(Xt[b])
            loss = (((ctrl - Yt[b]) ** 2).mean(1) * Wt[b]).sum() / Wt[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        net.eval()
        with torch.no_grad():
            v = ((dog_t + full_t * net(Xt[val]) - Yt[val]) ** 2).mean().item()
        log(f"  epoch {ep + 1}/{epochs}: train mse {tot / steps_per:.5f} rad^2, val {v:.5f} (rms {np.sqrt(v):.3f} rad)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=150000, help="labelled states per round (all workers)")
    ap.add_argument("--steps0", type=int, default=300000, help="round 0 (teacher rollouts)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--epochs0", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.08, help="max motor noise (rad) in teacher rollouts")
    ap.add_argument("--imitate", help="override the teacher set's dog-motion brain")
    ap.add_argument("--teachers", default="v2", help="teacher set (teachers.SETS)")
    ap.add_argument("--over", default="{}", help="JSON overrides of the teacher set (Master keyword arguments)")
    ap.add_argument("--hidden", default="512,512,256")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--resume", action="store_true", help="continue from out/data.npz and out/student.pt")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--init-data", help="start from this data.npz (its first --init-rows rows) instead of teacher rollouts")
    ap.add_argument("--init-rows", type=int, default=0)
    ap.add_argument("--init-student", help="start the network from this student.pt (weights and input normalizer) "
                    "instead of a fresh one (v4: the v3 student)")
    ap.add_argument("--beta-floor", type=float, default=0.0, help="teacher share of the executed actions never below this")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.workers)
    from env import Master
    from imitate_env import DOG
    from student import Net, save
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "dagger.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    over = json.loads(a.over)
    full = Master(seed=0, imitate=a.imitate, teachers=a.teachers, **over).full
    hidden = tuple(int(x) for x in a.hidden.split(","))
    student = out / "student.pt"
    if a.resume and (out / "data.npz").exists():
        z = np.load(out / "data.npz")
        X, Y = [z["obs"]], [z["label"]]
        r0 = int(z["round"]) + 1
        ck = torch.load(student, weights_only=False)
        net = Net(hidden=ck["hidden"])
        net.load_state_dict(ck["state"])
    else:
        X, Y, r0, net = [], [], 0, None
        if a.init_data:
            z = np.load(a.init_data)
            k = a.init_rows or len(z["obs"])
            X, Y = [z["obs"][:k]], [z["label"][:k]]
            r0 = 1
            log(f"init data: {k} rows of {a.init_data}")
    history = json.loads((out / "history.json").read_text()) if a.resume and (out / "history.json").exists() else []
    def fresh_net(Xa):
        n = Net(hidden=hidden)
        mean, std = Xa.mean(0), Xa.std(0)
        std[std < 1e-3] = 1.0
        for i in list(range(46, 57)) + [77]:   # one-hot and flag inputs stay 0/1
            mean[i], std[i] = 0.0, 1.0
        n.mean[:], n.std[:] = torch.as_tensor(mean), torch.as_tensor(std)
        return n

    if a.init_data and net is None:   # a first student on the given data, before any rollout of it
        net = fresh_net(X[0])
        t1 = time.time()
        train(net, X[0], Y[0], full, DOG, a.epochs0, a.lr, log)
        save(net, student, round=0, imitate=a.imitate, teachers=a.teachers, over=over)
        save(net, out / "student_r0.pt", round=0, imitate=a.imitate, teachers=a.teachers, over=over)
        log(f"init student trained in {time.time() - t1:.0f} s")
    ctx = mp.get_context("fork")
    for rnd in range(r0, a.rounds):
        t0 = time.time()
        steps = a.steps0 if rnd == 0 else a.steps
        beta = 1.0 if rnd == 0 else max(a.beta_floor, 0.5 - 0.15 * (rnd - 1))
        noise = a.noise if rnd == 0 else 0.0
        per = steps // a.workers
        jobs = [(w, a.seed * 100 + rnd * 10 + w, per, str(student) if rnd > 0 else None, beta, noise, a.imitate, a.teachers, over)
                for w in range(a.workers)]
        with ctx.Pool(a.workers) as pool:
            res = pool.map(collect, jobs)
        st = {k: sum(r["stats"][k] for r in res) for k in res[0]["stats"]}
        X += [r["obs"] for r in res]
        Y += [r["label"] for r in res]
        Xa, Ya = np.concatenate(X), np.concatenate(Y)
        X, Y = [Xa], [Ya]
        log(f"round {rnd}: beta {beta:.2f}, collected {steps} in {time.time() - t0:.0f} s, {st}; data {len(Xa)}")
        if net is None and a.init_student:
            ck = torch.load(a.init_student, weights_only=False)
            net = Net(hidden=ck["hidden"])
            net.load_state_dict(ck["state"])
            log(f"network from {a.init_student}")
        if net is None:
            net = fresh_net(Xa)
        t1 = time.time()
        train(net, Xa, Ya, full, DOG, a.epochs0 if rnd == 0 else a.epochs, a.lr if rnd == 0 else a.lr * 0.5, log)
        save(net, student, round=rnd, imitate=a.imitate, teachers=a.teachers, over=over)
        save(net, out / f"student_r{rnd}.pt", round=rnd, imitate=a.imitate, teachers=a.teachers, over=over)
        np.savez(out / "data.npz", obs=Xa, label=Ya, round=rnd)
        history.append(dict(round=rnd, beta=beta, data=len(Xa), collect_s=round(t1 - t0), train_s=round(time.time() - t1), **st))
        (out / "history.json").write_text(json.dumps(history, indent=1))
        log(f"round {rnd}: trained in {time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
