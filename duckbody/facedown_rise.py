"""Get up from lying face down (owner's plan): tuck the legs under, lift the body into the perch, then stand.

    python facedown_rise.py [--workers 2] [--seeds 20]

Step 3 (perch -> stand) already works: runtime.py's "perch rise" (MoveBrain, 20/20). This searches steps 1-2
in the real runtime, from a face-down start (nose down 90 deg, dropped and settled):
  tuck   glide the legs to (hip, knee, ankle) over g s, hold 0.5 s           (limbs aligned under the body)
  lift   MoveBrain(goal = perch servo targets, lean, t1 = 0.5, t2, sign)    (body picks up into the sit)
  hold   1 s in the perch, then "wake" -> perch rise -> stand brain
Reached the perch = up > 0.85, height < 0.8 x stand (perch or squat height), soles on the floor. Stood = up > 0.9, height > 0.9 x stand.
The best scripts are re-run on randomized bodies (mass +-10%, friction 0.4-1.0, random heading).
"""
import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np

import runtime as rtm
from poses import leg_q

HERE = Path(__file__).resolve().parent
BRAINS, CFG = rtm.load_brains()
STAND_H = CFG["stand_height"]


def leg_ctrl(m, base, h, k, a):
    q = np.zeros(m.nq)
    leg_q(m, q, h, k, a)
    c = base.copy()
    for side in ("left", "right"):
        for n in ("hip_pitch", "knee", "ankle"):
            c[m.actuator(f"{side}_{n}").id] = q[m.jnt_qposadr[m.joint(f"{side}_{n}").id]]
    return c


def face_down(be, rt, rng):
    """Lay the duck on its belly, nose down, legs at home, and let it settle."""
    m, d = be.model, be.data
    mujoco.mj_resetDataKeyframe(m, d, 0)
    yaw = rng.uniform(-np.pi, np.pi)
    qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    qp = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])                # +90 deg about y: nose down
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qp
    d.qpos[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    d.qpos[2] = 0.12
    d.ctrl[:] = m.key("home").ctrl
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=int(1.0 / m.opt.timestep))
    rt.targets = d.ctrl.copy()
    rt.last_action = np.zeros(len(rt.pol))


def episode(args):
    (h, k, a, g, lean, t2, sign), seed, randomize = args
    rng = np.random.default_rng(seed)
    be = rtm.SimBackend(CFG["servo_names"])
    m = be.model
    if randomize:
        f = rng.uniform(0.9, 1.1, m.nbody)
        m.body_mass[:] *= f
        m.body_inertia[:] *= f[:, None]
        m.geom_friction[:, 0] = rng.uniform(0.4, 1.0)
    rt = rtm.DuckRuntime(backend=be, brains=dict(BRAINS), config=CFG, log=None)
    face_down(be, rt, rng)
    perch = np.array(json.loads((HERE / "model" / "poses_mid.json").read_text())["perch"]["ctrl"])
    rt.brains["pose:tuck"] = rtm.PoseBrain(leg_ctrl(m, rt.home, h, k, a), rt, glide=g)
    rt.brains["pose:lift"] = rtm.MoveBrain(perch, rt, lean, 0.5, t2, sign)
    for b in ("pose:tuck", "pose:lift"):
        rt.quiet.add(b)
    rt.mode = "get up"
    rt.plan = [("pose:tuck", g + 0.5), ("pose:lift", 0.5 + t2 + 1.0), ("pose:perch", None)]
    rt.plan_all = list(rt.plan)
    rt._next_in_plan()
    rt.run(g + 0.5 + 0.5 + t2 + 1.0 + 0.2)
    p = be.posture()
    feet = be.read().feet.sum() == 2
    perched = bool(p["up"] > 0.85 and p["height"] < 0.8 * STAND_H and feet)     # perch or squat height: the perch rise works from both
    best_up = p["up"]
    stood = False
    if perched:
        rt.command("wake")
        mn = 1.0
        for _ in range(int(4.5 * rt.hz)):
            rt.tick()
            mn = min(mn, be.posture()["up"])
        q = be.posture()
        stood = bool(q["up"] > 0.9 and q["height"] > 0.9 * STAND_H and mn > 0.3)
    return args[0], perched, stood, round(float(best_up), 2), round(float(p["height"]), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "runs" / "facedown_rise.json"))
    a = ap.parse_args()
    grid = list(itertools.product((-1.2, -0.6, 0.0, 0.5), (1.6, 2.18), (-1.5, -0.8, 0.0), (0.5, 1.0),
                                  (-1.3, -0.6, 0.3), (1.0, 2.0), (-1, 1)))
    with Pool(a.workers) as pool:
        R = pool.map(episode, [(g, 0, False) for g in grid])
        perched = [r for r in R if r[1]]
        stood = [r for r in R if r[2]]
        print(f"face-down -> perch: {len(perched)}/{len(R)};  -> stood: {len(stood)}/{len(R)} (nominal body)", flush=True)
        best = sorted(R, key=lambda r: (-r[2], -r[1], -r[3]))[:8]
        for r in best:
            print("  ", r, flush=True)
        results = []
        for r in stood[:4]:
            RR = pool.map(episode, [(r[0], 100 + s, True) for s in range(a.seeds)])
            n_p, n_s = sum(x[1] for x in RR), sum(x[2] for x in RR)
            results.append(dict(script=dict(zip(("hip", "knee", "ankle", "tuck_s", "lean", "t2", "sign"), r[0])), perched=n_p, stood=n_s, seeds=a.seeds))
            print(f"randomized: {r[0]} perched {n_p}/{a.seeds}, stood {n_s}/{a.seeds}", flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(nominal=[dict(script=r[0], perched=r[1], stood=r[2], up=r[3], h=r[4]) for r in R],
                                           randomized=results), indent=1, default=float))


if __name__ == "__main__":
    main()
