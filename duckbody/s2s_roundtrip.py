"""Sit down into a seat the duck can stand up from, and stand up from exactly that seat (round trip).

    python s2s_roundtrip.py [--workers 4] [--seeds 20]

The stand-up works (s2s_move.py: MoveBrain from the measured perch seat, 20/20 on randomized bodies); the
sit-down did not end in that seat. This searches both halves on the REAL end state of the sit-down:
  1. sit-down = squat brain 2 s -> MoveBrain(goal = perch ctrl with the goal ankle swept, lean, t1, t2), then the
     ankles glide to the goal ankle over 0.5 s (SitBrain: MoveBrain alone keeps the soles flat and never holds it);
     grid lean in [-1.3, -0.2], t2 in [1, 4], goal ankle in [-1.57, 0] on the nominal body. Keep the ones
     that end up > 0.85, height < 0.65 x stand, and never fall (up stays > 0.5).
  2. for each surviving sit-down, from the state it actually lands in (the same deterministic episode
     continued, not a dropped pose), the rise MoveBrain grid: lean in [-1.3, -0.7], t1 in [0.3, 0.8],
     t2 in [1, 3], sign -1, then the stand brain. Stood = up > 0.9, height > 0.9 x stand, up > 0.3 throughout.
  3. the best pairs are run as full round trips on 20 randomized bodies (mass +-10%, friction 0.4-1.0);
     the bar is 18/20 seated and 18/20 stood.
  4. the winner is saved as "perch" in model/poses_mid.json with its sit-down and rise parameters
     (runtime.py PLANS: sit = squat -> pose:perch, wake from the perch = rise via MoveBrain).
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv        # s2s_move parses nothing, but keep its __main__ quiet
import s2s_move as sm                             # noqa: E402
sys.argv = _argv
from runtime import DuckRuntime, SimBackend       # noqa: E402

HERE = Path(__file__).resolve().parent
SEAT = dict(hip=-0.349, knee=2.182)               # the perch seat's legs (runs/perch_cands.json); ankle is swept
SQUAT_S, SIT_T1, SETTLE_S, RISE_EXTRA_S = 2.0, 0.5, 1.5, 2.0
STAND_H = sm.cfg["stand_height"]


class SitBrain(sm.MoveBrain):
    """MoveBrain keeps the soles flat to the end (its ankle rule also runs after t1 + t2, so the goal ankle is
    never held). For the sit-down the goal ankle is swept: after the move, glide the ankles from the
    flat-sole value to the goal ankle over ANKLE_S. ankle=None keeps MoveBrain's flat soles."""

    def __init__(self, rt, goal, lean, t1, t2, ankle):
        super().__init__(rt, goal, lean, t1, t2)
        self.ankle = ankle

    def __call__(self, obs):
        a = super().__call__(obs)
        rt = self.rt
        t = (rt.steps - self.t0) * rt.dt - (self.t1 + self.t2)
        if self.ankle is None or t <= 0:
            return a
        f = min(1.0, t / ANKLE_S)
        tgt = sm.home[rt.pol] + a * rt.scale
        for an in sm.L["ankle"]:
            j = list(rt.pol).index(an)
            tgt[j] = tgt[j] + (self.goal[an] - tgt[j]) * f
        return np.clip((tgt - sm.home[rt.pol]) / rt.scale, -1, 1)


ANKLE_S = 0.5


def body(seed, randomize):
    be = SimBackend(sm.names)
    m = be.model
    if randomize:
        rng = np.random.default_rng(seed)
        f = rng.uniform(0.9, 1.1, m.nbody)
        m.body_mass[:] = m.body_mass * f
        m.body_inertia[:] = m.body_inertia * f[:, None]
        m.geom_friction[:, 0] = rng.uniform(0.4, 1.0)
    rt = DuckRuntime(backend=be, brains=dict(sm.brains), config=sm.cfg, log=None)
    return be, rt


def sit_down(be, rt, sit):
    """squat -> MoveBrain to the perch. Returns (seated, up, height, min_up)."""
    lean, t2, ankle = sit
    goal = sm.perch_ctrl(be.model, dict(hip=SEAT["hip"], knee=SEAT["knee"], ankle=0.0 if ankle is None else ankle))
    rt.brains["pose:perch"] = SitBrain(rt, goal, lean, SIT_T1, t2, ankle)
    rt.quiet.add("pose:perch")
    rt.run(1.0)
    rt.mode = "sit"
    rt.plan = [("squat", SQUAT_S), ("pose:perch", None)]
    rt.plan_all = list(rt.plan)
    rt._next_in_plan()
    mn = 1.0
    for _ in range(int((SQUAT_S + SIT_T1 + t2 + ANKLE_S + SETTLE_S) * rt.hz)):
        rt.tick()
        mn = min(mn, be.posture()["up"])
    p = be.posture()
    return bool(p["up"] > 0.85 and p["height"] < 0.65 * STAND_H and mn > 0.5), p["up"], p["height"], mn


def rise(be, rt, rs):
    lean, t1, t2 = rs
    rt.brains["pose:rise2"] = sm.MoveBrain(rt, sm.home, lean, t1, t2, sign=-1)
    rt.quiet.add("pose:rise2")
    rt.mode = "rise"
    rt.plan = [("pose:rise2", t1 + t2 + 0.3), ("stand", None)]
    rt.plan_all = list(rt.plan)
    rt._next_in_plan()
    mn = 1.0
    for _ in range(int((t1 + t2 + RISE_EXTRA_S) * rt.hz)):
        rt.tick()
        mn = min(mn, be.posture()["up"])
    p = be.posture()
    return bool(p["up"] > 0.9 and p["height"] > 0.9 * STAND_H and mn > 0.3), p["up"], p["height"], mn


def job_sit(sit):
    be, rt = body(0, False)
    ok, up, h, mn = sit_down(be, rt, sit)
    return sit, ok, round(float(up), 3), round(float(h), 4), round(float(mn), 3), be.data.qpos.round(5).tolist(), be.data.ctrl.round(5).tolist()


def job_pair(args):
    sit, rs = args
    be, rt = body(0, False)
    seated = sit_down(be, rt, sit)[0]
    stood, up, h, mn = rise(be, rt, rs) if seated else (False, 0, 0, 0)
    return sit, rs, seated, stood, round(float(up), 3), round(float(mn), 3)


def job_round(args):
    sit, rs, seed = args
    be, rt = body(seed, True)
    seated = sit_down(be, rt, sit)[0]
    stood = rise(be, rt, rs)[0] if seated else False
    return seated, stood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--top", type=int, default=6, help="pairs to run on randomized bodies")
    ap.add_argument("--poses", default=str(HERE / "model" / "poses_mid.json"))
    ap.add_argument("--out", default=str(HERE / "runs" / "s2s_roundtrip.json"))
    a = ap.parse_args()
    sits = list(itertools.product(np.round(np.linspace(-1.3, -0.2, 6), 3), (1.0, 2.0, 3.0, 4.0), [None] + list(np.round(np.linspace(-1.571, 0.0, 7), 3))))
    rises = list(itertools.product((-1.3, -1.0, -0.7), (0.3, 0.5, 0.8), (1.0, 2.0, 3.0)))
    with Pool(a.workers) as pool:
        S = pool.map(job_sit, sits)
        ok = [s for s in S if s[1]]
        print(f"1. sit-downs: {len(ok)}/{len(S)} end seated (up > 0.85, height < 0.65 x stand, no fall)", flush=True)
        for s in sorted(ok, key=lambda s: s[3])[:10]:
            print("   lean %.2f t2 %.1f ankle %s -> up %.2f h %.1f mm min up %.2f" % (s[0][0], s[0][1], "flat" if s[0][2] is None else "%+.2f" % s[0][2], s[2], 1000 * s[3], s[4]))
        P = pool.map(job_pair, [(s[0], r) for s in ok for r in rises])
        good = [p for p in P if p[3]]
        print(f"2. rises from the real end states: {len(good)}/{len(P)} stood (nominal body)", flush=True)
        # rank: sit-downs whose end state many rises work from first (a forgiving seat), then the rise
        per_sit = {}
        for p in good:
            per_sit.setdefault(tuple(p[0]), []).append(p)
        cands = sorted(good, key=lambda p: (-len(per_sit[tuple(p[0])]), -p[5]))
        seen, pairs = set(), []
        for p in cands:
            if tuple(p[0]) not in seen:
                seen.add(tuple(p[0]))
                pairs.append((tuple(p[0]), tuple(p[1])))
            if len(pairs) >= a.top:
                break
        results = []
        for sit, rs in pairs:
            R = pool.map(job_round, [(sit, rs, 100 + s) for s in range(a.seeds)])
            n_sit, n_stood = sum(r[0] for r in R), sum(r[1] for r in R)
            results.append(dict(sit=dict(lean=sit[0], t1=SIT_T1, t2=sit[1], ankle=sit[2]), rise=dict(lean=rs[0], t1=rs[1], t2=rs[2], sign=-1),
                                seated=n_sit, stood=n_stood, seeds=a.seeds, rises_from_nominal_seat=len(per_sit[sit])))
            print(f"3. sit lean {sit[0]:.2f} t2 {sit[1]:.1f} ankle {sit[2]} | rise lean {rs[0]:.1f} t1 {rs[1]:.1f} t2 {rs[2]:.1f}: "
                  f"seated {n_sit}/{a.seeds}, stood {n_stood}/{a.seeds}", flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(sit_downs=[dict(sit=s[0], ok=s[1], up=s[2], height=s[3], min_up=s[4]) for s in S],
                                           round_trips=results), indent=1, default=float))
    if not results:
        print("no sit-down + rise pair worked on the nominal body")
        return
    best = max(results, key=lambda r: (min(r["seated"], r["stood"]), r["stood"]))
    bar = 0.9 * a.seeds
    print(f"best round trip: seated {best['seated']}/{a.seeds}, stood {best['stood']}/{a.seeds} (bar {bar:.0f}/{a.seeds} both ways)")
    if best["seated"] < bar or best["stood"] < bar:
        print("under the bar: not saved")
        return
    s = best["sit"]
    sit_key = (s["lean"], s["t2"], s["ankle"])
    end = next(x for x in S if tuple(x[0]) == sit_key)
    poses = json.loads(Path(a.poses).read_text())
    poses["perch"] = dict(qpos=end[5], ctrl=end[6], up=end[2], height=end[3],
                          legs=dict(hip=SEAT["hip"], knee=SEAT["knee"], ankle=s["ankle"]),
                          sit=best["sit"], rise=best["rise"],
                          round_trip=f"seated {best['seated']}/{a.seeds}, stood {best['stood']}/{a.seeds} (randomized bodies)",
                          note="floor seat on thighs and soles: sit = squat -> MoveBrain to this ctrl; wake = MoveBrain to home, then the stand brain")
    Path(a.poses).write_text(json.dumps(poses, indent=1))
    print("saved perch to", a.poses)


if __name__ == "__main__":
    main()
