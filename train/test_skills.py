"""How often does each skill switch work? Scripted sequences on one Bolt (skills.py), each repeated with
different seeds and physics randomization (motor strength, torso mass, floor friction, sensor noise, as the
envs randomize).

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python test_skills.py --n 10

A switch works when Bolt does not fall in the 2 s after it (jumps: and cleared the bar cleanly: over without
touching it, landed on its feet). A switch is only counted when Bolt reached it without falling earlier in
the same try (a fall starts "recover", and what comes after a recovery is a different test). Writes
skills_report.json (rates and, for every failure, what the body was doing) and skills_replays.json (the
first try of each sequence and its first failing try, as 25 Hz body poses in cloudwatch.py's replay shape).
"""
import argparse
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from skills import Bolt   # noqa: E402

HERE = Path(__file__).resolve().parent
WINDOW = 2.0   # s without a fall after a switch


class Tally:
    def __init__(self):
        self.ok, self.tries, self.skipped, self.fails = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(list)
        self.order = []

    def add(self, name, ok, why=None):
        if name not in self.order:
            self.order.append(name)
        if ok is None:            # not reached: an earlier fall in this try
            self.skipped[name] += 1
            return
        self.tries[name] += 1
        self.ok[name] += bool(ok)
        if not ok and why:
            self.fails[name].append(why)


def window(bolt, t0, seconds=WINDOW):
    """Fall in [t0, t0 + seconds)? Plus the lowest body height and uprightness in that window."""
    t1 = t0 + int(round(seconds / bolt.dt))
    tr = [x for x in bolt.trace if t0 <= x[0] < t1]
    fall = next((e for e in bolt.events if e["kind"] == "fall" and t0 <= e["t"] < t1), None)
    stats = dict(min_height=round(min(x[2] for x in tr), 3), min_up=round(min(x[3] for x in tr), 2)) if tr else {}
    if fall:
        stats.update(fell_after=round((fall["t"] - t0) * bolt.dt, 2), in_skill=fall["skill"],
                     height=fall["height"], up=fall["up"])
    return fall is None, stats


def clean_so_far(bolt):
    return not any(e["kind"] == "fall" for e in bolt.events)


# ---------- the sequences ----------
def seq_gaits(bolt, seed, tally):
    """walk -> trot -> canter -> trot -> walk, 3 s each."""
    bolt.start("walk", seed=seed)
    bolt.do("walk", 3.0)
    prev = "walk"
    for g in ("trot", "canter", "trot", "walk"):
        reached, t0 = clean_so_far(bolt), bolt.t
        bolt.do(g, 3.0)
        ok, why = window(bolt, t0)
        tally.add(f"{prev}->{g}", ok if reached else None, why)
        prev = g


def seq_rest(bolt, seed, tally):
    """walk -> sit -> lie (imitation clips) -> stand (posture brain: the clips have no way back up)."""
    bolt.start("walk", seed=seed)
    bolt.do("walk", 2.0)
    for prev, skill, secs in (("walk", "sit", 4.0), ("sit", "lie", 5.5), ("lie", "stand", 3.0)):
        reached, t0 = clean_so_far(bolt), bolt.t
        bolt.do(skill, secs)
        ok, why = window(bolt, t0)
        p = bolt.sr.posture()
        why = dict(why, end_height=round(float(p["height"]), 3), end_up=round(float(p["up"]), 2))
        if skill == "stand":   # really up: upright, body above 38 cm
            ok = ok and p["up"] > 0.9 and p["height"] > 0.38
        tally.add(f"{prev}->{skill}" + (" (posture)" if skill == "stand" else ""), ok if reached else None, why)


def seq_jump(height):
    def run(bolt, seed, tally):
        """trot 2 s -> hurdle placed 1.5-2 m ahead -> trot up, jump, land -> canter 0.6 s -> trot 2 s."""
        bolt.start("trot", seed=seed)
        bolt.do("trot", 2.0)
        reached = clean_so_far(bolt)
        ahead = float(np.random.default_rng(seed).uniform(1.5, 2.0))
        r = bolt.do("jump", height=height, ahead=ahead)
        cm = round(height * 100)
        why = {k: r.get(k) for k in ("cleared", "touched", "dropped", "landed", "fell")}
        tally.add(f"trot->jump {cm} cm", (r["clean"] and not r["fell"]) if reached else None, why)
        if reached and r["landed"] and not r["fell"]:
            ok, w = window(bolt, r["landed_at"], WINDOW + 0.6)
            tally.add(f"jump {cm} cm->(canter)->trot", ok, w)
        else:
            tally.add(f"jump {cm} cm->(canter)->trot", None)
    return run


def seq_shove(bolt, seed, tally):
    """walk 1.5 s -> a hard sideways shove -> (fall detector) recover -> walk 3 s."""
    bolt.start("walk", seed=seed)
    bolt.do("walk", 1.5)
    reached = clean_so_far(bolt)
    bolt.shove()
    bolt.do("walk", 2.0)    # the fall detector starts "recover" by itself
    fell = any(e["kind"] == "fall" for e in bolt.events)
    tally.add("shove -> fall", fell if reached else None)
    rec = next((e for e in bolt.events if e["kind"] in ("recovered", "not_recovered")), None)
    tally.add("fall -> recover (stood up)", (rec["kind"] == "recovered") if reached and rec else None,
              dict(end_height=rec.get("height")) if rec else None)
    if reached and rec and rec["kind"] == "recovered":
        t0 = bolt.t
        bolt.do("walk", 3.0)
        ok, why = window(bolt, t0)
        tally.add("recover -> walk", ok, why)
    else:
        tally.add("recover -> walk", None)


SEQUENCES = {"gaits": seq_gaits, "rest": seq_rest, "jump20": seq_jump(0.20), "jump30": seq_jump(0.30),
             "jump40": seq_jump(0.40), "shove": seq_shove}
LABELS = {"gaits": "Walk, trot, canter, trot, walk", "rest": "Walk, sit, lie down, stand up",
          "jump20": "Trot, jump a 20 cm bar, trot", "jump30": "Trot, jump a 30 cm bar, trot",
          "jump40": "Trot, jump a 40 cm bar, trot", "shove": "Shoved over, gets up, walks on"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="tries per sequence (different seeds and physics)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*", help="sequence names (default: all)")
    ap.add_argument("--report", default=str(HERE / "skills_report.json"))
    ap.add_argument("--replays", default=str(HERE / "skills_replays.json"))
    args = ap.parse_args()
    bolt = Bolt(seed=args.seed)
    tally, replays, progress, started = Tally(), {}, {}, time.time()
    for name in args.only or SEQUENCES:
        before = {k: tally.tries[k] - tally.ok[k] for k in tally.order}
        failed_kept = False
        for k in range(args.n):
            SEQUENCES[name](bolt, args.seed + 37 * k, tally)
            failed = any(tally.tries[x] - tally.ok[x] > before.get(x, 0) for x in tally.order)
            before = {x: tally.tries[x] - tally.ok[x] for x in tally.order}
            if k == 0:
                replays.update(bolt.replay(name))
                progress[name] = dict(best=float(not failed), stood=not failed,
                                      text="every switch worked" if not failed else "a switch failed")
            elif failed and not failed_kept:
                replays.update(bolt.replay(name + "_fail"))
                progress[name + "_fail"] = dict(best=0.0, stood=False, text=f"a failing try (seed {args.seed + 37 * k})")
                failed_kept = True
        print(f"{name}: done ({time.time() - started:.0f} s)", flush=True)
    rows = []
    for s in tally.order:
        n, ok = tally.tries[s], tally.ok[s]
        rows.append(dict(switch=s, ok=ok, tries=n, rate=round(ok / n, 2) if n else None, not_reached=tally.skipped[s],
                         failures=tally.fails[s]))
    print(f"\n{'switch':34s} {'works':>7s} {'rate':>6s} {'not reached':>12s}")
    for r in rows:
        rate = "-" if r["rate"] is None else f"{r['rate']:.0%}"
        print(f"{r['switch']:34s} {r['ok']:>3d}/{r['tries']:<3d} {rate:>6s} {r['not_reached']:>12d}")
    imitate = bolt.imitate_from
    report = dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n=args.n, seed=args.seed,
                  seconds=round(time.time() - started), window_s=WINDOW,
                  brains=dict(imitate=str(Path(imitate[0]).relative_to(HERE.parent) / imitate[1]),
                              jump="models/bolt-dog-jump-a1-16g-27M", selfright="models/bolt-dog-selfright-17e-16.5M",
                              posture="models/bolt-dog-posture-getup-3M"),
                  switches=rows)
    Path(args.report).write_text(json.dumps(report, indent=1))
    steps = int(imitate[1].split("_")[1])
    kinds = [[k, LABELS.get(k, LABELS.get(k.replace("_fail", ""), k) + " (a failing try)")] for k in progress]
    page = dict(updated=report["updated"], runs={"skills": dict(
        label="Bolt switching skills like a dog (skills.py)", where="this machine", goal="cloud", kinds=kinds,
        measure=dict(label="Switches that worked in this try", lo=0, hi=1, unit="%", lowText="none", highText="all"),
        words=["all switches worked", "a switch failed"], finished=True,
        checkpoints={str(steps): dict(steps=steps, exam={r["switch"]: r["rate"] for r in rows}, progress=progress, replays=replays)})})
    Path(args.replays).write_text(json.dumps(page, separators=(",", ":")))
    print(f"\nwrote {args.report} and {args.replays}")


if __name__ == "__main__":
    main()
