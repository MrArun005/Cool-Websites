"""How often does each skill switch work? Scripted sequences on one Bolt (skills.py), each repeated with
different seeds and physics randomization (motor strength, torso mass, floor friction, sensor noise, as the
envs randomize).

    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python test_skills.py --n 10

A switch works when Bolt does not fall in the 2 s after it (jumps: and cleared the bar cleanly: over without
touching it, landed on its feet). A switch is only counted when Bolt reached it without falling earlier in
the same try (a fall starts "recover", and what comes after a recovery is a different test); a switch not
reached in more than half of its tries is reported as "untested" (no rate), not as a pass rate. Writes
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

from skills import IMITATE_RUN, Bolt   # noqa: E402
from bolt_env import add_brake_args, set_motor_brake   # noqa: E402

HERE = Path(__file__).resolve().parent
WINDOW = 2.0   # s without a fall after a switch

# What the recordings show (run of 2026-10-27, N=10, dog-motion brain run 18b at 15M; rates in skills_report.json)
FINDINGS = [
    "Gait switches mostly work (walk->trot 10/10, trot->canter 9/10, canter->trot 7/9, trot->walk 7/7). The falls are "
    "sideways (roll ~1 rad when the uprightness drops below 0.5) 0.5-0.8 s after the switch, at 1.7-2 m/s. The 'canter' "
    "is not a real canter: Bolt makes ~1.2 m/s of the dog's 3.3 m/s, so the canter clip is mostly a different trot.",
    "walk->sit never falls but only 1 in 10 is a sit: at 1 m/s the sit clip (a dog standing still, then sitting) makes Bolt "
    "brake by folding the front legs, and it ends nose DOWN (pitch -0.5 rad, the clip has +0.6) at 20 cm. From that pose "
    "the lie clip tips it over (9/10 falls, 1.35 s in, nose down and rolled 0.6 rad). From standing still the same chain "
    "works every time (sit 10/10 in pose, lie 10/10, posture stand 10/10). Starting the sit clip at its start instead of "
    "the closest pose (8/8 fell), or braking first with the posture brain's stand (6-8 of 8 fell), did not help.",
    "Jumps from a trot: the jump brain learned from a standstill 45-65 cm away; handed over at 1.1 m from a 1.5-1.9 m/s trot "
    "it clears 20 cm 8/10 and 30 cm 7/10 cleanly, but 40 cm only 3/10 (every failure knocked the bar: the trailing legs "
    "clip it; 1 in 10 also crashes). The bigger problem is after the landing: Bolt lands at 3-4 m/s, faster than any clip "
    "the dog-motion brain can hold, and falls sideways within 0.5-2 s in 17 of 26 landings, at every height (a 0.6 s canter "
    "bridge did not measurably help).",
    "Recovery works: every shove topples Bolt, self-righting + tuck + stand stands it up 9/10 (the 10th ends at 38 cm, "
    "just under the bar), and walking off after it works 9/9.",
    "What would fix them: train the transitions instead of only the skills. (1) Imitation training that switches clips "
    "mid-try at the closest pose (exactly what skills.py does) and starts sit/lie from moving walk/trot states, or a "
    "'stop' clip, so it learns to brake before sitting. (2) The jump brain trained from a running approach (tries that "
    "start from recorded trot states 1-2 m before the bar) and continuing 2 s into the trot reward after landing, or the "
    "dog-motion brain given starts from the jump brain's landings (3-4 m/s). (3) A real canter (run 18c/22) so fast "
    "landings have a gait to go to.",
]


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


def rows(tally):
    """One row per switch. A switch that was not reached (an earlier fall in the same try) in more than half of
    its tries is "untested": its few tries say nothing about the switch (lie -> stand passed 1/1 with 9 of 10
    tries never reaching lie), so it gets status "untested" and no rate instead of a pass rate."""
    out = []
    for s in tally.order:
        n, ok, skipped = tally.tries[s], tally.ok[s], tally.skipped[s]
        untested = skipped > (n + skipped) / 2 or n == 0
        out.append(dict(switch=s, status="untested" if untested else "tested", ok=ok, tries=n,
                        rate=None if untested else round(ok / n, 2), not_reached=skipped, failures=tally.fails[s]))
    return out


def rate_text(r):
    return "untested" if r["status"] == "untested" else f"{r['rate']:.0%}"


def window(bolt, t0, seconds=WINDOW):
    """Fall in [t0, t0 + seconds)? With what the body did from t0 up to the fall (or the window's end): speed
    at the switch, lowest height, least upright, and the roll and pitch as it went over (which way it fell)."""
    t1 = t0 + int(round(seconds / bolt.dt))
    fall = next((e for e in bolt.events if e["kind"] == "fall" and t0 <= e["t"] < t1), None)
    tr = [x for x in bolt.trace if t0 <= x[0] < (fall["t"] if fall else t1)]
    stats = {}
    if tr:
        stats = dict(speed_at_switch=round(tr[0][4], 2), min_height=round(min(x[2] for x in tr), 3),
                     min_up=round(min(x[3] for x in tr), 2))
    if fall:
        last = tr[-1] if tr else None
        stats.update(fell_after=round((fall["t"] - t0) * bolt.dt, 2), in_skill=fall["skill"], height=fall["height"],
                     speed=round(last[4], 2) if last else None, roll=round(last[5], 2) if last else None,
                     pitch=round(last[6], 2) if last else None)
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


POSE_OK = {   # the pose the clip ends in, reached (clip: sit 23 cm, nose up 0.6 rad; lie 16-17 cm, level)
    "sit": lambda p: p["height"] < 0.27 and p["pitch"] > 0.3,
    "lie": lambda p: p["height"] < 0.19 and p["up"] > 0.9,
}


def seq_rest(start):
    def run(bolt, seed, tally):
        """walk (or stand still: the sit clip's first second) -> sit -> lie (imitation clips) -> stand (posture
        brain: the clips have no way back up). Also counts whether the sit and lie poses were really reached."""
        if start == "walk":
            bolt.start("walk", seed=seed)
            bolt.do("walk", 2.0)
            prev, tag = "walk", ""
        else:
            bolt.start("sit", tau=0.0, seed=seed)   # standing still at the start of the sit clip
            prev, tag = "still", "still: "
        for skill, secs in (("sit", 4.0), ("lie", 5.5), ("stand", 3.0)):
            reached, t0 = clean_so_far(bolt), bolt.t
            bolt.do(skill, secs)
            ok, why = window(bolt, t0)
            p = bolt.sr.posture()
            why = dict(why, end_height=round(float(p["height"]), 3), end_up=round(float(p["up"]), 2),
                       end_pitch=round(float(p["pitch"]), 2))
            if skill == "stand":   # really up: upright, body above 38 cm
                ok = ok and p["up"] > 0.9 and p["height"] > 0.38
            name = f"{tag}{prev}->{skill}" + (" (posture)" if skill == "stand" else "")
            tally.add(name, ok if reached else None, why)
            if skill in POSE_OK:
                tally.add(name + " [pose reached]", (ok and POSE_OK[skill](p)) if reached else None, why)
            prev = skill
    return run


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


SEQUENCES = {"gaits": seq_gaits, "rest": seq_rest("walk"), "rest_still": seq_rest("still"), "jump20": seq_jump(0.20), "jump30": seq_jump(0.30),
             "jump40": seq_jump(0.40), "shove": seq_shove}
LABELS = {"gaits": "Walk, trot, canter, trot, walk", "rest": "Walk, sit, lie down, stand up",
          "rest_still": "Standing still, sit, lie down, stand up",
          "jump20": "Trot, jump a 20 cm bar, trot", "jump30": "Trot, jump a 30 cm bar, trot",
          "jump40": "Trot, jump a 40 cm bar, trot", "shove": "Shoved over, gets up, walks on"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="tries per sequence (different seeds and physics)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*", help="sequence names (default: all)")
    ap.add_argument("--imitate-steps", type=int, help="dog-motion checkpoint (default: run 18b's newest)")
    ap.add_argument("--report", default=str(HERE / "skills_report.json"))
    ap.add_argument("--replays", default=str(HERE / "skills_replays.json"))
    add_brake_args(ap, default=False)
    args = ap.parse_args()
    print("motor brake (on, rad/s, N·m per rad/s):", set_motor_brake(args.motor_brake, args.brake_speed, args.brake_gain), flush=True)
    n = args.imitate_steps
    bolt = Bolt(seed=args.seed, imitate=(IMITATE_RUN, f"policy_{n}_steps.zip", f"policy_vecnormalize_{n}_steps.pkl") if n else None)
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
    table = rows(tally)
    print(f"\n{'switch':34s} {'works':>7s} {'rate':>8s} {'not reached':>12s}")
    for r in table:
        print(f"{r['switch']:34s} {r['ok']:>3d}/{r['tries']:<3d} {rate_text(r):>8s} {r['not_reached']:>12d}")
    imitate = bolt.imitate_from
    report = dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n=args.n, seed=args.seed,
                  seconds=round(time.time() - started), window_s=WINDOW,
                  brains=dict(imitate=str(Path(imitate[0]).relative_to(HERE.parent) / imitate[1]),
                              jump="models/bolt-dog-jump-a1-16g-27M", selfright="models/bolt-dog-selfright-17e-16.5M",
                              posture="models/bolt-dog-posture-getup-3M"),
                  switches=table, findings=FINDINGS)
    Path(args.report).write_text(json.dumps(report, indent=1))
    steps = int(imitate[1].split("_")[1])
    kinds = [[k, LABELS.get(k, LABELS.get(k.replace("_fail", ""), k) + " (a failing try)")] for k in progress]
    page = dict(updated=report["updated"], runs={"skills": dict(
        label="Bolt switching skills like a dog (skills.py)", where="this machine", goal="cloud", kinds=kinds,
        measure=dict(label="Switches that worked in this try", lo=0, hi=1, unit="%", lowText="none", highText="all"),
        words=["all switches worked", "a switch failed"], finished=True,
        checkpoints={str(steps): dict(steps=steps, exam={r["switch"]: r["rate"] for r in table}, progress=progress, replays=replays)})})
    Path(args.replays).write_text(json.dumps(page, separators=(",", ":")))
    print(f"\nwrote {args.report} and {args.replays}")


if __name__ == "__main__":
    main()
