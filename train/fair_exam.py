"""Fair exams on fresh seeds for a finished jump or recovery brain (a model folder or a run's best/ folder).

    # jump: 30 tries per bar height at each approach speed (fair agility score, fixed physics, straight approach)
    python fair_exam.py jump runs/jumptrot/best --speeds 1.5 2.0 --dist 1.5 --heights 0.40 0.50 0.60 0.70
    python fair_exam.py jump ../models/bolt-dog-jump-runup-16i-35.5M --speeds 1.5 2.0      # the old brain, same test
    # recovery: 30 tries per kind of fall, the whole recovery (self-right, tuck, stand) and self-righting alone
    python fair_exam.py recover runs/recover/best
    python fair_exam.py recover ../models/bolt-dog-selfright-17e-16.5M

Each try lasts --max-steps control steps (50 per second); the default 220 (4.4 s) is the setting the model cards'
fair 40/50/60 cm numbers were taken with (16i-35.5M: 100/77/3% at 2.0 m/s from 1.5 m, seed 7000). Before 28 Oct
this script used a fixed 200 (4.0 s), which gives 16i-35.5M 83%/7% at 50/60 cm on the same tries: a shorter try,
not different physics. Pass --max-steps 200 to reproduce those older fair_exam numbers.

Seeds 9000+ by default: the training runs examine on seeds 5000+ (jump) and 7000+ (recovery), so these tries
were never used to pick a brain. Writes <folder>/fair_exam_<what>.json unless --no-save.
"""
import argparse
import json
import time
from pathlib import Path
from bolt_env import add_brake_args, set_motor_brake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["jump", "recover"])
    ap.add_argument("folder")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.5, 2.0])
    ap.add_argument("--dist", type=float, default=1.5)
    ap.add_argument("--heights", type=float, nargs="+", default=[0.40, 0.50, 0.60, 0.70])
    ap.add_argument("--max-steps", type=int, default=220,
                    help="jump: control steps per try (50 per second); 220 = 4.4 s, the model cards' setting (before 28 Oct: 200)")
    ap.add_argument("--gait", action="store_true", help="jump: also start mid-stride (trot/canter pose) at each speed")
    ap.add_argument("--strict", action="store_true", help="jump: also the old strict test (randomized physics, askew)")
    ap.add_argument("--easy", action="store_true", help="recover: also the old falls and physics (17e's exam conditions)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="fixed control delay (ms) in every try (29 Oct)")
    add_brake_args(ap, default=False)
    a = ap.parse_args()
    print("motor brake (on, rad/s, N·m per rad/s):", set_motor_brake(a.motor_brake, a.brake_speed, a.brake_gain), flush=True)
    folder, out, t0 = Path(a.folder), {}, time.time()
    if a.latency_ms:
        import getup_env
        import jump_env
        jump_env.EXAM_LATENCY_MS = getup_env.EXAM_LATENCY_MS = a.latency_ms
        out["latency_ms"] = a.latency_ms
    if a.what == "jump":
        from jump_env import JumpPolicy, exam
        pol = JumpPolicy(folder)
        print(f"jump exam: {a.episodes} tries per height, seed {a.seed}, max steps {a.max_steps} ({a.max_steps / 50:.1f} s), "
              f"bar {a.dist} m away, heights {a.heights}", flush=True)
        for sp in a.speeds:
            for gait in ((False, True) if a.gait else (False,)):
                r = exam(pol, episodes=a.episodes, seed=a.seed, heights=tuple(a.heights), dist=a.dist, max_steps=a.max_steps,
                         speed=sp, strict_too=a.strict, gait=gait)
                tag = f"v{round(sp * 10)}{'_gait' if gait else ''}"
                out.update({f"{tag}_{k}": v for k, v in r.items()})
                print(tag, r, flush=True)
    else:
        from getup_env import EXAM_KINDS, GetUpPolicy, exam, exam_full
        pol = GetUpPolicy(folder)
        r = exam_full(pol, episodes=a.episodes, seed=a.seed, hard=True)
        print("full recovery, tough falls:", r, flush=True)
        out.update({f"full_{k}": v for k, v in r.items()})
        r = exam(pol, episodes=a.episodes, seed=a.seed, kinds=EXAM_KINDS, goal="selfright", hard=True)
        print("self-righting alone, tough falls:", r, flush=True)
        out.update({f"right_{k}": v for k, v in r.items()})
        if a.easy:
            r = exam_full(pol, episodes=a.episodes, seed=a.seed, kinds=("side", "back", "tipped", "standing"), hard=False)
            print("full recovery, old falls:", r, flush=True)
            out.update({f"old_{k}": v for k, v in r.items()})
        stood = [v for k, v in out.items() if k.startswith("full_stood_")]
        out["full_stood_mean"] = sum(stood) / len(stood)
    if a.what == "jump":
        out["max_steps"], out["dist"] = a.max_steps, a.dist
    out["motor_brake"] = dict(on=a.motor_brake, speed=a.brake_speed, gain=a.brake_gain)
    out["episodes"], out["seed"], out["minutes"] = a.episodes, a.seed, round((time.time() - t0) / 60, 1)
    print(json.dumps(out, indent=1))
    if not a.no_save:
        (folder / f"fair_exam_{a.what}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
