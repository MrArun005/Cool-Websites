"""Watch the cloud training runs for Bolt Learning Live: record each new checkpoint as 3D replays.

    python cloudwatch.py --out <site>/data/cloud.json

The cloud machines push their newest checkpoint to their own branches (train/remote/<run>/). Every few
minutes this fetches them, and for each checkpoint not seen yet records a few tries the way the games use
the brain: jumps over the real hurdle (run 16f), walking, running and turning (run 19), and copying the
real dog's motions (run 18). Each try is Bolt's body pose (qpos 0-18) at 25 Hz; jump tries also carry the
hurdle (its fixed parts once, the bar's pose per frame). The page draws them with the same robot model.
Kept separate from learnwatch.py's file so the two watchers never overwrite each other.
"""
import argparse
import ast
import datetime as dt
import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = {
    "jump16g": dict(branch="claude/train-jump16e", label="Run 16g: higher jumps, 30-50 cm (real A1 motors)", where="cloud machine 1",
                    kinds=[["j25", "25 cm bar"], ["j30", "30 cm bar"], ["j40", "40 cm bar"], ["j50", "50 cm bar"]],
                    measure=dict(label="Clean jumps in the exam (10 tries per bar height)", lo=0, hi=1, unit="%",
                                 lowText="none", highText="all clean"),
                    words=["clean jump", "not clean"]),
    "jump16h": dict(branch="claude/train-jump16e", label="Run 16h: 40-60 cm bars (real A1 motors)", where="cloud machine 1",
                    kinds=[["j30", "30 cm bar"], ["j40", "40 cm bar"], ["j50", "50 cm bar"], ["j60", "60 cm bar"]],
                    measure=dict(label="Clean jumps in the exam (10 tries per bar height)", lo=0, hi=1, unit="%",
                                 lowText="none", highText="all clean"),
                    words=["clean jump", "not clean"]),
    "jump16i": dict(branch="claude/train-jump16e", label="Run 16i: high jumps with a run-up (40-70 cm)", where="cloud machine 1",
                    kinds=[["j40", "40 cm bar"], ["j50", "50 cm bar"], ["j60", "60 cm bar"], ["j70", "70 cm bar"]],
                    measure=dict(label="Clean jumps in the exam (10 tries per bar height)", lo=0, hi=1, unit="%",
                                 lowText="none", highText="all clean"),
                    words=["clean jump", "not clean"]),
    "imitate18b": dict(branch="claude/train-run20", label="Run 18b: moving like a real dog, with the canter (dog running)", where="cloud machine 4 (moved at 18.5M)",
                      kinds=[["walk", "Walk"], ["trot", "Trot"], ["pace", "Pace"], ["canter", "Canter (run)"], ["turn_left", "Turn"], ["sit", "Sit"], ["lie", "Lie down"]],
                      measure=dict(label="How closely Bolt copies the real dog (1 = exactly)", lo=0, hi=1, unit="",
                                   lowText="not at all", highText="exactly"),
                      words=["copied", "fell"]),
    "imitate18d": dict(branch="claude/train-imitate18d", label="Run 18d: run 18b + getting up like a dog", where="cloud machine 7 (from run 18b at 14M)",
                      kinds=[["walk", "Walk"], ["trot", "Trot"], ["pace", "Pace"], ["canter", "Canter (run)"], ["turn_left", "Turn"], ["sit", "Sit"], ["lie", "Lie down"], ["getup", "Get up"]],
                      measure=dict(label="How closely Bolt copies the real dog (1 = exactly)", lo=0, hi=1, unit="",
                                   lowText="not at all", highText="exactly"),
                      words=["copied", "fell"]),
    "imitate18e": dict(branch="claude/train-walk19", label="Run 18e: switching between motions like a dog", where="cloud machine 2 (from run 18b at 15.5M)",
                      kinds=[["walk", "Walk"], ["trot", "Trot"], ["pace", "Pace"], ["canter", "Canter (run)"], ["turn_left", "Turn"], ["sit", "Sit"], ["lie", "Lie down"], ["getup", "Get up"]],
                      measure=dict(label="How closely Bolt copies the real dog (1 = exactly)", lo=0, hi=1, unit="",
                                   lowText="not at all", highText="exactly"),
                      words=["copied", "fell"]),
    "imitate18c": dict(branch="claude/train-run22b", label="Run 18c: running like a real dog (walk, trot, canter)", where="cloud machine 6 (from run 18b at 13.5M)",
                      kinds=[["walk", "Walk"], ["trot", "Trot"], ["canter", "Canter (run)"], ["turn_left", "Turn left"], ["turn_right", "Turn right"]],
                      measure=dict(label="How closely Bolt copies the real dog (1 = exactly)", lo=0, hi=1, unit="",
                                   lowText="not at all", highText="exactly"),
                      words=["copied", "fell"]),
}


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)


def newest(run):
    """(steps, folder path in the branch) of the newest checkpoint the branch holds, or None."""
    b = RUNS[run]["branch"]
    if b == "local":   # a run training on this machine
        zs = [int(p.stem.split("_")[1]) for p in (REPO / f"train/runs/{run}/checkpoints").glob("policy_*_steps.zip")]
        return (max(zs), f"train/runs/{run}/checkpoints") if zs else None
    names = git("ls-tree", "-r", "--name-only", f"origin/{b}", f"train/remote/{run}/").stdout.split()
    best = None
    for n in names:
        m = re.search(r"/policy_(\d+)_steps\.zip$", n)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), n.rsplit("/", 1)[0])
    return best


def fetch(run, steps, folder, into):
    """The checkpoint's files under the names the policy loaders expect (policy.zip, vecnormalize.pkl)."""
    b = RUNS[run]["branch"]
    if b == "local":
        for src, dst in ((f"policy_{steps}_steps.zip", "policy.zip"), (f"policy_vecnormalize_{steps}_steps.pkl", "vecnormalize.pkl")):
            shutil.copy(REPO / folder / src, into / dst)
        return
    for src, dst in ((f"policy_{steps}_steps.zip", "policy.zip"), (f"policy_vecnormalize_{steps}_steps.pkl", "vecnormalize.pkl")):
        blob = subprocess.run(["git", "show", f"origin/{b}:{folder}/{src}"], cwd=REPO, capture_output=True).stdout
        (into / dst).write_bytes(blob)


def exams(run):
    if RUNS[run]["branch"] == "local":
        p = REPO / f"train/runs/{run}/train.log"
        log = p.read_text() if p.exists() else ""
        return {int(m.group(1)): ast.literal_eval(m.group(2)) for m in re.finditer(r"checkpoint (\d+) ready: (\{.*?\})", log)}
    log = git("show", f"origin/{RUNS[run]['branch']}:train/remote/{run}/train.log").stdout
    return {int(m.group(1)): ast.literal_eval(m.group(2)) for m in re.finditer(r"checkpoint (\d+) ready: (\{.*?\})", log)}


def quat(xmat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, dtype=float))
    return q


# ---------- recording ----------
def record_jump(d, exam, heights=(0.15, 0.20, 0.25, 0.30), dist=0.55, steps=150, speed=None):
    from jump_env import JumpEnv, JumpPolicy
    p = JumpPolicy(d)
    env = JumpEnv(randomize=False, real=True, max_steps=steps)
    m = env.model
    hurdle_body = m.body("hurdle0").id
    rec, prog = {}, {}
    for k, h in ((f"j{round(h * 100)}", h) for h in heights):
        env.exam_bar, env.exam_askew = (dist, h), (0.0, 0.0)
        env.exam_speed = speed
        obs, _ = env.reset(seed=4321)
        static = [[int(m.geom_type[g]), np.round(m.geom_size[g], 4).tolist(), np.round(env.data.geom_xpos[g], 4).tolist(),
                   np.round(quat(env.data.geom_xmat[g]), 4).tolist()] for g in range(m.ngeom) if m.geom_bodyid[g] == hurdle_body]
        bar = m.geom_size[env.bar_geom]
        frames, bars, info, fell = [], [], {}, False
        for t in range(env.max_steps):
            if t % 2 == 0:
                frames.append(env.data.qpos[:19].copy()); bars.append(env.data.qpos[env.bar_q:env.bar_q + 7].copy())
            obs, _, fell, trunc, info = env.step(p.act(obs))
            if fell or trunc:
                break
        clean = bool(info.get("cleared") and not info.get("touched") and not fell and info.get("up", 0) > 0.9
                     and info.get("height", 0) > 0.35)
        rec[k] = np.round(np.concatenate(frames), 4).tolist()
        rec[k + "_props"] = dict(static=static, bar=[float(bar[0]), float(bar[1])], barpose=np.round(np.concatenate(bars), 4).tolist())
        rate = exam.get(f"jump_{round(h * 100)}")
        prog[k] = dict(best=rate if rate is not None else float(clean), stood=clean,
                       text=(f"{round(rate * 100)}% clean" if rate is not None else ("clean" if clean else "not clean")))
    return rec, prog


def record_walk(d):
    from bolt_env import BoltEnv
    from evaluate import Policy
    shutil.copy(REPO / "models/bolt-dog-walk-sharp-12M/gait.json", d / "gait.json")
    p = Policy(d)
    env = BoltEnv(randomize=False, max_steps=10_000, gait_generator="dog", turning=p.turning, speed_control=p.speed_control,
                  sharp_turns=p.sharp_turns)
    rec, prog = {}, {}
    for k, v, w in (("walk", 0.4, 0.0), ("run", 1.0, 0.0), ("turn", 0.4, 0.4), ("sharp", 0.4, 1.5)):
        env.forced_command = "walk"
        obs, _ = env.reset(seed=4321)
        if env.turning:
            env.set_turn(w)
        if env.speed_control:
            env.set_speed(v)
        obs = env._obs()
        frames, fell, start, yaw0 = [], False, env.data.qpos[:2].copy(), None
        R = env._rotation(); h0 = np.arctan2(R[1, 0], R[0, 0]); yaw, last = 0.0, h0
        for t in range(250):
            if t % 2 == 0:
                frames.append(env.data.qpos[:19].copy())
            obs, _, fell, trunc, info = env.step(p.act(obs))
            R = env._rotation(); hh = np.arctan2(R[1, 0], R[0, 0]); yaw += float(np.angle(np.exp(1j * (hh - last)))); last = hh
            if fell:
                break
        secs = len(frames) * 2 * env.dt
        dist = float(np.dot(env.data.qpos[:2] - start, [np.cos(h0), np.sin(h0)]))
        kept = (yaw / secs / w) if w else (dist / secs / v)
        rec[k] = np.round(np.concatenate(frames), 4).tolist()
        prog[k] = dict(best=round(float(np.clip(kept, -0.2, 1.2)), 3), stood=not fell,
                       text=("fell" if fell else (f"{abs(yaw / secs):.2f} rad/s" if w else f"{dist / secs:.2f} m/s")))
    return rec, prog


def record_imitate(d, run="imitate18"):
    from imitate_env import ImitateEnv, ImitatePolicy
    if (Path(d) / "params.pkl").exists():   # a brain trained on the GPU (mjx_imitate.py)
        from mjx_imitate import MjxImitatePolicy
        p = MjxImitatePolicy(Path(d) / "params.pkl")
    else:
        p = ImitatePolicy(d)
    env = ImitateEnv(randomize=False, clips=[k for k, _ in RUNS[run]["kinds"]])   # every clip shown on the page, canter included
    rec, prog = {}, {}
    for k, _ in RUNS[run]["kinds"]:
        if k not in env.clip_names:
            continue
        env.exam_start = (k, 0.0)
        obs, _ = env.reset(seed=4321)
        if env.clip.loop:
            env.limit = 200
        frames, total, fell = [], 0.0, False
        for t in range(env.limit):
            if t % 2 == 0:
                frames.append(env.data.qpos[:19].copy())
            obs, r, fell, done, info = env.step(p.act(obs))
            total += r
            if fell or done:
                break
        rec[k] = np.round(np.concatenate(frames), 4).tolist()
        score = total / env.limit
        prog[k] = dict(best=round(float(score), 3), stood=not fell, text=("fell" if fell else f"copies {score:.2f}"))
    return rec, prog


def record_run(d):
    from run_env import RunEnv, RunPolicy
    p = RunPolicy(d)
    env = RunEnv(randomize=False, pushes=False, max_steps=250)
    rec, prog = {}, {}
    for k, cmd in (("v1", (1, 0, 0)), ("v2", (2, 0, 0)), ("v3", (3, 0, 0)), ("v4", (4, 0, 0)), ("w3", (0.5, 0, 3.0))):
        env.exam_command = cmd
        obs, _ = env.reset(seed=4321)
        frames, got, fell = [], [], False
        for t in range(250):
            if t % 2 == 0:
                frames.append(env.data.qpos[:19].copy())
            obs, _, fell, _, info = env.step(p.act(obs))
            if t >= 100:
                got.append(info["w"] if k[0] == "w" else info["vx"])
            if fell:
                break
        kept = 0.0 if fell or not got else float(np.mean(got)) / (cmd[2] if k[0] == "w" else cmd[0])
        rec[k] = np.round(np.concatenate(frames), 4).tolist()
        prog[k] = dict(best=round(float(np.clip(kept, -0.2, 1.2)), 3), stood=not fell,
                       text="fell" if fell else (f"{np.mean(got):.2f} rad/s" if k[0] == "w" else f"{np.mean(got):.2f} m/s"))
    return rec, prog


def news(path, run, n, prog):
    if not path:
        return
    path = Path(path)
    N = json.loads(path.read_text()) if path.exists() else dict(items=[])
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    labels = dict(RUNS[run]["kinds"])
    parts = [f"{labels[k]}: {v['text']}" for k, v in prog.items()]
    N["items"].append(dict(t=now, kind="news", title=f"{RUNS[run]['label'].split(':')[0]}: checkpoint {n / 1e6:g}M (cloud)",
                           body="; ".join(parts) + ". Watch it in 3D below."))
    N["items"].sort(key=lambda x: x["t"], reverse=True)
    N["updated"] = now
    path.write_text(json.dumps(N, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--news")
    ap.add_argument("--once", action="store_true", help="one round, then stop (for testing)")
    args = ap.parse_args()
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else dict(runs={})
    while True:
        changed = False
        for run, spec in RUNS.items():
            if spec["branch"] != "local":
                git("fetch", "-q", "origin", spec["branch"])
            R = data["runs"].setdefault(run, dict(checkpoints={}))
            R.update(label=spec["label"], where=spec["where"], kinds=spec["kinds"], measure=spec["measure"], words=spec["words"], goal="cloud")
            got = newest(run)
            if not got or str(got[0]) in R["checkpoints"]:
                continue
            n, folder = got
            ex = exams(run).get(n, {})
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                fetch(run, n, folder, tmp)
                try:
                    rec, prog = (record_jump(tmp, ex) if run == "jump16f" else record_jump(tmp, ex, (0.25, 0.30, 0.40, 0.50)) if run == "jump16g" else record_jump(tmp, ex, (0.30, 0.40, 0.50, 0.60)) if run == "jump16h" else record_jump(tmp, ex, (0.40, 0.50, 0.60, 0.70), 1.5, 220, 2.0) if run == "jump16i" else record_walk(tmp) if run == "bolt19"
                                  else record_run(tmp) if run in ("run20", "run21", "run22", "run22b") else record_imitate(tmp, run))
                except Exception as e:   # a half-pushed or unreadable checkpoint: try again next round
                    print(f"{run} {n}: could not record ({e})", flush=True)
                    continue
            R["checkpoints"][str(n)] = dict(steps=n, exam=ex, progress=prog, replays=rec)
            R["finished"] = ((REPO / f"train/runs/{run}/policy.zip").exists() if spec["branch"] == "local" else
                             bool(git("ls-tree", "--name-only", f"origin/{spec['branch']}", f"train/remote/{run}/final/").stdout.strip()))
            changed = True
            news(args.news, run, n, prog)
            print(f"{run} {n}: {prog}", flush=True)
        if changed:
            data["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            tmpf = out.with_suffix(".tmp")
            tmpf.write_text(json.dumps(data, separators=(",", ":")))
            tmpf.replace(out)
        if args.once:
            return
        if data["runs"] and all(r.get("finished") for r in data["runs"].values()):
            print("finished", flush=True)
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
