"""Record the replays for the Bolt demo page: one short replay per clip / sequence of each final best brain.

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python gaitlab/record_demo.py --out <site>/data [--only gpu360 ...]

Writes <out>/<brain>.json ({replays: {key: flat qpos frames (19 per frame, 25 Hz), key_props: hurdle,
key_phase: skill per frame}, ok: {key: bool}}) and <out>/robot.json (the robot's parts plus the kinematic tree,
so the page poses Bolt from qpos with a few lines of forward kinematics instead of the 11 MB MuJoCo wasm).
Brains: gpu360 (Kaggle GPU, 9 clips), b18 (run 18b, 8 clips), jump16i (40/50 cm with a run-up),
switch18g (walk-trot-canter-trot-walk, teacher chain of run 18g), master (MasterMind student: gaits+rest, a jump).
A sequence is retried on new seeds until a try has no fall (the page states the measured success rates).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TRAIN = Path(__file__).resolve().parent.parent
REPO = TRAIN.parent
MODELS = REPO / "models"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(TRAIN / "mastermind"))

CLIPS9 = ["walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup"]


def flat(frames):
    return np.round(np.concatenate(frames), 4).tolist()


def imitate(policy, clips, seconds=4.0):
    from imitate_env import ImitateEnv
    env = ImitateEnv(randomize=False, clips=clips, slots=policy.slots)
    rec, ok = {}, {}
    for k in clips:
        env.exam_start = (k, 0.0)
        obs, _ = env.reset(seed=4321)
        if env.clip.loop:
            env.limit = int(seconds * 50)
        frames, fell = [], False
        for t in range(env.limit):
            if t % 2 == 0:
                frames.append(env.data.qpos[:19].copy())
            obs, r, fell, done, info = env.step(policy.act(obs))
            if fell or done:
                break
        rec[k], ok[k] = flat(frames), not fell
        print(k, len(frames), "fell" if fell else "ok", flush=True)
    return rec, ok


def jump16i():
    from cloudwatch import record_jump
    rec, prog = record_jump(MODELS / "bolt-dog-jump-runup-16i-35.5M", {}, (0.40, 0.50), 1.5, 220, 2.0)
    ok = {k: bool(v["stood"]) for k, v in prog.items()}
    print(ok, flush=True)
    return rec, ok


def sequence(teachers, brain_path, steps, tries=12, seed0=1000, randomize=True):
    """steps: [(skill, seconds) | ("jump", height)]. Returns (replay dict, ok) for the first try with no fall."""
    from drive import Driver
    from env import Master
    from student import Student
    m = Master(seed=seed0, teachers=teachers, randomize=randomize)
    brain = Student(brain_path, m) if brain_path else None
    d = Driver(m, brain)
    best = None
    for k in range(tries):
        seed = seed0 + 37 * k
        first = steps[0][0]
        d.start(first, seed=seed)
        good = True
        for skill, arg in steps:
            if skill == "jump":
                r = d.do("jump", height=arg, ahead=1.75)
                good = good and bool(r.get("clean")) and not r.get("fell") and not r.get("fell_after")
            else:
                good = d.do(skill, arg) and good
        good = good and not any(e["kind"] == "fall" for e in m.b.events)
        print(teachers, "seed", seed, "ok" if good else "failed", len(m.b.frames), flush=True)
        if good:
            return m.b.replay("x"), True
        if best is None:
            best = m.b.replay("x")
    return best, False


def rename(rep, key):
    return {key + k[1:]: v for k, v in rep.items()}


GAITS = [("walk", 3.0), ("trot", 3.0), ("canter", 3.0), ("trot", 3.0), ("walk", 3.0)]


def record(name):
    if name == "gpu360":
        from mjx_imitate import MjxImitatePolicy
        return imitate(MjxImitatePolicy(MODELS / "bolt-dog-imitate-gpu-360M/params.pkl"), CLIPS9)
    if name == "b18":
        from imitate_env import ImitatePolicy
        return imitate(ImitatePolicy(MODELS / "bolt-dog-imitate-18b-30M"), CLIPS9[:8])
    if name == "jump16i":
        return jump16i()
    if name == "switch18g":   # run 18g's brain drives walk/trot/canter (teacher set v4: clips -> 18g at 35.5M)
        rep, ok = sequence("v4", None, GAITS)
        return rename(rep, "gaits"), {"gaits": ok}
    if name == "master":
        st = TRAIN / "mastermind/export_v3/student.pt"
        rec, ok = {}, {}
        rep, ok["day"] = sequence("v3", st, GAITS + [("sit", 4.0), ("lie", 5.5), ("stand", 3.0)])
        rec.update(rename(rep, "day"))
        rep, ok["jump"] = sequence("v3", st, [("trot", 2.0), ("jump", 0.40), ("trot", 1.0)])
        rec.update(rename(rep, "jump"))
        return rec, ok
    raise ValueError(name)


def robot(out, src_path):
    """robot.json: the parts (as the Gait Lab draws them) and the kinematic tree of the robot's bodies, from a
    Gait Lab robot.json ({xml, geoms})."""
    import mujoco
    src = json.loads(Path(src_path).read_text())
    m = mujoco.MjModel.from_xml_string(src["xml"])
    used = {g["id"] for g in src["geoms"]}
    bodies = sorted({int(m.geom_bodyid[g]) for g in used})
    need = set()
    for b in bodies:
        while b > 0:
            need.add(b)
            b = int(m.body_parentid[b])
    kin = []
    for b in sorted(need):
        j = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
        kin.append(dict(id=b, parent=int(m.body_parentid[b]), pos=np.round(m.body_pos[b], 6).tolist(),
                        quat=np.round(m.body_quat[b], 6).tolist(), jtype=int(m.jnt_type[j]) if j >= 0 else -1,
                        qadr=int(m.jnt_qposadr[j]) if j >= 0 else -1, axis=np.round(m.jnt_axis[j], 6).tolist() if j >= 0 else None,
                        jpos=np.round(m.jnt_pos[j], 6).tolist() if j >= 0 else None))
    geoms = [dict(g, body=int(m.geom_bodyid[g["id"]]), gpos=np.round(m.geom_pos[g["id"]], 6).tolist(),
                  gquat=np.round(m.geom_quat[g["id"]], 6).tolist()) for g in src["geoms"]]
    Path(out, "robot.json").write_text(json.dumps(dict(kin=kin, geoms=geoms), separators=(",", ":")))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=["gpu360", "b18", "jump16i", "switch18g", "master"])
    ap.add_argument("--robot", help="a Gait Lab robot.json ({xml, geoms}): write <out>/robot.json with the kinematic tree")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.robot:
        robot(a.out, a.robot)
    for name in a.only:
        rec, ok = record(name)
        Path(a.out, f"{name}.json").write_text(json.dumps(dict(replays=rec, ok=ok), separators=(",", ":")))
        print(name, "written", flush=True)
