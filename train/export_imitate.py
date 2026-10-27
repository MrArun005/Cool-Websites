"""Package a dog-imitation brain (imitate_env.py) for the browser, as a self-contained model folder.

    python export_imitate.py runs/imitate18b/checkpoints/policy_17000000_steps.zip --name bolt-dog-imitate-18b-17M \
                             --reference /tmp/imitate_reference.json

Writes ../models/<name>/:
    brain.json         the brain as the pages load brains (live.js Brain: normalizer + layers, base64 float32)
    clips.json         the reference clips the browser plays (gaitlab/imitate.js), base64 float32 per frame
    policy.onnx        the same brain for other runtimes (normalization built in)
    policy.zip + vecnormalize.pkl, robot.xml   to continue training / the body it was trained on
    model.json, exam.json, MODEL.md            facts, the exam (imitate_env.exam), the model card

clips.json, per clip: fps, loop, n frames, length (s), and per frame
    joints (12), height (1), v_body (3, body frame), w_body (3, body frame; the brain sees only the yaw rate)
        what the brain's inputs need (ImitateEnv._obs)
    joint_vel (12), tilt (9, body rotation without yaw, row-major)
        to start Bolt inside the clip, moving as the dog was (reference state initialization, ImitateEnv.reset)
    gravity (3), paws (12, paw centres relative to the body in its heading frame)
        only to score the tracking (ImitateEnv.tracking), as the tests do
--reference writes Python runs with fixed physics for gaitlab/imitate_test.mjs: start states and every step's
inputs, action and pose (observation parity), a clip switch, and 6 s per clip closed loop (fall, tracking).
"""
import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

import bolt_env
from bolt_env import ACTION_SCALE, FEET, MODEL
from export_live import brain, f32
from imitate_env import ALL_CLIPS, DOG, HOLD, NEXT, ImitateEnv, ImitatePolicy, exam
from package import JOINTS, export_onnx

OBS_LAYOUT = [("gravity_in_body_frame", 3), ("angular_velocity_x0.25", 3), ("linear_velocity_body", 3),
              ("body_height", 1), ("joint_pos_minus_dog_stance", 12), ("joint_vel_x0.1", 12), ("previous_action", 12),
              ("clip_phase_sin_cos", 2), ("clip_one_hot", None), ("ref_joints_next_minus_dog_stance", 12),
              ("ref_height_next", 1), ("ref_velocity_body_next", 3), ("ref_yaw_rate_next", 1),
              ("seconds_since_clip_end", 1)]
FIELDS = ("joints", "height", "v_body", "w_body", "joint_vel", "tilt", "gravity", "paws")
PARITY = (("walk", 0.3), ("trot", 0.1), ("canter", 0.2), ("pace", 0.0), ("turn_left", 0.4), ("sit", 0.0), ("lie", 0.0))
LOOP_TEST = ("walk", "trot", "canter", "sit", "lie")


def clips_json(names):
    out = {}
    for name in names:
        c = ImitateEnv(randomize=False, clips=[name]).clips[name]
        out[name] = dict(fps=c.fps, loop=c.loop, n=c.n, length=c.length,
                         **{k: f32(np.asarray(getattr(c, k)).reshape(c.n, -1)) for k in FIELDS})
    return out


def check_onnx(policy, path, clips, samples=200):
    import onnxruntime as ort
    session, env = ort.InferenceSession(str(path)), ImitateEnv(randomize=True, clips=clips, slots=policy.slots)
    obs, _ = env.reset(seed=5)
    worst = 0.0
    for _ in range(samples):
        ref = policy.act(obs)
        worst = max(worst, float(np.abs(session.run(None, {"obs": obs[None].astype(np.float32)})[0][0] - ref).max()))
        obs, _, fell, trunc, _ = env.step(ref)
        if fell or trunc:
            obs, _ = env.reset()
    return worst


def state(env):
    d = env.data
    return dict(qpos=d.qpos.tolist(), qvel=d.qvel.tolist(), ctrl=d.ctrl.tolist(), clip=env.clip.name, tau0=env.tau0,
                t=env.t, last_action=np.asarray(env.last_action, dtype=float).tolist())


def parity_runs(policy, clips, steps=100):
    """Fixed physics: per clip a start state and every step (inputs, action, pose), as the browser must repeat."""
    env = ImitateEnv(randomize=False, clips=clips, slots=policy.slots)
    runs = []
    for k, (name, at) in enumerate(PARITY):
        env.exam_start = (name, at * env.clips[name].length)
        obs, _ = env.reset(seed=100 + k)
        run = dict(start=state(env), steps=[])
        for _ in range(steps):
            a = policy.act(obs)
            run["steps"].append(dict(obs=obs.tolist(), action=np.asarray(a, dtype=float).tolist()))
            obs, _, fell, _, _ = env.step(a)
            run["steps"][-1]["qpos"] = env.data.qpos.tolist()
            if fell:
                break
        runs.append(run)
    return runs


class _Pick:
    """A stand-in random generator that makes ImitateEnv._maybe_switch switch now, to a chosen clip."""

    def __init__(self, name):
        self.name = name

    def random(self):
        return 0.0

    def choice(self, names):
        assert self.name in names
        return self.name


def switch_run(policy, clips, plan=(("walk", 60), ("trot", 60), ("sit", 120))):
    """Walk, then (by ImitateEnv's own switching code) trot joined at the closest pose, then sit from its start."""
    env = ImitateEnv(randomize=False, clips=clips, slots=policy.slots)
    env.exam_start = (plan[0][0], 0.0)
    obs, _ = env.reset(seed=7)
    env.exam_start = None
    run = dict(start=state(env), plan=[list(p) for p in plan], steps=[], switches=[])
    for i, (name, n) in enumerate(plan):
        if i:
            env.rng, env.switch_rate = _Pick(name), 1e9
            env._maybe_switch()
            env.switch_rate = 0.0
            obs = env._obs()
            run["switches"].append(dict(at=len(run["steps"]), clip=name, tau0=env.tau0, limit=env.limit))
        for _ in range(n):
            a = policy.act(obs)
            run["steps"].append(dict(obs=obs.tolist(), action=np.asarray(a, dtype=float).tolist()))
            obs, _, fell, _, _ = env.step(a)
            run["steps"][-1]["qpos"] = env.data.qpos.tolist()
    return run


def fixed_exam(policy, clips, names=LOOP_TEST, seconds=6.0, starts=4, seed=9000):
    """What the browser test does, in Python: fixed physics, `seconds` per try (sit/lie keep holding their last
    pose), loops from evenly spaced points of the stride, sit/lie from their start. Per clip: falls and the mean
    tracking reward per step (0 for every step lost after a fall, as in exam())."""
    env = ImitateEnv(randomize=False, clips=clips, slots=policy.slots)
    n = int(round(seconds / env.dt))
    out = {}
    for name in names:
        c = env.clips[name]
        scores, falls = [], 0
        for k in range(starts):
            env.exam_start = (name, (k / starts) * c.length if c.loop else 0.0)
            obs, _ = env.reset(seed=seed + k)
            env.limit = n
            total, fell = 0.0, False
            for _ in range(n):
                obs, r, fell, done, _ = env.step(policy.act(obs))
                total += r
                if fell or done:
                    break
            falls += fell
            scores.append(total / n)
        out[f"track_{name}"] = round(float(np.mean(scores)), 3)
        out[f"fall_{name}"] = falls / starts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", help="default: the newest checkpoint of runs/imitate18b")
    ap.add_argument("--name", help="default: bolt-dog-imitate-<run>-<N>M")
    ap.add_argument("--episodes", type=int, default=4, help="exam tries per clip (training's exam: 4)")
    ap.add_argument("--reference", help="where to write the Python reference runs for gaitlab/imitate_test.mjs")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "models"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.checkpoint:
        ck = Path(args.checkpoint)
    else:
        cks = Path(__file__).with_name("runs").joinpath("imitate18b", "checkpoints").glob("policy_[0-9]*_steps.zip")
        ck = max(cks, key=lambda p: int(re.search(r"(\d+)_steps", p.name).group(1)))
    steps = int(re.search(r"(\d+)_steps", ck.name).group(1))
    run = ck.parent.parent.name
    name = args.name or f"bolt-dog-{run.replace('imitate', 'imitate-')}-{steps / 1e6:g}M"
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ck, out / "policy.zip")
    shutil.copy(ck.with_name(ck.name.replace("policy_", "policy_vecnormalize_", 1).replace(".zip", ".pkl")), out / "vecnormalize.pkl")
    shutil.copy(MODEL, out / "robot.xml")
    policy = ImitatePolicy(out)
    clips = ALL_CLIPS[:policy.slots]   # a brain from before the getup clip (8 slots) knows the first 8
    (out / "brain.json").write_text(json.dumps(dict(brain(out), source=name), separators=(",", ":")))
    env = ImitateEnv(randomize=False, clips=clips, slots=policy.slots)
    constants = dict(dt=env.dt, frame_skip=env.frame_skip, dog_pose=DOG.tolist(), action_scale=ACTION_SCALE.tolist(),
                     slots=policy.slots, all_clips=ALL_CLIPS, hold=HOLD, next={k: list(v) for k, v in NEXT.items()},
                     feet=FEET, motor=getattr(bolt_env, "MOTOR", None))
    (out / "clips.json").write_text(json.dumps(dict(constants=constants, clips=clips_json(clips)), separators=(",", ":")))
    export_onnx(policy, out / "policy.onnx")
    err = check_onnx(policy, out / "policy.onnx", clips)
    result = exam(policy, clips=clips, episodes=args.episodes)
    fixed = fixed_exam(policy, clips)
    (out / "exam.json").write_text(json.dumps(dict(exam=result, fixed_physics_6s=fixed), indent=2))
    layout = [(n, s if s else policy.slots) for n, s in OBS_LAYOUT]
    facts = dict(name=name, source=str(ck), steps=steps, packaged=dt.date.today().isoformat(), control_hz=50,
                 obs_size=sum(s for _, s in layout), action_size=12, clips=clips,
                 obs_layout=[dict(name=n, size=s) for n, s in layout], joints=JOINTS, dog_stance=DOG.tolist(),
                 action_scale=ACTION_SCALE.tolist(),
                 control="joint targets = clip(reference joints at the next step + action_scale * action, joint limits)",
                 onnx_max_difference=err, exam=result, fixed_physics_6s=fixed)
    (out / "model.json").write_text(json.dumps(facts, indent=2))
    def row(c):
        fx = (f"{fixed[f'track_{c}']:.3f} | {100 * fixed[f'fall_{c}']:.0f}%") if f"track_{c}" in fixed else "- | -"
        return f"| {c} | {result[f'track_{c}']:.3f} | {100 * result[f'fall_{c}']:.0f}% | {result.get(f'speed_{c}', '-')} | {fx} |"
    rows = "\n".join(row(c) for c in clips)
    (out / "MODEL.md").write_text(f"""# {name}

Bolt's dog-imitation brain: moves like a real dog by tracking retargeted motion-capture clips
(`train/imitate_env.py`, run {run} at {steps:,} steps). It knows {", ".join(clips)}.

## How to run it (gaitlab/imitate.js does this)

Every 1/50 s: advance the clip clock, build the {facts["obs_size"]} inputs below, run the brain (`brain.json` in
the pages, or `policy.onnx`), and set the joint targets to `clip(q_ref(next step) + action_scale * action)`: the
clip's next pose is the plan, the brain adds corrections. Switching clip: a gait joins at the frame whose pose is
closest to Bolt's; sit and lie play from their start and then hold their last pose (the brain practised 1 s).

Inputs: {", ".join(f"{n} ({s})" for n, s in layout)}.

## Exam

Training's exam (`imitate_env.exam`, {args.episodes} tries per clip, randomized physics, loops 4 s, sit/lie through their
hold): mean tracking reward per step (0..1), falls, and the share of the dog's pace kept. Fixed physics 6 s: the
browser test's protocol in Python (gaitlab/imitate_test.mjs runs the same in the browser physics).

| Clip | Tracking | Falls | Pace | Tracking, fixed 6 s | Falls, fixed 6 s |
|---|---|---|---|---|---|
{rows}

ONNX matches the PyTorch policy to {err:.1e}.
""")
    if args.reference:
        ref = dict(model=name, parity=parity_runs(policy, clips), switch=switch_run(policy, clips), fixed_physics_6s=fixed)
        Path(args.reference).write_text(json.dumps(ref, separators=(",", ":")))
    print(f"wrote {out}: exam {result}\nfixed physics 6 s {fixed}\nonnx error {err:.1e}")


if __name__ == "__main__":
    main()
