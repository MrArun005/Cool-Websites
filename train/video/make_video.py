"""Film Bolt (simulation) for the LinkedIn post: one continuous take of the whole story, rendered at up to 4K.

Story (MasterMind's world, mastermind/env.py Master, driven like test_skills.py's sequences):
    walk 3 s -> trot 3 s -> canter 3 s -> trot 2.5 s -> jump an agility bar (--jump-height, default 50 cm) -> trot 2 s -> walk 2 s
    -> sit 4 s -> lie down 5 s -> stand 3 s -> walk 2 s -> shoved over -> gets up by itself -> walk 3 s

Takes are tried on seeds --seed, --seed+1, ... until one is clean (no fall except after the shove, the bar cleared
without dropping it, back on its feet after the shove); the file name and video.json say which take of how many
tries it was, so the post can say so honestly ("take 3 of 5").

    cd train
    MUJOCO_GL=osmesa python video/make_video.py --brain teacher --teachers v5 --size 640x360 --out runs/video/preview.mp4
    MUJOCO_GL=osmesa python video/make_video.py --brain teacher --teachers v5 --size 3840x2160 --out runs/video/bolt_4k.mp4

--brain: "teacher" (the expert brains, MasterMind's teachers) or a MasterMind student.pt (one network).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "mastermind"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from drive import Driver  # noqa: E402
from env import Master  # noqa: E402
from student import Student  # noqa: E402

STORY = [("walk", 3.0), ("trot", 3.0), ("canter", 3.0), ("trot", 2.5), ("jump", None), ("trot", 2.0), ("walk", 2.0),
         ("sit", 4.0), ("lie", 5.0), ("stand", 3.0), ("walk", 2.0), ("shove", None), ("walk", 3.0)]


def take(m, brain, seed, jump_ahead, jump_h):
    """Play the story once, recording the world after every control step. Returns (frames, notes, clean)."""
    d = Driver(m, brain)
    frames = []
    step0 = m.step

    def step(ctrl):
        r = step0(ctrl)
        dd = m.data
        frames.append((dd.qpos.copy(), dd.mocap_pos.copy(), dd.mocap_quat.copy()))
        return r
    m.step = step
    notes, clean = [], True
    try:
        d.start("walk", seed=seed)
        for skill, arg in STORY:
            if skill == "jump":
                r = d.do("jump", height=jump_h, ahead=jump_ahead)
                ok = bool(r.get("clean")) and not r.get("fell") and not r.get("fell_after")
                notes.append(dict(skill=f"jump {round(jump_h * 100)} cm", ok=ok, **{k: r.get(k) for k in ("cleared", "dropped", "landed", "fell")}))
            elif skill == "shove":
                n_ev = len(d.events)
                d.shove()
                d.do("walk", 2.0)
                rec = [e for e in d.events[n_ev:] if e["kind"] in ("recovered", "not_recovered")]
                ok = bool(rec) and rec[0]["kind"] == "recovered"
                notes.append(dict(skill="shove -> get up", ok=ok))
            else:
                ok = d.do(skill, arg)
                notes.append(dict(skill=skill, ok=bool(ok)))
            clean = clean and notes[-1]["ok"]
            if not clean:
                break
    finally:
        m.step = step0
    return frames, notes, clean


def render(m, frames, size, out, fps, every):
    w, h = size
    model = m.data and m.w.model
    model.vis.global_.offwidth, model.vis.global_.offheight = max(w, model.vis.global_.offwidth), max(h, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 2.6, -12.0, 120.0
    look = None
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for i, (qpos, mpos, mquat) in enumerate(frames[::every]):
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = qpos, mpos, mquat
        mujoco.mj_forward(model, data)
        target = data.qpos[:3].copy()
        target[2] = 0.35
        look = target if look is None else look + 0.08 * (target - look)   # a smooth follow, not a locked camera
        cam.lookat[:] = look
        cam.azimuth = 120.0 + 25.0 * np.sin(i / (fps * 9.0))               # a slow drift around Bolt
        r.update_scene(data, camera=cam)
        ff.stdin.write(r.render().tobytes())
    ff.stdin.close()
    ff.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="teacher")
    ap.add_argument("--teachers", default="v5")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--jump-ahead", type=float, default=1.75)
    ap.add_argument("--jump-height", type=float, default=0.50)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=25, help="video frame rate (control runs at 50 Hz; 25 = every 2nd step)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    t0 = time.time()
    tries = []
    for k in range(a.tries):
        seed = a.seed + k
        m = Master(seed=seed, teachers=a.teachers)
        brain = None if a.brain == "teacher" else Student(a.brain, m)
        frames, notes, clean = take(m, brain, seed, a.jump_ahead, a.jump_height)
        tries.append(dict(seed=seed, clean=clean, notes=notes))
        print(f"take {k + 1} (seed {seed}): {'clean' if clean else 'not clean'} {notes[-1]}", flush=True)
        if clean:
            break
    else:
        print("no clean take", flush=True)
        sys.exit(1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    every = max(1, round(1 / (m.dt * a.fps)))
    render(m, frames, (w, h), out, a.fps, every)
    meta = dict(brain=a.brain, teachers=a.teachers, take=len(tries), of_tries=len(tries), seed=tries[-1]["seed"],
                tries=tries, seconds=round(len(frames) * m.dt, 1), size=a.size, fps=a.fps, simulation=True,
                minutes=round((time.time() - t0) / 60, 1), command="python " + " ".join(sys.argv))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{out}: take {len(tries)} of {len(tries)} tries, {meta['seconds']} s of simulation, {meta['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
