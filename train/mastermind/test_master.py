"""The skill-switch tests of test_skills.py (same sequences, seeds, windows) on the one MasterMind brain.

    cd train && MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python mastermind/test_master.py --brain mastermind/out/student.pt --n 10
    ... --brain teacher      the env's teacher chains (the skill manager inside the unified env, a sanity check)
Also --exam: imitate_env.exam per clip (the student driven through the Master env on a clip command, scored by
the ImitateEnv tracking reward) and jump clean rates from a trot (the trot->jump rows of the switch tests).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from baseline import run, show  # noqa: E402
from drive import Driver  # noqa: E402
from env import Master  # noqa: E402
from student import Student  # noqa: E402


def clip_exam(m, brain, episodes=4, seed=9000, clips=("walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie")):
    """imitate_env.exam's measure, through the Master env: 4 s per loop from evenly spaced points of the stride
    (sit/lie: from the start through the 1 s hold); per step the ImitateEnv tracking reward; 0 for every step lost
    after a fall; speed_ = forward distance (turns: angle) over the dog's."""
    from imitate_env import HOLD
    out = {}
    for name in clips:
        c = m.im.clips[name]
        scores, falls, pace = [], 0, []
        for k in range(episodes):
            m.start(name, tau=(k / episodes) * c.length if c.loop else 0.0, seed=seed + k)
            n = 200 if c.loop else int((c.length + HOLD) / m.dt)
            total, fell, done_by, yaw = 0.0, False, 0.0, 0.0
            for _ in range(n):
                R = m.w._rotation()
                p0, h0 = m.data.qpos[:2].copy(), np.arctan2(R[1, 0], R[0, 0])
                ctrl = brain.act(m.obs()) if brain is not None else None
                r = m.step(ctrl)
                R = m.w._rotation()
                done_by += float(np.dot(m.data.qpos[:2] - p0, [np.cos(h0), np.sin(h0)]))
                yaw += float(np.angle(np.exp(1j * (np.arctan2(R[1, 0], R[0, 0]) - h0))))
                terms, err = m.im.tracking()
                total += sum(m.im.weights[k2] * v for k2, v in terms.items())
                lost = abs(err["height_err"]) > 0.15 or err["tilt"] > 0.8
                if r["fell"] or lost:
                    fell = True
                    break
            falls += fell
            scores.append(total / n)
            if c.loop:
                turn = name.startswith("turn")
                want = n * m.dt * (c.w_body[:, 2].mean() if turn else c.v_body[:, 0].mean())
                pace.append((yaw if turn else done_by) / want)
        out[f"track_{name}"] = round(float(np.mean(scores)), 3)
        out[f"fall_{name}"] = falls / episodes
        if c.loop:
            out[f"speed_{name}"] = round(float(np.mean(pace)), 2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="teacher")
    ap.add_argument("--imitate", help="override the teacher set's dog-motion brain")
    ap.add_argument("--teachers", default="v2", help="teacher set (teachers.SETS): v2 for the v2 student, v3 for v3")
    ap.add_argument("--with-50", action="store_true", help="also the 50 cm jump sequence")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--exam", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    m = Master(seed=a.seed, imitate=a.imitate, teachers=a.teachers)
    brain = None if a.brain == "teacher" else Student(a.brain, m)
    t0 = time.time()
    res = dict(brain=a.brain, teachers=a.teachers, imitate=a.imitate, n=a.n, seed=a.seed)
    if a.exam:
        res["exam"] = clip_exam(m, brain)
        print(res["exam"], flush=True)
    rows, secs = run(Driver(m, brain), a.n, a.seed, a.only, a.with_50)
    show(rows)
    res.update(seconds=round(time.time() - t0), switches=rows)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
