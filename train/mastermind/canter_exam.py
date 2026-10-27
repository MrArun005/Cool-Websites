"""Canter copy exam (research/exam_canter.py's protocol: randomized physics, 4 s from evenly spaced stride points,
seeds 5000+k, N=30) for a student (or the teachers) through the Master env.

    cd train && python mastermind/canter_exam.py --brain <student.pt|teacher> --teachers v3 --out mastermind/results/canter_v3.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from env import Master  # noqa: E402
from student import Student  # noqa: E402
from test_master import clip_exam  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="teacher")
    ap.add_argument("--teachers", default="v3")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out")
    a = ap.parse_args()
    m = Master(seed=0, teachers=a.teachers)
    brain = None if a.brain == "teacher" else Student(a.brain, m)
    r = clip_exam(m, brain, episodes=a.n, seed=5000, clips=("canter", "trot", "walk"))
    r.update(brain=a.brain, teachers=a.teachers, n=a.n)
    print(r)
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
