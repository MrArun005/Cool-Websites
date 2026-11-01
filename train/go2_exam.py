"""Exam a Go2 brain (a GPU trainer params.pkl) as every result in deploy/PLAN.md was measured.

    BOLT_BODY=go2 MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python go2_exam.py <params.pkl>            # full exam
    BOLT_BODY=go2 OMP_NUM_THREADS=1 python go2_exam.py <params.pkl> --jump                      # jump only (8 tries)
    BOLT_BODY=go2 OMP_NUM_THREADS=1 python go2_exam.py <params.pkl> --moving                    # sit / lie from a walk and a trot, 12 tries

Full: imitate_env.exam over the 9 motion clips (4 tries) + exam_steer; a brain with the jump slot also gets the jump exam,
reported apart (not in MEAN TRACKING, so scores stay comparable with brains from before the jump clip).
"""
import json
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imitate_env import MOTION_CLIPS, exam, exam_steer, score  # noqa: E402
from mjx_imitate import MjxImitatePolicy  # noqa: E402

p = MjxImitatePolicy(sys.argv[1])
if "--jump" in sys.argv:
    r = exam(p, clips=["jump"], episodes=8, latency_sweep=())
    print(json.dumps({k: v for k, v in r.items() if k.endswith("_jump")}))
elif "--moving" in sys.argv:
    r = exam(p, clips=["walk", "trot", "sit", "lie"], episodes=12, latency_sweep=())
    print(json.dumps({k: v for k, v in r.items() if "moving" in k or k.startswith("fall_")}))
else:
    r = exam(p, clips=MOTION_CLIPS)
    tr = [v for k, v in r.items() if k.startswith("track_")]
    fa = [v for k, v in r.items() if k.startswith("fall_")]
    st = exam_steer(p, clips=("trot", "canter"), rates=(-0.8, -0.4, 0.4, 0.8))
    r.update(st)
    if p.slots >= 10:
        j = exam(p, clips=["jump"], episodes=8, latency_sweep=())
        r.update({k: v for k, v in j.items() if k.endswith("_jump")})
    print(json.dumps(r))
    print(f"MEAN TRACKING {np.mean(tr):.3f}  FALL RATE {np.mean(fa):.3f}  SCORE {score(r):.3f}  STEER {st['steer_mean']:.3f}"
          + (f"  JUMP track {r['track_jump']:.3f} fall {r['fall_jump']:.3f}" if "track_jump" in r else ""))
