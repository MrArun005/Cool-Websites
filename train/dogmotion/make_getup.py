"""The dog getting up: the motion-capture lie-down clip (stand, sit, lie, hold) played backwards, so
lying -> sit (front legs push up first) -> stand, the way dogs rise. Writes clips/getup.npz.

    python dogmotion/make_getup.py      (from train/)
"""
from pathlib import Path

import numpy as np

CLIPS = Path(__file__).with_name("clips")
z = dict(np.load(CLIPS / "lie.npz"))
out = {}
for k, v in z.items():
    if k in ("base_pos", "base_quat", "joints", "foot_pos", "foot_contact"):
        out[k] = v[::-1].copy()                       # same poses, opposite order
    elif k in ("joint_vel", "base_linvel", "base_angvel"):
        out[k] = -v[::-1].copy()                      # time runs backwards: velocities flip sign
    else:
        out[k] = v
out["name"] = np.array("getup")
out["source"] = np.array(str(z.get("source", "")) + " | reversed lie clip: lie -> sit -> stand")
np.savez(CLIPS / "getup.npz", **out)
print("getup.npz:", len(out["joints"]), "frames,", len(out["joints"]) / float(out["fps"]), "s")
