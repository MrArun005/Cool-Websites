"""Where the expert brains (the teachers) live."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
# run 18e (switch-trained dog-motion brain, 75 inputs incl. getup) arrives on branch claude/train-walk19:
#   mkdir -p $D && for f in policy_20000000_steps.zip policy_vecnormalize_20000000_steps.pkl; do
#     git show origin/claude/train-walk19:train/remote/imitate18e/$f > $D/$f; done
IMITATE18E = Path(os.environ.get("IMITATE18E", "/tmp/claude-0/-home-user-Cool-Websites/029cd38f-3b1a-5876-9645-932075813c4a/scratchpad/imitate18e"))


def imitate_brain(which="18e"):
    """(folder, zip, stats) of a dog-motion brain: '18e' (20M) or '18b' (newest checkpoint, as skills.py)."""
    if which == "18e":
        return IMITATE18E, "policy_20000000_steps.zip", "policy_vecnormalize_20000000_steps.pkl"
    if which == "18b":
        return TRAIN / "runs" / "imitate18b" / "checkpoints", "policy_15000000_steps.zip", "policy_vecnormalize_15000000_steps.pkl"
    raise ValueError(which)
