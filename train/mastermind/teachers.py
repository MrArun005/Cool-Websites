"""Where the expert brains (the teachers) live, and the teacher sets the MasterMind env can run.

v2 (REPORT.md, the first student): dog-motion run 18e at 20M for every clip, jump brain 16g handed over 1.1 m before
the bar, 0.6 s canter bridge after landing.
v3: dog-motion run 18e at 36M (models/bolt-dog-switching-18e-36M) for walk/trot/pace/turns/sit/lie, the running brain
18c at 24.5M for the canter clip (fair canter 65%, 1/30 falls), and the run-up jump brain 16i at 35.5M handed over
1.5 m before the bar (it trained from 0.5-1.5 m, arriving at up to 3 m/s, 200-step tries).
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
MODELS = TRAIN.parent / "models"
# run 18e at 20M (the v2 teacher) arrives on branch claude/train-walk19:
#   mkdir -p $D && for f in policy_20000000_steps.zip policy_vecnormalize_20000000_steps.pkl; do
#     git show origin/claude/train-walk19:train/remote/imitate18e/$f > $D/$f; done
IMITATE18E = Path(os.environ.get("IMITATE18E", "/tmp/claude-0/-home-user-Cool-Websites/029cd38f-3b1a-5876-9645-932075813c4a/scratchpad/imitate18e"))


def imitate_brain(which="18e"):
    """(folder, zip, stats) of a dog-motion brain: '18e' (20M), '18e36' (36M), '18c' (running, 24.5M), 'r4'
    (research student r4) or '18b' (newest checkpoint, as skills.py)."""
    if which == "18e":
        return IMITATE18E, "policy_20000000_steps.zip", "policy_vecnormalize_20000000_steps.pkl"
    if which == "18e36":
        return MODELS / "bolt-dog-switching-18e-36M", "policy.zip", "vecnormalize.pkl"
    if which == "18c":
        return MODELS / "bolt-dog-running-18c-24.5M", "policy.zip", "vecnormalize.pkl"
    if which == "r4":
        return MODELS / "bolt-dog-imitate-pps-student-r4", "policy.zip", "vecnormalize.pkl"
    if which == "18b":
        return TRAIN / "runs" / "imitate18b" / "checkpoints", "policy_15000000_steps.zip", "policy_vecnormalize_15000000_steps.pkl"
    raise ValueError(which)


# dog: the dog-motion brain for every clip; canter: a brain for the canter clip only (None: dog); land: a brain for
# the clips played in the first `land_s` seconds after a jump's landing (None: as usual); jump: jump brain folder,
# handed over `jump_from` m before the bar, `jump_steps` its try length (its clock input is t / jump_steps), at most
# `jump_limit` steps in charge; bridge: seconds of canter after landing before the gait comes back.
SETS = {
    "v2": dict(dog="18e", canter=None, land=None, land_s=0.0, jump=MODELS / "bolt-dog-jump-a1-16g-27M", jump_from=1.1,
               jump_steps=100, jump_limit=150, bridge=0.6),
    "v3": dict(dog="18e36", canter="18c", land=None, land_s=0.0, jump=MODELS / "bolt-dog-jump-runup-16i-35.5M",
               jump_from=1.5, jump_steps=200, jump_limit=200, bridge=0.6),
}


def teacher_set(name="v2", **over):
    s = dict(SETS[name])
    s.update({k: v for k, v in over.items() if v is not None})
    return s
