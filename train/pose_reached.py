"""What "the pose was reached" means: ONE definition, for the switch test and for training.

The thresholds are the switch test's (test_skills.py POSE_OK and its "really up" check, the numbers in
skills_report.json; research/run_switches.py runs the same test): the pose each clip ends in, with room to spare.
    sit    body below 27 cm and nose up more than 0.3 rad   (the sit clip ends at 23 cm, nose up 0.6 rad)
    lie    body below 19 cm and upright (z-axis up > 0.9)   (the lie clip ends at 16-17 cm, level)
    stand  body above 38 cm and upright (z-axis up > 0.9)   (test_skills' "really up" after the posture stand)
height: body height (m, qpos[2]); pitch: nose-up pitch (rad, arcsin of the rotation matrix's R[2, 0], as
BoltEnv.posture()); up: the body z-axis' up component (R[2, 2]).

pose_ok works on Python floats, numpy and JAX arrays alike (only comparisons and &), so the CPU env (imitate_env),
the MJX env (mjx_imitate) and the switch test all call the same function. CLIP_POSE: the pose a posture clip ends in.
No imports: Kaggle kernels carry this file next to mjx_imitate.py.
"""
POSE_OK = {
    "sit": dict(height_max=0.27, pitch_min=0.3),
    "lie": dict(height_max=0.19, up_min=0.9),
    "stand": dict(height_min=0.38, up_min=0.9),
}
CLIP_POSE = {"sit": "sit", "lie": "lie", "getup": "stand"}


def pose_ok(name, height, pitch, up):
    """True where the body is in pose `name` (a POSE_OK key)."""
    c = POSE_OK[name]
    ok = True
    if "height_max" in c:
        ok = ok & (height < c["height_max"])
    if "height_min" in c:
        ok = ok & (height > c["height_min"])
    if "pitch_min" in c:
        ok = ok & (pitch > c["pitch_min"])
    if "up_min" in c:
        ok = ok & (up > c["up_min"])
    return ok


def posture_ok(name, p):
    """pose_ok on a BoltEnv.posture() dict (test_skills.py)."""
    return bool(pose_ok(name, p["height"], p["pitch"], p["up"]))
