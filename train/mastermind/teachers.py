"""Where the expert brains (the teachers) live, and the teacher sets the MasterMind env can run.

v2 (REPORT.md, the first student): dog-motion run 18e at 20M for every clip, jump brain 16g handed over 1.1 m before
the bar, 0.6 s canter bridge after landing.
v3: dog-motion run 18e at 36M (models/bolt-dog-switching-18e-36M) for walk/trot/pace/turns/sit/lie, the running brain
18c at 29.5M for the canter clip (fair canter 68%, 1/30 falls), and the run-up jump brain 16i at 35.5M handed over
1.1 m before the bar (it trained from 0.5-1.5 m, arriving at up to 3 m/s, 200-step tries; see SETS), and research
student r4 for the 2.6 s after a landing.
v4: run 18g at 35.5M (gait switches practised) for walk/trot/canter/turns, the rest as v3 (REPORT.md, v4 section).
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
    """(folder, zip, stats) of a dog-motion brain: '18e' (20M), '18e36' (36M), '18c' (running, 29.5M), '18c24' (24.5M), 'r4'
    (research student r4) or '18b' (newest checkpoint, as skills.py)."""
    if which == "18e":
        return IMITATE18E, "policy_20000000_steps.zip", "policy_vecnormalize_20000000_steps.pkl"
    if which == "18e36":
        return MODELS / "bolt-dog-switching-18e-36M", "policy.zip", "vecnormalize.pkl"
    if which == "18c":
        return MODELS / "bolt-dog-running-18c-29.5M", "policy.zip", "vecnormalize.pkl"
    if which == "18c24":
        return MODELS / "bolt-dog-running-18c-24.5M", "policy.zip", "vecnormalize.pkl"
    if which == "18f31":
        return MODELS / "bolt-dog-running-18f-31M", "policy.zip", "vecnormalize.pkl"
    if which == "18g35":   # run 18g at 35.5M: walk/trot/canter/turns trained with mid-try gait switches (8-slot)
        return MODELS / "bolt-dog-canter-switching-18g-35.5M", "policy.zip", "vecnormalize.pkl"
    if which == "18d":   # run 18d's best (24.5M): all 9 clips, no switch training; sit 95%, lie 94%, getup 92%
        return MODELS / "bolt-dog-getup-18d-24.5M", "policy.zip", "vecnormalize.pkl"
    if which == "18g":   # run 18g's newest best (canter switching), fetched into $IMITATE18G
        return Path(os.environ.get("IMITATE18G", MODELS / "bolt-dog-canter-switching-18g")), "policy.zip", "vecnormalize.pkl"
    if which == "r4":
        return MODELS / "bolt-dog-imitate-pps-student-r4", "policy.zip", "vecnormalize.pkl"
    if which == "18b":
        return TRAIN / "runs" / "imitate18b" / "checkpoints", "policy_15000000_steps.zip", "policy_vecnormalize_15000000_steps.pkl"
    raise ValueError(which)


GPU360 = MODELS / "bolt-dog-imitate-gpu-360M" / "params.pkl"   # Kaggle GPU brain (MJX PPO, 360M steps, 9 clips)
GPU3 = MODELS / "bolt-dog-imitate-gpu3-617M" / "params.pkl"     # run gpu3: delay-proof, sits/lies from a trot (29 Oct)


def imitate_policy(which):
    """A loaded dog-motion brain (act(obs) on the ImitateEnv observation): any imitate_brain() name, 'gpu360' or 'gpu3'."""
    if which in ("gpu360", "gpu3"):
        from mjx_imitate import MjxImitatePolicy
        return MjxImitatePolicy(GPU360 if which == "gpu360" else GPU3)
    from imitate_env import ImitatePolicy
    return ImitatePolicy(*imitate_brain(which))


# dog: the dog-motion brain for every clip; canter: a brain for the canter clip only (None: dog); land: a brain for
# the clips played in the first `land_s` seconds after a jump's landing (None: as usual); jump: jump brain folder,
# handed over `jump_from` m before the bar, `jump_steps` its try length (its clock input is t / jump_steps), at most
# `jump_limit` steps in charge; bridge: seconds of canter after landing before the gait comes back; clips: {clip: brain}
# for single clips (wins over canter and dog; the landing brain still wins during its land_s).
SETS = {
    "v2": dict(dog="18e", canter=None, land=None, land_s=0.0, jump=MODELS / "bolt-dog-jump-a1-16g-27M", jump_from=1.1,
               jump_steps=100, jump_limit=150, bridge=0.6),
    # v3: 16i handed over 1.1 m before the bar, not at its training/exam distance of 1.5 m. It needs ~2 m/s at 1.5 m
    # (fair exam 40 cm 100% at 2.0 m/s, 35% at 1.6 m/s, 0% from standing) and a trot arrives at ~1.5 m/s; handed over
    # from a trot (teacher chain, N=30, seed 1000, clean 20/30/40/50 cm): 1.5 m 57/27/20/7%, 1.3 m 63/53/20/3%,
    # 1.1 m 80/70/53/20%, 1.0 m -/63/40/7%; a canter run-up (switch to canter for the approach) fell often (<25%);
    # gating the hand-over on the trot's phase gave no clear gain. land: research student r4 (I'm-possible) for
    # 2.6 s after landing (bridge + first 2 s of trot): landing -> trot 74/118 -> 94/118 on the same tries.
    # rec_cmd: the recovery phase is shown to the student on its command input (env.py). The first v3 DAgger run without
    # it lost the recovery in alternate rounds (stood up 0-85% of 12-20 shoves): lying belly-down, "keep tucking" and
    # "stand up" labels fall on the same states, and the student stayed down.
    "v3": dict(dog="18e36", canter="18c", land="r4", land_s=2.6, jump=MODELS / "bolt-dog-jump-runup-16i-35.5M",
               jump_from=1.1, jump_steps=200, jump_limit=200, bridge=0.6, rec_cmd=True),
}
# v4: run 18g at 35.5M (gait switches practised: canter->trot 93% in the v3 world) for walk/trot/canter/turns; the
# rest of v3 unchanged (pace/sit/lie: `dog`). v4d/v4g: other choices for sit/lie/pace, compared on dev seeds.
_G4 = {c: "18g35" for c in ("walk", "trot", "canter", "turn_left", "turn_right")}
SETS["v4"] = dict(SETS["v3"], canter=None, clips=_G4)
SETS["v4a"] = dict(SETS["v4"], approach="18e36")
SETS["v4d"] = dict(SETS["v4"], dog="18d")
SETS["v4g"] = dict(SETS["v3"], dog="18g35", canter=None)
# the GPU brain (arrived during the v4 DAgger run), compared in the teacher chain on dev seeds only
SETS["v4gpu_c"] = dict(SETS["v4"], clips=dict(_G4, canter="gpu360"))
SETS["v4gpu_s"] = dict(SETS["v4"], clips=dict(_G4, sit="gpu360", lie="gpu360"))
SETS["v4b"] = SETS["v4gpu_all"] = dict(SETS["v4"], clips={c: "gpu360" for c in ("walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie")})
# v5 (28 Oct): v4 with the jump brain trained from a trot (models/bolt-dog-jump-trot-55M, Kaggle jump-trot, re-picked
# on the mid-stride hand-over: fair mid-stride 1.5 m/s 40/50 cm 92/65% vs 16i 17/5%). Same hand-over and clock.
SETS["v5"] = dict(SETS["v4"], jump=MODELS / "bolt-dog-jump-trot-55M")
# v6 candidates (29 Oct): gpu3 (sit/lie from a walk or trot 100%, trained on the canter->trot change) where the v5
# teacher chain fails most (walk->sit / sit->lie 80%, canter->trot 87%); compared on dev seeds 20000.
SETS["v6s"] = dict(SETS["v5"], clips=dict(SETS["v5"]["clips"], sit="gpu3", lie="gpu3"))
SETS["v6c"] = dict(SETS["v5"], clips=dict(SETS["v5"]["clips"], canter="gpu3"))
SETS["v6"] = dict(SETS["v5"], clips=dict(SETS["v5"]["clips"], sit="gpu3", lie="gpu3", canter="gpu3"))
# land_hold: s the jump brain keeps control after landing (it slows Bolt down, as Bolt's Day Out's "land"), then the
# gait; instead of the canter bridge. Default 0 (the bridge).
SETS["v6h"] = dict(SETS["v6"], land_hold=0.6)
SETS["v6h1"] = dict(SETS["v6"], land_hold=1.0)
# v7p (29 Oct): v6h with the jump brain from the local hand-over pilot (runs/pilot_ho, 59.2M: held-out real hand-overs
# 84/90/87/73% at 20-50 cm vs 55M 81/77/61/31%); a dev-seed check before the Kaggle jump-handover run lands.
SETS["v7p"] = dict(SETS["v6h"], jump=MODELS / "bolt-dog-jump-handover-59M")
SETS["v7"] = SETS["v7p"]
# v7s: v7 with the canter back on 18g (v7's student lost canter look: tracking 0.695 -> 0.548 with gpu3 as teacher)
SETS["v7s"] = dict(SETS["v6s"], land_hold=0.6, jump=MODELS / "bolt-dog-jump-handover-59M")
# v8g (29 Oct): delay-proof teachers for a latency-trained MasterMind: gpu3 (trained with 0-40 ms delay) for every
# dog-motion clip; the rest as v7s. Checked in the teacher chain at 20 ms before any DAgger run.
SETS["v8g"] = dict(SETS["v7s"], clips={c: "gpu3" for c in ("walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie")})
# v9j (29 Oct): v8g with the jump brain trained on real hand-overs for 12 h on Kaggle (jump-handover 74M: held-out
# hand-overs 87/86/91/90% at 20-50 cm vs 84/90/87/73% for 59M)
SETS["v9j"] = dict(SETS["v8g"], jump=MODELS / "bolt-dog-jump-handover-74M")


def teacher_set(name="v2", **over):
    s = dict(SETS[name])
    s.update({k: v for k, v in over.items() if v is not None})
    return s
