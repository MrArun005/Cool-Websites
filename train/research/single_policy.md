# One policy with a command vector (Step 7, design note for review)

Status: **design only, for review.** No env code is written until the owner has read this. This is an
experiment on branch `claude/single-policy-design`, not a replacement: the 9-clip dog-motion brain, the posture
brain and the skill manager (`train/skills.py`) stay as they are, and the jump brain stays separate.

## What it replaces, and why try it

Today the walking side of Bolt is two brains switched by `skills.py`: the dog-motion brain (imitation of 9
clips, picked by a clip one-hot) and the posture brain (stand). A switch changes the clip and jumps the clip
clock to the closest pose. The skills report shows where that hurts: canter -> trot 7/9, walk -> sit pose
reached 1/10, sit -> lie 1/10 from a walk (`train/skills_report.json`, re-run with
`MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python test_skills.py --n 10 --seed 1000 --imitate-steps 15000000` in
`train/`). Each clip is a fixed speed, so there is nothing between walk (1.12 m/s) and trot (1.90 m/s).

One policy driven by a command vector (gait id, forward speed, turn rate, posture) would: switch by changing
the command instead of swapping clips, cover speeds between the clips, and own the posture changes (sit, lie,
get up) that today go through a second brain.

Clip numbers used below (mean body-frame forward speed, yaw rate, body height). Re-run in `train/` with:

```bash
python -c "
from imitate_env import Clip, ALL_CLIPS
for n in ALL_CLIPS:
    c = Clip(n)
    print(n, c.loop, round(c.length, 2), round(c.v_body[:, 0].mean(), 2), round(c.w_body[:, 2].mean(), 2),
          round(c.height.mean(), 3), round(c.height[-1], 3))"
```

| clip | loops | length | forward m/s | yaw rad/s | body height m |
|---|---|---|---|---|---|
| walk | yes | 0.66 s | 1.12 | -0.10 | 0.393 |
| trot | yes | 0.54 s | 1.90 | -0.09 | 0.377 |
| pace | yes | 0.64 s | 1.14 | -0.08 | 0.387 |
| canter | yes | 0.48 s | 3.31 | 0.26 | 0.431 |
| turn_left / turn_right | yes | 0.64 s | 0.88 | +1.03 / -1.03 | 0.414 |
| sit | no | 3.52 s | ~0 | ~0 | 0.296 (ends 0.234) |
| lie | no | 5.52 s | ~0 | ~0 | 0.245 (ends 0.169) |
| getup | no | 5.52 s | ~0 | ~0 | 0.245 (ends 0.408) |

Walk and pace have the same speed, so speed alone cannot pick the gait: the gait id has to be in the command.

## Command vector (10 floats)

| index | name | encoding | range |
|---|---|---|---|
| 0-4 | gait id | one-hot: none (standing / posture), walk, trot, pace, canter | |
| 5 | forward speed | v_cmd / 3.3 | 0 .. 1 (0 .. 3.3 m/s) |
| 6 | turn rate | w_cmd / 1.0 | -1.2 .. 1.2 (rad/s) |
| 7-9 | posture | one-hot: up, sit, lie | |

Rules that make every skill one command:

| skill | gait | v_cmd (m/s) | w_cmd (rad/s) | posture |
|---|---|---|---|---|
| stand | none | 0 | 0 | up |
| walk | walk | 0.6 - 1.4 | about 0 | up |
| trot | trot | 1.4 - 2.4 | about 0 | up |
| pace | pace | 0.8 - 1.4 | about 0 | up |
| canter | canter | 2.4 - 3.3 | about 0 | up |
| turn left / right | walk | 0.6 - 1.0 | +0.5 .. +1.2 / -0.5 .. -1.2 | up |
| sit | none | 0 | 0 | sit |
| lie | none | 0 | 0 | lie (from sit) |
| get up | none | 0 | 0 | up, given while lying |
| landing recovery (Step 1) | canter, then trot | the landing speed, clipped to 3.3, then 1.9 | 0 | up |

Landing recovery is not a command of its own. It is a start state: tries that begin in the run-up jump brain's
landing (Step 1's `research/results/landing_starts.npz`, made by `research/gen_landing_starts.py`) with the
command "canter" for the bridge, then "trot". The policy has to see from its own body that it is in the air or
just landed. That is how the skill manager hands over today, so the exam can compare the two directly.

Postures win over gaits: any posture other than up means gait none and v_cmd = w_cmd = 0. Lie is accepted only
from sit, and get up only from lie (the `NEXT` table in `imitate_env.py` already encodes the allowed order).

## Observation layout

The actor sees only what a real robot has, plus the command. It does not see the reference clip, so on the robot
it runs from commands alone, with no clip player.

| index | input | size |
|---|---|---|
| 0 | gravity in the body frame | 3 |
| 3 | angular velocity x 0.25 (body) | 3 |
| 6 | linear velocity (body frame) | 3 |
| 9 | body height (m) | 1 |
| 10 | joint angles - dog stance | 12 |
| 22 | joint velocities x 0.1 | 12 |
| 34 | previous action | 12 |
| 46 | gait clock sin, cos | 2 |
| 48 | command vector (above) | 10 |
| 58 | time since the command last changed (s, capped at 2) | 1 |
| | **actor total** | **59** |

Indices 0-45 are exactly the current imitation brain's, so the first layers can be copied from the 360M brain.

- **Gait clock.** Gaits advance it at the gait's cadence, scaled by v_cmd / clip speed. Postures use it as
  progress 0..1 through the posture clip and hold it at 1 when the clip ends, the same as the current phase
  input (index 46).
- **Linear velocity and body height** (indices 6 and 9) are kept so the layout matches the current brains. A
  real robot only estimates them. Removing them is a separate experiment, and the exam below reports whether it
  costs anything.
- **The critic sees more** (asymmetric actor-critic, training only). It gets the 59 actor inputs plus the
  reference the reward uses:
  - next-step reference joints (12), reference body height (1), reference body velocity (3) and reference yaw
    rate (1), as indices 57-73 of today's obs;
  - the clip one-hot (9);
  - the randomized motor strength, torso mass and friction factors (3).

  That is 59 + 26 + 3 = 88 inputs.

**Action.** The motor targets span the full range around the dog stance (ctrl = DOG + ACTION_SCALE * a), as in
the jump and get-up brains. They are not added to the clip's pose as a residual, because the actor no longer
sees the clip. Alternative B (keep the clip as a feed-forward plan and the residual action, with the reference
in the actor's obs) is easier to learn but needs a clip player on the robot. Choosing between A and B is
review question 1.

## How the 9-clip imitation reward maps onto it

The command picks the reference clip, so the reward is the same five-term imitation reward as today
(`imitate_env.py`: pose 0.50, joint velocity 0.05, paws 0.20, body pose 0.15, body velocity 0.10), measured
against that clip. Commands map to clips like this:

| command | reference clip | clip clock |
|---|---|---|
| walk with abs(w_cmd) < 0.5 | walk | cadence x v_cmd / 1.12 |
| walk with w_cmd >= 0.5 / <= -0.5 | turn_left / turn_right | cadence x v_cmd / 0.88 |
| trot | trot | cadence x v_cmd / 1.90 |
| pace | pace | cadence x v_cmd / 1.14 |
| canter | canter | cadence x v_cmd / 3.31 |
| none + sit / lie / up-from-lie | sit / lie / getup | real time, then hold |
| none + up (standing) | the sit clip's first, standing frames, held | held |

The cadence factor is clipped to 0.6-1.2, the same clip as `jump_env._gait_pose`. Outside that band the next
clip's range takes over. The reference body velocity in the reward becomes (v_cmd, 0, w_cmd) for gaits, so a
commanded speed between two clips is what the policy is paid for, not the clip's own speed.

Two terms are added, gaits only, and the five imitation terms are scaled to share the reward with them:

    r = 0.7 * r_imitation + 0.15 * exp(-4 (v_fwd - v_cmd)^2) + 0.15 * exp(-4 (w_yaw - w_cmd)^2)

Postures keep r = r_imitation, as today.

The terms that stay as they are:
- the early-termination rules (15 cm off the reference height, 0.8 rad tilt, body on the floor when the dog's
  was not);
- the landing grace from Step 1;
- physics randomization, sensor noise and the real motor model.

**Command switching in training.** This replaces switching clips (`switch_rate` in `imitate_env.py`). Every
2-4 s a try draws a new command, following the `NEXT` order. For gaits it also draws a new v_cmd and w_cmd
inside that gait's band, so a switch can be trot 1.6 -> trot 2.2 as well as trot -> canter. The reference clip
starts at its closest-pose moment, which is what `skills.py` does at a switch, so training sees the same jumps
the skill manager makes.

**Jump.** The jump brain stays separate. At the hand-over the single policy gives way to the jump brain exactly
as the dog-motion brain does today, and after the landing it takes over again through the landing starts above.

## What the exam looks like

All of these use fixed seeds (9000+, never used to pick a checkpoint), physics randomization on, and the real
motor model. Each one is run on the single policy and, on the same seeds, on the brains it would replace (the 9-clip
dog-motion brain plus the posture brain). By the "pick by exam" rule, the single policy is kept only if it is
not worse on any switch both can do.

1. **Command tracking.** Five-second tries for each gait at three speeds in its band and both turns. Each try
   reports:
   - whether it fell;
   - the mean speed and yaw-rate error;
   - the imitation score (`imitate_env.score`, today's number for picking a brain).

   On the 9-clip brain this is run only at each clip's own speed.
2. **Switches.** Every `test_skills.py` sequence (gaits, rest, rest_still, jump20/30/40, shove), with the
   single policy in place of the dog-motion and posture brains. The jump brain and the self-righting brain are
   unchanged. The same tally, the same 2 s window, and the Step 5 rule: a switch not reached in more than half
   its tries is reported as untested.
3. **Landing recovery.** The share of the Step 1 landing states that are still upright 2.6 s later (the
   "jump -> (canter) -> trot" window), against the dog-motion brain fine-tuned in Step 1.
4. **Speed sweep.** Commands from 0 to 3.3 m/s in 0.25 m/s steps, showing achieved speed against commanded
   speed. This shows whether the canter is real: the 18b brain makes about 1.2 m/s of the clip's 3.31
   (`skills_report.json` findings).
5. **Motor audit.** Peak joint speed, torque and power per command (`research/jump_audit.py`'s pattern),
   against the motor model, with the share of time above 21 rad/s. This is the real-body rule.

Every number in the exam report comes with the command that produced it.

## Questions for review before any env code

1. Action A (full range, no clip in the actor) or B (residual on the command's clip, clip player on the
   robot)? A is the cleaner robot interface. B is the safer first run.
2. Keep linear velocity and body height in the actor obs for the first run (to match the current brains), or
   drop them now?
3. Start from scratch, or copy indices 0-45 of the 360M GPU brain's first layer (fine-tune)? The rest of the
   input layer would start at zero.
4. Is gait id "none" plus posture "up" the right way to say "stand", or should stand be a gait of its own?
5. Which body is the target for this run, the 33.5 N·m code default or the 17 N·m in HANDOFF.md (see the Step
   8 report)?
