# Handoff: starting the new robot

This repo holds everything from the first phase. A new session starts the **new robot** from here.

## The new robot: the owner's idea

Not a dog, not a copy of an existing product. An original robot with:

- **A core body** that holds the hardware: computer, SSD storage, battery, motor drivers, and
  whatever else is needed.
- **Wheels for moving** (a pair of tires driven by motors) instead of legs, because good legs are
  hard. Legs could come later.
- **A head** on top that holds the "brain" side of the robot and moves on its own:
  spins a full **360°**, swings, and turns left↔right **while the core stays still**.
- The owner's description was cut off at *"And with the tires we can move, we can…"*.
  **Ask the owner to finish it before designing.**

Ground rules from the owner:
- Build it ourselves. Don't buy a ready-made robot, and don't copy one.
- Learn techniques from existing open-source robots and improve on them (see `docs/LESSONS.md`).
- **Digital first:** design the robot as a simulation ("digital physical robot"), train its
  brain there, then build the real one to match.
- After **every** training run, keep the trained model **and** its training data
  (`train/package.py` does this; output goes in `models/`).

## What already exists

| Path | What it is | Reusable for the new robot? |
|---|---|---|
| `pet/js/brain.js` | Pet brain: understands commands, personality, battery, memory, feedback log | Yes: swap the trick list for the new robot's skills |
| `pet/index.html` + `pet/js/` | 3D browser pet (three.js bundled, works offline), voice in/out | The UI, chat and voice parts, yes |
| `train/` | MuJoCo + PPO training on CPU: environment, training with resume, exam, video watcher | Yes: same pipeline, new robot model and environment |
| `train/watch.py` | Films and tests every checkpoint while training runs | Yes |
| `train/package.py` | Packages a run into `models/<name>/`: ONNX brain, checkpoint, exam, curve, recorded rollouts, model card | Yes |
| `models/bolt-dog-3M/` | Trained dog brain (stands and walks 100%, no sit/lie) + 3,000 recorded steps | Reference only |
| `docs/LESSONS.md` | Techniques from Microduck, Open Duck Mini, Petoi, Sesame and our own runs | Read first |
| `pet/ROADMAP.md` | The original plan: digital pet → physics training → understanding people & humor → body | Phases 3–4 still apply |
| `robot/` | First project: Arduino limb firmware + Python brain with SQLite memory, wheeled base | The wheel/serial code is a starting point |
| `hardware/mini/`, `duck/` | Earlier body designs (servo dog draft, Open Duck plan) | Reference only |

The dog's raw checkpoints were in `train/runs/` (not in git); the best one is preserved as
`models/bolt-dog-3M`.

## Environment setup (cloud container, CPU only, no GPU)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r train/requirements.txt onnx onnxruntime onnxscript
apt-get install -y libosmesa6 xvfb openscad   # offscreen MuJoCo video + CAD renders
export MUJOCO_GL=osmesa
cd train && python -m unittest test_env.py
node --test ../pet/test/*.test.js
```

- Training throughput here: about 2,000 steps/s with 8 parallel environments on 4 cores.
- Rendering video: `MUJOCO_GL=osmesa`. CAD previews: `xvfb-run -a openscad ...`.
- Big GPU training (thousands of robots at once) needs an NVIDIA GPU: Isaac Lab 3.0 or MuJoCo
  Playground / mjlab. Not available in this container.

## Suggested first steps

1. Get the rest of the owner's description; agree on size, speed and what the head must do.
2. Sketch the robot in MuJoCo: core box with realistic mass, two driven wheels (+ caster, or
   self-balancing on two wheels), a continuous-rotation pan joint and a tilt joint for the head.
3. Decide what is *trained* vs *programmed*: driving and head pointing can be classic control;
   learned skills could be self-balancing, following a person, expressive head motion.
4. Reuse `train/` patterns: environment, exam, watcher, packaging after every run.
5. Hook up `pet/js/brain.js` (or a language model) to command the new robot's skills.

## Standing rule: real body, real motors
Everything Bolt does must be something a real body with real motors could do (the user's rule). The
physics carries a real-motor model (train/bolt_env.py MOTOR: 17 N·m cap, 420 W, 21 rad/s, the Unitree A1's
class for a 12 kg dog) and real joint ranges (sideways hips ±48°, Go2). Check every new skill's peak joint
speed, torque and power against it (scratchpad audit.py pattern) before trusting any result, and remove
shortcuts no real robot has (teleports, kinematic carrying) or say plainly where one remains.

## Finish line: the best robot software when training ends
Training is not the goal; one working robot brain stack is. When runs 16e, 17d, 18 and 19 end:
1. **Pick by exam, not by last step.** Each run keeps its best checkpoint (best/). Re-run the exam on the best and last 2 checkpoints with 3 seeds; keep the winner. Never ship a brain that scores below the one it replaces.
2. **Prove it in the real-body physics.** Everything under the motor model (17 N·m, 420 W, 21 rad/s) and the ±0.84 rad hip range; no teleports, no kinematic shortcuts. A skill counts only if it passes in the browser physics too (live_check, getup_test, game tests).
3. **One stack, one controller.** Walk (19) and dog-like gaits (18), jump (16e), self-right then posture stand (17d), all switching through the same hand-over rules the games use. Test the switches: walk→jump→land→walk, fall→self-right→tuck→stand→walk, 60 simulated minutes on the pet page with zero stuck states.
4. **Package.** models/<name>/ with ONNX, exam.json, MODEL.md for each brain; one README table of skills, success rates and limits (licence: dog-motion brains are CC BY-NC 4.0).
5. **Ship.** Re-export the Gait Lab, castle, agility and pet pages with the new brains; remove the teleport fallback and the kinematic ball carry; republish; final news item on Bolt Learning Live.
If a run ends worse than its start (e.g. the motor limits cost too much), fix the reward or curriculum and retrain that skill only; do not ship the regression.

## Product goal: better together than the published research
Each published system leads at one skill (Peng 2020 gaits, ANYmal recovery, parkour jumps, Mini Cheetah speed).
Bolt's edge is the whole dog in one stack, measured honestly. Steps:
1. Skill manager (train/skills.py, test_skills.py): dog gaits by speed (walk/trot/canter from runs 18b-18d), jump
   (16g/16h), fall -> roll to belly (17e) -> get up front-first like a dog (18d getup clip), sit/lie. Target: 95%+
   of switches without a fall.
2. Robust for hardware: train with motor delay 0-20 ms, sensor noise, shoves; avoid inputs a real robot cannot measure.
3. Benchmarks on the papers' terms (gait imitation vs Peng, recovery vs ANYmal/AFR, jump heights vs parkour, pushes
   vs DreamWaQ, a Barkour-style course) and a scoreboard on Bolt Learning Live, each number marked sim or real.
4. Demo: pet page and games on the combined Bolt; 4K video pushed to git.
Do not claim to beat a specialist at its speciality unless the same test says so.

## Strict standing rules (set by the owner, 27 Oct)
0. (28 Oct) Always give times to the owner in Indian Standard Time (IST, UTC+5:30), e.g. "08:29 IST".
0b. (28 Oct) Kaggle runs stay LONG: size each job to use nearly all of Kaggle's 12 h session limit (time guard ~10.5-11.5 h), results once a day, rather than short 2-4 h runs.
1. Every 30 minutes, check everything, whether or not anyone reported: each training run's log is fresh (< 40 min),
   checkpoints arrive, exams improve; each agent is alive; this machine's background tasks are running; pages are
   published with the latest data (with a "what's new" table); git is committed and pushed.
2. Fix anything stalled or broken at once (restart, retune, replace); never leave a failure waiting for a report.
3. Never stop: keep training and improving all day until the targets are reached (dog motions >= 90% incl. canter
   >= 70% with no falls, switches >= 95%, clean jumps >= 50 cm with a run-up), then the GitHub release, game and 4K video.
4. Re-arm the 30-minute check every time; never let it lapse.
