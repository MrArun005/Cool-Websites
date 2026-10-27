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
