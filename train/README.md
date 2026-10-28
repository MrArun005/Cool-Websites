# Training Bolt

Teaching a robot to be a pet is two separate kinds of training, just like with a puppy:

| | Body: *how* to move | Brain: *what* to do and say |
|---|---|---|
| Puppy | Learns to walk and sit by trying, falling and trying again | Learns what "sit" means, and what makes you happy, from treats and your reactions |
| Bolt | **Reinforcement learning** in a physics simulator: millions of attempts, a score for each, the network adjusts toward what scored well | **A language model** maps what you say to tricks and replies; **your feedback log** ("good boy" / "no") teaches it what you like |
| Where | This folder (`train/`) | `pet/js/brain.js` today; a language model in Phase 3 |

## 1. Body training (this folder)

`bolt.xml` is Bolt as a MuJoCo physics model: 12 kg, 12 motors of 33.5 N·m each (the Unitree A1's joint motor; `bolt_env.MOTOR` also limits each to 420 W and 21 rad/s), real foot contacts and friction. Brains trained before 27 Oct (commit fd5199b) had a 17 N·m cap: each `models/*/MODEL.md` says which body it was trained in, and `BOLT_MOTOR_TORQUE=17` examines it in that body. Nothing is scripted; the network has to find out how to use its motors.

`bolt_env.py` is the training ground. 50 times a second the policy (a neural network with two 256-neuron layers) feels the body's tilt, spin, joint angles and the current command, and sets 12 motor targets. It's scored on:
- **doing what was asked**: the right height and posture for *stand*, *sit* and *lie*; moving forward at 0.5 m/s with a trot rhythm for *walk*
- **moving like a good robot**: penalties for jerky motion, wasted motor effort and falling over

Commands change every 4 seconds, so it also learns to go from one to another. Every episode randomizes body mass (±15%), floor friction (×0.6–1.3) and motor strength (±15%), so it learns the skill rather than one exact simulation. That's what makes it survive on a real robot later.

`train.py` runs PPO (Proximal Policy Optimization), the standard algorithm for this, with 8 simulated dogs in parallel.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m unittest test_env.py               # model + environment sanity checks
python train.py --steps 6000000              # ~55 min on 4 CPU cores
python evaluate.py runs/bolt --video bolt.mp4
tensorboard --logdir runs                     # watch the reward climb
python watch.py runs/bolt --log train.log --out runs/bolt/watch   # film + test every checkpoint
```

`watch.py` runs next to training (send train.py's output to `train.log`). Each time a checkpoint is saved, it films the dog obeying commands while still practising (exploration noise included, so you see it wobble and fall) and gives it a quick exam. It writes `data.json` and `clips/*.mp4` for a progress page.

On Linux without a display, set `MUJOCO_GL=osmesa` (install `libosmesa6`) for the video.

### Dog-motion brain on a GPU: skill-switch training flags (`mjx_imitate.py`)

Each new flag defaults to off (the motor brake excepted: on for training, owner decision 28 Oct), so a run without them
trains as before. CPU smoke test for each (the `--quick` path,
16 envs, 20k steps, JAX venv; the 360M brain's `params.pkl` has no reward scale, so pass the one the GPU runs use):

```bash
# landing starts: 30% of tries start in the run-up jump brain's landing (mid-air to touch-down), canter 0.6 s, then trot
python research/gen_landing_starts.py --tries 96          # once: research/results/landing_starts.npz (2398 states)
python mjx_imitate.py train --warm ../models/bolt-dog-imitate-gpu-360M/params.pkl --reward-scaling 0.246 \
    --steps 20000 --envs 16 --out /tmp/smoke_landing --quick --landing-rate 0.3
python -m unittest test_mjx_imitate                       # CPU stack: landing start keeps the 75-input layout
# (the smoke run takes ~18 min on 4 CPU cores, mostly compiling and the eval rollouts)
```
```bash
# posture commands while moving: 30% of tries start in walk/trot at 0.5-2 m/s and are told at once to sit, lie or get up
python mjx_imitate.py train --warm ../models/bolt-dog-imitate-gpu-360M/params.pkl --reward-scaling 0.246 \
    --steps 20000 --envs 16 --out /tmp/smoke_moving --quick --moving-rate 0.3
```
Moving-start flags: `--moving-rate` (share of tries, default 0), `--moving-speed LO HI` (m/s, default 0.5 2.0),
`--moving-gaits` (default walk trot: the one whose clip pace is nearest the drawn speed), `--moving-clips` (default
sit lie getup), `--pose-time CLIP=S ...` (time limit to reach the pose, default sit=4 lie=5.5 getup=5.5: the switch
test's times), `--pose-weight` (reward per step in the pose once reached in time, default 0.2). "Reached" is
`pose_reached.py`, the same function `test_skills.py` (and so `research/run_switches.py`) uses.
Motor brake: `--motor-brake` / `--no-motor-brake` (default ON for training), `--brake-speed` (default 25 rad/s, the
drive cutoff), `--brake-gain` (default 3.5 N·m per rad/s), as `bolt_env`; exams keep it off unless `--motor-brake`.

Landing flags: `--landing-rate` (share of tries, default 0), `--landing-file` (default
`research/results/landing_starts.npz`), `--landing-grace` (s without the 15 cm height check while still in the air,
default 0.5), `--landing-bridge` (s of canter before the switch, default 0.6 = `skills.BRIDGE`), `--landing-then`
(default trot). The CPU env (`imitate_env.ImitateEnv`) takes the same arguments.

### Results

See the results table below, filled in from `runs/bolt/eval.json`.

### What comes next for the body

- **Tricks with flight** (jump, backflip, somersault): the same recipe plus reference motions exported from the browser sim to imitate. These need far more samples: tens to hundreds of millions of steps, which means a GPU (MuJoCo MJX or Isaac Lab) instead of 4 CPU cores.
- **Sim-to-real**: identical observation and action layout on the physical robot, with the IMU providing tilt and spin.

## 2. Brain training (what makes it a *pet*)

1. **Collect real interactions.** Use the browser dog (`pet/index.html`). Every command, what it did and your 👍/👎 ("good boy" / "no") is saved. **Export training data** downloads it as JSON.
2. **Language model with tools.** Each trick becomes a tool; the model reads what you said, the dog's state and its memories, and picks tricks plus a reply. No training needed to start; the exported log becomes its examples and its exam (an evaluation set of phrasings with the right answer).
3. **Learn your preferences.** Feedback labels show which responses you liked. They first guide which examples the model sees; with a few thousand labels they can train a small model (fine-tuning) that runs on the robot itself, offline.
4. **Humor.** Rate every joke and bit. Keep a fixed set of situations ("you came home", "it failed a trick") and measure the funny rate from people outside the project, aiming for 50% or more.
