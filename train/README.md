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

```bash
# transition term: for 1 s after a mid-try clip switch, -0.3 per rad of roll/pitch beyond 0.3 rad of the reference's
python mjx_imitate.py train --warm ../models/bolt-dog-imitate-gpu-360M/params.pkl --reward-scaling 0.246 \
    --steps 20000 --envs 16 --out /tmp/smoke_transition --quick --transition-weight 0.3
```
Transition flags: `--transition-weight` (default 0 = off), `--transition-window` (s after the switch, default 1.0: the
canter -> trot falls come 0.5-0.8 s after it), `--transition-threshold` (rad of roll or pitch, each against the
reference's own, that is free; default 0.3). The five tracking terms are unchanged.

```bash
# real-robot gaps (mastermind/REPORT.md: latency is the biggest): 0-40 ms action latency per try, sensor noise, motor lag
python mjx_imitate.py train --warm ../models/bolt-dog-imitate-gpu-360M/params.pkl --reward-scaling 0.246 \
    --steps 20000 --envs 16 --out /tmp/smoke_real --quick --latency-max-ms 40 --obs-noise 1.0 --motor-lag-ms 10
```
Real-gap flags (all default 0 = off; the models are mastermind/env.py's stress test): `--latency-min-ms`/`--latency-max-ms`
(action latency drawn per try, at the 500 Hz physics rate), `--obs-noise` (x legged_gym's sensor noise levels,
`imitate_env.NOISE`), `--motor-lag-ms` (first-order lag time constant). `imitate_env.exam` adds a latency sweep,
`lat<ms>_track` / `lat<ms>_fall` at 0/10/20/40 ms (the 360M brain: falls 0 / 6% / 42% / 89%).

```bash
# rough terrain curriculum (0 -> 5 -> 10 cm at 0 / 30 / 60 % of the steps) and pushes (every 2 s, half the time, 1 m/s)
python mjx_imitate.py train --warm ../models/bolt-dog-imitate-gpu-360M/params.pkl --reward-scaling 0.246 \
    --steps 20000 --envs 16 --out /tmp/smoke_terrain --quick --terrain-cm 0 5 10 --terrain-at 0 0.3 0.6
python mjx_imitate.py train --warm ../models/bolt-dog-imitate-gpu-360M/params.pkl --reward-scaling 0.246 \
    --steps 20000 --envs 16 --out /tmp/smoke_push --quick --push-interval 2
```
Terrain (step 6): `bolt_env.TERRAIN` / `terrain_xml` add a 40 x 40 m height field (uniform random heights per 10 cm cell up
to `--terrain-cm`, field fixed by `--terrain-seed`) beside the flat floor, centred 40 m out, only when the height is
above 0 (so bolt.xml and every existing brain are untouched). `--terrain-clips` (default walk trot) tries start on it
within `--terrain-spread` m (5) of its middle; body height is measured from the ground under the body. Only paws and
shins touch the bumps (body/head/thigh boxes on a height field made MJX 6x slower; paws and shins cost 1.9x).
`--terrain-cm 0 5 10 --terrain-at 0 0.3 0.6` trains in phases, each warm-started from the last. Pushes:
`--push-interval` (s, default 0 = off), `--push-prob` (0.5), `--push-speed` (1.0 m/s), `--push-spin` (0 rad/s), the
switch test's shove (`bolt_env.apply_shove`, also used by `skills.Bolt.shove`). The target test:
```bash
python -c "from imitate_env import exam_terrain; from mjx_imitate import MjxImitatePolicy as P; print(exam_terrain(P('<brain.pkl>'), 5, clip='trot'))"
```
(30 tries of 4 s, randomized physics; a fall is the skill manager's: tipped past 60 degrees or the body down 0.1 s.)

CPU-vs-GPU parity of all of it (CPU venv dumps, JAX venv checks; needs a 75-input SB3 checkpoint):
```bash
python test_mjx_imitate.py dump runs/imitate18d_start/policy_14000000_steps.zip warm.pkl cases.pkl   # CPU stack
python test_mjx_imitate.py check warm.pkl cases.pkl                                                    # JAX stack
```
The Unitree Go2 body with steering (`mjx_imitate.py train --body go2 --steer-max 1.0`; Kaggle: `make_kernel.py --body go2`),
against `ImitateEnv(body="go2")`, steering off and on (needs a Go2 SB3 run folder with policy.zip + vecnormalize.pkl):
```bash
python mjx_imitate.py warm runs/go2_smoke/best/policy.zip warm_go2.pkl                                # CPU stack (records body go2)
python test_mjx_imitate.py dump-go2 runs/go2_smoke/best warm_go2.pkl cases_go2.pkl                    # CPU stack
python test_mjx_imitate.py check-go2 warm_go2.pkl cases_go2.pkl                                       # JAX stack
```
Known gap (accepted 28 Oct): at joint speeds above ~25 rad/s MJX's `implicitfast` substep differs from MuJoCo's by
~1 rad/s, brake or not (exactly 0 with the Euler integrator on both sides, where the brake also matches to 6e-6
rad/s; the `[brake]` lines show both). Below that the one-step physics agrees to ~5e-7 rad (median).

Motor audit of the new starts (every 500 Hz substep; `research/imitate_audit.py`, the jump_audit.py pattern):
```bash
python research/imitate_audit.py ../models/bolt-dog-imitate-gpu-360M/params.pkl --motor-brake \
    --out research/results/audit_imitate_360M_brake.json
```

Kaggle job with the three fixes (build and smoke-test locally; pushing starts it):
```bash
python kaggle/make_kernel.py --user <user> --job imitate --slug bolt-dog-gpu3 --warm-dataset <user>/bolt-imitate-360m \
    --steps 1000000000 --hours 10.5 --landing-rate 0.2 --moving-rate 0.2 --transition-weight 0.3 --evals 40 \
    --latency-min-ms 0 --latency-max-ms 40 --obs-noise 1.0 --extra "--reward-scaling 0.246 --switch-rate 0.3 --seed 3" -o /tmp/kg3
python /tmp/kg3/kernel.py --local /tmp/kg3_run <dir with warm.pkl>    # JAX venv: the same command, 20k steps, 16 envs
```

Landing flags: `--landing-rate` (share of tries, default 0), `--landing-file` (default
`research/results/landing_starts.npz`), `--landing-grace` (s without the 15 cm height check while still in the air,
default 0.5), `--landing-bridge` (s of canter before the switch, default 0.6 = `skills.BRIDGE`), `--landing-then`
(default trot). The CPU env (`imitate_env.ImitateEnv`) takes the same arguments.

### Dog-motion brain on a real robot's body (Unitree Go2; CPU env only so far)

The target robot is the Unitree Go2 EDU (README "Target robot"; plan: `research/real_robot_port.md`). Real robots are
entries of a table, `robot_bodies.BODIES` (model file, joint names in our order, stance, motor torque and speed, PD
gains, clip prefix, body-size limits, pose thresholds, folded legs); the leg kinematics are measured from the model. A
second robot (DEEP Robotics Lite3 is planned) is added as an entry, plus its model file under `robots/`.

```bash
python robot_bodies.py go2                                     # what the entry resolves to
PYTHONPATH=. python dogmotion/retarget_body.py --body go2 --bvh-dir <MANN bvh folder> --png /tmp/go2png   # go2_*.npz clips
python train_imitate.py --body go2 --name go2_smoke --clips walk trot --steps 300000 --envs 2 --exam-every 100000
BOLT_BODY=go2 python -c "from imitate_env import ImitatePolicy, exam; print(exam(ImitatePolicy('runs/go2_smoke/best'), clips=['walk', 'trot']))"
python -m unittest test_env.Go2BodyTest
```

`--body go2` (or `BOLT_BODY=go2`) swaps the model (Menagerie Go2 with position servos, kp 40 / kd 1.0), the motor model
(23.7 N·m and 30.1 rad/s hip/thigh, 45.43 N·m and 15.7 rad/s knee; drive fade, power and brake approximate), the stance
(thigh 0.85, knee -1.7, ~0.30 m) and the clips (`dogmotion/clips/go2_*.npz`); inputs, outputs and reward are as
Bolt's. Without it everything is Bolt, bit for bit. Smoke run (30 Oct, in simulation, 300k steps, walk + trot, 2 envs):
reward per step 0.23 -> 0.64; exam at 300k: walk 0.773 / trot 0.758 tracking, no falls, 35% / 70% of the dog's pace.
Kaggle CPU pilot from scratch, all 9 clips: `kaggle/make_train_kernel.py kernel --job go2-dog-pilot`.
Not yet on the Go2: the GPU trainer (`mjx_imitate.py`), landing starts, terrain, hurdles, the jump/get-up envs.

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
