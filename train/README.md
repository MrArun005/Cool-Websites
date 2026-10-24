# Training Bolt

Teaching a robot to be a pet is two separate kinds of training, just like with a puppy:

| | Body: *how* to move | Brain: *what* to do and say |
|---|---|---|
| Puppy | Learns to walk and sit by trying, falling and trying again | Learns what "sit" means, and what makes you happy, from treats and your reactions |
| Bolt | **Reinforcement learning** in a physics simulator: millions of attempts, a score for each, the network adjusts toward what scored well | **A language model** maps what you say to tricks and replies; **your feedback log** ("good boy" / "no") teaches it what you like |
| Where | This folder (`train/`) | `pet/js/brain.js` today; a language model in Phase 3 |

## 1. Body training (this folder)

`bolt.xml` is Bolt as a MuJoCo physics model: 12 kg, 12 motors limited to 17 N·m each, real foot contacts and friction. Nothing is scripted; the network has to find out how to use its motors.

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
```

On Linux without a display, set `MUJOCO_GL=osmesa` (install `libosmesa6`) for the video.

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
