# Prompt for Claude Code on your own GPU computer

Paste everything below the line into Claude Code (terminal, VS Code or desktop app) on the computer with the NVIDIA GPU.
It sets the project up, checks the GPU, and runs the two pending 40-minute pilots, then reports a before/after table.

---

You are helping me (Arun Mallikarjun) train "Bolt", a robot-dog brain for the Unitree Go2 EDU, in MuJoCo simulation.
Everything is simulation; say so in every result. No external AI services in the product. Never print, commit or send
any token or key file (e.g. ~/.kaggle/*).

## 1. Get the project
```bash
git clone -b claude/robot-build-memory-ju76nt https://github.com/MrArun005/Cool-Websites
cd Cool-Websites/train
```
(If it is already cloned: `git checkout claude/robot-build-memory-ju76nt && git pull`.)

## 2. Install (Linux or Windows WSL2 with an NVIDIA driver; Python 3.11-3.13)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install "jax[cuda12]==0.9.2" jaxlib==0.9.2 flax==0.12.6 optax==0.2.8 orbax-checkpoint==0.12.8 mujoco-mjx==3.17.0 brax==0.14.2
python -c "import jax; print(jax.devices())"
```
The last line must show a CUDA/GPU device. If it shows only CPU, stop and tell me (driver or WSL2 GPU support missing);
do not train on CPU. A 1-minute smoke run after step 3's COMMON is set: `python mjx_imitate.py train $COMMON --warm
../models/bolt-go2-damp-798M/params.pkl --out runs/smoke --steps 20000 --envs 16 --quick --evals 3` must finish without errors.

## 3. Two pilots, 40 minutes each (one at a time; read train/deploy/PLAN.md "Log" for the background)
Common flags (all one line):
```
COMMON="--body go2 --steps 2000000000 --envs 4096 --evals 10 --hours 0.7 --moving-rate 0.5 --latency-min-ms 0 --latency-max-ms 40 --steer-max 0.8 --root-global 0.5 --steer-weight 0.6 --switch-rate 0.3 --moving-gaits walk trot --moving-clips sit lie --reward-scaling 0.26880143601458134 --blind-vel --drop-p 0.05 --damping-range 0.1 2.0"
```
a. Sitting from a trot (current robot brain 798M falls 3 in 12 going trot -> sit):
```
python mjx_imitate.py train $COMMON --moving-rate 0.75 --seed 51 --warm ../models/bolt-go2-damp-798M/params.pkl --out runs/sitpilot
```
b. Jump (never learned yet: track 0.114, falls 8/8):
```
python mjx_imitate.py train $COMMON --seed 43 --warm ../models/bolt-go2-jump-warm/params.pkl --out runs/jumppilot --clips walk trot pace canter turn_left turn_right sit lie getup jump --clip-weight canter=3 jump=4
```
(`--moving-rate` given twice in a: the later 0.75 wins. If argparse complains, drop it from COMMON for that run.)

c. New training ideas (A/B, same seed and length; only the flags differ):
```
python mjx_imitate.py train $COMMON --seed 61 --warm ../models/bolt-go2-damp-798M/params.pkl --out runs/base798
python mjx_imitate.py train $COMMON --seed 61 --warm ../models/bolt-go2-damp-798M/params.pkl --out runs/privhist --privileged-critic --history 3
```
`--privileged-critic`: the training-only value network sees true speed, friction and paw contacts. `--history 3`: the
brain reads its last 3 steps of sensor inputs. Score both with go2_exam.py (full and --moving); privhist wins only if it
beats base798, not just 798M.

## 4. Score (CPU is fine; set `export BOLT_BODY=go2 OMP_NUM_THREADS=1 MUJOCO_GL=egl`)
- Sit pilot vs the current brain, both: `python go2_exam.py <params.pkl> --moving` and `python go2_exam.py <params.pkl>`
  on `runs/sitpilot/params.pkl` and `../models/bolt-go2-damp-798M/params.pkl`.
- Jump pilot: `python go2_exam.py runs/jumppilot/params.pkl --jump` (before: track 0.114, fall 1.0).
- Before/after table. "Clearly better" = sit: fewer trot->sit falls with the full exam's mean tracking >= 0.81 and falls
  <= 2%; jump: track_jump rising well above 0.114 or falls below 8/8. If clearly better, run the same command with
  `--hours 3.5 --evals 40` (warm from the pilot's params.pkl) and score again. If not, stop and report why.
- A brain only replaces the robot brain after the vendor rehearsal in deploy/PLAN.md passes too
  (`deploy/lockstep_vendor.py`, `deploy/lockstep_recovery.py`; needs `bash deploy/setup_go2.sh --sim` first).

## 5. Save
Commit results (params.pkl, a MODEL.md with the numbers, exam output) under models/<name>/ and a PLAN.md log line, on
branch claude/robot-build-memory-ju76nt, then push. Never commit keys or tokens.
