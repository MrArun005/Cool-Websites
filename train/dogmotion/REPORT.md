# Moving like a real dog: imitation of dog motion capture (run 18 prep)

Goal: Bolt's gaits and postures copied from a real dog, the way Peng et al. 2020 ("Learning Agile Robotic Locomotion Skills by Imitating Animals") taught a Laikago robot: retarget dog mocap onto the robot, then train a brain with a DeepMimic-style tracking reward. Everything below is in place and smoke-tested; the long run (run 18) has not been started.

## 1. Data
The real dog motion capture of Zhang, Starke et al. (SIGGRAPH 2018, "Mode-Adaptive Neural Networks for Quadruped Motion Control"): 51 BVH files, 60 Hz, one dog, about 30 minutes, from the AI4Animation page (MotionCapture.zip, 33 MB). **License CC BY-NC 4.0** (non-commercial, attribution); details, URLs and the exact segments are in `SOURCES.md`. The raw BVH stays out of the repo; the 8 retargeted clips (about 250 kB) in `clips/` carry the same license.

Gait segments are the ones Peng et al. chose (their keypoint files match our BVH forward kinematics to 3e-6 m). Sit and lie were found by classifying every second of every file by hip/shoulder height and speed.

| clip | what | length | dog speed |
|---|---|---|---|
| walk | 4-beat walk, one stride | 0.66 s loop | 1.12 m/s |
| trot | trot | 0.54 s loop | 1.90 m/s |
| pace | pace (same-side legs together) | 0.64 s loop | 1.15 m/s |
| canter | canter (kept, not trained by default) | 0.48 s loop | 3.37 m/s |
| turn_right / turn_left | walking turn and its mirror image | 0.64 s loop | 0.88 m/s, 1.03 rad/s |
| sit | stand 1 s, sit down, hold | 3.5 s | |
| lie | stand, sit, lie down, hold | 5.5 s | |

## 2. Retargeting (`retarget.py`)
`python dogmotion/retarget.py --bvh-dir <unzipped MotionCapture>` (from `train/`, PYTHONPATH=.) writes `clips/<name>.npz` (50 Hz: base_pos, base_quat (wxyz), joints (12, package.JOINTS order), joint_vel, base_linvel/base_angvel (world), foot_pos (4x3), foot_contact, fps, loop, cycle_pos, cycle_yaw, source) and `clips/check.json`. `preview.py` renders a contact sheet and joint curves per clip.

1. BVH forward kinematics, Z-up metres, resampled to 50 Hz, lightly smoothed.
2. Body rotation from the dog's trunk, so its pitch and roll carry over. Start at the origin facing +x.
3. Scale per leg pair from a standing dog onto Bolt's dog stance (0.385 m hip to paw): front x1.116, rear x1.048.
4. Paw targets = Bolt's hip + the scaled dog hip-to-paw vector, but paw height = the dog's paw height above the floor (planted paws stay planted). The dog's sideways paw offset (paws under its midline) is removed, or Bolt's thighs would cut into its body.
5. A planted paw out of reach: the body is lowered smoothly (4-7 cm in the gaits).
6. Loops closed and paws smoothed before leg IK (smoothing joint angles afterwards pushed paws 3 cm into the floor); joint limits enforced; a leg may roll under the body at most 0.1 rad.
7. Sit and lie blend into Bolt's own measured leg fold (the dog's long paws folded flat wedged Bolt's knees under its body); nose-up pitch capped at 0.6 rad.

Kinematic check (MuJoCo, qpos set frame by frame):

| clip | planted paw height err mean / range (mm) | paw miss mean / max (mm) | joint clamp (rad) | floor pen. (mm) | self contact frames | body height (m) |
|---|---|---|---|---|---|---|
| walk | 2.8 / -3.0..+9.7 | 23 / 43 | 0.13 | 0.0 | 0 | 0.39-0.40 |
| trot | 2.5 / -2.4..+10.5 | 44 / 72 | 0.21 | 0.0 | 0 | 0.36-0.39 |
| pace | 1.9 / -0.8..+8.8 | 13 / 46 | 0.20 | 0.0 | 0 | 0.38-0.40 |
| canter | 1.4 / -3.4..+6.9 | 39 / 64 | 0.38 | 0.0 | 0 | 0.41-0.45 |
| turn_left/right | 3.4 / -6.1..+10.2 | 51 / 94 | 0.32 | 0.1 | 0 | 0.40-0.44 |
| sit | 6.8 / -0.3..+67 | 75 / 125 | 0.0 | 0.0 | 0 | 0.23-0.41 |
| lie | 12.9 / -3.2..+79 | 183 / 338 | 0.45 | 0.0 | 0 | 0.16-0.41 |

## 3. Training environment (`../imitate_env.py`) and trainer (`../train_imitate.py`)
- ImitateEnv(BoltEnv), 74 inputs (layout in the module docstring): body/joint senses, clip phase sin/cos, a one-hot over the 8 clips, the next reference step (joints, height, body velocity, yaw rate), and time since a sit/lie ended.
- Motors: `ctrl = clip(q_ref(next) + ACTION_SCALE * a)`: the dog's pose as feed-forward plus the brain's corrections.
- Reward (DeepMimic/Peng weights): 0.5 pose, 0.05 joint velocity, 0.2 paws, 0.15 body height+tilt, 0.1 body velocity; every term exp(-k * error) in 0..1, nothing negative, so falling is never a way out.
- Reference state initialization at a random moment of a random clip, physics randomization and sensor noise as in the other envs, early termination when far off the reference.
- Exam per clip: tracking, fall rate, and `speed_` = share of the dog's distance (turns: angle) covered. Best brain chosen by tracking x min(1, speed_).
- PPO: 2048 steps per env per update, batch 4096, 5 epochs, lr 3e-4, target KL 0.03, ELU [512, 256], log_std -1, gamma 0.95.

## 4. Smoke test (300k steps, 2 envs, alongside run 17)

| | reference playback, no brain | 100k | 200k | 300k |
|---|---|---|---|---|
| mean tracking (exam) | 0.255 | 0.366 | 0.421 | 0.564 |
| fall rate (exam) | 0.79 | 0.29 | 0.14 | 0.04 |
| training reward per step | 0.22 | 0.29 | 0.37 | 0.40-0.42 |

A random brain scores 0.02 and falls every time. **It does not keep the dog's pace yet**: at 300k walk and pace cover 19% of the dog's distance, trot 31%, turns 15-30%. Poses come first, pace later.

## 5. Full run
`python3 tools/keepalive/keepalive.py start train/jobs/run18-imitate.json` (from the repo root, once the machine is free): `train_imitate.py --name imitate18 --steps 30000000`, ~4.5-5.5 h on 4 cores. If tracking rises but `speed_walk` / `speed_pace` stay under ~0.6 by 5M, restart with `--root-vel-weight 0.25`.

## 6. Risks
- License: CC BY-NC 4.0, so clips and brains trained on them are non-commercial, with credit.
- The dog trots at 1.9 m/s; Bolt's best trot is 1.2 m/s on 17 N·m motors. Canter is left out.
- One multi-clip brain may trade clips against each other; train subsets if a clip stays at 0.

## 7. Porting to the browser later
Export each clip to JSON (per-frame joints, height, body-frame velocity, yaw rate), keep a clip clock in JS, build the 74 inputs exactly as `ImitateEnv._obs`, and send `q_ref(next) + ACTION_SCALE * action` to the motors.
