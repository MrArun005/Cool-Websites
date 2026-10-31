"""Build a Kaggle CPU kernel that fine-tunes one Bolt brain with PPO for a whole free session (~10.5 h on 4 CPUs).
The same recipe as make_mastermind_kernel.py: a private data set holding the code and brains as one tar (named
.pack, which Kaggle leaves packed), pip installs over Kaggle's torch, everything written to /kaggle/working as it
goes (train/runs is a link into it, so every checkpoint and best/ is saved the moment it is written), and a time
guard (--hours) that ends training cleanly before Kaggle's 12 h limit.

Jobs (JOBS below):
  jump-trot  train_jump.py from the run-up brain 16i-35.5M: jump from a trot to a canter (1.2-2.4 m/s, most tries
             mid-stride), practice bars up to 75 cm; exams at 1.5 m/s and 2.0 m/s from 1.5 m, 40-70 cm, still legs;
             best/ picked (from 28 Oct) on the mid-stride exam: 1.7 m/s from 1.1 m, 40/50/60 cm, legs mid-stride.
  jump-run60 train_jump.py from jump-trot-55M: run-up only (1.8-3.2 m/s from 1.5-4 m), bars 55-75 cm, brake on;
             best/ picked mid-stride at 2.5 m/s from 3 m over 55/60/65 cm.
  recover    train_getup.py from self-righting 17e-16.5M: kicks, bad landings, ice, a second kick, wider physics;
             exam = the whole recovery (self-right, tuck, stand) from those falls.
  go2-dog-pilot  train_imitate.py --body go2 FROM SCRATCH (start None): the dog-motion brain on the Unitree Go2 body
             (robot_bodies.py, clips dogmotion/clips/go2_*.npz). The data tar must be built from a commit that has
             robot_bodies.py and the go2_ clips (build_data packs git HEAD; it takes every robots/*/*_nomesh.xml and
             LICENSE, so a new body's model goes in by its file name).

    python kaggle/make_train_kernel.py data -o /tmp/td          # code + brains from git HEAD, one tar
    kaggle datasets create -p /tmp/td                             # later: kaggle datasets version -p /tmp/td -m new
    python kaggle/make_train_kernel.py kernel --job jump-trot -o /tmp/k1
    kaggle kernels push -p /tmp/k1
    kaggle kernels status arunmallikarjuna/bolt-jump-trot
    kaggle kernels output arunmallikarjuna/bolt-jump-trot -p runs/kaggle-jump-trot

The output: runs/<name>/ (checkpoints/, best/ with best.json, train.log), fair_exam_*.txt (30 tries per test on
fresh seeds 9000+, for the best brain and the starting brain; fair_exam_<tag>_2.txt the second final exam, each
file starting with its command), summary.json and kernel.log. To re-pick a finished run's brain on the mid-stride
exam afterwards: python research/reselect_jump.py runs/kaggle-jump-trot/runs/jumptrot

Local smoke test of the exact kernel script, Kaggle's folders faked (tiny settings: --train-h 0.06 --every 8192
--envs 2 --n-steps 1024 --exam-episodes 2 --final-episodes 2):
    KAGGLE_INPUT=/tmp/fake/input KAGGLE_WORKING=/tmp/fake/working BOLT_ROOT=/tmp/fake/bolt NO_PIP=1 python /tmp/k1/kernel.py
"""
import argparse
import io
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TAR = "bolt-train.pack"   # a tar under a name Kaggle does not unpack (it unpacks .tar AND the policy.zip files inside)
MODELS = ["bolt-dog-jump-runup-16i-35.5M", "bolt-dog-jump-trot-55M", "bolt-dog-jump-handover-59M", "bolt-dog-jump-run60-74M", "bolt-dog-jump-handover-74M", "bolt-dog-recover-delay-30M", "bolt-dog-jump-delay-76M", "bolt-dog-jump-gaps-80.5M", "bolt-dog-selfright-17e-16.5M", "bolt-dog-posture-getup-3M", "bolt-dog-canter-speed-38M", "bolt-dog-jump-runup-88M"]
PIP = ["numpy>=2", "mujoco==3.17.0", "stable-baselines3==2.9.0", "gymnasium==1.4.0"]   # torch: Kaggle's own

JOBS = {
    "jump-trot": dict(
        name="jumptrot", script="train_jump.py", start="../models/bolt-dog-jump-runup-16i-35.5M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.45", "--bar-max", "0.75",
              "--exam-speed", "1.5", "--exam-dist", "1.5", "--exam-heights", "0.40", "0.50", "0.60", "0.70",
              "--exam2", "2.0", "1.5",
              # best/ is picked on the mid-stride hand-over (28 Oct, research/handover_gap.py): 1.7 m/s from 1.1 m,
              # legs mid-stride as MasterMind's trot hands over; the still-legs exams above are still logged
              "--select", "midstride", "--select-speed", "1.7", "--select-dist", "1.1",
              "--select-heights", "0.40", "0.50", "0.60"],
        # final fair exams (fair_exam.py, seeds 9000+): still legs and mid-stride at 1.5/2.0 m/s from 1.5 m, then the
        # trot hand-over case; tries of 220 steps (4.4 s), the model cards' setting
        finals=[["jump", "--speeds", "1.5", "2.0", "--dist", "1.5", "--heights", "0.40", "0.50", "0.60", "0.70", "--gait",
                 "--max-steps", "220"],
                ["jump", "--speeds", "1.7", "--dist", "1.1", "--heights", "0.40", "0.50", "0.60", "--gait",
                 "--max-steps", "220"]]),
    "jump-run60": dict(
        # 60 cm needs a run-up: every try starts running (1.8-3.2 m/s, 1.5-4 m out, mostly mid-stride), bars 55-75 cm;
        # from the trot jump brain (jump-trot 55M). Motor brake on (train_jump default; owner rule 0c for 60 cm work).
        # Same 200-step try as its training (the clock input is t / max_steps), long enough for a 4 m run-up.
        name="jumprun60", script="train_jump.py", start="../models/bolt-dog-jump-trot-55M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "1.5", "4.0", "--approach-speed", "3.2", "--approach-speed-min", "1.8", "--run-frac", "1.0",
              "--gait-start", "0.8", "--bar-start", "0.55", "--bar-max", "0.75",
              "--exam-speed", "2.5", "--exam-dist", "3.0", "--exam-heights", "0.50", "0.60", "0.70",
              "--exam2", "2.0", "1.5",
              "--select", "midstride", "--select-speed", "2.5", "--select-dist", "3.0",
              "--select-heights", "0.55", "0.60", "0.65"],
        finals=[["jump", "--speeds", "2.0", "2.5", "3.0", "--dist", "3.0", "--heights", "0.50", "0.60", "0.70", "--gait",
                 "--max-steps", "220", "--motor-brake"],
                ["jump", "--speeds", "1.7", "--dist", "1.1", "--heights", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "jump-run60b": dict(
        # run60's lesson (research/results/jump_run60/VERDICT.md): its bar never left 55 cm and it lost the trot
        # hand-over. Mix the old short approaches with the long run-ups and start the bar at 40 cm; best/ picked on
        # 60 cm from a long run-up AND the trot hand-over.
        name="jumprun60b", script="train_jump.py", start="../models/bolt-dog-jump-trot-55M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "4.0", "--approach-speed", "3.2", "--approach-speed-min", "1.2", "--run-frac", "0.9",
              "--gait-start", "0.7", "--bar-start", "0.40", "--bar-max", "0.75",
              "--exam-speed", "2.5", "--exam-dist", "3.0", "--exam-heights", "0.50", "0.60", "0.70",
              "--exam2", "1.7", "1.1",
              "--select", "midstride", "--select-speed", "2.5", "--select-dist", "2.0",
              "--select-heights", "0.50", "0.55", "0.60"],
        finals=[["jump", "--speeds", "2.0", "2.5", "3.0", "--dist", "3.0", "--heights", "0.50", "0.60", "0.70", "--gait",
                 "--max-steps", "220", "--motor-brake"],
                ["jump", "--speeds", "1.7", "--dist", "1.1", "--heights", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "jump-runup": dict(
        # 29 Oct: run60b (mixed run-ups) learned nothing from 3 m; run60's 74M is the only brain that clears 50 cm from a
        # long run-up (43% at 2.5 m/s). A separate run-up brain from it: long run-ups only, practice bar from 50 cm.
        # Pass: mid-stride 50 cm from 3 m at 2.5 m/s >= 50% in the final fair exam (30 tries, seed 9000).
        name="jumprunup", script="train_jump.py", start="../models/bolt-dog-jump-run60-74M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "1.5", "4.0", "--approach-speed", "3.2", "--approach-speed-min", "1.8", "--run-frac", "1.0",
              "--gait-start", "0.8", "--bar-start", "0.50", "--bar-max", "0.75",
              "--exam-speed", "2.5", "--exam-dist", "3.0", "--exam-heights", "0.50", "0.60", "0.70",
              "--exam2", "2.0", "3.0",
              "--select", "midstride", "--select-speed", "2.5", "--select-dist", "3.0",
              "--select-heights", "0.50", "0.55", "0.60"],
        finals=[["jump", "--speeds", "2.0", "2.5", "3.0", "--dist", "3.0", "--heights", "0.50", "0.60", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "steer-pilot2": dict(
        # 30 Oct: steer-pilot (weight 0.15, +-1 rad/s) learned no steering in 4.4M steps (steer_mean 0.12-0.16 vs 0.15
        # at the start; trot/canter tracking kept, canter 0.715 -> 0.728 at 85% speed). The pose and paw terms (0.7 of the
        # reward) pay for the straight reference legs, so turning costs more than it earns. Steer term weight 0.6 and
        # the global root terms (the reference path turns with the command, so drifting off it costs too); commands
        # +-0.8 rad/s on most tries. Win: steer_mean >= 0.6, trot >= 0.85, canter >= 0.69, no falls.
        name="steer2", script="train_imitate.py", start="../models/bolt-dog-canter-speed-38M",
        args=["--slots", "8", "--clips", "trot", "canter", "--switch-rate", "0.3", "--lr", "1e-4",
              "--root-vel-weight", "0.6", "--steer-max", "0.8", "--steer-share", "0.8", "--steer-weight", "0.6",
              "--root-global", "0.5"],
        finals=[]),
    "jump-runup90": dict(
        # 30 Oct, owner: every category 90+. runup 88M: 50 cm from 3 m at 2.0/2.5/3.0 m/s 57/63/17% (flat after 88M).
        # Narrow it to the speeds that work (1.8-2.6 m/s, no 3 m/s) and to 50 cm first (practice bars 45-58 cm), so
        # the brain masters one jump before 60 cm. Win: 50 cm at 2.0 and 2.5 m/s >= 80% in the pilot (goal 90%).
        name="jumprunup90", script="train_jump.py", start="../models/bolt-dog-jump-runup-88M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "1.5", "4.0", "--approach-speed", "2.6", "--approach-speed-min", "1.8", "--run-frac", "1.0",
              "--gait-start", "0.8", "--bar-start", "0.45", "--bar-max", "0.58",
              "--exam-speed", "2.5", "--exam-dist", "3.0", "--exam-heights", "0.50", "0.55",
              "--exam2", "2.0", "3.0",
              "--select", "midstride", "--select-speed", "2.25", "--select-dist", "3.0",
              "--select-heights", "0.50", "0.55"],
        finals=[["jump", "--speeds", "2.0", "2.5", "--dist", "3.0", "--heights", "0.50", "0.60", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "canter-fast": dict(
        # 30 Oct, owner: canter 90+. The A1 drive (21 rad/s rated, fades out by 25) caps the canter at ~0.71 tracking
        # (research/results/canter_limit/NOTE.md). A Go2-class drive (~30 rad/s rated, out by 35, 700 W; an assumed,
        # approximate spec) as a separate body track: canter-speed 38M's recipe with the faster motors. Today's brain
        # on the faster motors without retraining: canter 0.631 vs 0.630 (8 tries), so only training can tell.
        # Win: canter tracking >= 0.80 at >= 85% of the dog's speed, trot >= 0.87, no falls.
        name="canterfast", script="train_imitate.py", start="../models/bolt-dog-canter-speed-38M",
        args=["--slots", "8", "--clips", "trot", "canter", "--switch-rate", "0.3", "--lr", "1e-4",
              "--root-vel-weight", "0.6", "--motor-speed", "30", "35", "700"],
        finals=[]),
    "go2-dog-pilot": dict(
        # 30 Oct: Go2 is the target robot (README "Target robot"). First dog-motion brain on the Go2 body: run 18e's
        # recipe (all 9 clips, clip switches 0.3/s, 9-slot layout) from scratch, brake on (the Go2's per-joint brake),
        # 0-40 ms of control delay per try (real_robot_port.md: delay in every training run), exams without delay as
        # Bolt's. Local smoke (300k steps, walk + trot, 2 envs): the reward rose and the exam ran. Pilot win line: mean
        # tracking >= 0.70 over walk/trot/pace/turns/sit/lie/getup with no falls (Bolt 18e at 36M: 0.83-0.93 on those).
        name="go2dog", script="train_imitate.py", start=None,
        args=["--body", "go2", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "40"],
        finals=[]),
    "go2-jump-pilot": dict(
        # 31 Oct, owner's Go2 order: motion (done, models/bolt-go2-gpu-241M) -> jumps. First hurdle-jump brain on the Go2
        # body (jump_env body sizes: bars x0.70, distances x0.86 of Bolt's), from scratch like Bolt's first jump brains
        # (jump-6.5M .. 10M), with 0-40 ms control delay per try as every Go2 run. Exam bars 10/14/17/21 cm at 0.47 m.
        # Early gate (~25%): any exam height above 0. Pilot win: jump >= 0.5 on the 10 and 14 cm bars.
        name="go2jump", script="train_jump.py", start=None,
        args=["--body", "go2", "--latency-min-ms", "0", "--latency-max-ms", "40"],
        finals=[]),
    "go2-getup-pilot": dict(
        # 31 Oct, owner's Go2 order: motion -> jumps -> shove recovery. First get-up brain on the Go2 body (getup_env
        # body sizes: heights x0.70, "standing" above 0.22 m, shoves x0.80), from scratch, stand goal (selfright needs
        # a Go2 posture brain), tough falls (--hard: kicks, drops, ice, second kick), 0-40 ms control delay per try.
        # Pilot win: stands up from every start kind of the exam >= 0.5 (Bolt recover-delay 30M: 1.0 on all).
        name="go2getup", script="train_getup.py", start=None,
        args=["--body", "go2", "--goal", "stand", "--hard", "--latency-min-ms", "0", "--latency-max-ms", "40"],
        finals=[]),
    "go2-damp-pilot": dict(
        # 30 Oct: go2-dog-pilot's exact recipe with the Go2 joint damping at 0.5 instead of Menagerie's 2.0 N·m·s/rad (a
        # generic value; at a trot's 10.5 rad/s thigh speed 2.0 eats ~21 of the 23.7 N·m drive). A/B against the 2-h
        # go2dog pilot (mean tracking 0.767, canter 0.32, falls 5.8% at 3.3M). Which damping a real Go2 has is a
        # measurement for the hardware; this says how much the choice matters.
        name="go2damp", script="train_imitate.py", start=None,
        args=["--body", "go2", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "40", "--joint-damping", "0.5"],
        finals=[]),
    "jump-handover": dict(
        # 29 Oct pilot (2M steps, local): practising from REAL MasterMind hand-overs (research/record_handover.py)
        # took the held-out hand-over exam from 81/77/61/31% to 88/87/81/68% at 20/30/40/50 cm (brake on).
        # Win: held-out 30 cm >= 90%, 40 cm >= 85%, 50 cm >= 70% with the still-legs exam not worse than 55M.
        name="jumpho", script="train_jump.py", start="../models/bolt-dog-jump-trot-55M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states.npz", "--handover-rate", "0.6",
              "--exam-speed", "1.7", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50",
              "--exam2", "2.0", "1.5"],
        finals=[["jump", "--speeds", "1.5", "2.0", "--dist", "1.5", "--heights", "0.40", "0.50", "0.60", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "jump-handover-mix": dict(
        # 29 Oct: jump-handover 74M (trained on v6h hand-overs, 1.69 m/s) was better alone but worse inside MasterMind:
        # v8g's gpu3 trot hands over at 1.50 m/s (research/results/handover_states_v8g*.npz). Practise from both sets
        # (handover_states_mix.npz: 400 v6h + 400 v8g). Verdict on handover_states_v8g_test.npz AND the v8g teacher
        # chain; pick on handover_states_v8g_dev.npz. 59M on v8g test: 85/90/85/70% at 20/30/40/50 cm.
        name="jumphomix", script="train_jump.py", start="../models/bolt-dog-jump-handover-74M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states_mix.npz", "--handover-rate", "0.7",
              "--exam-speed", "1.5", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50",
              "--exam2", "1.7", "1.1"],
        finals=[["jump", "--speeds", "1.5", "1.7", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "jump-gaps": dict(
        # 29 Oct: every real-body gap that hurt the jump (realbody_stress: delay 20 ms, motor lag 20 ms; lost packets)
        # drawn per try, from the hand-over brain 59M, on both hand-over sets (v6h and v8g trots). The in-training exams
        # run with 20 ms delay so best/ is picked under delay. Verdict: handover_exam on handover_states_v8g_test.npz at
        # --latency-ms 20 and at 0 (59M: 0 ms 85/90/85/70%, 20 ms ~0%), then the teacher chain.
        name="jumpgaps", script="train_jump.py", start="../models/bolt-dog-jump-handover-59M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states_mix.npz", "--handover-rate", "0.7",
              "--latency-min-ms", "0", "--latency-max-ms", "30", "--motor-lag-max-ms", "25", "--drop-max", "0.1",
              "--exam-latency-ms", "20",
              "--exam-speed", "1.5", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50"],
        finals=[["jump", "--speeds", "1.5", "1.7", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake", "--latency-ms", "20"],
                ["jump", "--speeds", "1.5", "1.7", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "jump-dyndelay": dict(
        # owner 30 Oct: "delay should be a dynamic value". The delay sweep (research/results/delay_sweep/) showed every jump
        # brain is good only near the delays it trained on (93M: 0 ms; delay 76M: 10-20 ms; jump-gaps: 20-30 ms, worst 27%).
        # One brain for all of it: delay 0-40 ms per try, wobbling +- up to 10 ms per step, spikes, motor lag, lost
        # packets; continues jump-gaps (same run folder) via --resume-kernel. Judge with research/delay_sweep.py: worst and
        # average over 0-40 ms steady and jittered (jump-gaps 76.8M: worst 27%, average 53%; delay 76M: 8% / 53%).
        name="jumpgaps", script="train_jump.py", start="../models/bolt-dog-jump-handover-59M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states_mix.npz", "--handover-rate", "0.7",
              "--latency-min-ms", "0", "--latency-max-ms", "40", "--latency-jitter-ms", "10", "--latency-spike-p", "0.03",
              "--motor-lag-max-ms", "25", "--drop-max", "0.1",
              "--exam-latency-ms", "20", "--exam-jitter-ms", "8",
              "--exam-speed", "1.5", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50"],
        finals=[["jump", "--speeds", "1.5", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake", "--latency-ms", "20", "--jitter-ms", "8"],
                ["jump", "--speeds", "1.5", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "jump-dyndelay2": dict(
        # 30 Oct: jump-gaps 80.5M won the delay sweep alone (avg 58%, worst 34%) but lost inside the 20 ms chain (jumps
        # 17/15/14/5 vs 26/22/15/5 of 30): a delayed trot hands over much messier (heading sd 0.24-0.34 rad vs 0.04,
        # off line up to 27 cm). Practise also from hand-overs recorded in the v10d chain at 0-40 ms
        # (handover_states_mix2.npz = mix + 383 delayed). Continues jump-dyndelay (same run folder).
        name="jumpgaps", script="train_jump.py", start="../models/bolt-dog-jump-handover-59M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states_mix2.npz", "--handover-rate", "0.75",
              "--latency-min-ms", "0", "--latency-max-ms", "40", "--latency-jitter-ms", "10", "--latency-spike-p", "0.03",
              "--motor-lag-max-ms", "25", "--drop-max", "0.1",
              "--exam-latency-ms", "20", "--exam-jitter-ms", "8",
              "--exam-speed", "1.5", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50"],
        finals=[["jump", "--speeds", "1.5", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake", "--latency-ms", "20", "--jitter-ms", "8"]]),
    "jump-delayaware": dict(
        # 30 Oct, owner: "any delay should be recognised by robot and act accordingly". jump-dyndelay2's exact recipe
        # (the control arm, running alongside) plus one input: the delay Bolt measures (packet timestamps, +-2 ms clock
        # error; bolt_env.measured_delay_ms). From jump-gaps 80.5M widened 51 -> 52 inputs (the new input starts at
        # zero weight: 73/96 at 40 cm, 20 ms +-8, the same as 80.5M). Win: delay_sweep average and worst both above
        # 80.5M's (58% / 34%) and above jump-dyndelay2's, and 0 ms not below 80.5M's 45%.
        name="jumpaware", script="train_jump.py", start="../models/bolt-dog-jump-gaps-80.5M",
        args=["--real", "--delay-input", "--widen-from", "../models/bolt-dog-jump-gaps-80.5M",
              "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states_mix2.npz", "--handover-rate", "0.75",
              "--latency-min-ms", "0", "--latency-max-ms", "40", "--latency-jitter-ms", "10", "--latency-spike-p", "0.03",
              "--motor-lag-max-ms", "25", "--drop-max", "0.1",
              "--exam-latency-ms", "20", "--exam-jitter-ms", "8",
              "--exam-speed", "1.5", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50"],
        finals=[["jump", "--speeds", "1.5", "--dist", "1.1", "--heights", "0.30", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake", "--latency-ms", "20", "--jitter-ms", "8"]]),
    "steer-pilot": dict(
        # 30 Oct, Bolt Run: Bolt cannot turn at a trot (steering exam: 38M follows a turn command 0.15 of 1). From
        # canter-speed 38M (its recipe: trot + canter, switches 0.3/s, body-velocity weight 0.6, lr 1e-4) plus a
        # yaw-rate command of +-1 rad/s. Win: steer_mean >= 0.6 with trot tracking >= 0.85, canter >= 0.69, no falls.
        name="steer", script="train_imitate.py", start="../models/bolt-dog-canter-speed-38M",
        args=["--slots", "8", "--clips", "trot", "canter", "--switch-rate", "0.3", "--lr", "1e-4",
              "--root-vel-weight", "0.6", "--steer-max", "1.0"],
        finals=[]),
    "jump-delay": dict(
        # owner 29 Oct: "train delay into everything". At 20 ms the jump brain cleared 0/100 real hand-overs; a local
        # pilot with 0-30 ms delay moved it to 8/100 (30 cm) after 0.5M steps. Win: held-out real hand-overs at
        # 20 ms 30/40 cm >= 70/60%, without delay not below 85/80% (now 90/87%).
        name="jumpdelay", script="train_jump.py", start="../models/bolt-dog-jump-handover-59M",
        args=["--real", "--lr", "1e-4", "--target-kl", "0.015", "--n-steps", "2048", "--max-steps", "200",
              "--approach", "0.8", "2.0", "--approach-speed", "2.4", "--approach-speed-min", "1.2", "--run-frac", "0.85",
              "--gait-start", "0.7", "--bar-start", "0.30", "--bar-max", "0.65",
              "--handover-file", "research/results/handover_states.npz", "--handover-rate", "0.6",
              "--latency-min-ms", "0", "--latency-max-ms", "30",
              "--exam-speed", "1.7", "--exam-dist", "1.1", "--exam-heights", "0.30", "0.40", "0.50"],
        finals=[["jump", "--speeds", "1.5", "2.0", "--dist", "1.5", "--heights", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake", "--latency-ms", "20"],
                ["jump", "--speeds", "1.5", "2.0", "--dist", "1.5", "--heights", "0.40", "0.50", "--gait",
                 "--max-steps", "220", "--motor-brake"]]),
    "recover-delay": dict(
        # 17e stood up ~85% without delay, ~68% at 20 ms (fair_exam recover, 10 tries per fall, seed 9000).
        # Win: at 20 ms >= 85% stood up, without delay not below 17e.
        name="recoverdelay", script="train_getup.py", start="../models/bolt-dog-selfright-17e-16.5M",
        args=["--goal", "selfright", "--hard", "--full-exam", "--lr", "1e-4", "--target-kl", "0.015",
              "--latency-min-ms", "0", "--latency-max-ms", "30"],
        finals=[["recover", "--latency-ms", "20"], ["recover"]]),
    "recover-gaps": dict(
        # 30 Oct: recover-delay 30M stood up 87.3% / 74.5% (0 / 20 ms, 180 tries) vs 17e 81.7 / 63.7. Next: every real-body
        # gap that matters (delay 0-30 ms, motor lag 0-25 ms, 10% lost packets), exams at 20 ms so best/ is picked under
        # delay, the motor brake (as trained). Win: >= 85% stood up at 20 ms, not below 30M without delay.
        name="recovergaps", script="train_getup.py", start="../models/bolt-dog-recover-delay-30M",
        args=["--goal", "selfright", "--hard", "--full-exam", "--lr", "1e-4", "--target-kl", "0.015",
              "--latency-min-ms", "0", "--latency-max-ms", "30", "--motor-lag-max-ms", "25", "--drop-max", "0.1",
              "--exam-latency-ms", "20"],
        finals=[["recover", "--latency-ms", "20", "--motor-brake"], ["recover", "--motor-brake"]]),
    "recover-dyndelay": dict(
        # owner 30 Oct: one recovery brain for any delay: 0-40 ms per try, wobbling +- up to 10 ms per step, spikes, motor
        # lag, lost packets; exams at 20 ms with jitter. From recover-delay 30M (stood up 87.3 / 74.5% at 0 / 20 ms).
        # Judge with fair_exam recover at 0/10/20/30/40 ms (+ --jitter-ms 8): worst and average must beat 30M's.
        name="recoverdyn", script="train_getup.py", start="../models/bolt-dog-recover-delay-30M",
        args=["--goal", "selfright", "--hard", "--full-exam", "--lr", "1e-4", "--target-kl", "0.015",
              "--latency-min-ms", "0", "--latency-max-ms", "40", "--latency-jitter-ms", "10", "--latency-spike-p", "0.03",
              "--motor-lag-max-ms", "25", "--drop-max", "0.1", "--exam-latency-ms", "20", "--exam-jitter-ms", "8"],
        finals=[["recover", "--latency-ms", "20", "--jitter-ms", "8", "--motor-brake"], ["recover", "--motor-brake"]]),
    "recover-delayaware": dict(
        # 30 Oct: recover-dyndelay's exact recipe plus the measured delay as input 47 (as jump-delayaware), from
        # recover-delay 30M widened 46 -> 47 (zero-weight column: the same exam results as 30M). Judge with the
        # recovery sweep at 0/10/20/30/40 ms (+ jitter): average and worst must beat 30M (70.8 / 46.7) and recover-dyndelay.
        name="recoveraware", script="train_getup.py", start="../models/bolt-dog-recover-delay-30M",
        args=["--goal", "selfright", "--hard", "--full-exam", "--lr", "1e-4", "--target-kl", "0.015",
              "--delay-input", "--widen-from", "../models/bolt-dog-recover-delay-30M",
              "--latency-min-ms", "0", "--latency-max-ms", "40", "--latency-jitter-ms", "10", "--latency-spike-p", "0.03",
              "--motor-lag-max-ms", "25", "--drop-max", "0.1", "--exam-latency-ms", "20", "--exam-jitter-ms", "8"],
        finals=[["recover", "--latency-ms", "20", "--jitter-ms", "8", "--motor-brake"], ["recover", "--motor-brake"]]),
    "recover": dict(
        name="recover", script="train_getup.py", start="../models/bolt-dog-selfright-17e-16.5M",
        args=["--goal", "selfright", "--hard", "--full-exam", "--lr", "1e-4", "--target-kl", "0.015"],
        finals=[["recover", "--easy"]]),
}

KERNEL = r'''# Bolt robot dog: PPO fine-tuning on a Kaggle CPU (made by train/kaggle/make_train_kernel.py)
import glob, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
T0 = time.time()
CFG = @CFG@
INP = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
ROOT = Path(os.environ.get("BOLT_ROOT", "/tmp/bolt"))
WORK.mkdir(parents=True, exist_ok=True)
log = open(WORK / "kernel.log", "a")


def say(s):
    s = f"[{(time.time() - T0) / 3600:5.2f} h] {s}"
    print(s, flush=True); log.write(s + "\n"); log.flush()


if not os.environ.get("NO_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + CFG["pip"], check=True)
import numpy, torch, mujoco, stable_baselines3  # noqa: E401
say(f"python {sys.version.split()[0]} numpy {numpy.__version__} torch {torch.__version__} mujoco {mujoco.__version__} "
    f"sb3 {stable_baselines3.__version__} cpus {os.cpu_count()}")
tars = sorted(glob.glob(str(INP / "**" / CFG["tar"]), recursive=True))
if not (ROOT / "train" / CFG["script"]).exists():
    assert tars and Path(tars[0]).is_file(), f"no {CFG['tar']} under {INP}"
    with tarfile.open(tars[0]) as t:
        t.extractall(ROOT, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
TRAIN = ROOT / "train"
(WORK / "runs").mkdir(exist_ok=True)
runs = TRAIN / "runs"
if runs.is_symlink() or not runs.exists():
    if runs.is_symlink():
        runs.unlink()
    runs.symlink_to(WORK / "runs", target_is_directory=True)   # every checkpoint lands in /kaggle/working at once
else:
    say(f"warning: {runs} is a real folder, outputs copied at the end only")
RUN = TRAIN / "runs" / CFG["name"]
RUN.mkdir(parents=True, exist_ok=True)
say(f"code in {TRAIN}, run in {RUN.resolve()}")
env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
env.pop("MUJOCO_GL", None)   # no rendering


# chunked runs (owner 30 Oct: short pieces, each continuing the last): copy the previous piece's newest checkpoint
# and best/ from its kernel output (kernel_sources mounts it under the input folder) before training resumes
if CFG.get("resume_kernel"):
    prev = sorted(glob.glob(str(INP / "**" / "runs" / CFG["name"] / "checkpoints" / "policy_*_steps.zip"), recursive=True),
                  key=lambda c: int(Path(c).stem.split("_")[1]))
    prev = [c for c in prev if os.path.exists(c.replace("policy_", "policy_vecnormalize_", 1)[:-4] + ".pkl")]
    assert prev, f"no checkpoint of {CFG['name']} under {INP} (resume_kernel {CFG['resume_kernel']})"
    (RUN / "checkpoints").mkdir(parents=True, exist_ok=True)
    c = prev[-1]
    for f in (c, c.replace("policy_", "policy_vecnormalize_", 1)[:-4] + ".pkl"):
        shutil.copy(f, RUN / "checkpoints" / Path(f).name)
    pb = Path(c).parent.parent / "best"
    if pb.is_dir() and not (RUN / "best").exists():
        shutil.copytree(pb, RUN / "best")
    for f in ("train.log",):
        if (Path(c).parent.parent / f).exists() and not (RUN / f).exists():
            shutil.copy(Path(c).parent.parent / f, RUN / f)
    say(f"continuing piece from {CFG['resume_kernel']}: {Path(c).name}")


def latest():
    cks = glob.glob(str(RUN / "checkpoints" / "policy_*_steps.zip"))
    cks = [c for c in cks if os.path.exists(c.replace("policy_", "policy_vecnormalize_", 1)[:-4] + ".pkl")]
    return max(cks, key=lambda c: int(Path(c).stem.split("_")[1])) if cks else None


rc = 0
for attempt in range(CFG["attempts"]):
    left = CFG["train_h"] - (time.time() - T0) / 3600
    if left < min(0.25, CFG["train_h"] / 4):
        say(f"{left:.2f} h of training time left: no more attempts")
        break
    ck = latest()
    cmd = [sys.executable, "-u", CFG["script"], "--name", CFG["name"], "--steps", str(CFG["steps"]),
           "--envs", str(CFG["envs"]), "--every", str(CFG["every"]), "--hours", f"{left:.3f}"] + CFG["args"]
    cmd += ["--resume", ck] if ck else ["--init-from", CFG["start"], "--exam-first"] if CFG["start"] else []   # None: from scratch
    say(("resume: " if ck else "start: ") + " ".join(cmd[2:]))
    with open(RUN / "train.log", "a") as tl:
        p = subprocess.Popen(cmd, cwd=TRAIN, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True); tl.write(line); tl.flush()
            if time.time() - T0 > CFG["hard_h"] * 3600:
                say("hard time limit: killing training")
                p.kill()
                break
        rc = p.wait()
    say(f"{CFG['script']} exit code {rc}")
    if rc == 0:
        break
# fair exams on fresh seeds (9000+): the best brain and the starting brain, the same tries
best = RUN / "best"
for i, final in enumerate(CFG["finals"]):
    for tag, folder in (("best", best),) + ((("start", Path(CFG["start"])),) if CFG["start"] else ()):
        if (TRAIN / folder / "policy.zip").exists() and time.time() - T0 < CFG["hard_h"] * 3600 - 600:
            cmd = [sys.executable, "-u", "fair_exam.py"] + [final[0], str(folder)] + final[1:] + \
                  ["--episodes", str(CFG["final_episodes"]), "--no-save"]
            say(f"fair exam {i + 1} of the {tag} brain: " + " ".join(cmd[2:]))
            r = subprocess.run(cmd, cwd=TRAIN, env=env, capture_output=True, text=True, timeout=max(60, CFG["hard_h"] * 3600 - (time.time() - T0) - 300))
            (WORK / f"fair_exam_{tag}{'' if i == 0 else f'_{i + 1}'}.txt").write_text(" ".join(cmd[2:]) + "\n" + r.stdout + r.stderr)
            print(r.stdout[-3000:], r.stderr[-2000:], flush=True)
summary = dict(hours=round((time.time() - T0) / 3600, 2), exit_code=rc, latest=latest(), config=CFG)
if (best / "best.json").exists():
    summary["best"] = json.loads((best / "best.json").read_text())
(WORK / "summary.json").write_text(json.dumps(summary, indent=1))
if not runs.is_symlink():
    shutil.copytree(RUN, WORK / "runs" / CFG["name"], dirs_exist_ok=True)
say("done: " + json.dumps(summary.get("best", {}))[:500])
sys.exit(0 if (best / "best.json").exists() else 1)
'''


def build_data(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "--", "train", "models"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split()
    keep = []
    for f in files:
        p = Path(f)
        if (p.parent == Path("train") and p.suffix in (".py", ".xml")) or p.parent == Path("train/dogmotion/clips") or \
           (p.parts[:2] == ("train", "robots") and (p.name.endswith("_nomesh.xml") or p.name == "LICENSE")) or \
           f in ("train/research/results/handover_states.npz", "train/research/results/handover_states_mix.npz",
                  "train/research/results/handover_states_mix2.npz") or \
           (len(p.parts) == 3 and p.parts[0] == "models" and p.parts[1] in MODELS and p.suffix in (".zip", ".pkl", ".md", ".json", ".txt")):
            keep.append(f)
    missing = [m for m in MODELS if not any(f == f"models/{m}/policy.zip" for f in keep)]
    assert not missing, missing
    with tarfile.open(out / TAR, "w") as t:
        for f in keep:
            blob = subprocess.run(["git", "show", f"HEAD:{f}"], cwd=REPO, capture_output=True, check=True).stdout
            info = tarfile.TarInfo(f)
            info.size, info.mode = len(blob), 0o644
            t.addfile(info, io.BytesIO(blob))
    (out / "dataset-metadata.json").write_text(json.dumps(dict(
        id=f"{a.user}/{a.dataset}", title=a.dataset, licenses=[dict(name="CC0-1.0")]), indent=1))
    print(f"{len(keep)} files, {(out / TAR).stat().st_size / 1e6:.1f} MB in {out / TAR}")


def build_kernel(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    job = JOBS[a.job]
    args = list(job["args"])
    if a.n_steps:
        args = [x for i, x in enumerate(args) if not (x == "--n-steps" or (i and args[i - 1] == "--n-steps"))]
        args += ["--n-steps", str(a.n_steps)]
    if a.exam_episodes and job["script"] == "train_getup.py":
        args += ["--exam-episodes", str(a.exam_episodes)]
    cfg = dict(pip=PIP, tar=TAR, script=job["script"], name=job["name"], start=job["start"], args=args,
               steps=a.steps, envs=a.envs, every=a.every or 1_000_000,
               train_h=a.train_h, hard_h=a.hard_h, attempts=3, resume_kernel=a.resume_kernel, finals=job["finals"], final_episodes=a.final_episodes)
    (out / "kernel.py").write_text(KERNEL.replace("@CFG@", repr(cfg)))
    slug = a.slug or f"bolt-{a.job}"
    (out / "kernel-metadata.json").write_text(json.dumps(dict(
        id=f"{a.user}/{slug}", title=slug, code_file="kernel.py", language="python", kernel_type="script",
        is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
        dataset_sources=[f"{a.user}/{a.dataset}"], competition_sources=[],
        kernel_sources=[f"{a.user}/{a.resume_kernel}"] if a.resume_kernel else []), indent=1))
    print(f"kernel in {out}: push with  kaggle kernels push -p {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "kernel"])
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--job", choices=sorted(JOBS), default="jump-trot")
    ap.add_argument("--user", default="arunmallikarjuna")
    ap.add_argument("--dataset", default="bolt-train-data")
    ap.add_argument("--slug", help="kernel name (default bolt-<job>)")
    ap.add_argument("--steps", type=int, default=400_000_000, help="a cap far beyond the session: the time guard ends it")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--every", type=int, help="steps between checkpoints and exams (default 1M)")
    ap.add_argument("--n-steps", type=int, help="override PPO steps per env per update (smoke tests)")
    ap.add_argument("--exam-episodes", type=int, default=15, help="recover: tries per kind in the checkpoint exam")
    ap.add_argument("--final-episodes", type=int, default=30, help="tries per test in the final fair exams")
    ap.add_argument("--train-h", type=float, default=10.6, help="training ends (time guard) this long after the start")
    ap.add_argument("--hard-h", type=float, default=11.6, help="kill anything still running at this time")
    ap.add_argument("--resume-kernel", help="continue from this kernel's output (its newest checkpoint and best/): "
                    "the next piece of a chunked run, e.g. --train-h 3.5 --hard-h 4.2 --slug bolt-<job>-p2")
    a = ap.parse_args()
    build_data(a) if a.what == "data" else build_kernel(a)


if __name__ == "__main__":
    main()
