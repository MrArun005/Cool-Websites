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
MODELS = ["bolt-dog-jump-runup-16i-35.5M", "bolt-dog-jump-trot-55M", "bolt-dog-jump-handover-59M", "bolt-dog-jump-run60-74M", "bolt-dog-jump-handover-74M", "bolt-dog-recover-delay-30M", "bolt-dog-jump-delay-76M", "bolt-dog-jump-gaps-80.5M", "bolt-dog-selfright-17e-16.5M", "bolt-dog-posture-getup-3M", "bolt-dog-canter-speed-38M", "bolt-dog-jump-runup-88M", "bolt-lite3-dog-20.9M", "bolt-lite3-dog-38.8M", "bolt-lite3-dog-61.8M", "bolt-lite3-dog-63.8M", "bolt-lite3-dog-87.3M", "bolt-lite3-dog-112M", "bolt-lite3-dog-steer-candidate-130M", "bolt-lite3-dog-149M", "bolt-lite3-dog-blind-149M", "bolt-lite3-dog-blind-hist-start", "bolt-lite3-dog-hist-penalty-164M", "bolt-lite3-dog-hist-fast-172M", "bolt-lite3-dog-hist-196M", "bolt-lite3-selfright-28M"]
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
    "lite3-dog-pilot": dict(
        # 1 Nov: DEEP Robotics Lite3, the second target robot (robot_bodies "lite3", robots/deeprobotics_lite3). Its first
        # dog-motion brain, exactly go2-dog-pilot's recipe (which led to the Go2 GPU brains): all 9 clips, switches
        # 0.3/s, brake on, 0-40 ms delay, from scratch. Win line as Go2's: mean tracking >= 0.70 with no falls.
        name="lite3dog", script="train_imitate.py", start=None,
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "40"],
        finals=[]),
    "lite3-dog-2": dict(
        # 2 Nov: lite3-dog-pilot's brain (20.9M, mean tracking 0.771, canter 0.36, moving sit/lie and delays not yet
        # learned) trained on a second free CPU session, same recipe, so the GPU trainer starts from a stronger brain.
        # Win: mean tracking >= 0.80 and canter >= 0.45 without losing the others.
        name="lite3dog2", script="train_imitate.py", start="../models/bolt-lite3-dog-20.9M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "40"],
        finals=[]),
    "lite3-dog-3": dict(
        # 2 Nov: lite3-dog-2's brain (38.8M: tracking 0.813, canter 0.49, no falls on the 9 clips) still falls sitting or
        # lying down from a moving start (trot -> sit 3/4, walk -> lie 3/4, trot -> lie 4/4). Half the tries now start
        # moving and are told to sit or lie at once (the Go2 GPU recipe's moving starts), steering +-0.8 rad/s as the Go2.
        # Win: moving trot -> sit and walk -> lie falls <= 1 in 4, mean tracking >= 0.80.
        name="lite3dog3", script="train_imitate.py", start="../models/bolt-lite3-dog-38.8M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "40", "--moving-rate", "0.5", "--steer-max", "0.8"],
        finals=[]),
    "lite3-dog-4": dict(
        # 3 Nov: lite3-dog-3's brain (61.8M: moving sit/lie 0/16 falls, 0/3 falls on DEEP's MJCF) on the corrected leg
        # collision shapes (lite3_nomesh.xml, 3 Nov) with up to 60 ms of control delay: on DEEP's model it still falls
        # 3/3 with 5-15 ms of delay. Win: DEEP-model rehearsal with 5-15 ms delay <= 1/3 falls, plain 0/3, exam tracking
        # >= 0.81, moving sit/lie falls <= 1 in 16.
        name="lite3dog4", script="train_imitate.py", start="../models/bolt-lite3-dog-61.8M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.5", "--steer-max", "0.8"],
        finals=[]),
    "lite3-dog-4b": dict(
        # 3 Nov, A/B with lite3-dog-4: the same, but the Lite3's motors 4x faster (rated speed, cutoff, power), i.e. close
        # to DEEP's simulator, whose motors give 30 N·m at any speed. Question: is the brain's sensitivity to 3-15 ms of
        # delay on DEEP's model the motor model? If 4b survives that delay there and 4 does not, the GPU run's
        # --motor-speed-range is the right fix. Not a robot brain by itself (real motors are speed-limited).
        name="lite3dog4b", script="train_imitate.py", start="../models/bolt-lite3-dog-61.8M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.5", "--steer-max", "0.8",
              "--motor-speed-x", "4"],
        finals=[]),
    "lite3-dog-5": dict(
        # 3 Nov evening: lite3-dog-4 (63.8M, 0-60 ms delay) became the Lite3 brain: 5-15 ms delay on DEEP's model 6/12 ->
        # 4/12 falls, 3-6 ms 1/24 -> 0/24; the faster-motor A/B (4b) did not help. More of what worked: from 63.8M, delay
        # 0-80 ms per try. Win: DEEP-model 5-15 ms falls <= 2/12 with the routine 0/3 and moving sit/lie 0/48.
        name="lite3dog5", script="train_imitate.py", start="../models/bolt-lite3-dog-63.8M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "80", "--moving-rate", "0.5", "--steer-max", "0.8"],
        finals=[]),
    "lite3-dog-6": dict(
        # 4 Nov: from the Lite3 robot brain 87.3M (routine with 3-6 ms delay 1/6, trot->sit 5-15 ms 2/24, every remaining
        # fall in the canter): the canter drawn 2x as often and 0-80 ms of delay. Late checkpoints are tested under delay
        # on DEEP's model (best/ is picked by an exam with no delay). Win: routine with 3-6 ms 0/6 and 5-15 ms <= 2/24.
        name="lite3dog6", script="train_imitate.py", start="../models/bolt-lite3-dog-87.3M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "turn_left", "turn_right", "sit", "lie",
              "getup", "--switch-rate", "0.3", "--latency-ms", "0", "80", "--moving-rate", "0.5", "--steer-max", "0.8"],
        finals=[]),
    "lite3-dog-7": dict(
        # 4 Nov night: fine-tunes from 87.3M at lr 1e-4 have plateaued (dog-5/6, workers 1-6: none beat it on DEEP's
        # model; w4 collapsed past ~105M). A gentler step: lr 3e-5 and target-kl 0.01, the delay DEEP's model needs
        # (0-40 ms), the canter left as is. Win: trot->sit 5-15 ms < 2/24 with the no-canter routine 0/12.
        name="lite3dog7", script="train_imitate.py", start="../models/bolt-lite3-dog-87.3M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "40", "--moving-rate", "0.5", "--steer-max", "0.8",
              "--lr", "3e-5", "--target-kl", "0.01"],
        finals=[]),
    "lite3-dog-8": dict(
        # 4 Nov night: as lite3-dog-7 (lr 3e-5, target-kl 0.01) with the sit drawn 2x and moving starts 0.6: the
        # remaining 87.3M falls under delay are trot -> sit (2/24 twice). Win: trot->sit 5-15 ms <= 1/24 on two runs,
        # no-canter routine 0/12.
        name="lite3dog8", script="train_imitate.py", start="../models/bolt-lite3-dog-87.3M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "sit", "lie",
              "getup", "--switch-rate", "0.3", "--latency-ms", "0", "40", "--moving-rate", "0.6", "--steer-max", "0.8",
              "--lr", "3e-5", "--target-kl", "0.01"],
        finals=[]),
    "lite3-dog-9": dict(
        # 5 Nov: from the Lite3 brain's parent 112M (trot->sit 5-15 ms 0/48 on DEEP's model; its blend with worker 7 is
        # the robot brain). Worker 7 (steer weight 0.25, lr 3e-5) reached tracking 0.800 / steer 0.583 but lost the sits
        # (9/24): the same steering push from 112M, gentler (lr 2e-5), with 112M's moving starts 0.75 kept. Win: our exam
        # steer >= 0.50 with trot->sit 0/24 and the no-canter routine 0/12 under delay on DEEP's model.
        name="lite3dog9", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8",
              "--steer-weight", "0.25", "--lr", "2e-5", "--target-kl", "0.01"],
        finals=[]),
    "lite3-dog-10": dict(
        # 5 Nov: from 112M, the canter (5/12 falls under 3-6 ms on DEEP's model; 87.3M 2/12) drawn 3x, 0-60 ms, lr 2e-5,
        # moving 0.75 kept. Win: canter routine <= 1/12 with trot->sit 0/24 and the no-canter routine 0/12.
        name="lite3dog10", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "2e-5", "--target-kl", "0.01"],
        finals=[]),
    "lite3-dog-11": dict(
        # 5 Nov night: lite3-dog-9 (112M + steer weight 0.25, lr 2e-5) steered best (0.580) but lost sits (trot->sit 3/24
        # under 5-15 ms on DEEP's model); lite3-dog-10 (canter x3) kept sits 0/48 and steered 0.516. The steering push
        # with the sit drawn 2x to hold the sits. Win: steer >= 0.52 with trot->sit 0/24 and the routine 0/12.
        name="lite3dog11", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "sit", "lie",
              "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8",
              "--steer-weight", "0.2", "--lr", "2e-5", "--target-kl", "0.01"],
        finals=[]),
    "lite3-dog-12": dict(
        # 5 Nov night: canter falls under delay on DEEP's model stay (current brain 2/12; dog-10's canter x3 made it worse,
        # 7/12). The canter falls come with 3-6 ms of delay, so: delay drawn only in the range a real robot has, 2-20 ms
        # (not 0-60), canter x2, lr 2e-5, from 112M. Win: canter routine <= 1/12 with sits 0/24 and the routine 0/12.
        name="lite3dog12", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "turn_left", "turn_right", "sit", "lie",
              "getup", "--switch-rate", "0.3", "--latency-ms", "2", "20", "--moving-rate", "0.75", "--steer-max", "0.8",
              "--lr", "2e-5", "--target-kl", "0.01"],
        finals=[]),
    "lite3-dog-13": dict(
        # 6 Nov: lite3-dog-10's recipe (112M + canter x3, lr 2e-5) gave the best steering that keeps the sits (130M:
        # steer 0.516, trot->sit 0/48) but drove the joints past 1.2x rated speed twice as often live on DEEP's simulator
        # (17-25 steps a run vs 7-14). Same recipe with the motors' rated speed, cutoff and power x0.85 in training, so the
        # brain learns to stay inside a slower drive. Win: steer >= 0.50, sits 0/24, routine 0/12, live over-speed <= 14.
        name="lite3dog13", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "2e-5", "--target-kl", "0.01", "--motor-speed-x", "0.85"],
        finals=[]),
    "lite3-dog-14": dict(
        # 6 Nov: as lite3-dog-13 but continuing from the steering candidate 130M itself (lr 1e-5): teach the steerer to
        # slow its joints rather than re-learning the steering. Win: as dog-13.
        name="lite3dog14", script="train_imitate.py", start="../models/bolt-lite3-dog-steer-candidate-130M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "0.85"],
        finals=[]),
    "lite3-dog-15": dict(
        # 6 Nov: the steering candidate 130M over-speeds live on DEEP's simulator (17-25 steps a run over 1.2x rated vs
        # 7-14) but not in our training env (6.6% of steps over rated vs the robot brain's 9.5%): our motor model fades
        # the drive above rated speed, DEEP's model has no speed limit, so training never saw the excess. Train it with a
        # faster drive (x1.3, closer to DEEP's) AND the new over-speed penalty (imitate_env overspeed_weight), lr 1e-5.
        # Win: live over-speed <= 14 with steer >= 0.50, sits 0/24, routine 0/12.
        name="lite3dog15", script="train_imitate.py", start="../models/bolt-lite3-dog-steer-candidate-130M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.3",
              "--overspeed-weight", "0.5"],
        finals=[]),
    "lite3-dog-16": dict(
        # 6 Nov: lite3-dog-15's idea (a DEEP-like faster drive + the over-speed penalty) from the sit-safe 112M instead of
        # the steerer, with a mild steering push (weight 0.2), lr 2e-5. Win: steer >= 0.50, live over-speed <= 14, sits
        # 0/24, routine 0/12.
        name="lite3dog16", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8",
              "--steer-weight", "0.2", "--lr", "2e-5", "--target-kl", "0.01", "--motor-speed-x", "1.3",
              "--overspeed-weight", "0.5"],
        finals=[]),
    "lite3-dog-17": dict(
        # 6 Nov: the stronger form of lite3-dog-15: 130M with a x1.5 drive and overspeed weight 1.0, lr 1e-5. Win: as 15.
        name="lite3dog17", script="train_imitate.py", start="../models/bolt-lite3-dog-steer-candidate-130M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-18": dict(
        # 7 Nov: lite3-dog-17's strong over-speed fix (drive x1.5, overspeed weight 1.0) gave 149M: sits 0/24, routine
        # 0/12, steer 0.534, live over-speed 5-12. The same fix from the sit-safe 112M with a mild steering push (0.2),
        # lr 1e-5 (dog-16 at x1.3 / 0.5 / lr 2e-5 lost sits 1-2/24). Win: as dog-17's 149M, sits 0/48.
        name="lite3dog18", script="train_imitate.py", start="../models/bolt-lite3-dog-112M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8",
              "--steer-weight", "0.2", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-19": dict(
        # 7 Nov: lite3-dog-17 again from 130M with a stronger penalty (1.5) and a faster drive (x1.7): does more of the
        # fix keep helping? Win: as dog-17's 149M with live over-speed lower still.
        name="lite3dog19", script="train_imitate.py", start="../models/bolt-lite3-dog-steer-candidate-130M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.7",
              "--overspeed-weight", "1.5"],
        finals=[]),
    "lite3-dog-20": dict(
        # 7 Nov: from the robot brain 149M (sits 0/48, steer 0.534, live over-speed 5-12, but canter routine 9/12 under
        # 3-6 ms on DEEP's model). The open goal is the canter: drawn 3x, the over-speed fix kept (drive x1.5, weight 1.0),
        # lr 1e-5. Win: canter routine <= 2/12 with sits 0/48, routine 0/24, live over-speed <= 12, steer >= 0.50.
        name="lite3dog20", script="train_imitate.py", start="../models/bolt-lite3-dog-149M",
        args=["--body", "lite3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right",
              "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-21": dict(
        # 9 Nov: from the blind robot brain (distill_blind.py copy of 149M: no velocity input; on DEEP's model with the
        # leg estimate it falls 0/24 where 149M fell 2-7/24, canter routine 1/24, but steered canters fall 4 of 8 in our
        # exam, steer_canter 0.255 against 0.433). PPO with --blind-vel, canter drawn 3x, 149M's settings. Win: steered
        # canter falls <= 1/8 and steer_canter >= 0.40 with every 0 of the blind brain's record kept.
        name="lite3dog21", script="train_imitate.py", start="../models/bolt-lite3-dog-blind-149M",
        args=["--body", "lite3", "--blind-vel", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left",
              "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-22": dict(
        # 9 Nov: lite3-dog-21 with the steering reward weighted (0.2, as dog-18). Win: as dog-21.
        name="lite3dog22", script="train_imitate.py", start="../models/bolt-lite3-dog-blind-149M",
        args=["--body", "lite3", "--blind-vel", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left",
              "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0", "--steer-weight", "0.2"],
        finals=[]),
    "lite3-dog-23": dict(
        # 9 Nov: lite3-dog-21 at lr 3e-5: does a faster rate move the canter within one session? Win: as dog-21.
        name="lite3dog23", script="train_imitate.py", start="../models/bolt-lite3-dog-blind-149M",
        args=["--body", "lite3", "--blind-vel", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left",
              "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75",
              "--steer-max", "0.8", "--lr", "3e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-24": dict(
        # 10 Nov: dog-21..23 (blind PPO, one frame) gave nothing on fresh seeds; the cause was no speed information in one
        # frame. From the blind brain with 3 frames of history (distill_hist.py; fresh-seed steered canter 0.35-0.39
        # with 5-6 falls of 24, live over-speed 8-12): PPO with --history 3 --blind-vel, 149M's settings. Win, judged on
        # FRESH seeds (ws/steer_fresh.py), not best/'s exam: steered canter falls <= 3/24, stress tests all 0, live
        # over-speed <= 7.
        name="lite3dog24", script="train_imitate.py", start="../models/bolt-lite3-dog-blind-hist-start",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-25": dict(
        # 10 Nov: lite3-dog-24 with the over-speed penalty doubled (the history brain runs its joints faster live than the
        # blind brain: 8-12 steps over 1.2x rated against 4-7). Win: as dog-24.
        name="lite3dog25", script="train_imitate.py", start="../models/bolt-lite3-dog-blind-hist-start",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "2.0"],
        finals=[]),
    "lite3-dog-26": dict(
        # 10 Nov: lite3-dog-24 at lr 3e-5. Win: as dog-24.
        name="lite3dog26", script="train_imitate.py", start="../models/bolt-lite3-dog-blind-hist-start",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "3e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "1.0"],
        finals=[]),
    "lite3-dog-27": dict(
        # 11 Nov: dog-25 (history, over-speed penalty 2.0) is the best candidate: stress all 0, canter routine 0/36, live
        # over-speed 5-9, fresh-seed steered canter 0.408 / 6 falls of 24; its flaw is 2 plain-trot falls of 76 in our
        # exam (robot brain 0). On from it with the trot drawn 3x. Win: trot falls 0 of 64 fresh tries, the rest kept.
        name="lite3dog27", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-penalty-164M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "trot", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5",
              "--target-kl", "0.01", "--motor-speed-x", "1.5", "--overspeed-weight", "2.0"],
        finals=[]),
    "lite3-dog-28": dict(
        # 11 Nov: dog-26 (lr 3e-5) has the best steered canter (fresh seeds 0.457, 1 fall of 24) but live over-speed
        # 13-21. On from it with the penalty at 3.0. Win: live over-speed <= 8 with steered canter falls <= 2/24, stress 0.
        name="lite3dog28", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-fast-172M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5",
              "--target-kl", "0.01", "--motor-speed-x", "1.5", "--overspeed-weight", "3.0"],
        finals=[]),
    "lite3-dog-29": dict(
        # 11 Nov: dog-25's brain at dog-26's rate (lr 3e-5), penalty 2.0. Win: steered canter falls <= 2/24, the rest kept.
        name="lite3dog29", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-penalty-164M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter", "turn_left", "turn_right", "sit", "lie", "getup",
              "--switch-rate", "0.3", "--latency-ms", "0", "60", "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "3e-5",
              "--target-kl", "0.01", "--motor-speed-x", "1.5", "--overspeed-weight", "2.0"],
        finals=[]),
    "lite3-dog-30": dict(
        # 11 Nov: on from the robot brain hist-196M (dog-28's last checkpoint: fresh steered canter 0.501 / 1 fall of 24,
        # stress 0, live over-speed 3-12) with the same settings: it was still improving when the session ended. Win,
        # on fresh seeds: steered canter falls 0/24 and exam-seed falls <= 1/8, live over-speed <= 8, everything kept.
        name="lite3dog30", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-196M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "3.0"],
        finals=[]),
    "lite3-dog-31": dict(
        # 11 Nov: lite3-dog-30 with the over-speed penalty at 4.0 (3.0 brought live over-speed from 13-21 to 3-12).
        name="lite3dog31", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-196M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "4.0"],
        finals=[]),
    "lite3-dog-32": dict(
        # 11 Nov: lite3-dog-30 with the steering reward at 0.3 (default 0.15): the steered canter is the open part.
        name="lite3dog32", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-196M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "3.0", "--steer-weight", "0.3"],
        finals=[]),
    "lite3-dog-33": dict(
        # 12 Nov: dog-30..32 (more of the same from hist-196M) gave nothing: fresh steered canter 3-6 falls of 24 against
        # 196M's 1 (and 1 of 24 on a second seed). Probe on 196M, 24 tries per rate: canter asked for +0.5 / +1.0 rad/s
        # turns 0.41 / 0.61 with 0 falls; asked for -0.5 / -1.0 it turns -0.07 / -0.32 with 2 falls each. The canter
        # turns left only. Here right turns are drawn in 7 of 10 steered canters. Win: canter at -0.5 reaches <= -0.25
        # with 0 falls of 24, left turns and everything else kept.
        name="lite3dog33", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-196M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "3.0", "--steer-canter-right", "0.7"],
        finals=[]),
    "lite3-dog-34": dict(
        # 12 Nov: lite3-dog-33 with the steering reward at 0.3 as well.
        name="lite3dog34", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-196M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "1e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "3.0", "--steer-canter-right", "0.7", "--steer-weight", "0.3"],
        finals=[]),
    "lite3-dog-35": dict(
        # 12 Nov: lite3-dog-33 with right turns in 9 of 10 steered canters and lr 3e-5.
        name="lite3dog35", script="train_imitate.py", start="../models/bolt-lite3-dog-hist-196M",
        args=["--body", "lite3", "--blind-vel", "--history", "3", "--clips", "walk", "trot", "pace", "canter", "canter", "canter",
              "turn_left", "turn_right", "sit", "lie", "getup", "--switch-rate", "0.3", "--latency-ms", "0", "60",
              "--moving-rate", "0.75", "--steer-max", "0.8", "--lr", "3e-5", "--target-kl", "0.01", "--motor-speed-x", "1.5",
              "--overspeed-weight", "3.0", "--steer-canter-right", "0.9"],
        finals=[]),
    "lite3-selfright-pilot": dict(
        # 2 Nov: go2-selfright-pilot's recipe on the Lite3 (getup_env sizes: heights x0.80, kicks x0.90, standing above
        # 0.22 m): selfright goal, from scratch, tough falls, 0-40 ms delay. Win: righted from side and back >= 0.5.
        name="lite3selfright", script="train_getup.py", start=None,
        args=["--body", "lite3", "--goal", "selfright", "--hard", "--latency-min-ms", "0", "--latency-max-ms", "40"],
        finals=[]),
    "lite3-selfright-2": dict(
        # 4 Nov: the Lite3 self-right brain (28M) was trained before the leg collision shapes were corrected (3 Nov:
        # thicker thighs, shanks down to the foot). Live in DEEP's simulator, after a fall from a trot, it twice got stuck
        # at ~57 degrees for 4-8 s. Fine-tune it on the corrected body with tough falls and 0-40 ms delay. Win: still
        # 10/10 from side, back and tipped, and no stuck tries in lockstep_recovery on DEEP's model.
        name="lite3selfright2", script="train_getup.py", start="../models/bolt-lite3-selfright-28M",
        args=["--body", "lite3", "--goal", "selfright", "--hard", "--latency-min-ms", "0", "--latency-max-ms", "40"],
        finals=[]),
    "go2-jump-pilot": dict(
        # 31 Oct, owner's Go2 order: motion (done, models/bolt-go2-gpu-241M) -> jumps. First hurdle-jump brain on the Go2
        # body (jump_env body sizes: bars x0.70, distances x0.86 of Bolt's), from scratch like Bolt's first jump brains
        # (jump-6.5M .. 10M), with 0-40 ms control delay per try as every Go2 run. Exam bars 10/14/17/21 cm at 0.47 m.
        # Early gate (~25%): any exam height above 0. Pilot win: jump >= 0.5 on the 10 and 14 cm bars.
        name="go2jump", script="train_jump.py", start=None,
        args=["--body", "go2", "--latency-min-ms", "0", "--latency-max-ms", "40"],
        finals=[]),
    "go2-jump-pilot2": dict(
        # 1 Nov: go2-jump-pilot (still starts, 18M steps, 10.6 h) cleared 0 bars at every height: it learned to hop in
        # place (body 0.19 -> 0.48 m) without moving forward (0.05-0.30 m of the 0.39-0.56 m to the bar). jump-trot's
        # recipe instead, Go2-sized: 85% of tries start running at the bar (0.8-1.8 m/s; the Go2 trot clip is 1.4), 70%
        # of those mid-stride, bar 0.6-1.5 m away; exams at 1.2 m/s from 1.0 m. From scratch, 0-40 ms delay.
        # Gate (end of the 3.5 h piece): any bar > 0; win: >= 0.5 on 10 and 14 cm.
        name="go2jump2", script="train_jump.py", start=None,
        args=["--body", "go2", "--latency-min-ms", "0", "--latency-max-ms", "40", "--max-steps", "200",
              "--approach", "0.6", "1.5", "--approach-speed", "1.8", "--approach-speed-min", "0.8", "--run-frac", "0.85",
              "--gait-start", "0.7", "--exam-speed", "1.2", "--exam-dist", "1.0",
              "--exam-heights", "0.10", "0.14", "0.17", "0.21"],
        finals=[]),
    "go2-jump-pilot3": dict(
        # 1 Nov: pilot 2 (running starts) refused the bar: it ran up, stopped ~0.3 m short (research/results/go2_jump/
        # PIECES.md). The coach's floor (7 cm on the Go2) is already a real jump for it, so it never met a bar it could
        # step over. Pilot 2's recipe + --bar-floor 0.02 --bar-start 0.03: start at stepping-over bars, the coach raises
        # them as clean crossings come. Gate (end of 3.5 h): any exam bar > 0, or the coach's bar above 7 cm.
        name="go2jump3", script="train_jump.py", start=None,
        args=["--body", "go2", "--latency-min-ms", "0", "--latency-max-ms", "40", "--max-steps", "200",
              "--approach", "0.6", "1.5", "--approach-speed", "1.8", "--approach-speed-min", "0.8", "--run-frac", "0.85",
              "--gait-start", "0.7", "--exam-speed", "1.2", "--exam-dist", "1.0",
              "--exam-heights", "0.10", "0.14", "0.17", "0.21", "--bar-floor", "0.02", "--bar-start", "0.03"],
        finals=[]),
    "go2-getup-pilot": dict(
        # 31 Oct, owner's Go2 order: motion -> jumps -> shove recovery. First get-up brain on the Go2 body (getup_env
        # body sizes: heights x0.70, "standing" above 0.22 m, shoves x0.80), from scratch, stand goal (selfright needs
        # a Go2 posture brain), tough falls (--hard: kicks, drops, ice, second kick), 0-40 ms control delay per try.
        # Pilot win: stands up from every start kind of the exam >= 0.5 (Bolt recover-delay 30M: 1.0 on all).
        name="go2getup", script="train_getup.py", start=None,
        args=["--body", "go2", "--goal", "stand", "--hard", "--latency-min-ms", "0", "--latency-max-ms", "40"],
        finals=[]),
    "go2-selfright-pilot": dict(
        # 1 Nov: go2-getup-pilot (stand goal, 23M steps, 10.6 h) never got up from the side or back (0.0 at every
        # checkpoint): the stand goal moves the joints only ACTION_SCALE around the stance, too little to roll a body
        # over. Bolt's way instead: the selfright goal (full motor range; belly down, legs tucked, still), then the
        # motion brain's getup clip stands it up (models/bolt-go2-gpu-489M: getup 0.93). From scratch, tough falls,
        # 0-40 ms delay. Gate (end of the 3.5 h piece): righted from side or back > 0; win: >= 0.5 from both.
        name="go2selfright", script="train_getup.py", start=None,
        args=["--body", "go2", "--goal", "selfright", "--hard", "--latency-min-ms", "0", "--latency-max-ms", "40"],
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
