# Draft: asking a Go2 EDU owner for five minutes (not sent; Arun decides where and whether to post)

Two versions of the same ask. Both are honest about the state: simulation only, AI-assisted, non-commercial motion data.

## Short post (Reddit r/robotics, Unitree Discord, LinkedIn)

**Title:** Open-source dog-mocap brain for the Go2 EDU, simulation only so far. Looking for one owner to run a 5-minute check that never moves the robot

I'm Arun. I direct an open project called Bolt: one learned controller that makes a Go2 EDU walk, trot, canter, sit,
lie down and get up like a dog, trained in MuJoCo from dog motion capture. The code and training were produced with AI
assistance under my direction. I don't own a Go2, so **everything is simulation**: in Unitree's own simulator the
robot program finishes a 41-second routine 10 times out of 10, also with 0-19 ms command delay and 5% lost steps.

I'm not asking anyone to let it drive their robot. The first step is a script that **only listens**: it has no
command publisher. You run it with the robot lying down, tilt the robot and move each leg by hand when asked, and it
checks the IMU frame, joint order, joint signs and timing against what the brain was trained on, then writes a small
report (readings and verdicts, nothing personal).

```
git clone -b claude/robot-build-memory-ju76nt https://github.com/MrArun005/Cool-Websites && cd Cool-Websites
bash train/deploy/setup_go2.sh && source bolt-deps/env.sh && cd train
python deploy/first_run.py --iface <your ethernet> --domain 0
```

If you can run it and send me the report folder, that alone tells me whether the assumptions made in simulation hold on
real hardware. Criticism of the approach is just as welcome. Dog motion data is CC BY-NC, so this is non-commercial.

Repo: https://github.com/MrArun005/Cool-Websites/tree/claude/robot-build-memory-ju76nt

## Email to a lab or club that has a Go2 EDU

Subject: 5-minute read-only check on a Go2 EDU for an open-source learned controller (simulation only so far)

Hello,

I direct a small open-source project, Bolt: a single learned controller for the Unitree Go2 EDU (walk, trot, canter,
sit, lie, get up, fall recovery), trained in MuJoCo from dog motion capture and run through unitree_sdk2. The code and
training were produced with AI assistance under my direction. I have no robot; every result is simulation (Unitree's
simulator: routine 10/10, also with 0-19 ms delay and 5% lost control steps; details and commands in the repo).

Would someone in your group be willing to run one script on your Go2 EDU? It only subscribes to `rt/lowstate`: it
contains no publisher and cannot command the robot. Over about five minutes it asks you to tilt the robot and move
each leg by hand, checks frames, joint order, signs and loop timing, and writes a report I would be grateful to
receive. Anything beyond that (a hanging test at half gains, with your emergency stop in hand) would be entirely your
call, following the staged checklist in `train/deploy/HARDWARE_DAY.md`.

The motion data is CC BY-NC 4.0; the project is non-commercial. I would credit your group in the repo for any help.

Repo: https://github.com/MrArun005/Cool-Websites/tree/claude/robot-build-memory-ju76nt

Thank you,
Arun Mallikarjun
