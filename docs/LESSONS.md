# Lessons from robots that already work

Techniques collected from open-source robots that made the jump from simulation to a real
body, in our own words, so we can learn from them and do better rather than copy them.
Main sources: Pollen Robotics' Microduck training notes
([microduck_rl](https://github.com/pollen-robotics/microduck_rl), Apache-2.0), Open Duck Mini's
sim-to-real guide ([Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini), Apache-2.0),
Petoi Bittle/Nybble, Sesame, and our own training of Bolt.

## 1. The simulation must match the body
- **Weigh what you'll actually build.** Printed parts with infill weigh less than solid CAD
  parts; use the slicer's mass estimate for every part in the simulation model.
- **Measure the motors.** Feed the simulator the real motor's stiffness, damping, friction,
  armature and torque limit (Open Duck uses the BAM tool for this). If simulated motors behave
  differently from real ones, the trained brain fails or moves chaotically on the robot.
- **Model the slop.** Cheap gears have backlash (a couple of degrees of play); simulate it.
- **Check the physics before training.** A target pose must actually be stable: hold it for a
  few seconds from slightly random starts and check the tilt, not just the height. Measure target
  heights from the simulated robot; never guess them. (Microduck lost days to a target 5 mm off.)

## 2. Make the brain robust (domain randomization)
- Vary mass, friction, motor strength, sensor noise, sensor mounting angle and control delay
  every episode, so the brain learns the skill and not one exact simulation.
- Randomization must **reset** every episode, never pile up; a randomizer that slowly drifted
  the centre of mass silently ruined Microduck runs for months.
- Randomization teaches tolerance to random error, not to a fixed offset: a crooked IMU must be
  calibrated on the robot.

## 3. Training setup that transfers
- Control at **50 Hz**. Train with PPO. Export to **ONNX** and run it on the robot's computer.
- **Bake the observation normalizer into the exported model.** Simulation playback applies it
  anyway and hides the bug; a hand-converted model without it fails on hardware.
  (Our `train/package.py` does this and checks the export against the original.)
- Whatever filtering or smoothing the brain is trained with, the robot must use exactly the
  same, and vice versa.
- Keep one input layout shared by all skills (walk, sit, trick) so the robot can swap brains
  instantly; skills that don't use an input get zeros, but the input stays.
- Always run a tiny smoke test (a few iterations) before a long run.

## 4. Reward design: the hard-won rules
- **RL optimizes exactly what you wrote, not what you meant.** Any freedom left open gets
  exploited (whipping instead of rolling, propping up on the head instead of standing). Define
  the trick with hard checks on contacts and orientation, not small nudges.
- **No jackpots.** A reward paid every step once a goal is reached pays the robot to get there
  as violently as possible. Track a target that moves at a fixed rate instead.
- **Never reward being in a bad state** (fallen, low): the robot will lie there and farm it.
  Reward *progress* (getting more upright pays, staying flat pays nothing).
- **Adding rewards together creates compromises.** If leaning halfway earns 80% of every term,
  the robot leans. Multiplying the terms makes any single failure cost everything.
  *This is likely why our Bolt dog never learned sit and lie: standing still already earned most
  of the sum under every command.*
- **Introduce smoothness penalties after the skill exists.** Early on, any cost for trying makes
  "do nothing" the winning strategy.
- **Check the sign of every penalty.** A penalty with the wrong sign becomes a reward for the
  bad behaviour.
- **Gaussian tracking width ≈ the error you still care about.** Too loose and there's no signal;
  too tight on something unavoidable (a heavy head must bob while walking) and the robot freezes.
- **Train the idle command explicitly** (all zeros = "stand still"); random sampling almost never
  produces it. Rare but important commands (turn in place) need their own share of practice.

## 5. Curricula
- Make things harder only after the current skill has consolidated. If a metric drops exactly
  when a stage starts, stretch the stages.
- To learn the last part of a trick, start some episodes near the end of it, so the hard part
  gets practice.
- Resuming the same task keeps its curriculum progress; starting a new task from another task's
  brain should restart the curriculum.

## 6. When to stop a run (our rule)
- **Keep the best checkpoint, not the last one.** Learning wobbles: a skill can appear and then
  be lost again (our run 2 stood on four feet at 1M steps and lost it by 1.5M).
- **Stop when there's no new best for 4 checkpoints in a row** (2M steps here), then fix the cause
  and train again. Don't stop at the first dip; scores naturally go up and down a little.
- This is automatic: `watch.py --patience 4` scores every checkpoint, copies the best to
  `runs/<name>/best/`, and writes `runs/<name>/STOP`, which makes `train.py` stop.
  Then package `runs/<name>/best/`.

## 7. Reading a run
- Total reward can rise from side terms while the actual trick never happens: watch the main
  task term itself.
- **Measure before theorizing.** When a run "fails", test the actual checkpoint headless (exam,
  end-state clusters) before changing rewards. Several "failures" were really early checkpoints
  or a bad success test.
- **Watch the video.** Numbers can pass while the motion looks wrong.
- Report what the rollouts show ("rolls but face-plants 1 in 3"), not "it works".

## 8. Hardware choices that small robots converge on
- **Smart serial bus servos** (Feetech STS3215, Dynamixel XL330) over plain hobby servos: one wire
  for all motors, position and load feedback, tunable stiffness. That feedback is what the brain
  observes.
- Small computer in the robot (Raspberry Pi Zero 2 W / CM-class, Rockchip boards) running the
  ONNX brain at 50 Hz; heavy thinking (speech, language model) can run elsewhere over Wi-Fi.
- An **IMU** for tilt and spin is essential; foot or contact switches help.
- A **screen or LED eyes** is the cheapest way to make a robot feel alive (Sesame, Microduck,
  Agile ONE).
- Frames: 3D-printed PLA, laser-cut wood that slots together (Nybble), or interlocking plastic
  (Bittle); TPU for grippy feet. Keep weight low: it decides what the motors can do.
- **Physics sets the pace.** A 25 cm robot naturally tumbles at 3.5–5.5 rad/s; don't cap speeds
  with human intuitions, punish impacts and thrashing instead.

## 9. Our own experience so far
- 4 CPU cores trained stand and walk in ~3M steps (~1 h); sit and lie never emerged (see the
  reward-composition lesson above).
- Learning wobbles: walk was 100% at 3M, 80% with more falls at 3.5M. Keep every checkpoint and
  pick by exam, not by "latest".
- Filming and testing every checkpoint (`train/watch.py`) made progress and regressions obvious.
- Run 2 (measured poses, multiplied reward, honest exam): sit and lie at 100% from 500k steps,
  stand on four feet at 1M, then lost; walk never appeared because the walk speed term was so
  strict that standing still and trying to move both scored zero (fixed for run 3).
  Best saved as `models/bolt-dog-run2-1M`.
- The run-2 brain drives a hand-size (19 cm, 188 g) copy of the dog with no retraining when the
  body is scaled by the physics rules (`scale_robot.py`, `BoltEnv(scale=0.25)`).
- On 4 CPU cores we get ~6M steps an hour; GPU teams train a walk with ~100M steps. Walking from
  scratch is slow here; imitating a reference gait is the next tool if run 3 doesn't walk.
- Runs 5-6 (walk on the built-in trot): legs coordinated from the first step and the exam said
  100% from 500k steps, but the 3D replay (`train/gaitlab/`) showed what the exam missed: the
  walk was a Z-shaped crouch, the 500k brain put a knee on the floor ~15% of the time, and the
  2M brain fell 9 times in 10 when it took over from the standing brain. **An exam that a bad
  walk can pass is the real bug; fix the exam before training more.**
- Run 7 fixes: a taller, dog-like stance under the trot (front legs straighter than rear; held
  open-loop: body 40 cm, knees 19 cm up, no scrapes, faster, less motor force than the crouch),
  a reward cost per leg with a knee/shin/thigh on the floor, an eased start from the standing
  pose at a random point in the stride, and an exam that fails knee contact, crouching, and
  half its walks start as a hand-over from standing. `supervise.sh` keeps the run going
  (restarts from the latest checkpoint after any crash) with early stopping turned off at the
  owner's request; the best checkpoint is still kept in `runs/<name>/best/`.
- Runs 8-9 (dog gait, tall stand): planning each paw's path (like real dogs and cats) kept the
  lower legs ~20-25° from upright instead of 38-65°. Two hand-over lessons:
  - **Ease into a new skill from the pose that skill practised**, not from the previous brain's
    last motor commands (those push past the legs to hold the weight: 8/10 falls) nor from the
    sagging actual joint angles (9/10). Easing from the dog stance: 0/10.
  - **Test the hand-over with the real brains.** The walk exam handed over from a perfect
    motor-held stand; the 5.5M walk passed it 100% yet sank into a knee-dragging crouch in 7 of 10
    hand-overs from the real standing brain, while the 5M checkpoint did 10/10. Keep the best
    checkpoint by the real test, not the latest. (Next: let the exam hand over from a given brain.)
- Reuse what already works: the old posture brain already sat and lay 100%, so run 9c started
  from it and only had to learn the new stand. Centre a new pose on its own resting point: with
  the tall stand centred on the old crouch, the brain's outputs pinned at their limits and stalled.

## Runs 13-14 and the games: getting up, hurdles

- **A game is a good test.** Bolt's Castle played 40 games headless and showed what the exam never did:
  standing straight up from a sit fell 1 time in 3 (also in Python). Run 13 practised getting up, with no
  spinning allowed, and fell in none of 48 tries. Easing the motor targets instead made it worse: 12 falls in 12.
- **Check a hand-made motion's direction before training on it.** The bounding leap's "spring" put the paws
  in front of the hips, which pushes the body backwards. Run 14 then learned the safe local optimum: hover in
  front of the bar and hop in place (no touch, no fall). My open-loop test had counted those as clean,
  because it only checked for touches and falls, not whether Bolt got past the bar.
- **Look for the easy physics first.** Bolt's belly is ~36 cm up, so only the legs need to clear a bar up to
  25 cm. Lifting the trot's paws near a bar cleared 48 in 48 with no training, while 64 hand-made leaps all failed.
- **Handlers matter as much as brains.** In the agility course, point-to-point steering zigzagged through the
  weaves at 0.3 m/s; a smooth line through the poles went 0.8 m/s with no misses. High-stepping while turning
  hard toppled Bolt twice in 24 rounds; a straight run-up and gentle steering over the bar fixed it.
- **Never move a plain world geom at run time.** MuJoCo missed collisions with a hurdle box moved through
  `model.geom_pos`: a ball dropped on it fell through, and Bolt's legs passed through bars uncounted. That made
  "stepping over 25 cm bars: 48 in 48" and the clean game hurdles fake. Movable things (the bar, the ice) are now
  mocap bodies moved through `data.mocap_pos`; a dropped ball and a bar under the dog both collide. Honest result:
  the trot touches even a 5 cm bar every time (it does not place its paws around the bar), so hurdles wait
  for the jump brain. Test a new physics object with a simple drop test before trusting any result on it.
- **Pay for what you want, not for what comes before it.** The jump brain learned to stand still when standing
  was paid (run 15b), and to dive over and crash when clearing paid more than a fall cost (early 15c). It jumped
  for real once most of the prize waited for a clean landing, touching the bar cost at once, and paws over the
  bar were paid for being above it (a tuck).
- **A coach that raises the bar too fast can break a brain.** Run 15c jumped 15 cm bars clean 11 in 12 at 6.5M;
  its coach then raised the practice bar from 13 to 20 cm in 1M steps, and by 7M it jumped nothing, even 10 cm.
  Keep the best checkpoint, exam on bars the brain can nearly clear (not only ones it cannot), and resume from
  the last good one with smaller steps (run 15d).
- **Start a learned skill the way it was practised.** The jump brain always began from 0.5 s of holding the dog
  stance. Jumping straight after the standing brain fell 1 time in 3 at 15 cm; holding the stance first, 0 in 6.
  Stopping 0.83 m from the bar (training used 0.45-0.65 m) still mostly worked, but stopping at 0.6 m was better.
