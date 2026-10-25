# Hybrid duck: Microduck body + the owner's duck head

## Main robot: every skill, through one runtime (`runtime_hybrid.py`)

The hybrid is now the main robot. `runtime_hybrid.py` turns commands ("walk", "run", "squat", "sit", "swag",
"sleep", "wake", "lie down", "get up", "hop", "jump", "roll", "idle", `say(text)`, `look(yaw, pitch)`) into plans of
brains: Microduck's ONNX policies (unchanged) and scripted moves (`moves_hybrid.py`, parameters in
`moves_hybrid.json` from the cross-entropy searches in `search_moves.py`). It picks the transition from where the duck
is (seated -> stand up with alpha_sitstand first, squatting -> rise first, walking -> stop first, fallen -> velstand
gets up), keeps last_action across brain switches like Microduck's daemon, slews the head command, and drives the
mouth servo from speech (syllables of the text, or `say_audio()` loudness through `../mouth.py`'s MouthSync).

**Results: `test_skills_hybrid.py`, every skill through the runtime on 10 randomized bodies** (seeds 0-9; per body:
mass and inertia x0.95-1.05 per part, trunk and head CoM +-3 mm, floor friction x0.7-1.3, servo kp x0.85-1.15,
battery 6.5-8.2 V, command delay 3-6 physics steps, Microduck's observation noise; `harness.MicroduckSim(randomize=True)`).
Body: `scene_hybrid_allcollisions_noantenna.xml` (every part collides; antennas drawn only). Raw numbers:
`results_skills_hybrid.json`. Re-checked after the game changes to the runtime (drive mode, zero twist on a fall, unstick): get up x2, walk, run, roll, hop still 10/10.

| skill | how | passed | key numbers (median) |
|---|---|---|---|
| idle: stand, look around, talk | velstand + head command (new gaze every 1-2.5 s), jaw from the words | **10/10** | head yaw sweeps 1.4 rad, jaw opens to 0.65 rad (37 deg), tilt < 1.5 deg |
| walk | velstand, command 0.4 + IMU heading hold | **10/10** | **0.17 m/s** straight, 0.01 m/s sideways |
| run | velstand at command **1.35** (trained up to 0.4) + heading hold, 1 s ramp | **10/10** | **0.49 m/s** (1.5: 8/10 at 0.50) |
| squat | scripted: leg IK (soles flat), ankle + hip IMU balance; velstand rises | **10/10** | trunk 5.0 cm down, tilt 5 deg |
| sit, then stand (wake) | alpha_sitstand (flag 1 / 0), then velstand | **10/10** | seat at 6.2 cm |
| swag sit | alpha_sitstand seated + leg offsets (knees splayed, left knee open), head cocked, beak a little open | **10/10** | left knee opens 0.72 rad; stands up after |
| sleep, wake | alpha_sitstand seated, head drooped (neck -0.9 command), slow breathing nod; "wake" stands | **10/10** | neck droops 0.58 rad |
| lie down, get up | settle, squat, searched keyframes fold it forward onto bill, chest and knees; velstand gets up | **10/10** | rests at ~67 deg, gets up in 0.24 s |
| get up, face down | velstand | **10/10** | 0.24 s |
| get up, from its back | velstand | **10/10** | 0.92 s |
| hop / jump (hop after hop) | scripted, searched: crouch, push, release on the knee encoder, legs snap up, land in a crouch | **10/10** | **1.0 hop/s**, both feet off the floor 85 ms per hop (~6 mm), landing ~4 x body weight, tilt < 11 deg |
| forward roll | roulade until it has turned 300 deg and is upright (max 2 s), then velstand | **10/10** | 358 deg, upright after |

Notes, honestly:
- **Walk speed was a measuring problem, not a harness problem.** velstand drifts in yaw (+0.1-0.2 rad/s) so the earlier
  "distance along the start heading" undercounted. Measured along the heading it does 0.18 m/s at 0.4, matching
  Pollen's 0.17. The runtime holds the heading with wz = -1.5 x (IMU-integrated yaw), so it now also walks straight.
- **Run is velstand past its training range** (commands up to 1.35 vs 0.4 trained). It is a fast walk (no flight phase),
  10/10 in sim; on the robot start with 1.2 (0.46 m/s, also 10/10 here, less tilt) before 1.35.
- **Squat**: Microduck's own body_pose z command is not learned by velstand (Pollen's docs: <= 1 mm for -2 cm), so the
  squat is scripted. Holding Microduck's home pose without a brain falls forward in ~1 s, so every scripted leg move
  has IMU balance: ankle += 3 x pitch + 0.1 x pitch rate, hip += pitch (the hip term took the squat from 8/10 to 10/10).
- **Hop**: the XL330 legs can hop, but only with the right timing. A time-based push never left the floor (the knee
  slows down near straight); releasing on the knee encoder and snapping the legs up gives 55-90 ms flights. The search
  (`search_moves.py hop`, 14 generations x 24 candidates x 4 bodies) scored flight time, upright landing and landing
  force. Hops chain from the landing crouch, about 1 per second. Landing peaks are 2-6 x body weight (short contact
  spikes); that is the "soft" we have, there is no compliant landing brain.
- **Swag sit** is the sit brain plus fixed servo offsets, so the brain keeps the balance. Without help 4/10 bodies slowly
  leaned back past 20 deg and fell over when standing up; a guard fades the offsets when the IMU says it leans back
  past 6 deg (they stay at about 90 %).
- **Lie down** ends belly-down resting on the bill, chest and knees at ~67 deg (this head is 14 cm long, the trunk can't
  get flat). The IMU sees peaks of ~3 g in the fold (median), and velstand's face-down get-up takes it from there.
- Microduck's other downloaded brains (alpha_ground_pick, roller, roller_crouch) are in `../microduck_port/policies/`
  but not used (roller ones need the roller feet); ball_kick_left/right are used by duck soccer (`soccer.py`).
- No new policy was trained: every skill works with Microduck's brains plus scripted moves.

Run:
```
PY=/home/user/cool-websites/.venv/bin/python
$PY duckbody/hybrid/runtime_hybrid.py --script "say hello; walk; run; squat; stand; sit; swag; stand; sleep; wake; lie down; get up; jump; roll"
$PY duckbody/hybrid/test_runtime_hybrid.py                 # unit tests, ~10 s
$PY duckbody/hybrid/test_skills_hybrid.py [--skills hop,roll] [--page DIR]   # 10 bodies per skill, ~3 min per group of 4
$PY duckbody/hybrid/search_moves.py hop|swag|lie_down       # the searches (3 processes, 15-25 min each)
$PY duckbody/hybrid/page_data.py DIR                        # live page data from results_skills_hybrid.json
```

## Games: the hurdle course and following a ball (`course.py`, `follow.py`)

### The hardest hurdle game (`course.py`)

A MuJoCo course in front of the duck: a 1 m wide deck (15 cm side walls) over a floor 6 cm lower, so the hole is
a real hole. Stages in order: start, **steps** (up, 30 cm, down), **ramps** (up/flat/down twice), **bars** (three low
bars across), **rough** (random small boxes), **narrow** beam (1 cm high), **hole** (walk round it on a side lane),
**shoves** (velocity kicks in random directions), finish (6.2-6.6 m). Levels:

| level | step | ramps | bars | rough | beam width | hole | shoves |
|---|---|---|---|---|---|---|---|
| 1 | 1 cm | 5 / 10 deg | 0.5, 1.0, 1.5 cm | 0.3-0.8 cm | 18 cm | 20 x 24 cm | 2 x 0.3 m/s |
| 2 | 1.5 cm | 7.5 / 12.5 deg | 1.0, 1.5, 2.0 cm | 0.5-1.2 cm | 15 cm | 25 x 30 cm | 3 x 0.4 m/s |
| 3 | 2 cm | 10 / 15 deg | 1.5, 2.0, 2.5 cm | 0.8-1.6 cm | 12 cm | 30 x 36 cm | 4 x 0.5 m/s |

(velstand was trained on steps up to 1.5 cm, a 1 cm random grid and slopes up to 5.7 deg, so levels 2-3 go past it.)
Score: 1000 x fraction reached + 10 per second left of 90 s at the finish - 50 per fall. A fall costs time:
velstand gets up and the duck walks on. Steering is pure pursuit on the centre line with the **simulator's trunk
position** (on a real course this needs the camera on the course edges or a tag; the yaw part is what the IMU gives).

**Results, 10 randomized bodies per cell (finished / 10; falls summed over the 10 runs)** (`results_course_hybrid.json`):

| level | walk (0.4 everywhere) | course mode | course mode + stumble reflex | median time (reflex) | falls (reflex) | where the non-finishers stopped |
|---|---|---|---|---|---|---|
| 1 | 8/10, 1 fall | **10/10**, 1 fall | **10/10**, 1 fall | 22 s (walk: 43 s) | 1 | walk: bars, hole (timed out) |
| 2 | 0/10 (stalls on the 12.5 deg ramp) | 9/10, 27 falls | **9/10**, 25 falls | 37 s | 25 | bars (1) |
| 3 | 0/10 (2 cm step, 15 deg ramp) | 6/10, 71 falls | **7/10**, 66 falls | 63 s | 66 | ramps, rough, hole (1 each) |

Stages cleared at level 3 with the reflex: steps 10, ramps 9, bars 9, rough 8, narrow 8, hole 7, shoves 7 (of 10).
Falls at level 3 happen on the 15 deg ramp (30), the 2-2.5 cm bars (17), the step (6), rough (6), shoves (5).

What made the difference (all in `runtime_hybrid.py`, no training):
- **Speed per obstacle** (`COURSE_SPEED`, from `course.py --probe`: every obstacle x 5 speed commands x 10 bodies,
  `probe_course_hybrid.json`). Slow walks stall on ramps past 10 deg (0.4: 0/10 at 12.5 deg) and trip on 1.5 cm
  bars (0.4: 3/10, 0.8: 8/10); a run trips on steps and rough ground. So: run 1.35 on the flat, 1.0 on ramps, 0.8 on
  steps / bars / shoves, 0.7 rough, 0.6 beam and hole. That alone takes level 1 from 8/10 in 43 s to 10/10 in 21 s.
- **Turn on the spot** when the heading is off by > 0.7 rad, and **back up** 1 s when a turn is blocked (IMU yaw does
  not move for 1.5 s while turning: a foot caught behind a bar).
- **Get-up fixes** (help every skill): a fall now zeroes the twist (velstand got the walk command while getting up and
  propped itself on step edges), and an **unstick sequence**: 2 s not upright -> the lie-down keyframes, then a 0.8 s
  roulade kick, then a 1.5 s one, velstand after each. Two trap poses it needs: crouched with the head on the floor
  and a foot on a step (tilt 35-50 deg, velstand holds it forever), and lying on its side propped by the flat 14 cm
  head. Twist wiggles, head swings and the sit pose did not free them.
- **Stumble reflex** (`course_r`): tilt > 12 deg -> stand still 0.6 s. Small but real at the hard levels (L2 37 s vs
  43 s, L3 7/10 vs 6/10, fewer falls); no effect at level 1.
- **Hopping over bars does not help**: the searched hop is in place with ~6 mm of flight, and the bars that trip it
  are 1.5 cm and up. Higher foot lift would need training (velstand's foot-clearance target is 2 cm).

### Follow a ball with the head camera (`follow.py`)

Microduck's MJCF `head_camera` ends up inside the owner's longer head shell (and its frame looks backwards), so the sim
adds `eye` on the front of the owner's head above the bill (Pi camera v2 field of view, 62 x 49 deg), rendered
160 x 120 at 10 Hz (OSMesa). Detection is an OpenCV HSV threshold for the magenta ball + largest blob (0.2 ms per
frame here; a few ms on a Pi Zero 2 W). The ray to the ball goes into the trunk frame through the head joints'
kinematics, giving bearing, elevation and (from the pixel radius) distance; the head looks straight at it, the body
turns (wz = 1.5 x bearing) and walks (vx = 1.2 x (distance - 15 cm), at least 0.15, back up if it comes too close),
turns on the spot to search after 0.6 s without it, and hops when it has been within 18 cm of a still ball for 1.5 s.
Ball paths: rolls at 4-12 cm/s with stops, once rolls fast (0.7 m/s) round the duck to 0.5 m behind it, stops for
the last 8 s; 60 s per run, 10 randomized bodies and paths (`results_follow_hybrid.json`):

| ball in view (median) | final distance camera -> ball (median; target 15 cm) | within 25 cm | falls | hops | lost and found again |
|---|---|---|---|---|---|
| **95 %** (58-97 % per run) | **16.8 cm** (trunk -> ball 27 cm) | 9/10 | 0 | 33 | 10/10 runs, median 2.3 s |

Honestly: in 3 runs it took 9-18 s to find the ball again once (it turned the wrong way, or the ball sat close
behind it at the edge of view). The first version servoed the head on the pixel error and integrated on the
*commanded* head angle; with the head's lag it overshot and lost the ball (25 % in view). Next step for real toys:
a tiny CNN detector (96 x 96 int8, FOMO/SSD-style) in place of the colour threshold.

Run:
```
PY=/home/user/cool-websites/.venv/bin/python
$PY duckbody/hybrid/course.py [--levels 1,2,3] [--modes walk,course,course_r]   # ~2 min, 3 processes
$PY duckbody/hybrid/course.py --probe                                           # single obstacles x speeds, ~7 min
MUJOCO_GL=osmesa $PY duckbody/hybrid/follow.py                                   # ~2 min
MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --replays                  # page: hurdles, hurdles_typical, follow
```

## Duck soccer (`soccer.py`)

The duck plays alone on a **2.5 x 1.5 m pitch** with 15 cm boards (6 cm walls kept the ball in, but the get-up rolls
tumbled the duck over them), a **goal 50 cm wide, 15 cm high** (yellow posts and crossbar, light grey net) at one end
and a **7 cm, 20 g ball** with rolling friction (from 1 m/s it rolls ~1.7 m). It finds the ball with its head camera,
walks round behind it, lines it up at one foot, kicks with **Microduck's `ball_kick_left` / `ball_kick_right`**
(unchanged), chases it and repeats; after a goal it hops and the ball goes back to the centre spot. It gets up by
itself after falls (the runtime's get-up). Score: goals in a 2-minute match.

**The kick brains on this duck** (`--probe`, `probe_kick_hybrid.json`; ball placed relative to the kicking foot's
trained spot, 9 cm ahead of the trunk, 4.2 cm to the side; 3 bodies per cell, 0.5 s window as in the manifest):

| ball centre ahead of the trunk | hits (within +-2 cm sideways) | ball speed | rolled in 3 s |
|---|---|---|---|
| 6 cm | **18/18** | 1.19 m/s | 1.7 m |
| 9 cm (trained) | **18/18** | 1.21 m/s | 1.7 m |
| 12 cm | 8/18 | 0.42 m/s | 0.6 m |
| 15, 18 cm | 0/18 | - | - |

Up to +-4 cm sideways still hits (59/60 at 6-9 cm), but steers it: in line with the foot it goes ~12 deg outwards,
2 cm inside ~15 deg inwards, 1 cm inside about straight (the aim). Two things the probe found: the window is only
~4 cm deep, and **the kick brains do nothing at all if the head is not at home**: with the head down (how the camera
sees the ball at the feet) 0/4 kicked; 0.6 s after raising it 2/4; 0.8 s 4/4. So the duck lines up with its head
down, then raises it for 1 s, then kicks. They are blind (as on the robot), so all the aiming is the approach.

How it plays (details in the module docstring):
- **Ball**: magenta blob in the 160 x 120 head camera at 10 Hz; the ray through its centre meets the floor plane ->
  ball position relative to the trunk (head joint kinematics + IMU tilt), ~1 cm at the feet. With the head bent
  down (neck -0.6, head pitch 1.1, inside velstand's trained +-1.1) the camera sees 5-15 cm in front of the feet.
- **Goal** (camera, not given): the lowest yellow pixel at each side of the goal frame = the two post bases. A
  landmark with a known position and direction, so one sighting gives position and heading (+-5 deg close, +-12 deg
  from the far end), blended in 30 %, outliers dropped. In between: heading from the IMU gyro about gravity
  (200 Hz), position from the twist sent to velstand through its measured speed map. Nets had to be a different
  colour from the posts (from far away the back corners looked like posts: 25 deg errors).
- **Behaviour**: search (turn, head nodding from far to the feet) -> approach / chase a point 24 cm behind the ball on
  the line to the goal, orbiting the ball at 28 cm when it is not behind it -> align with the head down (step until
  the ball is 8 cm ahead and 3.2 cm to the side of the trunk, nearer foot, heading within 0.15 rad; velstand does
  not step below ~0.3 forward / 0.4 sideways / 0.5 turning, so it nudges with commands just above that) -> ready
  (head home 1 s) -> kick -> chase. Can't line up in 8 s (ball against a board): barge into it. Turning blocked 1.5 s
  (against a board): back up 1 s.

**Results, 10 randomized bodies and kick-offs, 2 minutes each** (`results_soccer_hybrid.json`):

| goals per match (median, range) | matches with a goal | kicks connected (ball > 0.3 m/s) | falls | ball in view (camera frames, median) |
|---|---|---|---|---|
| **3** (0-4), 25 in all | 9/10 | **36/48** (75 %) | 1 (got up by itself) | 50 % |

4 of the 25 goals were walked or bounced in rather than kicked within 6 s before. Honestly, what does not work yet:
- **Corners**: a ball in a corner next to the goal cannot be got behind (the trunk can't get between it and the
  board), so the duck barges at it until it pops out; the 0-goal match spent most of its 2 minutes there.
- **Misses** (12/48), most likely: the ball rolls or the feet shuffle a few cm while the head comes up (it can't see the ball
  then), and from the 4 cm deep kick window; misses cost a re-approach, not a fall.
- **Slow**: a full cycle (find, go round, align, kick) takes 15-25 s; the alignment nudges take most of it. A walk
  brain that steps in place at small commands, or a kick brain that accepts a head-down pose, would cut it a lot.
- Scoring uses the simulator's ball position (it went in the goal); the duck itself never uses ground truth. The
  sim gyro has no bias; on the robot the goal sightings would also have to correct heading drift (they already can).

Run:
```
PY=/home/user/cool-websites/.venv/bin/python
MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer.py --probe            # kick placement grid, ~1 min, 3 processes
MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer.py --seeds 0-9        # 10 matches, ~5 min, 3 processes
MUJOCO_GL=osmesa $PY duckbody/hybrid/soccer.py --trace 1          # one match with a printed trace
MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --soccer    # page: the soccer replay (the best match, cut to
                                                                  # 60 s when over 3 MB) + tests/games tables
```

## 5-a-side match: 11 hybrids, a referee and a GameController (`match.py`)

Blue (5) against Yellow (5) plus a referee duck in black: every one is the same hybrid (owner's head on Microduck's
body) running the same runtime and Microduck's brains, all in **one MuJoCo model** (11 copies attached with MjSpec
frames, names prefixed `b1_` ... `y5_`, `ref_`), with contacts between ducks on: shoulder-to-shoulder duels, bumps
and falls happen in the physics, and fallen ducks get up by themselves (the runtime's get-up and unstick sequence).

**Pitch** (RoboCup Humanoid KidSize field x 0.5). KidSize robots are 40-90 cm tall on a 9 x 6 m field; this duck is
~25 cm and its steps and kicks (ball rolls ~1.7 m) are about half theirs, while its speed (0.17-0.5 m/s) is like
theirs. So: **4.5 x 3 m**, 2.5 cm white lines, halfway line, 0.75 m centre circle, goal areas 0.5 x 1.5 m, penalty
areas 1 x 2.5 m, penalty marks 0.75 m out, **goals 1.3 m wide, 0.6 m high, 0.3 m deep** (white 5 cm posts and
crossbar, a net of back, sides and roof that collides, so the ball stops in it), a 0.5 m border strip and 15 cm
boards (a venue barrier; RoboCup has none). Ball 7 cm, 20 g (KidSize uses a ~15 cm ball). Turf friction 0.7 with
each duck's own randomized friction (0.7-1.3) on its geoms (MuJoCo takes the larger of the two).

**Physics, same as single-duck.** 5 ms steps, 50 Hz brains, BAM's XL330 m6 servo model for all 11 x 14 servos
(`MultiBam`: BAM's `MujocoController.update()` vectorised over 154 servos, with the voltage drop, kp and battery kept
per robot, and BAM's friction lookup kept exactly as it is per robot), 3-6 step command delay, observation noise,
Microduck's domain randomization per duck. **Check** (`--check`): one duck in this model against
`harness.MicroduckSim`, 4 s of walking and turning: **0.0000 mm** difference. The visible-only meshes are dropped
and the 11 ducks share one copy of the collision meshes (MuJoCo would hold 11 copies: 2.9 GB per process -> 1 GB).

**Speed**: 11 ducks alone run at **3.2 x real time** (physics + BAM 1.2 s, brains 0.3 s per 5 s); with the 4 head
cameras at 10 Hz **0.8-0.86 x real time** on a free CPU (rendering 18 ms per camera frame, OSMesa). The matches
below ran 3 at a time next to other jobs on 4 CPUs: **0.39-0.40 x real time**, ~15.5 min per match (6 min of play
plus stoppages).

**Players** (`Player`, 50 Hz):
- **Roles**: goalkeeper, 2 defenders, 2 attackers. Every 0.5 s each team picks the field player best placed for the
  ball (distance, turn needed, fallen +8, a defender when the ball is in the other half +3, the current one -1.5) as
  striker: only he goes for the ball. The others take spots relative to the ball (attackers 0.5 m behind it, 0.75 m
  to each side; defenders halfway back to their goal) with teammate repulsion (0.4 m). Both teams' strikers go for
  the same ball, so they meet and push.
- **Ball play** (soccer.py's): approach a point behind the ball on the line to the aim, align it at a foot with the
  head down, head home, kick with `ball_kick_left/right`. The aim is the goal (the side away from the keeper); if an
  opponent blocks the line and a teammate further up is open, a **pass** to him; from the own half or by the keeper,
  a clearance. **Duel**: with an opponent within 0.45 m of the ball there is no time to line up: walk through the
  ball towards the aim, with the ball in front of one foot, and kick at once if it is in that foot's kick window
  (6-11 cm ahead, +-3 cm) and the aim within 0.45 rad (0.8 s head-up first); pinned 3 s: step round it.
- **Goalkeeper**: stays in front of its goal on the line ball -> goal centre (0.32 m out), facing the ball; goes and
  clears when the ball is in its penalty area near it (or it is the closest of its team).
- **Team world model** (RoboCup team communication): only the **striker and the goalkeeper render their head
  cameras** (160 x 120, 10 Hz, soccer.py's colour blob and floor ray); they share the ball over "wifi" (5 per second,
  200 ms late, 2 cm noise) and the others play on that shared estimate. Rendering 10 cameras would halve the speed
  again. Honestly: the head cameras draw the pitch, the ball and **box proxies of the ducks** (one box per part, for
  occlusion), not the full meshes (OSMesa: 1.2 s per frame with 11 full ducks, 18 ms with boxes).

**Referee and GameController** (`Referee`, `Game.rules`, on the simulator's truth like the human referee + GameController):
the referee duck walks with play ~0.8 m from the ball at its side (off the line of play), keeps 0.4 m from players,
turns its head to the ball, hops after a goal. Calls: **kick-off** (teams on their spots, opponents outside the
centre circle until the ball moves 15 cm or 10 s), **goal** (ball fully over the line between the posts under the
bar; scorer and teammates within 0.8 m hop; then READY: players walk back to their kick-off spots, 14 s, then SET:
whoever is still off his spot by more than 0.35 m or down is carried there, RoboCup's manual placement), **throw-in**
(ball over the touch line: placed on the line where it went out for the other team of the last toucher, opponents
0.5 m away for 10 s or until it is played), **goal kick / corner** (over the goal line, by who touched it last),
**dropped ball** (ball pinned between ducks 20 s: both teams 0.5 m away for 3 s; our rule), **pick-up** (a duck down
12 s or off the pitch 8 s is taken off for 10 s and put back at the touch line in its own half), two halves of
**3 min playing time** with ends changed at half-time (ducks carried to the new spots). Every call is logged with a
"whistle".

**Results, RESULTS_N matches (seeds RESULTS_SEEDS), 2 x 3 min** (`results_match_hybrid.json`):

RESULTS_TABLE

Falls per duck over all matches: RESULTS_FALLS.

What is realistic, and what isn't:
- Realistic: the physics and servos are exactly single-duck's (checked), every duck uses Microduck's brains through
  the hybrid runtime, ducks collide, push and fall over each other and get up by themselves, the players who need the
  ball use their own head camera, set pieces and pick-ups follow the RoboCup Humanoid League procedures (scaled).
- **Self-localization is simulated**: each duck's pose is the simulator's truth plus a slowly drifting error (5 cm,
  3 deg, 2 s correlation), standing in for the line/goal particle filter a RoboCup robot runs (soccer.py has a real
  one on a single goal; with two identical white goals and 11 ducks that is a project of its own).
- **Opponents' positions** come from the truth with 6 cm noise (stands in for robot detection), used for duels and
  for choosing a pass. Teammates' positions are their own (noisy) estimates, as over team communication.
- On a referee placement (kick-off, throw-in...) the teams are told where the ball is.
- The referee is choreography; its calls are the GameController's, made on the truth.
- Low scoring: shots connect but from 1.5-2 m a 1.2 m/s kick rolls out before the line or the keeper is in the way,
  duels pin the ball for long stretches, and passes don't arrive (the receiver isn't facing it, opponents step in):
  RESULTS_PASSES. Real KidSize games are low scoring too, but these numbers are mostly the slow kick cycle (lining
  up takes seconds) and the blind 0.5 s kick brains.
- Not modelled: the ball leaving the ground, penalty kicks, fouls (pushing is legal here), robot heat / battery drain.

Run:
```
PY=/home/user/cool-websites/.venv/bin/python
MUJOCO_GL=osmesa $PY duckbody/hybrid/match.py --check                       # physics == harness, page FK, speed
MUJOCO_GL=osmesa $PY duckbody/hybrid/match.py --one 0 --out m0.json         # one match per process (~1 GB each)
$PY duckbody/hybrid/match.py --merge m0.json m1.json ...                    # -> results_match_hybrid.json
MUJOCO_GL=osmesa $PY duckbody/hybrid/match.py --trace 0 --t 60 --halves 1   # one short match with a printed trace
MUJOCO_GL=osmesa $PY duckbody/hybrid/page_data.py DIR --match               # page: 90 s of the best match
```
Page replay (`rollout_match.json`, ~2 MB): frames carry per duck the free joint and 15 joint angles (ints), the page
rebuilds every body by forward kinematics from `kin` (checked against MuJoCo: `--check`) and draws the 11 ducks from
one light mesh (`duck_mesh_match.json`) with team colours and numbers, the pitch, a broadcast camera following the
ball, and a scoreboard with the match clock and the referee's last call.

## Microduck's brains on the hybrid head (first study)

**What this is.** A simulated duck built from Pollen Robotics' **Microduck** (legs, hips, trunk, neck, all
14 joints, joint ranges, XL330 servo model, IMU, default pose) with **the owner's duck head on top**: the v3
head shell, bottom sheet, antennas and holders (Open Duck Mini v2 parts at the owner's 0.7 "mid" scale, the
same parts as `../v3/duck_mesh_v3_head60_jaw22.glb`), the owner's orange bill, and a working **mouth**
(lower jaw on a hinge, 0 to 35 degrees, on its own small servo). The question was whether Microduck's
official pre-trained brains (their ONNX policies, unchanged) still work on it.

**Short answer.** Yes, **as long as the antennas can't hit the floor**. With the head mass matched to
Microduck's and the antennas left off (or made soft or detachable), every skill works as on Microduck's own
body. The one weak spot is roulade inside the 1 s window the robot gives it: 7/10 instead of 10/10. With
the 2 s window it is 10/10. With the rigid antennas the duck gets stuck on its back. It gets up from there
only 3/10 times and roulade fails.

![hybrid getting up from its back](getup_back_hybrid_noantenna.png)

`getup_back_hybrid_noantenna.png`: hybrid (no-antenna collision) on its back, velstand brain, recovered in 1.0 s.
`getup_back_fail.png`: the same start with rigid antennas. It props itself on the head and antennas and
stays at 79 degrees of tilt.

## Results (10 randomized starts each, same seeds, harness and success rules as `../microduck_port/evaluate.py`)

| test | Microduck's body | hybrid (rigid antennas) | hybrid, owner mass | **hybrid, no antennas** | hybrid, owner mass, no antennas | hybrid, Microduck head collision (control) | hybrid small head (x2/3) |
|---|---|---|---|---|---|---|---|
| stand (velstand, from standing) | 10/10 | 10/10 | 10/10 | **10/10** | 10/10 | 10/10 | 10/10 |
| get up face down | 10/10 (0.32 s) | 10/10 (0.25 s) | 10/10 | **10/10 (0.25 s)** | 10/10 | 10/10 | 10/10 (0.42 s) |
| get up face down, random joints | 10/10 | 8/10 | 9/10 | **9/10** | 10/10 | 10/10 | 10/10 |
| get up from back (face up) | 10/10 (1.00 s) | **3/10** | **1/10** | **10/10 (0.89 s)** | 10/10 | 10/10 | 10/10 (1.14 s) |
| get up from back, random joints | 10/10 | 3/10 | 4/10 | **10/10** | 9/10 | 10/10 | 10/10 |
| get up from sitting (velstand) | 6/10 | 5/10 | 5/10 | **6/10** | 6/10 | 6/10 | 6/10 |
| sitstand: stand from sitting | 9/10 | 9/10 | 10/10 | **9/10** | 10/10 | 9/10 | 9/10 |
| sitstand: sit, then stand | 10/10 | 10/10 | 10/10 | **10/10** | 10/10 | 10/10 | 10/10 |
| walk vx 0.15 / 0.3 / 0.4 (velstand), no fall | 10/10 x3 | 10/10 x3 | 10/10 x3 | **10/10 x3** | 10/10 x3 | 10/10 x3 | 10/10 x3 |
| walk speed at 0.3 / 0.4 m/s command | 0.053 / 0.111 | 0.061 / 0.121 | 0.074 / 0.138 | **0.060 / 0.120** | 0.076 / 0.139 | 0.059 / 0.122 | 0.059 / 0.123 |
| alpha_walking 0.15, no fall | 10/10 | 10/10 | 10/10 | **10/10** | 10/10 | 10/10 | 10/10 |
| roulade, 1 s window (as the robot runs it) | 10/10 (360 deg) | **0/10** (128 deg) | 0/10 | **7/10 (318 deg)** | 10/10 | 10/10 | 8/10 |
| roulade, 2 s window | 10/10 | 5/10 | 4/10 | **10/10** | 10/10 | 10/10 | 10/10 |
| stand 10 s: mean / max tilt, rate RMS | 0.70 / 0.89 deg, 0.014 rad/s | 0.72 / 0.90, 0.014 | 0.90 / 1.07, 0.015 | **0.72 / 0.89, 0.014** | 0.90 / 1.07, 0.015 | 0.72 / 0.89, 0.014 | 0.72 / 0.88, 0.014 |
| stand + mouth flapping 3 Hz | n/a | 10/10 | 10/10 | **10/10** | 10/10 | 10/10 | 10/10 |
| walk 0.15 + mouth flapping | n/a | 10/10 | 10/10 | **10/10** (see note) | 10/10 | 10/10 | 10/10 |

Raw numbers, per run: `results_hybrid.json`. The `microduck` column reproduces `../microduck_port/results.json`.

**Reading it.**
- **Head mass and CoM aren't the problem.** The owner-mass variant has a 31 g lighter head, CoM 8.8 mm away.
  On Microduck's body the brains don't notice it, apart from a slightly larger standing lean: tilt 0.9 vs
  0.7 degrees. Without antennas it scores like Microduck everywhere, and it even rolls better.
- **Head shape is the problem, specifically the antennas.** The control variant draws the owner's head but
  keeps Microduck's head collision, and it scores exactly like Microduck. Turn on the owner's head hull
  without the antennas and only roulade-in-1 s drops. Add rigid antennas (6 cm rods above the head) and
  the back get-up and roulade break. The duck lies on its back propped on the antennas and the flat,
  14 cm wide head, at about 79 degrees of tilt, and the brain's roll-over move can't get past them.
- **Wobble.** Standing and walking wobble is the same as Microduck's own: tilt under 1 degree, angular-rate
  RMS 0.014 (standing) and 0.037 rad/s (walking). There is no extra shaking. The only "wobble" is that the
  wider, longer head rolls slower: roulade turns 294-361 degrees in its 1 s slot instead of 360, so 3/10
  don't finish the roll in time. The walk brain takes over after the slot and the duck still ends up
  standing (2 s window: 10/10).
- Note: in one variant (no antennas, matched mass), flapping the mouth while walking at 0.15 m/s doubled
  the angular-rate RMS to 0.11 rad/s, with no falls. In the other variants it changed nothing. Treat it as
  a small sensitivity, not a failure.

**Caveats that apply to Microduck's own body too (harness, not the hybrid):**
- "Walk 10/10" means "didn't fall". At a 0.15 m/s command neither Microduck nor the hybrid actually moves
  forward in this harness: about 7 mm in 10 s, stepping in place. At 0.3 / 0.4 m/s they reach 0.05-0.14 m/s,
  and both drift sideways about 0.7-0.9 m over 10 s, measured in the start frame. Pollen's own sim reports
  0.17 m/s at 0.4. Whatever causes this (no backlash model, no yaw remap, and so on), it is the same on
  both bodies, so the comparison holds. Absolute walking speed needs checking on the robot.
- velstand from a sitting start is 6/10 on both bodies. alpha_sitstand is the right brain for that, and it
  scores 9-10/10.

## Mass and centre of mass

The head link is Microduck's `jaw_soft` (everything after `head_roll`) plus the owner's jaw, at the STAND pose
(`head_report.json`):

| | mass | head CoM (world, m) | diff vs Microduck | whole-robot CoM diff |
|---|---|---|---|---|
| Microduck head | 0.1888 kg | (0.0235, 0.0012, 0.2372) | - | - |
| hybrid, matched (default) | 0.1888 kg (head 0.1843 + jaw 0.0045) | identical | 0 g, 0 mm, inertia identical | 0 mm |
| hybrid, owner mass | 0.1580 kg (owner's `duck_mid.xml` estimate) | (0.0286, -0.0027, 0.2431) | **-30.7 g (-16 %)**, **+5.1 / -3.9 / +5.9 mm (8.8 mm)** | +0.2 / -0.9 / -2.8 mm |

Microduck's training randomizes head CoM by about +-3 mm, ramped up to about +-8 mm, and mass/inertia by +-5 %.
The matched variant sits inside that by construction. The owner-mass variant's CoM shift is at the edge
of it, and its -16 % mass is outside it. It still works, as the table shows. Total robot: 0.737 kg
(Microduck) vs 0.707 kg (owner mass).

Size: the owner's head shell is 140 x 139 x 42 mm (L x W x H), 104 mm tall with antennas. Microduck's head is
123 x 92 x 59 mm. Microduck's head electronics (Pi Zero 2 W + hat, camera and lens holder, speaker, mouth
servo, motor support) fill a 111 x 80 x 41 mm box. The owner's shell, centred on Microduck's head, covers
that box in length and width but is **2-3 mm short in height** (bottom 1.6 mm, top 2.8 mm). The x2/3 "small"
head (93 x 92 x 28 mm) works in sim but cannot hold those parts.

## How it is built

`build_hybrid.py` writes every model from two inputs: `microduck/` (Microduck's MJCF and meshes, copied
unchanged) and `owner_head/` (the owner's head meshes) plus the bill and jaw from `../model/duck_mid.xml`.

- Below the head, nothing changes: Microduck's bodies, joints, ranges, classes, sensors, keyframes (plus
  the mouth joint, closed) and collision sets. `scene_hybrid_allcollisions*.xml` mirrors
  `scene_allcollisions.xml` (velstand was trained on it). `scene_hybrid*.xml` mirrors `scene.xml`, the
  ground-contact set (sitstand, roulade, alpha_walking).
- Head link: Microduck's head visuals are moved to geom group 4. They stay in the model for packaging
  checks but aren't drawn. Microduck's head collision meshes are replaced by convex hulls of the owner's
  head parts plus the two bill ellipsoids. The owner's head has the same orientation as Microduck's
  head and is shifted so the two shells' bounding-box centres coincide. That puts it where Microduck's
  head is, not 6 cm further back where the Open Duck neck carries it.
- Mouth: `owner_jaw` body, `mouth` hinge, `mouth` position actuator (kp 2, +-0.2 N m). It is not a
  policy joint. The harness leaves it alone and it holds closed at ctrl 0. Microduck itself has a mouth
  servo in its head, which the policy also doesn't control.

| variant | robot files | scenes |
|---|---|---|
| hybrid (rigid antennas, matched mass) | `hybrid_allcollisions.xml`, `hybrid_groundcontact.xml` | `scene_hybrid_allcollisions.xml`, `scene_hybrid.xml` |
| `_noantenna` (**recommended**) | `hybrid_*_noantenna.xml` | `scene_hybrid*_noantenna.xml` |
| `_ownermass`, `_ownermass_noantenna` | ... | ... |
| `_mdcollision` (control), `_small` | ... | ... |

## How to run

```
PY=/home/user/cool-websites/.venv/bin/python
$PY duckbody/hybrid/build_hybrid.py                         # (re)build all variants + head_report.json
$PY duckbody/hybrid/evaluate_hybrid.py                      # all bodies, ~2.5 min each, -> results_hybrid.json
$PY duckbody/hybrid/evaluate_hybrid.py --bodies hybrid_noantenna --n 10
MUJOCO_GL=osmesa $PY duckbody/hybrid/replay.py --body hybrid_noantenna --max-seeds 1   # replay + PNG strip
```
This needs the `.deps/` links of `../microduck_port`: Rhoban/bam for the servo model, and microduck_rl
for the `microduck` control body. It also needs `../microduck_port/policies/*.onnx`. The hybrid models
themselves only use files in this folder.
`harness.MicroduckSim` now also accepts an absolute scene path and only turns the 14 policy servos into
BAM motors, so the hybrid's mouth actuator stays a plain MuJoCo position servo. Microduck's own results
are unchanged by this.

Replays: `replay_getup_back_hybrid_noantenna.json` (success) and `replay_getup_back_fail.json` (rigid
antennas). Both are `dict(fps=25, seconds=6, frames=[[x,y,z,qw,qx,qy,qz] * n_bodies flattened per frame],
bodies=[names], ...)`, in the body order of the scene (world excluded).

## Fine-tuning: needed?

- **Recommended build (no rigid antennas, head mass matched within about 5 % and CoM within about 8 mm): no
  fine-tuning needed** for stand, both get-ups, walk or sit/stand. Roulade works, but 3/10 times it hasn't
  finished turning when its 1 s slot ends. Either give it a 1.5-2 s slot in the manifest (`duration_s`),
  which needs no training, or fine-tune.
- **Fine-tuning would be needed for:** rigid antennas, or a head of a very different size. That means
  continuing their PPO runs from the shipped checkpoints in `pollen-robotics/microduck_rl` (mjlab, GPU):
  - velstand (`Mjlab-VelStand-Rough-Backlash-MicroDuck`, from run `0045i2si` @ 2749) with the hybrid's
    allcollisions MJCF swapped in for the robot. The prone and supine spawns are what matter: the face-up
    get-up has to learn to roll over the antennas and wide head. Add head-collision randomization (antenna
    length and stiffness, head width +-10 %), and widen head mass DR to about +-20 % if the owner's real head
    mass differs. Expect a short run of a few hundred iterations: walking and standing already work and the
    walk-anchor BC keeps them.
  - roulade (`microduck_roulade_env_cfg.py`) on the hybrid groundcontact MJCF, so the roll finishes in
    1 s with the longer head.
  - sitstand needs nothing.
  Check before and after with `evaluate_hybrid.py`, aiming for the `microduck` column.

## Physical build

| part | from |
|---|---|
| trunk, battery (NP-F970), power support, legs (hip yaw/roll/pitch, knee, ankle), feet/soles, neck (neck_pitch, head_pitch, head_yaw, head_roll brackets), 14 XL330 servos, IMU, firmware/daemon, ONNX brains | **Microduck** (Pollen Robotics) |
| head internals kept inside the new shell: Pi Zero 2 W + robot hat, camera + M12 lens holder, speaker, mouth XL330 + motor support | **Microduck** |
| head shell + bottom sheet (0.7 scale), bill (upper bill), lower jaw on the mouth servo | **owner** (v3 head) |
| antennas | owner's, but **leave them off or make them soft / fold-away** (rigid ones break get-up from the back and the roll) |

To do for the real head:
1. Re-model the owner's shell about 3 mm taller (42 to 45+ mm inside) so Microduck's head stack fits. Mount it
   on Microduck's head_roll servo with the shell spanning 47 mm behind to 93 mm ahead of the roll axis and
   8 mm below to 33 mm above it at the standing pose (`owner_shell_box_rel_head_roll_world_mm`). That is
   52 mm further forward than the Open Duck neck carries it.
2. Drive the lower jaw from Microduck's mouth servo, a 0-35 degree hinge 9 mm inside the face.
3. Weigh the finished head. Ballast it to Microduck's 189 g with the CoM within about 5 mm of
   Microduck's: about 33 mm ahead of and 5 mm above the roll axis at the standing pose. The brains work
   without this (owner-mass variant), but it keeps you inside their training randomization.

## Licenses

`microduck/`: Pollen Robotics' microduck_rl at commit `b02fdea`. MJCF is Apache-2.0. Meshes are
CC BY-NC-SA 4.0 (non-commercial, share-alike). See `microduck/SOURCE.md`, `LICENSE` and
`LICENSE-MODELS.md`. The hybrid MJCFs are adaptations of Pollen's MJCF and reference those meshes, so
the same terms apply to anything that ships them. `owner_head/` is Open Duck Mini v2 meshes
(Apache-2.0), see `owner_head/SOURCE.md`. The policies are Pollen's, from the Hub
`pollen-robotics/microduck-policies`, used unchanged.
