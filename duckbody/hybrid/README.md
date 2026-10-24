# Hybrid duck: Microduck body + the owner's duck head

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
