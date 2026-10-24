# Robot dog roadmap

The plan: get the dog right in simulation first, teach it to understand people (and be funny), and only build a real body once the simulation and the brain hold up. Each phase has an exit test. We move on when the test passes, not when it feels done.

| Phase | What | Where it runs | Status |
|---|---|---|---|
| 1 | Digital dog: tricks, chat, voice, physics-checked moves, memory | Browser (`pet/index.html`) | ✅ Built |
| 2 | Full physics sim: motors, contacts, friction; tricks *learned* instead of scripted | MuJoCo on a PC/GPU | Next |
| 3 | Understanding people: free-form language, pet-human interaction, humor | Language model + your logged data | Planned |
| 4 | Physical body: first a small servo dog, then a brushless one that can flip | Hardware | Planned |

---

## Phase 1: Digital dog ✅

Open `pet/index.html`. Type or say commands like *sit*, *jump*, *jump around*, *backflip*, *somersault*, *roll over*, *play dead*, *shake*, *beg*, *bow*, *spin*, *dance*, *come here*, *talk*, *tell me a joke*, or chains like *sit then backflip*.

**What's real physics and what's simplified.** This matters, because Phase 2 exists to remove the simplifications.

| Real | Simplified |
|---|---|
| Jumps are ballistic: gravity (9.81 m/s²) alone decides air time and arc | Leg push is a planned acceleration, not motor torque through contacts |
| Spin uses conservation of angular momentum; tucking the legs really lowers inertia (computed from leg positions), so the dog spins ~1.8× faster | Body mass is lumped; legs are point masses for inertia |
| Motor limits: launch speed ≤ 3.6 m/s, spin momentum ≤ 5 kg·m²/s. That's why a double backflip under-rotates and crashes | No friction or slipping; feet stick where they land |
| Leg force, energy and push-off time are computed for every jump (a backflip needs ~445 N, 3.8× body weight, in 116 ms) | Standing balance uses spring-damper "muscles" on the body, not per-joint torques |
| Leg angles come from real two-link inverse kinematics | 2 joints per leg (no sideways hip joint yet) |

**Memory.** Mood and battery live in RAM and reset on reload. The dog's name and a log of every interaction, including your feedback ("good boy" = +1, "no" = −1 on the trick it was about), are saved in the browser. **Export training data** downloads that log as JSON, which is the seed dataset for Phase 3.

**Code:** `pet/js/physics.js` (pure physics), `dog.js` (body simulation and tricks), `brain.js` (language, personality and memory), `render.js` (3D), `app.js` (wiring and voice). Tests: `node --test pet/test/*.test.js`.

**Exit test:** all tests pass, every trick looks right to you, and you've logged ~200 interactions with feedback.

---

## Phase 2: Full physics simulation, with learned tricks

Swap the scripted motions for a physics engine where **every motor torque, foot contact and friction force is simulated**, then let the dog *learn* each trick.

1. **Model the dog in MuJoCo** (free and open source; what DeepMind and most legged-robot labs use). Use the same dimensions and masses as Phase 1, plus a sideways hip joint, giving 3 motors per leg and 12 in total. Use the real torque and speed limits of the motors we'd buy in Phase 4.
2. **Reference motions:** export the Phase 1 trajectories (sit, flips, roll) as targets to imitate. That's the DeepMimic approach, and it's why Phase 1 records them.
3. **Reinforcement learning** (PPO), thousands of dogs in parallel on a GPU (MuJoCo MJX or Isaac Lab). One reward per trick: match the pose, rotate a full turn, land on the feet, use little energy.
4. **Domain randomization:** vary mass (±15%), friction, motor strength, sensor noise and control delay during training, so the policies survive contact with a real body.
5. Plug the trained policies back into the browser (ONNX runtime) so you can compare scripted and learned versions side by side.

**Exit test**, over 1,000 randomized episodes per trick:
- Stand, sit, lie, walk, roll over, shake: ≥ 98% success
- Jump and somersault: ≥ 95%. Backflip: ≥ 90% clean landings
- It gets back up after every fall
- Peak motor torque stays below 80% of the rating (headroom for the real world)

---

## Phase 3: Understanding people, and humor

1. **Language → tricks with a language model.** Keep the body API exactly as it is (each trick is a "tool"). Replace the rules in `brain.js` with a model (e.g. Claude via tool use) that gets what you said plus the dog's state (battery, mood, memory) and returns the same step list. That unlocks open-ended requests: *"show off a bit"*, *"you look tired"*, *"do something cute for my sister"*.
2. **Pet-human interaction data.** Your exported logs (what was said, what the dog did, your feedback) become:
   - an **evaluation set**: phrasings the dog must understand, with the right answer
   - **examples** for the model, and later **fine-tuning data** for a small model that can run on the robot offline
3. **Humor, made measurable.** "Funny" isn't one thing, so we pin it down:
   - a **persona and style guide**: playful, self-deprecating about being a robot, physics puns, slapstick
   - **timing and physical comedy**: a pause before the punchline, a dramatic play-dead, a (safe) deliberate fail
   - a 👍/👎 button on every joke and bit. Ratings go into the log
   - a **humor eval**: a fixed set of situations ("you came home", "it failed a trick"), with responses rated by people who aren't us
4. **Voice:** better speech-to-text (e.g. Whisper), a consistent character voice for text-to-speech, and a wake word ("Hey Bolt").

**Exit test:**
- ≥ 90% correct on held-out commands phrased in new ways
- ≥ 50% of jokes and bits rated funny by people outside the project
- Reply starts within 1.5 s

---

## Phase 4: Physical body

**The honest hardware reality.** The simulation says a backflip needs ~445 N of leg force delivered in ~0.1 s. Hobby servos can't do that. Servo robot dogs can sit, walk, beg, roll and do small hops, but **robot dogs that backflip use brushless "quasi-direct-drive" motors** (like the MIT Mini Cheetah). So the body comes in two steps:

**4a. Small servo dog (≈ $300–600).** Proves the whole pipeline (sim → policy → real motors) cheaply.
- 12 servos (3 per leg), PCA9685 servo driver, IMU (BNO085), Raspberry Pi 5 brain (reuse the one from `robot/`), mic and speaker, 2S Li-ion pack.
- Frame: 3D-printed (open designs like SpotMicro), or start from a kit like Petoi Bittle.
- Can do: stand, sit, lie, shake, beg, bow, walk, turn, talk, small hops. Not flips.

**4b. Brushless dog that can flip (≈ $2,000–6,000).** Prices change, so check current ones.
- DIY: an open-source brushless quadruped design, e.g. the Stanford Pupper family or ODRI Solo-12.
- Or buy a commercial robot dog with an SDK and run our brain on it.

**Before the first real backflip:**
- The Phase 2 policy passes with randomized physics *and* with the measured mass and motor data of the real robot
- Emergency-stop button, fall detection, joint and torque soft-limits
- First attempts on a tether, over foam mats, with people and pets well clear

**Exit test:** the real robot does every trick the sim does (within its hardware limits) and answers you out loud.

---

## Rough timeline (part-time)

Phase 2: 1–3 months · Phase 3: 1–2 months (can overlap with Phase 2) · Phase 4a: ~1 month · Phase 4b: 3–6 months.
