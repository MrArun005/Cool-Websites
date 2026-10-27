# Gait Lab

A 3D replay page for Bolt's trained brains (stand, sit, lie, walk). The page draws a robot-dog body
on top of the simulator's own part positions, and marks paws, knees and shins that touch the floor.

Make the clips (run from `train/`), then serve this folder (e.g. `python -m http.server`):

    P="--brain posture=../models/bolt-dog-run2-1M --brain walk2M=<run>/checkpoints/<2M> --brain walk500k=../models/bolt-dog-walk-500k --gait-generator"
    python record3d.py $P --label "Full routine" -o gaitlab/data/routine.json \
        --script stand:2:posture sit:4:posture stand:3:posture lie:4:posture stand:3:posture walk:8:walk500k
    python record3d.py $P --label "Stand" --script stand:6:posture -o gaitlab/data/stand.json
    python record3d.py $P --label "Sit" --script stand:1.5:posture sit:5:posture -o gaitlab/data/sit.json
    python record3d.py $P --label "Lie down" --script stand:1.5:posture lie:5:posture -o gaitlab/data/lie.json
    python record3d.py $P --label "Walk, 2M brain" --script walk:8:walk2M -o gaitlab/data/walk.json
    python record3d.py $P --label "Walk, 500k brain" --script walk:8:walk500k -o gaitlab/data/walk-500k.json
    python record3d.py $P --label "Stand, then walk" --script stand:3:posture walk:6:walk2M -o gaitlab/data/handover.json

New walk (run 7, tall stance; note `--gait-generator tall` for these):

    P7="--brain posture=../models/bolt-dog-run2-1M --brain walk7=<run7 checkpoint folder> --gait-generator tall"
    python record3d.py $P7 --label "Full routine, new walk" -o gaitlab/data/routine7.json \
        --script stand:2:posture sit:4:posture stand:3:posture lie:4:posture stand:3:posture walk:8:walk7
    python record3d.py $P7 --label "Walk, run 7" --script walk:8:walk7 -o gaitlab/data/walk7.json
    python record3d.py $P7 --label "Stand, then walk, run 7" --script stand:3:posture walk:6:walk7 -o gaitlab/data/handover7.json

The tail, ears and blinking are decoration; the simulated dog has none of them.

## Live mode: pick who Bolt follows

The first clip, "Live", runs Bolt in the browser instead of replaying a recording. It uses MuJoCo's
official WebAssembly build (`@mujoco/mujoco`, the same 3.17.0 version as the Python training) and the
trained brains. Five people wander around, and Bolt follows whichever one you pick. You can switch
at any moment, with the buttons or by tapping a person.

- `live.js` is a port of `BoltEnv`'s control loop (dog gait, turning, speed control), the brains
  (normalizer plus two ELU layers) and the follow layer. The page and the Node tests use the same file.
- `../export_live.py` writes `data/bolt_live.json` (the robot, both brains' weights, and the start
  pose) and a Python reference run:

      python export_live.py --posture ../models/bolt-dog-posture-dog-2.5M \
          --walk ../models/bolt-dog-walk-run-8.5M -o gaitlab/data/bolt_live.json \
          --reference /tmp/bolt_live_reference.json

- `live_check.mjs` replays the reference run in the port and compares step by step. After 650
  steps the body was less than 0.001 mm from Python's, and brain outputs differed by under 1e-6:

      npm pack @mujoco/mujoco@3.17.0 && tar xzf mujoco-mujoco-3.17.0.tgz
      node gaitlab/live_check.mjs ./package/mujoco.js gaitlab/data/bolt_live.json /tmp/bolt_live_reference.json

- `live_follow_test.mjs` runs the page's follow logic headless, switching person every 3 to 15 s:

      node gaitlab/live_follow_test.mjs ./package/mujoco.js gaitlab/data/bolt_live.json 40 120

  This test found two falls in 80 simulated minutes, and both are now fixed:
  - Stopping dead from a run tipped Bolt over. It now brakes gradually and trots slowly for
    0.5 s before it stands.
  - Speeding up while turning toward someone behind it also tipped Bolt over. It now turns
    first, then speeds up.

  Bolt's speed is capped at 1.0 m/s, because 1.2 m/s isn't steady yet. After these fixes:
  0 falls in 40 two-minute sessions.

To serve the page, copy `mujoco.js` and `mujoco.wasm` from the npm package into `mujoco/`, and
`live.js` next to `index.html`.

## Bolt's Agility Run (you are the handler)

`agilityrun.js` + `agilityrun_template.html`: three agility courses (a straight line; a curve with slalom markers and
a down-stay; a full course with 30-40 cm bars). The player calls walk / trot / run (canter) / jump / sit / down /
stand and turns; auto-steer follows the course line and auto-jump takes each hurdle (both can be switched off).
Scoring is time plus 5 s per fault (knocked bar, refusal, missed obstacle), against a standard course time.
One physics world, several brains: the dog-imitation brain (run 18b, `imitate.js`) for the gaits, run 12's walk
brain for sharp turns, the real-hurdle jump brain (16g, 51 inputs), the posture brain for stand/sit/down, and the
get-up chain (17e) after falls. The comments at the top of `agilityrun.js` list the measured hand-overs.

Bundle and page (from `train/`; S = the scratchpad):

    python export_live.py --posture ../models/bolt-dog-posture-getup-3M --walk ../models/bolt-dog-walk-sharp-12M \
        --jump ../models/bolt-dog-jump-a1-16g-27M --getup ../models/bolt-dog-selfright-17e-16.5M \
        -o $S/agilityrun/site/data/bolt_run.json --reference /tmp/ref.json
    cp ../models/bolt-dog-imitate-18b-17.5M/brain.json $S/agilityrun/site/data/imitate_brain.json
    cp ../models/bolt-dog-imitate-18b-17.5M/clips.json $S/agilityrun/site/data/imitate_clips.json
    cp gaitlab/agilityrun_template.html $S/agilityrun/template.html && python gaitlab/build_pages.py $S agilityrun
    cp gaitlab/{live,game,imitate,agilityrun,hurdles3d,props3d}.js $S/agilityrun/site/   # plus mujoco/ as the other sites

Headless: `node gaitlab/agilityrun_test.mjs <mujoco.js> $S/agilityrun/site/data 6 1,2,3` (a scripted handler runs
each level) and `... 8 drill` (each command in turn on an empty stretch). The page takes `?level=2`, `?demo`
(the scripted handler plays) and `?speed=4` for tests.
