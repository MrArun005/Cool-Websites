// Check the dog-imitation brain in the browser physics (imitate.js) against Python (imitate_env.py).
//   python export_imitate.py <checkpoint> --reference /tmp/imitate_reference.json       (from train/)
//   node gaitlab/imitate_test.mjs <@mujoco/mujoco mujoco.js> ../models/<name> /tmp/imitate_reference.json
// 1. Parity: the same start states as Python (fixed physics), then step by step: the brain's inputs, its output
//    on Python's inputs, and the body's position. Plus a clip switch (walk -> trot at the closest pose -> sit).
// 2. Closed loop: walk, trot, canter, sit, lie for 6 s each from 4 points of the clip (Bolt spawned in the dog's
//    pose, moving as it was), falls and mean tracking reward, next to Python's numbers for the same protocol and
//    training's exam (randomized physics, 4 s).
import fs from "fs";
import path from "path";
import { Bolt } from "./live.js";
import { Imitate } from "./imitate.js";

const [mujocoJs, folder, referencePath, secondsArg = "6"] = process.argv.slice(2);
const mj = await (await import(mujocoJs)).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(folder, f)));
const clips = read("clips.json"), brain = read("brain.json"), model = read("model.json");
const ref = JSON.parse(fs.readFileSync(referencePath));
const bundle = { xml: fs.readFileSync(path.join(folder, "robot.xml"), "utf8"), constants: clips.constants, brains: {} };
const bolt = new Bolt(mj, bundle);
const dog = new Imitate(bolt, brain, clips);
const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);

function load(s) {
  const d = bolt.data;
  mj.mj_resetData(bolt.model, d);
  d.qpos.set(s.qpos); d.qvel.set(s.qvel); d.ctrl.set(s.ctrl);
  mj.mj_forward(bolt.model, d);
  dog.clip = dog.clips[s.clip]; dog.tau0 = s.tau0; dog.t = s.t; dog.lastAction = Float64Array.from(s.last_action);
}
// replay Python's steps in the port; returns the worst differences
function replay(steps, from = 0, to = steps.length, w = { obs0: null, obs: 0, act: 0, pos: 0, joints: 0 }) {
  for (let k = from; k < to; k++) {
    const r = steps[k], o = dog.obs(), diff = maxDiff(o, r.obs);
    if (w.obs0 === null) w.obs0 = diff;
    w.obs = Math.max(w.obs, diff);
    w.act = Math.max(w.act, maxDiff(dog.brain.act(r.obs), r.action));
    dog.step(dog.brain.act(o));
    const q = bolt.data.qpos;
    w.pos = Math.max(w.pos, Math.hypot(q[0] - r.qpos[0], q[1] - r.qpos[1], q[2] - r.qpos[2]));
    w.joints = Math.max(w.joints, maxDiff(Array.from(q.slice(7, 19)), r.qpos.slice(7, 19)));
  }
  return w;
}
const fmt = (w) => `inputs ${w.obs0.toExponential(1)} at the start, worst ${w.obs.toExponential(1)}; brain on the same input ` +
  `${w.act.toExponential(1)}; body ${(w.pos * 1000).toFixed(3)} mm, joints ${w.joints.toExponential(1)} rad`;

console.log(`${model.name} (${brain.inputs} inputs, clips ${Object.keys(dog.clips).join(" ")})`);
console.log("1. parity with Python (fixed physics, same start state, step by step):");
let worst = { obs0: 0, obs: 0, act: 0, pos: 0 };
for (const run of ref.parity) {
  load(run.start);
  const w = replay(run.steps);
  console.log(`   ${run.start.clip.padEnd(10)} ${String(run.steps.length).padStart(3)} steps: ${fmt(w)}`);
  for (const k of Object.keys(worst)) worst[k] = Math.max(worst[k], w[k]);
}
{   // switching: walk 60 steps, play("trot") joins at the closest pose, play("sit") from its start
  const S = ref.switch;
  load(S.start);
  let w = { obs0: null, obs: 0, act: 0, pos: 0, joints: 0 }, k = 0;
  const notes = [];
  S.plan.forEach(([name, n], i) => {
    if (i) {
      dog.play(name);
      const sw = S.switches[i - 1];
      notes.push(`${name}: clip time ${dog.tau().toFixed(3)} s (Python ${(sw.tau0 + dog.t * dog.dt).toFixed(3)})`);
    }
    w = replay(S.steps, k, k + n, w); k += n;
  });
  console.log(`   switch walk->trot->sit, ${S.steps.length} steps: ${fmt(w)}; joined ${notes.join(", ")}`);
}
console.log(`   worst over all: inputs at the start ${worst.obs0.toExponential(1)}, along the runs ${worst.obs.toExponential(1)}, ` +
  `brain ${worst.act.toExponential(1)}, body ${(worst.pos * 1000).toFixed(3)} mm`);

const seconds = +secondsArg, n = Math.round(seconds / dog.dt), starts = 4;
console.log(`2. closed loop in the browser physics, ${seconds} s per try, ${starts} starts per clip (fixed physics):`);
console.log("   clip     browser: track  falls  pose rms | Python same protocol: track  falls | training exam (random physics, 4 s): track  falls");
for (const name of ["walk", "trot", "canter", "sit", "lie"]) {
  const c = dog.clips[name];
  let scores = [], falls = 0, rms = [];
  for (let k = 0; k < starts; k++) {
    dog.spawn(name, c.loop ? (k / starts) * c.length : 0, 0);
    let total = 0, fell = false;
    for (let i = 0; i < n; i++) {
      dog.tick();
      const tr = dog.tracking();
      total += tr.reward; rms.push(tr.poseRms);
      if ((fell = dog.fell(tr))) break;
    }
    falls += fell; scores.push(total / n);
  }
  const py = ref.fixed_physics_6s, ex = model.exam, mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  console.log(`   ${name.padEnd(8)} ${mean(scores).toFixed(3).padStart(14)}  ${(falls / starts).toFixed(2).padStart(5)}  ${mean(rms).toFixed(3).padStart(8)} |` +
    ` ${(py[`track_${name}`] ?? NaN).toFixed(3).padStart(26)}  ${(py[`fall_${name}`] ?? NaN).toFixed(2).padStart(5)} |` +
    ` ${ex[`track_${name}`].toFixed(3).padStart(42)}  ${ex[`fall_${name}`].toFixed(2).padStart(5)}`);
}
// 3. As a game would use it: from standing still (the sit clip's first frame, held 0.5 s on the motors) play a gait,
// switch gaits, then sit and lie; falls along the way.
console.log("3. from standing still, as a game: play(clip) for the time given, in order");
for (const plan of [[["walk", 6]], [["trot", 6]], [["walk", 2], ["trot", 2], ["pace", 2], ["walk", 1], ["sit", 4.5], ["lie", 6.5]],
  [["trot", 2], ["canter", 2], ["trot", 2]]]) {
  dog.spawn("sit", 0, 0); dog.stop();
  for (let i = 0; i < 25; i++) bolt.restTick();
  const log = [];
  let fellAt = null, t = 0;
  for (const [name, secs] of plan) {
    dog.play(name);
    let total = 0;
    const m = Math.round(secs / dog.dt);
    for (let i = 0; i < m; i++, t++) {
      dog.tick(); const tr = dog.tracking(); total += tr.reward;
      if (fellAt === null && dog.fell(tr)) fellAt = `${name} at ${(t * dog.dt).toFixed(2)} s`;
    }
    log.push(`${name} ${secs}s track ${(total / m).toFixed(3)}`);
  }
  const q = bolt.data.qpos;
  console.log(`   ${log.join(", ")}; fell: ${fellAt || "no"}; ended at (${q[0].toFixed(2)}, ${q[1].toFixed(2)}), body ${q[2].toFixed(3)} m up`);
}
