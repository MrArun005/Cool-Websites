// Check the GPU dog-motion brain as Bolt's Day Out runs it (live.js Brain on dog_gpu360.json, imitate.js inputs)
// against Python (export_doglife.py --reference: ImitateEnv, fixed physics, MjxImitatePolicy).
//   node gaitlab/doglife_brain_test.mjs <mujoco.js> <data folder> <reference.json> <robot.xml>
// 1. the brain on Python's own inputs -> Python's actions (worst difference)
// 2. the browser's inputs from Python's start state and along the run (the port of ImitateEnv._obs + physics)
import fs from "fs";
import path from "path";
import { Bolt } from "./live.js";
import { Imitate } from "./imitate.js";

const [mujocoJs, dataDir, refPath, robotXml] = process.argv.slice(2);
const mj = await (await import(path.resolve(mujocoJs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f)));
const clips = read("dog_clips.json"), brain = read("dog_gpu360.json"), ref = JSON.parse(fs.readFileSync(refPath));
const bolt = new Bolt(mj, { xml: fs.readFileSync(robotXml, "utf8"), constants: clips.constants, brains: {} });
const dog = new Imitate(bolt, brain, clips);
const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
let act = 0, obs0 = 0, obs = 0, pos = 0, n = 0;
for (const run of ref.parity) {
  const s = run.start, d = bolt.data;
  mj.mj_resetData(bolt.model, d);
  d.qpos.set(s.qpos); d.qvel.set(s.qvel); d.ctrl.set(s.ctrl); mj.mj_forward(bolt.model, d);
  dog.clip = dog.clips[s.clip]; dog.tau0 = s.tau0; dog.t = s.t; dog.lastAction = Float64Array.from(s.last_action);
  let w = { act: 0, obs: 0, pos: 0 };
  run.steps.forEach((r, k) => {
    const o = dog.obs();
    if (k === 0) obs0 = Math.max(obs0, maxDiff(o, r.obs));
    w.obs = Math.max(w.obs, maxDiff(o, r.obs));
    w.act = Math.max(w.act, maxDiff(dog.brain.act(r.obs), r.action));
    dog.step(dog.brain.act(o)); n++;
    const q = d.qpos; w.pos = Math.max(w.pos, Math.hypot(q[0] - r.qpos[0], q[1] - r.qpos[1], q[2] - r.qpos[2]));
  });
  console.log(`  ${s.clip.padEnd(10)} ${run.steps.length} steps: brain on Python's inputs ${w.act.toExponential(1)}, inputs ${w.obs.toExponential(1)}, body ${(w.pos * 1000).toFixed(2)} mm`);
  act = Math.max(act, w.act); obs = Math.max(obs, w.obs); pos = Math.max(pos, w.pos);
}
console.log(`GPU brain (${brain.inputs} inputs), ${n} steps: worst action error on the same inputs ${act.toExponential(2)}; inputs at the start ${obs0.toExponential(1)}, along the runs ${obs.toExponential(1)}; body drift ${(pos * 1000).toFixed(2)} mm`);
process.exit(act < 1e-4 ? 0 : 1);
