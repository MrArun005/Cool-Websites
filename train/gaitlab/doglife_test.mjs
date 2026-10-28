// Bolt's Day Out headless: N simulated seconds of the mind + body on the live physics, with the viewer's
// buttons pressed at set times. Reports what Bolt did, how it moved between brains, falls and recoveries.
//   node doglife_test.mjs <mujoco.js> <site data folder> [seconds=300] [seed=1] [viewer=1]
import fs from "fs";
import path from "path";
import * as live from "./live.js";
import { makeLife } from "./doglife.js";

const [mujocoJs, dataDir, secsArg = "300", seedArg = "1", viewerArg = "1"] = process.argv.slice(2);
const mj = await (await import(path.resolve(mujocoJs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f)));
const bundle = read("bolt_run.json");
const brains = { gpu: read("dog_gpu360.json"), g18: read("dog_18g.json"), jump: read("jump_trot.json") };
const clips = read("dog_clips.json");
const { bolt, body, mind } = makeLife(live, mj, bundle, brains, clips, +seedArg);
const secs = +secsArg, dt = bolt.dt, viewer = viewerArg === "1";
const modeTime = {}, trans = {}, brainTime = {};
let last = body.mode, seen = 0, t0 = Date.now();
const acts = viewer ? [[40, () => mind.call()], [80, () => { const w = bolt.where(); mind.throwBall([w.x, w.y], mind.inYard([w.x + 7, w.y + 3])); }],
  [130, () => mind.push(100)], [170, () => mind.setJump()], [230, () => mind.push(-90)]] : [];
for (let i = 0; i * dt < secs; i++) {
  const t = i * dt;
  while (acts.length && t >= acts[0][0]) acts.shift()[1]();
  const w = bolt.where(); mind.person = [w.x + 4, w.y - 3];
  mind.step(dt); body.step(dt);
  modeTime[body.mode] = (modeTime[body.mode] || 0) + dt;
  brainTime[body.brain()] = (brainTime[body.brain()] || 0) + dt;
  if (body.mode !== last) { const k = `${last}>${body.mode}`; trans[k] = (trans[k] || 0) + 1; last = body.mode; }
  while (body.events.length) { const e = body.events.shift(); mind.say(e.text + (e.text.startsWith("Bolt fell") && process.env.DEBUG ? " " + JSON.stringify(body.fallInfo) : ""), e.kind); }
  while (seen < mind.log.length) { const e = mind.log[seen++], m = Math.floor(e.t / 60), s = (e.t % 60).toFixed(0).padStart(2, "0"); console.log(`  ${m}:${s} [${body.mode}] ${e.text}`); }
}
const r = (o) => Object.entries(o).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${typeof v === "number" && v % 1 ? v.toFixed(0) + "s" : v}`).join(", ");
console.log(`seed ${seedArg}: ${secs} s in ${((Date.now() - t0) / 1000).toFixed(0)} s wall; falls ${body.falls}, recoveries ${body.recoveries}; ${JSON.stringify(mind.stats)}; kerb stops ${mind.kerbStops}; energy ${mind.energy.toFixed(2)}`);
console.log(`  tasks: ${r(mind.counts)}`);
console.log(`  modes: ${r(modeTime)}`);
console.log(`  brains: ${r(brainTime)}`);
console.log(`  transitions: ${r(trans)}`);
console.log(`  fell in: ${JSON.stringify(body.fellLog || {})}`);
