// Check the browser port (live.js) against Python, step by step, on the reference run in the bundle.
//   node live_check.mjs <path to @mujoco/mujoco/mujoco.js> <bolt_live.json> <bolt_live_reference.json>
import fs from "fs";
import { Bolt } from "./live.js";

const [mujocoJs, bundlePath, referencePath] = process.argv.slice(2);
const mj = await (await import(mujocoJs)).default();
const bundle = JSON.parse(fs.readFileSync(bundlePath));
const ref = JSON.parse(fs.readFileSync(referencePath));
const bolt = new Bolt(mj, bundle);
bolt.reset(ref.start);
const maxDiff = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
let k = 0, worstObs = 0, worstBrain = 0;
for (const [command, n, turn, speed, hurdle] of ref.script) {
  bolt.setCommand(command); bolt.setTurn(turn); bolt.setSpeed(speed);
  if (hurdle !== undefined) bolt.placeHurdle(1.5, hurdle);
  for (let i = 0; i < n; i++, k++) {
    const r = ref.steps[k], obs = bolt.obs();
    const brain = command === "walk" ? bolt.brains.walk : bolt.brains.posture;
    worstObs = Math.max(worstObs, maxDiff(obs, r.obs));
    worstBrain = Math.max(worstBrain, maxDiff(brain.act(r.obs), r.action));  // same input: brain port only
    bolt.step(brain.act(obs));
    if ([0, 9, 99, 199, 349, 499, 649].includes(k)) {
      const q = bolt.data.qpos;
      console.log(`step ${String(k + 1).padStart(3)} (${command}, turn ${turn}, speed ${speed}): body position off by ` +
        `${(Math.hypot(q[0] - r.qpos[0], q[1] - r.qpos[1], q[2] - r.qpos[2]) * 1000).toFixed(3)} mm, joints by ` +
        `${maxDiff(Array.from(q).slice(7, 19), r.qpos.slice(7, 19)).toExponential(1)} rad`);
    }
  }
}
console.log(`worst difference: what the brain feels ${worstObs.toExponential(1)}, brain output on the same input ${worstBrain.toExponential(1)}`);
const q = bolt.data.qpos, r = ref.steps[ref.steps.length - 1].qpos;
console.log(`end: Python at (${r[0].toFixed(3)}, ${r[1].toFixed(3)}), browser port at (${q[0].toFixed(3)}, ${q[1].toFixed(3)}), fell: ${bolt.fell()}`);
if (ref.jump && bolt.brains.jump) {   // the jump: same start, same bar, step by step
  const J = ref.jump;
  bolt.reset(J.start);
  const yaw = Math.atan2(J.start.bar_dir[1], J.start.bar_dir[0]);
  if (J.start.real) { bolt.placeReal(0, J.start.bar_at, yaw, J.start.bar_h); bolt.aim(0); }   // a real hurdle (run 16 on)
  else bolt.setHurdle(J.start.bar_at, yaw, J.start.bar_h);
  bolt.startJump();
  let wObs = 0, wAct = 0, wPos = 0;
  J.steps.forEach((s) => {
    const o = bolt.jumpObs();
    wObs = Math.max(wObs, maxDiff(o, s.obs)); wAct = Math.max(wAct, maxDiff(bolt.brains.jump.act(s.obs), s.action));
    bolt.jumpStep(bolt.brains.jump.act(o));
    const q = bolt.data.qpos; wPos = Math.max(wPos, Math.hypot(q[0] - s.qpos[0], q[1] - s.qpos[1], q[2] - s.qpos[2]));
  });
  console.log(`jump (${J.steps.length} steps, Python: cleared ${J.cleared}, touched ${J.touched}${J.start.real ? `, dropped ${J.dropped}; browser dropped ${bolt.barDropped(0)}` : ""}): worst difference in what the jump brain feels ` +
    `${wObs.toExponential(1)}, its output ${wAct.toExponential(1)}, body position ${(wPos * 1000).toFixed(3)} mm`);
}
if (ref.getup && bolt.brains.getup) {   // getting up: the same fall (settled for 1 s), then the get-up step by step
  const G = ref.getup;
  bolt.reset(G.start);
  bolt.startGetUp();
  let wObs = 0, wAct = 0, wPos = 0, stoodAt = null;
  const marks = [];
  G.steps.forEach((s, i) => {
    const o = bolt.getUpObs();
    wObs = Math.max(wObs, maxDiff(o, s.obs)); wAct = Math.max(wAct, maxDiff(bolt.brains.getup.act(s.obs), s.action));
    bolt.getUpStep(bolt.brains.getup.act(o));
    const q = bolt.data.qpos, off = Math.hypot(q[0] - s.qpos[0], q[1] - s.qpos[1], q[2] - s.qpos[2]);
    wPos = Math.max(wPos, off);
    if (stoodAt === null && bolt.standing()) stoodAt = i + 1;
    if ([0, 24, 49, 99, 149, 199].includes(i)) marks.push(`${i + 1}: obs ${wObs.toExponential(1)}, body ${(off * 1000).toFixed(2)} mm`);
  });
  console.log(`getup (from its ${G.start.kind}, ${G.steps.length} steps; Python: standing at the end ${G.standing}, first stood at step ${G.stood_at}; ` +
    `browser: standing ${bolt.standing()}, first stood at step ${stoodAt}): worst difference in what the get-up brain feels ` +
    `${wObs.toExponential(1)}, its output on the same input ${wAct.toExponential(1)}, body position ${(wPos * 1000).toFixed(3)} mm`);
  console.log(`  by step: ${marks.join(" | ")}`);
}
