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
for (const [command, n, turn, speed] of ref.script) {
  bolt.setCommand(command); bolt.setTurn(turn); bolt.setSpeed(speed);
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
        `${maxDiff(Array.from(q).slice(7), r.qpos.slice(7)).toExponential(1)} rad`);
    }
  }
}
console.log(`worst difference: what the brain feels ${worstObs.toExponential(1)}, brain output on the same input ${worstBrain.toExponential(1)}`);
const q = bolt.data.qpos, r = ref.steps[ref.steps.length - 1].qpos;
console.log(`end: Python at (${r[0].toFixed(3)}, ${r[1].toFixed(3)}), browser port at (${q[0].toFixed(3)}, ${q[1].toFixed(3)}), fell: ${bolt.fell()}`);
