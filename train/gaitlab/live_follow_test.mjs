// Headless test of the live page's logic: Bolt follows one of five wandering people, and the choice
// switches every few seconds. Counts falls and how far behind Bolt stays.
//   node live_follow_test.mjs <mujoco.js> <bolt_live.json> [sessions] [seconds]
import fs from "fs";
import { Bolt, Follow, Crowd, followStep } from "./live.js";

const [mujocoJs, bundlePath, sessions = "10", seconds = "90"] = process.argv.slice(2);
const mj = await (await import(mujocoJs)).default();
const bundle = JSON.parse(fs.readFileSync(bundlePath));
const bolt = new Bolt(mj, bundle);
let falls = 0, dists = [], top = 0;
for (let s = 0; s < +sessions; s++) {
  bolt.reset(bundle.start);
  const crowd = new Crowd(5, 100 + s), follow = new Follow((v) => bolt.maxTurn(v), bundle.constants.sharp_turns);
  let rng = 7 + s, target = 0, nextSwitch = 0;
  const rand = () => ((rng = (rng * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let k = 0; k < +seconds * 50; k++) {
    if (k >= nextSwitch) { target = Math.floor(rand() * 5); nextSwitch = k + 50 * (3 + 12 * rand()); }
    const c = followStep(bolt, crowd, follow, target);
    if (k > 250) dists.push(c.dist);
    const v = bolt.data.qvel; top = Math.max(top, Math.hypot(v[0], v[1]));
    if (bolt.fell()) { falls++; console.log(`session ${s}: fell at ${(k / 50).toFixed(1)} s (${c.command}, asked ${c.speed.toFixed(2)} m/s)`); break; }
  }
}
dists.sort((a, b) => a - b);
console.log(`${sessions} sessions x ${seconds} s, target switched every 3-15 s: fell ${falls}, ` +
  `distance to the chosen person median ${dists[dists.length >> 1].toFixed(2)} m, top speed ${top.toFixed(2)} m/s`);
