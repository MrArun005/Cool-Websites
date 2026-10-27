// Knock Bolt over in the browser physics many times and let it get up by itself, the way the games do it
// (game.js GetUp: the self-righting brain rolls it onto its belly, the posture brain stands it up).
//   node getup_test.mjs <mujoco.js> <bolt_live.json> [tries per kind] [wait: now | rest] [tuck: lie | motors | none] [carry | zeros]
// Two kinds of fall:
//   lying:  started on its side, back or half-tipped (like GetUpEnv: dropped from 0.5 m in a random fallen
//           pose, then 1 s lying still with the motors holding it), exactly how the brain practised;
//   shoved: walking at 0.3-1.0 m/s, shoved 150 N sideways for 0.2 s until it falls (as in the games).
// wait: "rest" lies still (motors keep their last targets) until Bolt has come to rest, up to 1.5 s, before
// the get-up brain takes over (GetUpEnv always started from rest); "now" hands over the moment it fell.
// tuck: how the legs fold under the body before standing (game.js GETUP_TUCK).
// carry: the standing brain starts from the get-up brain's last action; zeros: from zeros.
// Reports per kind: righted (belly down), standing (2 tries, 10 s), still standing 2 s later, failures.
import fs from "fs";
import { Bolt } from "./live.js";
import { GETUP, GetUp, setGetUpCarry, setGetUpTuck, setGetUpWait } from "./game.js";

const [mujocoJs, bundlePath, triesArg = "20", wait = "now", tuck = "lie", carry = "zeros"] = process.argv.slice(2);
if (process.env.GETUP) Object.assign(GETUP, JSON.parse(process.env.GETUP));   // e.g. GETUP='{"tuck":1.5}' to try other timings
setGetUpWait(wait); setGetUpTuck(tuck); setGetUpCarry(carry === "carry");
const mj = await (await import(mujocoJs)).default();
const bundle = JSON.parse(fs.readFileSync(bundlePath));
const bolt = new Bolt(mj, bundle);
const c = bolt.c, tries = +triesArg;
let s = 12345;
const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
const uni = (a, b) => a + (b - a) * rand();
const mul = (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];

// GetUpEnv.reset without randomized physics: a fallen pose, dropped from 0.5 m, 1 s to land and lie still
function lying(kind) {
  bolt.reset(bundle.start);
  const sign = rand() < 0.5 ? -1 : 1;
  const [roll, pitch] = kind === "side" ? [sign * uni(1.2, 1.9), uni(-0.4, 0.4)] : kind === "back" ? [sign * uni(2.4, Math.PI), uni(-0.4, 0.4)] : [sign * uni(0.5, 1.2), uni(-0.6, 0.6)];
  const yaw = uni(-Math.PI, Math.PI), half = (a) => [Math.cos(a / 2), Math.sin(a / 2)];
  const qr = [half(roll)[0], half(roll)[1], 0, 0], qp = [half(pitch)[0], 0, half(pitch)[1], 0], qy = [half(yaw)[0], 0, 0, half(yaw)[1]];
  const d = bolt.data;
  d.qpos.set([0, 0, 0.5, ...mul(qy, mul(qp, qr))], 0);
  const legs = c.dog_pose.map((v, i) => Math.max(bolt.lo[i], Math.min(bolt.hi[i], 0.5 * v + 0.5 * uni(bolt.lo[i], bolt.hi[i]))));
  for (let i = 0; i < 12; i++) d.qpos[7 + i] = legs[i];
  d.ctrl.set(legs); d.qvel.fill(0);
  mj.mj_forward(bolt.model, d);
  for (let k = 0; k < 50; k++) bolt.restTick();
  return true;
}
// walking, then a cannon-style shove from a random side until it falls (false: it stayed up)
function shoved() {
  bolt.reset(bundle.start);
  for (let k = 0; k < 50; k++) bolt.run({ command: "stand" });
  const v = uni(0.3, 1.0), turn = uni(-0.5, 0.5), walk = Math.floor(uni(75, 150));
  for (let k = 0; k < walk; k++) bolt.run({ command: "walk", turn, speed: Math.min(v, 0.2 + 0.6 * k * bolt.dt) });
  bolt.shove((rand() < 0.5 ? -1 : 1) * 150, 0, 0.2);
  for (let k = 0; k < 100; k++) { bolt.run({ command: "walk", turn, speed: v }); if (bolt.fell()) return true; }
  return false;
}
// from the moment it fell: the games' recovery (game.js GetUp: rest, self-right, stand), then 2 s more standing
let misses = {};
function getUp() {
  const g = new GetUp(bolt);
  let rightedAt = null;
  while (!g.done && !g.failed) {
    const cmd = g.step(bolt.dt);
    if (rightedAt === null && (g.stage === "tuck" || g.stage === "stand")) rightedAt = g.t;
    if (!g.failed) bolt.run(cmd);
  }
  for (const m of g.misses) misses[m] = (misses[m] || 0) + 1;
  if (!g.done) return { up: false, righted: rightedAt !== null, tries: g.tries };
  let fellAgain = false;
  for (let k = 0; k < 100; k++) { bolt.run({ command: "stand" }); if (bolt.fell()) fellAgain = true; }
  return { up: true, righted: true, secs: g.secs, rightedAt, tries: g.tries, steady: !fellAgain && bolt.standing() };
}

const med = (a) => (a.length ? a.slice().sort((x, y) => x - y)[a.length >> 1].toFixed(2) : "-");
console.log(`get-up brain ${bundle.brains.getup.source} (goal ${bundle.brains.getup.goal || "stand"}), wait: ${wait}, tuck: ${tuck}, ${JSON.stringify(GETUP)}, standing brain starts from ${carry === "carry" ? "its last action" : "zeros"}`);
for (const kind of ["side", "back", "tipped", "shoved"]) {
  misses = {};
  let n = 0, up = 0, righted = 0, steady = 0, stayedUp = 0, second = 0, secs = [], rightSecs = [];
  for (let k = 0; k < tries; k++) {
    if (kind === "shoved" ? !shoved() : !lying(kind)) { stayedUp++; continue; }
    const r = getUp();
    n++; righted += r.righted;
    if (r.up) { up++; secs.push(r.secs); rightSecs.push(r.rightedAt); second += r.tries > 1; if (r.steady) steady++; }
  }
  console.log(`${kind.padEnd(7)} falls ${n}${stayedUp ? ` (${stayedUp} shoves did not topple it)` : ""}: righted ${righted}/${n}, standing ${up}/${n} ` +
    `(${second} on the 2nd try; median ${med(secs)} s from the fall, slowest ${secs.length ? Math.max(...secs).toFixed(2) : "-"} s; belly down after ${med(rightSecs)} s), ` +
    `still standing 2 s later ${steady}/${n}, failed ${n - up}; failed tries: ${JSON.stringify(misses)}`);
}
