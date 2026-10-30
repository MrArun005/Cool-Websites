// Bolt Run headless (simulation). Two parts:
//   auto: an autoplayer for N simulated seconds per seed. It sees the track's rows ahead (as the player sees the
//         screen), picks a lane with a way through, presses Up when a bar is 2-3.5 m ahead, and holds canter on
//         clear stretches when the catcher is near. Reports metres, jumps tried and clean (bar not knocked, all paws
//         past the bar, no fall), bumps into solid things, falls, and when (if) Bolt was caught.
//   idle: a do-nothing player (Go, then no input): Bolt trots straight on, so it must get caught.
//   node run_test.mjs <mujoco.js> <site data folder> auto [seconds=120] [seed=1]
//   node run_test.mjs <mujoco.js> <site data folder> idle [seconds=120] [seed=1]
import fs from "fs";
import path from "path";
import * as live from "./live.js";
import { RunLife, TRACK, JUMP, setLook, setBand } from "./run.js";
if (process.env.BAND) setBand(+process.env.BAND, +(process.env.PRED || 3));
if (process.env.LOOK) setLook(+process.env.LOOK);

const [mujocoJs, dataDir, what = "auto", a1, a2] = process.argv.slice(2);
const mj = await (await import(path.resolve(mujocoJs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f)));
const bundle = read("bolt_run.json");
const brains = { gpu: read("dog_gpu360.json"), g18: read("dog_18g.json"), jump: read("jump.json") };
const clips = read("dog_clips.json");
const secs = +(a1 || 120), seed = +(a2 || 1), t0 = Date.now();
const L = new RunLife(live, mj, bundle, brains, clips, seed);
const b = L.bolt, dt = b.dt;
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
let seen = 0;
const printLog = () => { while (seen < L.log.length) { const e = L.log[seen++]; if (process.env.QUIET !== "1") console.log(`  ${fmt(e.t)} [${L.body.mode}] ${e.text}`); } };

// the autoplayer: the rows within 10 m ahead, in order; take the lane nearest the current one that is open in
// the next row (and, if possible, the row after); hurdles: centre lane, Up at 2-3.5 m
function autoplay() {
  const w = b.where(), rows = L.segs.filter((s) => s.x + 0.6 > w.x && s.x < w.x + 10);
  const blocked = (s, l) => s.kind === "hurdle" ? l !== 0 : s.kind === "narrow" ? (s.blocked.includes(l) || (l === s.shut)) : s.blocked.includes(l);
  const next = rows[0];
  if (next) {
    let want = L.lane;
    if (blocked(next, want)) {
      const order = [L.lane, L.lane - 1, L.lane + 1, L.lane - 2, L.lane + 2].filter((l) => l >= -1 && l <= 1);
      const ok = order.filter((l) => !blocked(next, l));
      const both = ok.filter((l) => !rows[1] || !blocked(rows[1], l));
      want = (both[0] ?? ok[0] ?? want);
    }
    if (want !== L.lane) L.steer(Math.sign(want - L.lane));
    if (next.kind === "hurdle" && L.body.armed === null && !L.body.busy) { const a = next.x - w.x; if (a < 3.5 && a > 2.0) L.jump(); }
  }
  const clear = !next || next.x - w.x > 7;
  L.setFast(clear && L.gap() < 4 && Math.abs(w.y - L.lane * TRACK.lane) < 0.25);
}

L.start();
let caught = null, minGap = 99, speedSum = 0, n = 0;
for (let i = 0; i * dt < secs; i++) {
  if (what === "auto") autoplay();
  L.step(); printLog();
  if (process.env.TRACE && i % 25 === 0) { const w = b.where(), r = L.segs.filter((s) => s.x > w.x - 0.5).slice(0, 2).map((s) => `${s.kind}@${s.x.toFixed(1)}:${(s.blocked || []).join('')}`); console.log(`${(i * dt).toFixed(1)} x ${w.x.toFixed(2)} y ${w.y.toFixed(2)} yaw ${w.yaw.toFixed(2)} ${L.body.mode} lane ${L.lane} v ${L.body.speed.toFixed(2)} gap ${L.gap().toFixed(1)} ${r.join(' ')}`); }
  speedSum += L.body.speed; n++; minGap = Math.min(minGap, L.gap());
  if (L.phase === "caught") { caught = +L.caughtAt.toFixed(1); break; }
}
const S = L.stats;
console.log("RESULT " + JSON.stringify({ what, seed, secs, wall_s: Math.round((Date.now() - t0) / 1000), metres: Math.round(L.dist), score: L.score(), caught_s: caught,
  jumps: S.jumps, clean: S.clean, knocked: S.knocked, ran_through: S.knockedWalking, run_outs: S.runOuts, too_early: S.tooEarly, not_lined: S.notLined, bumps: S.bumps, touch_s: +S.touchS.toFixed(1), falls: S.falls,
  mean_speed: +(speedSum / n).toFixed(2), jump_detail: L.jumpLog || [], min_gap: +minGap.toFixed(2), rows: L.segN }));
