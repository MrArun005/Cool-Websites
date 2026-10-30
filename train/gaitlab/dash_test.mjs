// Bolt Dash headless (simulation). Plays real runs through the same Dash class as the page, pressing only the
// player's buttons (left, right, jump, down):
//   auto     an autoplayer that sees the rows ahead (as the player sees the screen), picks a lane with a way
//            through and hops each bar at the right moment, with no reaction delay
//   human    a human-like player: sees 12 m ahead, reacts 0.25 s (wall clock) after something comes into view,
//            hop timing off by up to +-0.08 s, now and then late (1 in 8) or picking the wrong lane (1 in 25)
//   idle     Go, then no input: Bolt runs straight on into things, so it must get caught
//   latency  on a clear stretch: lane changes and hops, the time from the key to visible motion
//   node dash_test.mjs <mujoco.js> <site data folder> auto|human|idle|latency [seconds=180] [seeds=1]
// Prints one RESULT line per run: metres, score, treats, bars cleared/seen/knocked, crashes, falls, caught (s), and
// the cost of the physics plus brains: ms per control step (mean, p95, max) and per 60 Hz frame at the run's clock.
import fs from "fs";
import path from "path";
import { performance } from "perf_hooks";
import * as live from "./live.js";
import { Dash, TRACK, ASSIST } from "./dash.js";

const [mujocoJs, dataDir, what = "auto", a1, a2] = process.argv.slice(2);
const mj = await (await import(path.resolve(mujocoJs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f)));
const bundle = read("bolt_run.json");
const brains = { gpu: read("dog_gpu360.json"), g18: read("dog_18g.json") };
const clips = read("dog_clips.json");
const secs = +(a1 || 180), seeds = String(a2 || "1").split(",").map(Number);
const L = new Dash(live, mj, bundle, brains, clips, seeds[0]);
const b = L.bolt, dt = b.dt;
const HOP_LEAD = +(process.env.HOP_LEAD || 0.3);   // s before the torso reaches a bar that the autoplayer hops

// a row's gates: the lanes it shuts first (at row.x) and, for two-part rows, next (at row.at2); a bar shuts none
function gates(row, x) {
  const g = [];
  if (row.blocked && row.blocked.length && row.x + 0.5 > x) g.push(new Set(row.blocked));
  if (row.blocked2 && row.at2 + 0.5 > x) g.push(new Set(row.blocked2));
  return g;
}
// the lane to be in for the gates in sight: the nearest one open at the next gate, preferring one open at the gate after
function plan(look) {
  const x = b.data.qpos[0], g = L.ahead(look).flatMap((r) => gates(r, x));
  if (!g.length) return L.lane;
  const s0 = g[0], s1 = g[1] || new Set();
  const order = [L.lane, L.lane - 1, L.lane + 1, L.lane - 2, L.lane + 2].filter((l) => l >= -1 && l <= 1);
  const ok = order.filter((l) => !s0.has(l)), both = ok.filter((l) => !s1.has(l));
  return both[0] ?? ok[0] ?? L.lane;
}
const goTo = (lane) => { while (L.lane < lane && L.left()); while (L.lane > lane && L.right()); };

// a player: decide(now) is called every step; inputs go out after the player's reaction delay
function player(kind, seed) {
  let r = seed * 9301 + 49297; const R = () => ((r = (r * 1103515245 + 12345) >>> 0) / 4294967296);
  const queue = [], seen = new Set(), hopped = new Set();
  const press = (at, f) => queue.push({ at, f });
  return (t) => {
    const v = Math.max(1, L.body.speed), x = b.data.qpos[0], clock = L.clock();
    if (kind === "auto") {
      goTo(plan(v * 1.3 + 1.2));
      for (const row of L.ahead(6)) if (row.kind === "bar" && !hopped.has(row.n) && row.x - x - 0.1 < v * HOP_LEAD) { hopped.add(row.n); L.jump(); }
    } else if (kind === "human") {
      const react = 0.25 * clock;   // 0.25 s of wall clock is more simulated time when the clock runs fast
      for (const row of L.ahead(12)) {
        if (!seen.has(row.n)) {   // something new came into view: decide the lane once, react after the delay
          seen.add(row.n);
          const late = R() < 1 / 8 ? 0.2 * clock : 0, wrong = R() < 1 / 25;
          press(t + react + late, () => goTo(wrong ? [-1, 0, 1][Math.floor(R() * 3)] : plan(v * 1.3 + 1.2)));
        }
        // a bar: the press is timed by eye, so it is a little early or late
        if (row.kind === "bar" && !hopped.has(row.n) && row.x - x - 0.1 < v * (HOP_LEAD + react)) {   // seen early enough to press on time
          hopped.add(row.n); press(t + react + (R() - 0.5) * 0.16 * clock, () => L.jump());
        }
      }
      // keep re-checking the lane a moment later too (people correct themselves)
      if (Math.round(t / dt) % 25 === 0) press(t + react, () => goTo(plan(v * 1.1 + 1.0)));
    }
    queue.sort((a, c) => a.at - c.at);
    while (queue.length && queue[0].at <= t) queue.shift().f();
  };
}

function stats(a) { const s = [...a].sort((p, q) => p - q); return { mean: +(s.reduce((p, q) => p + q, 0) / s.length).toFixed(2), p95: +s[Math.floor(0.95 * (s.length - 1))].toFixed(2), max: +s[s.length - 1].toFixed(2) }; }

if (what === "latency") {
  // a lane change: ms until the torso has moved 5 cm sideways and until it is within 10 cm of the new lane; a hop:
  // ms until it has risen 3 cm. In simulated time; the page adds at most one frame (17 ms) before the next step.
  L.reset(seeds[0]); L.start();
  for (const row of L.rows) for (const o of row.items) L.park(o);
  L.rows = []; L.nextX = 1e9;   // a clear track
  for (let i = 0; i < 150; i++) L.step();
  const lane = [], inLane = [], hop = [];
  for (let k = 0; k < 16; k++) {
    const y0 = b.data.qpos[1], z0 = b.data.qpos[2];
    if (k % 3 === 2) { L.jump(); let n = 0; while (b.data.qpos[2] < z0 + 0.03 && n < 50) { L.step(); n++; } hop.push(n * dt * 1000); for (let i = 0; i < 60; i++) L.step(); continue; }
    const dir = L.lane === 1 ? -1 : L.lane === -1 ? 1 : (k % 2 ? 1 : -1);
    L.steer(dir); const ty = L.lane * TRACK.lane;
    let n = 0, got = null;
    while (n < 100) { L.step(); n++; if (got === null && Math.abs(b.data.qpos[1] - y0) > 0.05) got = n; if (Math.abs(b.data.qpos[1] - ty) < 0.1) break; }
    lane.push(got * dt * 1000); inLane.push(n * dt * 1000);
    for (let i = 0; i < 40; i++) L.step();
  }
  console.log("RESULT " + JSON.stringify({ what, lane_visible_ms: stats(lane), lane_in_ms: stats(inLane), hop_visible_ms: stats(hop), falls: L.stats.falls, mode: L.body.mode, assist: ASSIST }));
  process.exit(0);
}

for (const seed of seeds) {
  L.reset(seed);
  const t0 = performance.now(), cost = [], play = what === "idle" ? () => {} : player(what, seed);
  let caught = null, minGap = 99, speedSum = 0, n = 0, steps = 0;
  L.start();
  for (let i = 0; i * dt < secs; i++) {
    play(L.t);
    const c0 = performance.now(); L.step(); cost.push(performance.now() - c0); steps++;
    speedSum += L.body.speed; n++; minGap = Math.min(minGap, L.gap);
    for (const e of L.events.splice(0)) if (process.env.LOG && !["treat", "hop", "land"].includes(e.type)) console.log(`  ${e.t.toFixed(1)} ${e.type} ${e.kind || e.h || ""} gap ${L.gap.toFixed(1)} x ${b.data.qpos[0].toFixed(1)} y ${b.data.qpos[1].toFixed(2)} lane ${L.lane} [${L.body.mode}]`);
    if (process.env.TRACE && i % 25 === 0) console.log((i * dt).toFixed(1), L.body.mode, b.data.qpos[0].toFixed(1), b.data.qpos[1].toFixed(2), b.data.qpos[2].toFixed(2), L.lane, L.body.speed.toFixed(2), L.gap.toFixed(1), L.ahead(8).map((r) => `${r.kind}@${(r.x - b.data.qpos[0]).toFixed(1)}:${(r.blocked || []).join("")}`).join(" "));
    if (L.phase === "caught") { caught = +L.caughtAt.toFixed(1); break; }
  }
  const S = L.stats, c = stats(cost), clock = L.clock();
  console.log("RESULT " + JSON.stringify({ what, seed, secs, metres: Math.round(L.dist), score: L.score(), caught_s: caught, treats: S.treats,
    bars: `${S.bars}/${S.barsSeen}`, knocked: S.knocked, crashes: S.crashes, falls: S.falls, hops: S.hops, best_mult: S.bestMult, lane_changes: S.lanes,
    mean_speed: +(speedSum / n).toFixed(2), top_speed: +S.topSpeed.toFixed(2), end_clock: +clock.toFixed(2), min_gap: +minGap.toFixed(2),
    step_ms: c, frame_ms_at_clock: +(c.mean * clock * 50 / 60).toFixed(2), wall_s: +((performance.now() - t0) / 1000).toFixed(1), sim_steps: steps }));
}
