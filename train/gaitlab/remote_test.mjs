// Bolt Remote headless (simulation). Two parts:
//   auto: Auto self mode for N simulated seconds per seed; Bolt only knows what its scanner has seen. Reports falls,
//         bumps into solid things (any contact of Bolt with a prop), kerb crossings (body centre onto the road),
//         hurdles seen and how well it estimated them, jumps tried and clean (bar not knocked, all paws past the REAL
//         bar line, no fall), how much of the yard it mapped, what it did.
//   remote: command handling: drive at a trot by a 20 Hz command stream, then stop sending (dead-man), switch to
//         auto and back mid-stride, and drive straight at the house at a trot (the collision guard must stop it).
//   node remote_test.mjs <mujoco.js> <site data folder> auto [seconds=240] [seed=1]
//   node remote_test.mjs <mujoco.js> <site data folder> remote [seed=1]
import fs from "fs";
import path from "path";
import * as live from "./live.js";
import { RemoteLife } from "./remote.js";
import { YARD } from "./doglife.js";

const [mujocoJs, dataDir, what = "auto", a1, a2] = process.argv.slice(2);
const mj = await (await import(path.resolve(mujocoJs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f)));
const bundle = read("bolt_run.json");
const brains = { gpu: read("dog_gpu360.json"), g18: read("dog_18g.json"), jump: read("jump.json") };
const clips = read("dog_clips.json");
const seed = +(what === "auto" ? a2 || 1 : a1 || 1);
const L = new RemoteLife(live, mj, bundle, brains, clips, seed);
const b = L.bolt, dt = b.dt, t0 = Date.now();
const fmt = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
let seen = 0;
const printLog = (quiet) => { while (seen < L.log.length) { const e = L.log[seen++]; if (!quiet) console.log(`  ${fmt(e.t)} [${L.body.mode}] ${e.text}`); } };
// solid things touched: contacts between Bolt and a prop (the yard's solid objects), counted as new touches
// bumps: episodes of touching (contacts less than 1 s apart count as one), plus the total time touching
let lastTouch = -9, bumps = 0, touchTime = 0, simT = 0, touching = false;
const bumpCheck = () => { simT += b.dt; if (b.touchingProps().length) { if (simT - lastTouch > 1.0) bumps++; lastTouch = simT; touchTime += b.dt; } };
const paws = () => L.gpu.feet.map((g) => [b.data.geom_xpos[3 * g], b.data.geom_xpos[3 * g + 1]]);

if (what === "auto") {
  const secs = +(a1 || 240);
  L.setMode("auto");
  let onRoad = 0, wasRoad = false, minY = 99, jumpTrue = [], cur = null;
  for (let i = 0; i * dt < secs; i++) {
    L.step(i * dt);
    bumpCheck();
    const w = b.where(), road = w.y < YARD.kerb; minY = Math.min(minY, w.y);
    if (road && !wasRoad) onRoad++; wasRoad = road;
    // the true outcome of each jump: the real bar's line, the real bar
    const m = L.body.mode;
    if (m === "jump" && !cur) { const k = L.body.jumpInfo.k, H = b.real[k]; cur = { k, H: { at: H.at.slice(), yaw: H.yaw, h: H.h }, est: { at: b.hurdleAt.slice(), dir: b.hurdleDir.slice(), h: b.hurdleH }, fell: false, past: false, falls0: L.body.falls }; }
    if (cur) {
      const ux = Math.cos(cur.H.yaw), uy = Math.sin(cur.H.yaw), side = b.hurdleDir[0] * ux + b.hurdleDir[1] * uy > 0 ? 1 : -1;
      if (paws().every(([x, y]) => side * ((x - cur.H.at[0]) * ux + (y - cur.H.at[1]) * uy) > 0.05)) cur.past = true;
      if (!["jump", "land", "settle", "bridge"].includes(m) && L.body.mode !== "jump") {
        const dropped = b.barDropped(cur.k), fell = L.body.falls > cur.falls0;
        const eDir = Math.atan2(cur.est.dir[1], cur.est.dir[0]), tDir = cur.H.yaw;
        jumpTrue.push({ k: cur.k, clean: cur.past && !dropped && !fell, past: cur.past, dropped, fell,
          estErr: { pos_cm: Math.round(100 * Math.hypot(cur.est.at[0] - cur.H.at[0], cur.est.at[1] - cur.H.at[1])), yaw_deg: Math.round(Math.abs(Math.atan2(Math.sin(2 * (eDir - tDir)), Math.cos(2 * (eDir - tDir)))) / 2 * 180 / Math.PI), h_cm: Math.round(100 * (cur.est.h - cur.H.h)) } });
        cur = null;
      }
    }
    printLog(process.env.QUIET === "1");
  }
  // mapping coverage of the lawn and sidewalk; hurdle estimates against the truth
  const Y = YARD, M = L.map; let cells = 0, known = 0;
  for (let x = Y.lawn.x0 + 0.5; x < Y.lawn.x1; x += 0.5) for (let y = Y.sidewalk.y0 + 0.2; y < Y.lawn.y1 - 0.5; y += 0.5) { cells++; if (M.known(M.idx(x, y))) known++; }
  const est = [...M.hurdles.keys()].map((k) => { const e = M.hurdle(k), H = b.real[k]; return e && { k, pos_cm: Math.round(100 * Math.hypot(e.at[0] - H.at[0], e.at[1] - H.at[1])), h_est: e.h && +e.h.toFixed(2), h_true: H.h, from: e.from, width: +e.width.toFixed(2), points: e.n }; }).filter(Boolean);
  const S = L.auto.stats, clean = jumpTrue.filter((j) => j.clean).length;
  const res = { seed, secs, wall_s: Math.round((Date.now() - t0) / 1000), falls: L.body.falls, recoveries: L.body.recoveries, bumps, touch_s: +touchTime.toFixed(1), road_crossings: onRoad, min_y: +minY.toFixed(2), kerb: Y.kerb,
    kerb_stops: S.kerbStops, mapped_pct: Math.round((100 * known) / cells), hurdles_seen: est, jumps: jumpTrue.length, jumps_clean_true: clean, jumps_detail: jumpTrue,
    mind_stats: S, tasks: L.auto.counts, distance_m: Math.round(S.distance) };
  console.log("RESULT " + JSON.stringify(res));
} else {
  // ---- remote command handling
  const out = {};
  let now = 0, seq = 0;
  const send = (c) => L.pilot.receive({ mode: "manual", vx: 0, turn: 0, seq: ++seq, t: now * 1000, ...c }, now);
  const run = (secs, each) => { for (let i = 0; i * dt < secs; i++) { now += dt; if (each) each(i); L.step(now); bumpCheck(); printLog(process.env.QUIET === "1"); } };
  // 1) open lawn: point Bolt east from the start spot, trot by a 20 Hz command stream for 4 s
  b.teleport(bundle.start, [-1, -8.5], 0); L.reset(seed); b.teleport(bundle.start, [-1, -8.5], 0);
  run(1.5);   // look around first (the map starts empty)
  run(4, (i) => { if (i % 3 === 0) send({ vx: 0.65 }); });
  const v0 = L.body.speed, mode0 = L.body.mode;
  // 2) the remote goes silent: time until Bolt notices (0.5 s) and until it is standing
  const silentAt = now; let noticed = null, stood = null;
  run(5, () => { if (noticed === null && L.pilot.deadman) noticed = now - silentAt; if (stood === null && L.body.mode === "stand" && L.body.speed < 0.15) stood = now - silentAt; });
  out.deadman = { speed_before: +v0.toFixed(2), gait_before: mode0, noticed_s: noticed && +noticed.toFixed(2), standing_after_s: stood && +stood.toFixed(2), falls: L.body.falls };
  // 3) mode switches: 10 rounds of Auto (8-20 s) then manual (a random walk or trot command, maybe turning, 4 s);
  //    a fall within 3 s of a switch counts against that switch
  let sw = { to_auto: 0, to_manual: 0, falls_after_auto: 0, falls_after_manual: 0, gaits_at_switch: {} }, rng = seed * 7919;
  const rnd = () => ((rng = (rng * 1103515245 + 12345) % 2147483648) / 2147483648);
  for (let r = 0; r < 10; r++) {
    L.setMode("auto"); sw.to_auto++; let f = L.body.falls; run(3); sw.falls_after_auto += L.body.falls - f; run(5 + 12 * rnd());
    const g = L.body.mode; sw.gaits_at_switch[g] = (sw.gaits_at_switch[g] || 0) + 1;
    const c = { vx: rnd() < 0.5 ? 0.35 : 0.65, turn: rnd() < 0.5 ? 0 : (rnd() - 0.5) * 0.8 };
    L.setMode("manual"); sw.to_manual++; f = L.body.falls;
    run(3, (i) => { if (i % 3 === 0) send(c); }); sw.falls_after_manual += L.body.falls - f; run(1, (i) => { if (i % 3 === 0) send(c); });
  }
  out.mode_switch = sw;
  // 4) collision guard: from 8 m south of the house wall, trot straight at it (the command keeps asking full speed)
  const house = YARD.house, start = [-3.5, house.y0 - 0.5 - 8];
  L.reset(seed); b.teleport(bundle.start, start, Math.PI / 2); bumps = 0; touchTime = 0; lastTouch = -9;
  run(1.0);
  let closest = 99, stopT = null;
  run(14, (i) => { if (i % 3 === 0) send({ vx: 1.0 }); const w = b.where(); closest = Math.min(closest, house.y0 - w.y); if (stopT === null && L.pilot.guard && L.pilot.guard.cap === 0) stopT = now; });
  out.guard_wall = { target: "house wall (porch in the way at y 7.4-9)", closest_body_centre_to_wall_m: +closest.toFixed(2), bumps, touch_s: +touchTime.toFixed(1), falls: L.body.falls, guard: L.pilot.guard && L.pilot.guard.why, end_y: +b.where().y.toFixed(2) };
  // 5) kerb guard: on the sidewalk side of the lawn, walk straight south at the road
  L.reset(seed); b.teleport(bundle.start, [6, -10], -Math.PI / 2); bumps = 0; touchTime = 0; lastTouch = -9;
  run(1.0);
  let minY = 99;
  run(12, (i) => { if (i % 3 === 0) send({ vx: 0.35 }); minY = Math.min(minY, b.where().y); });
  out.guard_kerb = { kerb_y: YARD.kerb, min_body_centre_y: +minY.toFixed(2), nose_margin_m: +(minY - 0.35 - YARD.kerb).toFixed(2), falls: L.body.falls, guard: L.pilot.guard && L.pilot.guard.why };
  out.wall_s = Math.round((Date.now() - t0) / 1000);
  console.log("RESULT " + JSON.stringify(out));
}
