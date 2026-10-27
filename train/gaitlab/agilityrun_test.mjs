// Bolt's Agility Run headless: a scripted handler (agilityrun.js Bot) runs each level, with auto-steer and
// auto-jump on, and the judge scores it. Reports per level: finished, time, faults, falls, jumps, ranks, and how
// often each command was carried out (Bolt in that gait or pose within 3 s, without falling on the way).
//   node agilityrun_test.mjs <mujoco.js> <site data folder> [runs per level] [levels, e.g. 1,2,3] [canter]
import fs from "fs";
import path from "path";
import { Bolt } from "./live.js";
import { rng, posture } from "./game.js";
import { LEVELS, buildCourse, Run, Bot, makeDog } from "./agilityrun.js";

const [mujocoJs, dataDir, runsArg = "4", levelsArg = "1,2,3", style = ""] = process.argv.slice(2);
const mj = await (await import(path.resolve(mujocoJs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dataDir, f)));
const bundle = read("bolt_run.json"), bolt = new Bolt(mj, bundle);
const dog = makeDog(bolt, read("imitate_brain.json"), read("imitate_clips.json"));
const reached = (r, c) => (["sit", "lie"].includes(c) ? posture(bolt) === c : c === "stand" ? r.body.mode === "stand" : r.body.gait === c);
for (const n of levelsArg.split(",").map(Number)) {
  const rows = [], cmds = {};
  for (let k = 0; k < +runsArg; k++) {
    const course = buildCourse(n - 1), run = new Run(bolt, dog, course, bundle.start), bot = new Bot(run, { style: style || "walk", rand: rng(((n * 1000 + k) * 2654435761) >>> 0) });
    const noise = rng(k * 7919 + n);   // a different start each run: tiny joint-speed noise, as a sensor would have
    for (let i = 6; i < 18; i++) bolt.data.qvel[i] += 0.1 * (noise() - 0.5);
    const orig = run.command.bind(run);
    let pending = null;
    const settle = (ok) => { if (!pending) return; const c = cmds[pending.c] || (cmds[pending.c] = { ok: 0, n: 0 }); c.n++; c.ok += ok; pending = null; };
    run.command = (c) => { settle(false); orig(c); if (c !== "jump") pending = { c, t: run.t, falls: run.stats.falls }; };
    const t0 = Date.now();
    let lastMode = null, seen = 0;
    while (!run.over && run.t < 400) {
      bot.step(); run.step();
      if (process.env.DEBUG) {
        const q = run.local(), b = run.body;
        if (b.mode !== lastMode) { console.log(`    ${run.t.toFixed(2)} mode ${b.mode} cmd ${run.cmd} s ${run.lineS.toFixed(2)} off ${run.lineOff.toFixed(2)} v ${b.speed.toFixed(2)} steer ${(b.want.steer ?? 0).toFixed(2)} armed ${b.armed} ahead ${bolt.hurdleAhead().toFixed(2)}`); lastMode = b.mode; }
        while (seen < run.events.length) { const e = run.events[seen++]; console.log(`    ${run.t.toFixed(2)} EVENT ${e.text}`); }
      }
      if (pending) {
        if (run.stats.falls > pending.falls) settle(false);
        else if (reached(run, pending.c)) settle(true);
        else if (run.t - pending.t > (["sit", "lie"].includes(pending.c) ? 4 : 3)) settle(false);
      }
    }
    settle(false);
    const res = run.result();
    rows.push(res);
    console.log(`  L${n} run ${k + 1}: ${res.finished ? "finished" : "NOT finished"} time ${res.time.toFixed(1)} s, faults ${res.faults.length} [${res.faults.join("; ")}], total ${res.total.toFixed(1)} / SCT ${res.sct} -> ${res.rank}; falls ${res.falls} ${JSON.stringify(res.fellIn)}, jumps ${res.cleanJumps}/${res.jumps} clean  (${((Date.now() - t0) / 1000).toFixed(0)} s wall)`);
  }
  const mean = (a) => a.reduce((s, x) => s + x, 0) / Math.max(1, a.length), fin = rows.filter((r) => r.finished);
  const ranks = {}; rows.forEach((r) => (ranks[r.rank] = (ranks[r.rank] || 0) + 1));
  console.log(`L${n} ${LEVELS[n - 1].name}: finished ${fin.length}/${rows.length}, mean time ${mean(fin.map((r) => r.time)).toFixed(1)} s, mean faults ${mean(rows.map((r) => r.faults.length)).toFixed(1)}, ` +
    `falls ${rows.reduce((s, r) => s + r.falls, 0)} in ${rows.length} runs ${JSON.stringify(rows.reduce((a, r) => { for (const [m, c] of Object.entries(r.fellIn)) a[m] = (a[m] || 0) + c; return a; }, {}))}, clean jumps ${rows.reduce((s, r) => s + r.cleanJumps, 0)}/${rows.reduce((s, r) => s + r.jumps, 0)}, ranks ${JSON.stringify(ranks)}`);
  console.log(`   commands carried out: ${Object.entries(cmds).map(([c, v]) => `${c} ${v.ok}/${v.n}`).join(", ")}`);
}
