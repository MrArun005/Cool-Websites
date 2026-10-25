// Run Bolt's Agility Cup headless: every competitor in the bundle runs the same courses.
//   node agility_test.mjs <mujoco.js> <bolt_live.json> [courses] [standard|championship]
import fs from "fs";
import { Bolt } from "./live.js";
import { buildAgility, Round } from "./agility.js";

const [mujocoJs, bundlePath, courses = "10", cls = "standard"] = process.argv.slice(2);
const mj = await (await import(mujocoJs)).default();
const bundle = JSON.parse(fs.readFileSync(bundlePath));
const bolt = new Bolt(mj, bundle);
const dogs = bundle.competitors || { [bundle.brains.walk.source]: { ...bundle.brains.walk, sharp_turns: bundle.constants.sharp_turns } };
for (const [name, spec] of Object.entries(dogs)) {
  bolt.useWalk(spec, spec.sharp_turns);
  let clear = 0, elim = 0, times = [], faults = [], why = {};
  for (let c = 0; c < +courses; c++) {
    const course = buildAgility(500 + c, { cls });
    const r = new Round(bolt, course, bundle.start, name);
    while (!r.over && r.t < 300) r.step();
    if (!r.over) why["timed out at " + (r.obstacle ? r.obstacle.title : "finish")] = (why["timed out at " + (r.obstacle ? r.obstacle.title : "finish")] || 0) + 1;
    if (r.phase === "eliminated") { elim++; why["fell at " + (r.obstacle ? r.obstacle.title : "finish")] = (why["fell at " + (r.obstacle ? r.obstacle.title : "finish")] || 0) + 1; continue; }
    if (r.phase !== "finished") continue;
    times.push(r.clock); faults.push(r.total); clear += r.total === 0;
    for (const n of r.notes) why[n] = (why[n] || 0) + 1;
    if (r.timeFaults) why["time faults"] = (why["time faults"] || 0) + 1;
  }
  const med = (a) => a.sort((x, y) => x - y)[a.length >> 1];
  console.log(`${name.padEnd(14)} clear rounds ${clear}/${courses}, eliminated ${elim}, median time ${med(times)?.toFixed(1)} s, median faults ${med(faults)}  ${JSON.stringify(why)}`);
}
