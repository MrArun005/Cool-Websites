// node doglife_jumptest.mjs <mujoco.js> <site data folder> <seed> [seconds=240]
// force jump tasks repeatedly: how often Bolt lines up, clears, and falls
import fs from "fs"; import path from "path";
import * as live from "./live.js";
import * as D from "./doglife.js";
const [mjs, dir, seed, secs = "240"] = process.argv.slice(2);
const mj = await (await import(path.resolve(mjs))).default();
const read = (f) => JSON.parse(fs.readFileSync(path.join(dir, f)));
const L = D.makeLife(live, mj, read("bolt_run.json"), { gpu: read("dog_gpu360.json"), g18: read("dog_18g.json"), jump: read(process.env.JUMP || "jump_trot.json") }, read("dog_clips.json"), +seed);
L.mind.choose = () => "jump"; if (process.env.LAND) D.setLanding(process.env.LAND); L.mind.energy = 1; L.mind.drift = () => {};
let seen = 0; const fm = {}; const say0 = L.body.say.bind(L.body); L.body.say = (t, k) => { if (t.startsWith("Bolt fell")) { const f = L.body.fallInfo || {}; fm[f.mode + "<" + f.prev] = (fm[f.mode + "<" + f.prev] || 0) + 1; } say0(t, k); };
for (let i = 0; i < 50 * +secs; i++) { L.mind.step(0.02); L.body.step(0.02); while (L.body.events.length) { const e = L.body.events.shift(); L.mind.say(e.text, e.kind); } }

const c = {}; for (const e of L.mind.log) { const k = e.text.replace(/\d+ cm/, "N cm").replace(/\(.*\)/, ""); c[k] = (c[k] || 0) + 1; }
console.log(`seed ${seed}:`, JSON.stringify(L.mind.stats), "falls", L.body.falls, JSON.stringify(fm), JSON.stringify(c));
