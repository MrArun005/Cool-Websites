// Bolt at home, headless: ten minutes (or more) of pet life with a scripted person who comes home,
// gives commands, praises, ignores Bolt for a while, throws the ball, goes out and comes back, then
// gives random commands. Reports falls, how often and how long each behaviour ran, the moods, and
// command obedience (how many were carried out, and how long it took).
//   node home_test.mjs <mujoco.js> <bolt_live.json> [minutes=10] [seed=1] [-v]
import fs from "fs";
import { Bolt } from "./live.js";
import { Pet, Memory, memoryStore, understand } from "./home.js"; import * as H from "./home.js";

const args = process.argv.slice(2), verbose = args.includes("-v");
const [mujocoJs, bundlePath, minutes = "10", seed = "1"] = args.filter((a) => a !== "-v");
const mj = await (await import(mujocoJs)).default();
const bundle = JSON.parse(fs.readFileSync(bundlePath));
const bolt = new Bolt(mj, bundle);
const pet = new Pet(bolt, bundle.start, { memory: new Memory(memoryStore()), seed: +seed });

let s = (+seed * 7919) >>> 0;
const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
const say = (text) => () => pet.hear(text);
const walk = (pts) => () => { pet.walkRoute = pts.slice(); };
// the scripted person (times in s)
const script = [
  [25, () => pet.arrive()],
  [45, say("Bolt, sit please")], [58, say("good boy")], [65, say("come here buddy")], [85, say("lie down")],
  [100, say("stay")], [118, say("okay, go play")], [135, () => pet.throwBall()], [175, () => pet.throwBall()],
  [210, say("follow me")], [211, walk([[-2.5, 0.5], [-2.5, -2.0], [2.0, -0.5], [3.5, 1.5], [0.5, 1.5], [0, -2.2]])],
  [260, say("jump!")], [285, say("good boy!")],
  // 290-450: the person stands still and ignores Bolt
  [450, say("hey Bolt, who's a good boy")], [453, say("come here and sit")], [468, say("you look tired")], [500, say("fetch!")],
  [525, () => pet.leave()], [560, () => pet.arrive()], [575, say("my name is Arun")], [578, say("what's my name?")],
];
const RANDOM = ["sit", "lie down", "come here", "stand up", "spin around", "are you hungry?", "fetch", "jump", "follow me", "stay", "come", "sit down please", "Bolt, down"];
let next = 585, seen = 0, logged = 0;
const T = +minutes * 60;
const wall = Date.now();
while (pet.t < T) {
  while (script.length && pet.t >= script[0][0]) script.shift()[1]();
  if (pet.t >= next && pet.person.present) {
    const text = RANDOM[Math.floor(rand() * RANDOM.length)];
    pet.hear(text);
    if (/follow/.test(text)) pet.walkRoute = [[rand() * 6 - 3, rand() * 4 - 2], [rand() * 6 - 3, rand() * 4 - 2]];
    if (rand() < 0.3) pet.hear("good boy");
    next = pet.t + 14 + 10 * rand();
  }
  if (pet.walkRoute && pet.walkRoute.length && !pet.person.goal) pet.walkPerson(pet.walkRoute.shift());
  pet.step(); if (pet.t > 450 && pet.t < 457 && Math.round(pet.t*50) % 5 === 0) { const w = bolt.where(); console.log("TR", pet.t.toFixed(1), pet.current?.name, pet.current?.intent||'', pet.body.state, pet.body.pose, pet.cmd.command, pet.cmd.speed?.toFixed(2), pet.cmd.turn?.toFixed(2), w.x.toFixed(2), w.y.toFixed(2), w.yaw.toFixed(2), 'z', w.z.toFixed(2), 'up', bolt.up().toFixed(2), 'person', pet.person.x.toFixed(2), pet.person.y.toFixed(2), 'ball', JSON.stringify(H.ballPos(bolt).map(v=>+v.toFixed(2)))); }
  if (verbose) while (seen < pet.events.length) { const e = pet.events[seen++]; console.log(e.t.toFixed(1).padStart(6), e.kind.padEnd(6), e.text); }
  if (verbose && pet.t - logged >= 10) {
    logged = pet.t; const w = bolt.where();
    console.log(`        ${pet.t.toFixed(0)}s ${pet.label()} | ${pet.mood()} | ${pet.body.state}/${pet.body.pose} at (${w.x.toFixed(1)}, ${w.y.toFixed(1)}) | ` +
      Object.entries(pet.drives).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" "));
  }
}
const pct = (x) => `${Math.round(100 * x)}%`;
console.log(`\n${minutes} min of pet life (seed ${seed}), ${((Date.now() - wall) / 1000).toFixed(0)} s to simulate: falls ${pet.falls}`);
console.log("behaviours (started / share of time):");
const total = Object.values(pet.stats.time).reduce((a, b) => a + b, 0);
for (const [k, v] of Object.entries(pet.stats.time).sort((a, b) => b[1] - a[1])) console.log(`  ${k.padEnd(9)} ${String(pet.stats.runs[k] || "-").padStart(3)}  ${pct(v / total).padStart(4)}`);
console.log("moods:", Object.entries(pet.stats.moods).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${pct(v / total)}`).join(", "));
const cmds = pet.stats.commands, by = {};
// a command counts once Bolt has carried it out; ones replaced by a newer command (or by the person
// going out) before Bolt finished are listed apart, not counted as obeyed or failed
const judged = cmds.filter((c) => c.done !== null || !c.superseded), cut = cmds.length - judged.length;
for (const c of judged) { const b = (by[c.intent] = by[c.intent] || { n: 0, ok: 0, lat: [] }); b.n++; if (c.done !== null) { b.ok++; b.lat.push(c.latency); } }
const ok = judged.filter((c) => c.done !== null), lats = ok.map((c) => c.latency).sort((a, b) => a - b);
console.log(`commands: obeyed ${ok.length}/${judged.length} (${pct(ok.length / Math.max(1, judged.length))}), median latency ${lats[lats.length >> 1]?.toFixed(1)} s, 90th percentile ${lats[Math.floor(0.9 * (lats.length - 1))]?.toFixed(1)} s; ${cut} replaced before done`);
for (const [k, b] of Object.entries(by)) console.log(`  ${k.padEnd(8)} ${b.ok}/${b.n}  latency ${b.lat.map((x) => x.toFixed(1)).join(", ") || "-"}`);
console.log("memory:", JSON.stringify(pet.memory.data));
// the parser on phrasings it was not written for
const CHECK = [["could you sit down for me", "sit"], ["lay down", "lie"], ["come here buddy", "come"], ["you look tired", "nap"], ["let's play!", "play"],
  ["good boy, now stay", "praise,stay"], ["bring me the ball", "fetch"], ["over here Bolt", "come"], ["time for bed", "nap"], ["wanna go for a walk?", "follow"],
  ["Bolt!", "attend"], ["hop over the bar", "jump"], ["are you hungry?", "bowl"], ["no, bad dog", "scold"], ["chase your tail", "spin"], ["hold still", "stay"],
  ["get up", "stand"], ["I'm home!", "greet"], ["call me Sam", "name_set"], ["play dead", "lie"]];
const right = CHECK.filter(([t, want]) => understand(t, {}).map((x) => x.intent).join(",") === want);
console.log(`parser: ${right.length}/${CHECK.length} phrasings understood${right.length < CHECK.length ? "; missed: " + CHECK.filter((c) => !right.includes(c)).map(([t, w]) => `"${t}" -> ${understand(t, {}).map((x) => x.intent)} (want ${w})`).join("; ") : ""}`);
