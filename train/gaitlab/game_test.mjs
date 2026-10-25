// Play Bolt's Castle headless: many games with new courses, counting passes, falls and time.
//   node game_test.mjs <mujoco.js> <bolt_live.json> [games] [level]
import fs from "fs";
import { Bolt } from "./live.js";
import { Game } from "./game.js";

const [mujocoJs, bundlePath, games = "20", level = "1"] = process.argv.slice(2);
const mj = await (await import(mujocoJs)).default();
const bundle = JSON.parse(fs.readFileSync(bundlePath));
const bolt = new Bolt(mj, bundle);
const tally = {}, notes = {};
let falls = 0, finished = 0, times = [], scores = [], livesLost = 0, tumbles = 0;
for (let g = 0; g < +games; g++) {
  const game = new Game(bolt, 1000 + g, bundle.start, { level: +level });
  while (!game.over && game.t < 400) game.step();
  livesLost += 3 - Math.max(0, game.lives); tumbles += game.falls;
  if (game.phase === "eliminated") falls++;
  if (game.phase === "finished") { finished++; times.push(game.t); }
  if (!game.over) console.log(`game ${g}: still going after 400 s at ${game.station.title} (${game.task.stage})`);
  scores.push(game.score);
  for (const r of game.results) {
    const k = r.kind === "shout" ? r.title : r.kind === "fall" ? "falls" : r.kind;
    tally[k] = tally[k] || [0, 0]; tally[k][0] += r.passed; tally[k][1] += 1;
    if (!r.passed) (notes[k] = notes[k] || []).push(r.note);
  }
}
console.log(`level ${level}: ${games} games: cleared ${finished}, eliminated ${falls}, lives lost ${livesLost}, falls ${tumbles}, median time ${times.sort((a, b) => a - b)[times.length >> 1]?.toFixed(0)} s, median score ${scores.sort((a, b) => a - b)[scores.length >> 1]}`);
for (const [k, [p, n]] of Object.entries(tally)) console.log(`  ${k.padEnd(18)} passed ${p}/${n}${notes[k] ? "  fails: " + [...new Set(notes[k])].join(" | ") : ""}`);
