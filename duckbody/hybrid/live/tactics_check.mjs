// tactics.js against tactics.py on the world models tactics_check.py recorded: every command field, every tick.
//   node duckbody/hybrid/live/tactics_check.mjs OUT.json
import fs from 'fs';
import { TeamTactics } from './tactics.js';

class ParkMiller {               // = tactics_check.ParkMiller
  constructor(seed = 0) { const s = Number.isInteger(seed) ? seed : Math.trunc(seed * 1000003); this.s = (s % 2147483646) + 1; }
  random() { this.s = (this.s * 16807) % 2147483647; return this.s / 2147483647; }
  next() { return this.random(); }
  uniform(a, b) { return a + (b - a) * this.random(); }
  choice(a) { return a[Math.min(a.length - 1, Math.floor(this.random() * a.length))]; }
}
const rec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const tac = {};
for (const [tn, seed] of Object.entries(rec.seeds)) tac[tn] = new TeamTactics(new ParkMiller(seed), r => new ParkMiller(r.random()));
let worst = 0, mism = 0, n = 0, first = null;
const close = (a, b) => (a === null && b === null) || (typeof a === 'number' && typeof b === 'number' && Math.abs(a - b) < 1e-9);
rec.ticks.forEach((tk, k) => {
  for (const tn of Object.keys(tk.wm)) {
    const out = tac[tn].step(tk.wm[tn]);
    for (const [name, want] of Object.entries(tk.cmds[tn])) {
      const got = out[name]; n++;
      let bad = [];
      for (const f of ['vx', 'vy', 'wz']) { worst = Math.max(worst, Math.abs(got[f] - want[f])); if (!close(got[f], want[f])) bad.push(f); }
      for (const f of ['kick', 'kind', 'mate', 'cam', 'state', 'role']) if ((got[f] ?? null) !== (want[f] ?? null)) bad.push(f);
      const gh = got.head, wh = want.head;
      if ((gh === null) !== (wh === null) || (gh && !gh.every((v, i) => close(v, wh[i])))) bad.push('head');
      if (bad.length) { mism++; if (!first) first = { tick: k, team: tn, name, bad, got, want }; }
    }
  }
});
console.log(`${n} commands compared over ${rec.ticks.length} ticks: ${mism} differ, max twist difference ${worst.toExponential(2)}`);
if (first) console.log('first difference', JSON.stringify(first));
process.exit(mism ? 1 : 0);
