// A live match outside the browser (same code as the page): node match_node.mjs PAGE_DIR [5v5|3v3] [sim_seconds=120] [seed=1]
// Run from a folder whose node_modules has @mujoco/mujoco@3.15.0.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { Game } from './live_game.js';
import { MlpPolicy } from './onnx_mlp.js';

const require = createRequire(path.join(process.cwd(), 'x.js'));
const loadMujoco = (await import(require.resolve('@mujoco/mujoco'))).default;
const [dir, tag = '5v5', secArg = '120', seedArg = '1'] = process.argv.slice(2);
const mj = await loadMujoco();
const pols = {};
for (const b of ['velstand', 'ball_kick_left', 'ball_kick_right', 'roulade']) {
  const buf = fs.readFileSync(path.join(dir, 'policies', b + '.onnx'));
  pols[b] = new MlpPolicy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), b);
}
const runBrains = (name, X, N) => pols[name].runBatch(X, N);
const g = new Game(mj, fs.readFileSync(path.join(dir, `live_model_${tag}.xml`), 'utf8'), JSON.parse(fs.readFileSync(path.join(dir, `live_meta_${tag}.json`), 'utf8')),
  JSON.parse(fs.readFileSync(path.join(dir, 'live_moves.json'), 'utf8')), runBrains, { seed: +seedArg });
g.onEvent = e => console.log(`[${e.t.toFixed(1)} s | ${e.half}H ${e.clock.toFixed(0)}] ${e.kind} ${e.text}`);
const N = Math.round(+secArg / 0.02), t0 = performance.now();
for (let k = 0; k < N && g.phase !== 'end'; k++) {
  g.tick();
  if (k % 500 === 0) {
    const [b] = g.world.ball();
    console.log(`t ${g.t.toFixed(1)} ${g.phase} ball ${b[0].toFixed(2)},${b[1].toFixed(2)} ` + g.players.map(p => `${p.name}:${(p.state || '').slice(0, 6)}`).join(' '));
  }
}
const wall = (performance.now() - t0) / 1000, s = g.summary();
console.log(JSON.stringify({ ...s, players: undefined }));
console.log(`${s.sim_s} s sim in ${wall.toFixed(1)} s wall = ${(s.sim_s / wall).toFixed(2)} x real time; mj_step ${(g.world.stepMs / s.sim_s).toFixed(0)} ms/s, brains ${(g.brainMs / s.sim_s).toFixed(0)} ms/s`);
for (const [n, p] of Object.entries(s.players)) console.log(n, JSON.stringify(p));
