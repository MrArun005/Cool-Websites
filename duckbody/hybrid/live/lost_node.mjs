// node lost_node.mjs PAGE_LIVE_DIR SECONDS SEED : a live 5v5 match in node with the page's own code (PAGE_LIVE_DIR/live_game.js),
// measuring ball-lost-at-the-feet (trunk within 15 cm of the ball, no own camera fix for 0.3 s), stalls (ball within
// 15 cm of its anchor > 3 s), dropped balls, kicks.  Run from a folder whose node_modules has @mujoco/mujoco@3.15.0.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
const [dir, secArg = '120', seedArg = '1'] = process.argv.slice(2);
const { Game } = await import(path.resolve(dir, 'live_game.js'));
const { MlpPolicy } = await import(path.resolve(dir, 'onnx_mlp.js'));
const require = createRequire(path.join(process.cwd(), 'x.js'));
const loadMujoco = (await import(require.resolve('@mujoco/mujoco'))).default;
const mj = await loadMujoco();
const pols = {};
for (const b of ['velstand', 'ball_kick_left', 'ball_kick_right', 'roulade']) {
  const buf = fs.readFileSync(path.join(dir, 'policies', b + '.onnx'));
  pols[b] = new MlpPolicy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), b);
}
const g = new Game(mj, fs.readFileSync(path.join(dir, 'live_model_5v5.xml'), 'utf8'), JSON.parse(fs.readFileSync(path.join(dir, 'live_meta_5v5.json'), 'utf8')),
  JSON.parse(fs.readFileSync(path.join(dir, 'live_moves.json'), 'utf8')), (n, X, N) => pols[n].runBatch(X, N), { seed: +seedArg, halfS: +secArg, halves: 1 });
const L = { near: 0, lost: 0, search: 0, standing: 0 }, st = {};
let anchor = null, t0 = 0, stalls = 0, stall_s = 0, play = 0;
const t1 = performance.now();
while (g.phase !== 'end') {
  g.tick();
  if (g.phase !== 'playing') { anchor = null; continue; }
  play++;
  const [b] = g.world.ball();
  if (g.setpiece) { anchor = null; }
  else if (!anchor || Math.hypot(b[0] - anchor[0], b[1] - anchor[1]) > 0.15) {
    if (anchor && g.t - t0 > 3) { stalls++; stall_s += g.t - t0; }
    anchor = [b[0], b[1]]; t0 = g.t;
  }
  for (const p of g.players) {
    if (!p.active()) continue;
    const [q] = p.duck.pose();
    if (Math.hypot(q[0] - b[0], q[1] - b[1]) >= 0.15) continue;
    L.near++;
    if (g.t - p.ball_t <= 0.3) continue;
    L.lost++;
    const s = (p.state || '').split('/')[0]; st[s] = (st[s] || 0) + 1;
    const c = p.cmd;
    if (s.startsWith('search')) L.search++;
    else if (c && c.vx === 0 && c.vy === 0 && c.wz === 0 && !['kick left', 'kick right', 'get up', 'hop'].includes(p.rt.mode)) L.standing++;
  }
}
const sm = g.summary(), wall = (performance.now() - t1) / 1000;
const r = x => +(x * 0.02).toFixed(1);
console.log(JSON.stringify({ seed: +seedArg, play_s: r(play), rtf: +(sm.sim_s / wall).toFixed(2), score: sm.score, kicks: sm.kicks, connected: sm.connected, falls: sm.falls,
  dropped: g.events.filter(e => e.kind === 'dropped ball').length, stalls, stall_s: +stall_s.toFixed(1),
  near_s: r(L.near), lost_near_s: r(L.lost), lost_search_s: r(L.search), lost_standing_s: r(L.standing),
  lost_states_s: Object.fromEntries(Object.entries(st).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([k, v]) => [k, r(v)])) }));
