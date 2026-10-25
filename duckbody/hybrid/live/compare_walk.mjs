// compare_walk.py's run in MuJoCo WASM + onnx_mlp.js (the page's brain runner) + the JS BAM port (live_sim.js): one duck, velstand,
// no noise, no delay, no randomization. Run from a folder whose node_modules has @mujoco/mujoco@3.15.0:  node compare_walk.mjs DIR   (DIR from export_live.py --one) -> DIR/walk_js.json
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { World, DT } from './live_sim.js';
import { MlpPolicy } from './onnx_mlp.js';

const require = createRequire(path.join(process.cwd(), 'x.js'));
const loadMujoco = (await import(require.resolve('@mujoco/mujoco'))).default;
const dir = process.argv[2];
const mj = await loadMujoco();
const xml = fs.readFileSync(path.join(dir, 'live_model_one.xml'), 'utf8');
const meta = JSON.parse(fs.readFileSync(path.join(dir, 'live_meta_one.json'), 'utf8'));
const vb = fs.readFileSync(path.join(dir, 'policies', 'velstand.onnx'));
const pol = new MlpPolicy(vb.buffer.slice(vb.byteOffset, vb.byteOffset + vb.length), 'velstand');
const w = new World(mj, xml, meta, { seed: 0, randomize: false });
const dk = w.ducks[0];
dk.obsNoise = false; dk.delay = false; dk.resetState();
w.put(dk, 0, 0, 0);
mj.mj_forward(w.m, w.d);
function schedule(k) {
  const t = k * DT;
  if (t < 0.5) return [0, 0, 0];
  if (t < 4.0) return [0.4, 0, 0];
  if (t < 6.5) return [0.4, 0, 0.4];
  if (t < 8.5) return [0, 0.35, 0];
  return [0, 0, 0];
}
const out = { trunk: [], yaw: [], q: [], obs0: null, act0: null, mujoco: mj.mj_versionString() };
const o = new Float32Array(61);
const t0 = performance.now();
for (let k = 0; k < 500; k++) {
  dk.command.fill(0); dk.command.set(schedule(k));
  dk.obs(o);
  const a = pol.run(o);
  if (k === 0) { out.obs0 = Array.from(o); out.act0 = Array.from(a); }
  dk.setAction(a);
  w.physics(false);
  const [p, yaw] = dk.pose();
  const x = w.d.xpos, tr = 3 * dk.meta.trunk;
  out.trunk.push([x[tr], x[tr + 1], x[tr + 2]].map(v => +v.toFixed(5)));
  out.yaw.push(+yaw.toFixed(5));
  out.q.push(dk.meta.qadr.map(i => +w.d.qpos[i].toFixed(4)));
}
fs.writeFileSync(path.join(dir, 'walk_js.json'), JSON.stringify(out));
console.log(`wasm walk done in ${((performance.now() - t0) / 1000).toFixed(2)} s, end`, out.trunk[499]);
