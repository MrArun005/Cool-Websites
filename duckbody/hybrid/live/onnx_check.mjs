// onnx_mlp.js against onnxruntime (Python, onnx_check.py): node onnx_check.mjs REF.json POLICY_DIR
import fs from 'fs';
import path from 'path';
import { MlpPolicy } from './onnx_mlp.js';
const [ref, dir] = process.argv.slice(2), R = JSON.parse(fs.readFileSync(ref, 'utf8'));
let worst = 0;
for (const [b, { X, Y }] of Object.entries(R)) {
  const buf = fs.readFileSync(path.join(dir, b + '.onnx'));
  const pol = new MlpPolicy(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length), b);
  let w = 0; const t0 = performance.now();
  X.forEach((x, k) => { const y = pol.run(x); y.forEach((v, j) => { w = Math.max(w, Math.abs(v - Y[k][j])); }); });
  console.log(`${b}: max |JS - onnxruntime| = ${w.toExponential(2)} over ${X.length} observations, ${((performance.now() - t0) / X.length).toFixed(3)} ms each`);
  worst = Math.max(worst, w);
}
process.exit(worst < 1e-4 ? 0 : 1);
