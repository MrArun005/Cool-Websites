// Microduck's policies, run straight from their ONNX files (unchanged) in plain JavaScript.
// Every brain is the same graph: Sub(mean) -> Div(std) -> (Gemm -> Elu) x 3 -> Gemm (61 -> 512 -> 256 -> 128 -> 14).
// This reads the .onnx protobuf, checks the graph is exactly that (op types, transB = 1, alpha = 1, shapes) and
// evaluates it. Why not onnxruntime-web: the artifact's sandbox only lets scripts (not fetches) come from the CDN, so
// its 14 MB WebAssembly runtime would have to be published with the page, for four small matrix products.
// Checked against onnxruntime (Python) in onnx_check.mjs: max difference ~1e-6.

function reader(buf) {
  let p = 0;
  const u8 = buf;
  const varint = () => { let r = 0, s = 1, b; do { b = u8[p++]; r += (b & 0x7f) * s; s *= 128; } while (b & 0x80); return r; };
  return {
    get pos() { return p; }, set pos(v) { p = v; },
    varint,
    fields(end, cb) {               // cb(fieldNo, wireType, value|[start,end])
      while (p < end) {
        const key = varint(), f = Math.floor(key / 8), wt = key & 7;
        if (wt === 0) cb(f, wt, varint());
        else if (wt === 2) { const n = varint(), s = p; p += n; cb(f, wt, [s, s + n]); }
        else if (wt === 5) { cb(f, wt, [p, p + 4]); p += 4; }
        else if (wt === 1) { cb(f, wt, [p, p + 8]); p += 8; }
        else throw new Error('onnx: wire type ' + wt);
      }
    },
  };
}
const utf8 = new TextDecoder();

export function parseOnnx(arrayBuffer) {
  const u8 = new Uint8Array(arrayBuffer), R = reader(u8);
  let graph = null;
  R.fields(u8.length, (f, wt, v) => { if (f === 7 && wt === 2) graph = v; });
  if (!graph) throw new Error('onnx: no graph');
  const nodes = [], inits = {};
  const str = ([a, b]) => utf8.decode(u8.subarray(a, b));
  const sub = ([a, b], cb) => { const save = R.pos; R.pos = a; R.fields(b, cb); R.pos = save; };
  sub(graph, (f, wt, v) => {
    if (f === 1 && wt === 2) {                     // NodeProto: input 1, output 2, op_type 4, attribute 5
      const n = { inputs: [], outputs: [], op: '', attrs: {} };
      sub(v, (g, w2, x) => {
        if (g === 1) n.inputs.push(str(x)); else if (g === 2) n.outputs.push(str(x)); else if (g === 4) n.op = str(x);
        else if (g === 5) {                        // AttributeProto: name 1, f 2 (float), i 3 (int)
          let name = '', val = null;
          sub(x, (h, w3, y) => { if (h === 1) name = str(y); else if (h === 2 && w3 === 5) val = new DataView(u8.buffer, u8.byteOffset + y[0], 4).getFloat32(0, true); else if (h === 3 && w3 === 0) val = y; });
          n.attrs[name] = val;
        }
      });
      nodes.push(n);
    } else if (f === 5 && wt === 2) {              // TensorProto: dims 1, data_type 2, name 8, raw_data 9
      const t = { dims: [], type: 0, name: '', raw: null };
      sub(v, (g, w2, x) => {
        if (g === 1) { if (w2 === 0) t.dims.push(x); else { const save = R.pos; R.pos = x[0]; while (R.pos < x[1]) t.dims.push(R.varint()); R.pos = save; } }
        else if (g === 2) t.type = x; else if (g === 8) t.name = str(x); else if (g === 9) t.raw = x;
      });
      if (t.type !== 1 || !t.raw) throw new Error('onnx: expected float raw_data in ' + t.name);
      const bytes = u8.slice(t.raw[0], t.raw[1]);
      t.data = new Float32Array(bytes.buffer);
      inits[t.name] = t;
    }
  });
  return { nodes, inits };
}

export class MlpPolicy {
  constructor(arrayBuffer, name = '') {
    const { nodes, inits } = parseOnnx(arrayBuffer);
    const ops = nodes.map(n => n.op).join(',');
    if (ops !== 'Sub,Div,Gemm,Elu,Gemm,Elu,Gemm,Elu,Gemm') throw new Error(`${name}: unexpected graph ${ops}`);
    this.mean = inits[nodes[0].inputs[1]].data; this.std = inits[nodes[1].inputs[1]].data;
    this.layers = [];
    for (const n of nodes) {
      if (n.op === 'Gemm') {
        if (n.attrs.transB !== 1 || (n.attrs.alpha ?? 1) !== 1 || (n.attrs.beta ?? 1) !== 1) throw new Error(`${name}: Gemm attributes`);
        const W = inits[n.inputs[1]], b = inits[n.inputs[2]];
        this.layers.push({ W: Float64Array.from(W.data), out: W.dims[0], inp: W.dims[1], b: Float64Array.from(b.data), elu: false });
      } else if (n.op === 'Elu') {
        if ((n.attrs.alpha ?? 1) !== 1) throw new Error(`${name}: Elu alpha`);
        this.layers[this.layers.length - 1].elu = true;
      }
    }
    if (this.layers[0].inp !== 61 || this.layers[this.layers.length - 1].out !== 14) throw new Error(`${name}: shapes`);
    this.buf = this.layers.map(l => new Float64Array(l.out));
    this.x0 = new Float64Array(61);
  }
  // obs: 61 floats -> Float32Array(14)
  run(obs) {
    let x = this.x0;
    for (let i = 0; i < 61; i++) x[i] = (Math.fround(obs[i]) - this.mean[i]) / this.std[i];
    for (let l = 0; l < this.layers.length; l++) {
      const L = this.layers[l], y = this.buf[l], W = L.W, n = L.inp;
      for (let o = 0; o < L.out; o++) {
        const r = o * n;
        let s0 = 0, s1 = 0, s2 = 0, s3 = 0, i = 0;
        for (; i + 3 < n; i += 4) { s0 += W[r + i] * x[i]; s1 += W[r + i + 1] * x[i + 1]; s2 += W[r + i + 2] * x[i + 2]; s3 += W[r + i + 3] * x[i + 3]; }
        for (; i < n; i++) s0 += W[r + i] * x[i];
        const s = L.b[o] + ((s0 + s1) + (s2 + s3));
        y[o] = L.elu && s < 0 ? Math.expm1(s) : s;
      }
      x = y;
    }
    return Float32Array.from(x);
  }
  // N observations (N*61) -> N*14, the batched interface live_sim.brainsTick expects (synchronous). Each weight row
  // is used for all N ducks while it is in cache (about 3x faster than N single calls).
  runBatch(X, N) {
    let x = new Float64Array(61 * N);
    for (let k = 0; k < N; k++) for (let i = 0; i < 61; i++) x[61 * k + i] = (Math.fround(X[61 * k + i]) - this.mean[i]) / this.std[i];
    for (const L of this.layers) {
      const n = L.inp, W = L.W, y = new Float64Array(L.out * N);
      for (let o = 0; o < L.out; o++) {
        const r = o * n, bo = L.b[o];
        for (let k = 0; k < N; k++) {
          const c = k * n;
          let s0 = 0, s1 = 0, i = 0;
          for (; i + 1 < n; i += 2) { s0 += W[r + i] * x[c + i]; s1 += W[r + i + 1] * x[c + i + 1]; }
          if (i < n) s0 += W[r + i] * x[c + i];
          const s = bo + s0 + s1;
          y[k * L.out + o] = L.elu && s < 0 ? Math.expm1(s) : s;
        }
      }
      x = y;
    }
    return Float32Array.from(x);
  }
}
