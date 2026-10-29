// Bolt Remote: drive Bolt from a phone, or switch on Auto self mode and let it look after itself. Simulation.
//
// The body is Bolt's Day Out's (doglife.js DogBody: the same trained brains and hand-overs). What is new:
//   Scanner   a head-mounted depth scanner: 279 rays (31 across 120 degrees x 9 rows, +2 to -50 degrees), 5 sweeps a
//             second, cast with MuJoCo's own mj_ray against the physics geometry. Each ray reports how far it went
//             and what kind of surface it hit (ground, road surface, obstacle, hurdle), as a depth camera with a
//             ground/object classifier would. The rows are levelled by the IMU (a stabilised scanner).
//   SenseMap  Bolt's own map, built only from its scans: a 0.25 m occupancy grid (free / obstacle / road / hurdle,
//             unknown until seen), hurdles estimated from the points seen on them (position, line, bar height).
//   Planner   A* on the map (unknown counts as passable but dearer), then a local "best open heading" steer.
//   AutoMind  what Bolt does on its own: explore what it has not seen, sniff things it found, trot, zoomies on a
//             long open line it has seen, sit, nap beside something big, watch the road from the kerb, and jump
//             hurdles it found, with the drives (energy, curiosity, play) of the Day Out mind. Every place and
//             obstacle comes from SenseMap, never from the yard plan (YARD is only used to build the world).
//   Pilot     manual driving from the remote: joystick (speed, turn), skill buttons, a 0.5 s dead-man stop and a
//             collision guard from the scanner that slows and stops Bolt before obstacles and the kerb.
// Bolt's own position comes from the simulator (a perfect odometry/SLAM stand-in; a real robot's estimate drifts).
// No DOM here: remote_test.mjs runs it headless, remote_template.html draws it.
import { DogBody, YARD, yardProps, TAKEOVER } from "./doglife.js";
import { Imitate } from "./imitate.js";
import { propQuat } from "./live.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const hyp = (x, y) => Math.hypot(x, y);
const D2R = Math.PI / 180;

// ---------------------------------------------------------------- the world
// The yard of Bolt's Day Out, plus a road surface the scanner can tell from the sidewalk: in this physics the kerb
// has no height, so the road is a 1 mm non-colliding layer (the scanner sees "road", nothing trips on it).
export function setupRemoteYard(bolt, bundle, Y = YARD) {
  bolt.reset(bundle.start);
  const props = yardProps(Y);
  bolt.props = props;
  const geoms = props.map((p, i) => `<geom name="prop${i}" type="${p.kind === "cyl" ? "cylinder" : "box"}" size="${p.size.join(" ")}" ` +
    `pos="${p.pos.join(" ")}" quat="${propQuat(p).join(" ")}" rgba="0.8 0.8 0.8 1"/>`);
  const R = Y.road, half = 19;
  // the edge of the simulated world (the floor ends at +-20 m): walls nothing is drawn for, that the scanner sees
  for (const [x, y, hx, hy] of [[0, 19.6, 20, 0.1], [0, -19.6, 20, 0.1], [19.6, 0, 0.1, 20], [-19.6, 0, 0.1, 20]])
    geoms.push(`<geom name="edge${x}_${y}" type="box" size="${hx} ${hy} 0.6" pos="${x} ${y} 0.6" rgba="0.8 0.8 0.8 1"/>`);
  geoms.push(`<geom name="road" type="box" size="${half} ${(R.y1 - R.y0) / 2} 0.0005" pos="0 ${(R.y0 + R.y1) / 2} 0.0005" contype="0" conaffinity="0" rgba="0.3 0.3 0.32 1"/>`);
  bolt.build(`<body name="props">${geoms.join("")}</body>`);
  bolt.teleport(bundle.start, Y.start.at, Y.start.yaw);
  for (let k = 0; k < bolt.real.length; k++) bolt.placeReal(k, null, 0, 0);
  for (const j of Y.jumps) bolt.placeReal(j.k, j.at, j.yaw, j.h);
}

// ---------------------------------------------------------------- the scanner
export const SCAN = { cols: 31, fov: 120, rows: [2, -4, -9, -14, -19, -25, -32, -40, -50], range: 10, hz: 5 };
export const KIND = { robot: 0, ground: 1, road: 2, obstacle: 3, hurdle: 4 };

export class Scanner {
  constructor(bolt, headGeom) {
    this.bolt = bolt; this.head = headGeom; this.mj = bolt.mj;
    const m = bolt.model, mj = bolt.mj, n = m.ngeom;
    const road = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, "road");
    const rootOf = (g) => m.body_rootid[m.geom_bodyid[g]];
    const robotRoot = m.body_rootid[bolt.torso];
    this.kind = new Uint8Array(n); this.hk = new Int8Array(n).fill(-1);
    for (let g = 0; g < n; g++) {
      if (rootOf(g) === robotRoot) this.kind[g] = KIND.robot;
      else if (g === bolt.floor) this.kind[g] = KIND.ground;
      else if (g === road) this.kind[g] = KIND.road;
      else this.kind[g] = KIND.obstacle;
    }
    bolt.real.forEach((H, k) => { for (const g of H.geoms) { this.kind[g] = KIND.hurdle; this.hk[g] = k; } });
    // geoms parked out of the world (the single game bar, the ice slab): never in view, skip them
    for (const g of [bolt.hurdle, bolt.ice]) if (g >= 0) this.kind[g] = KIND.robot;
    this.rays = [];
    for (const e of SCAN.rows) for (let c = 0; c < SCAN.cols; c++) this.rays.push({ az: (-SCAN.fov / 2 + (SCAN.fov * c) / (SCAN.cols - 1)) * D2R, el: e * D2R });
    // the hit geom comes back through the bindings' own buffers (a plain Int32Array is not written back)
    this.next = 0; this.gid = new mj.IntBuffer(1); this.nrm = new mj.DoubleBuffer(3); this.sweeps = 0; this.last = [];
  }
  origin() {
    const d = this.bolt.data, h = this.head;
    const w = this.bolt.where(), f = [Math.cos(w.yaw), Math.sin(w.yaw)];
    return { p: [d.geom_xpos[3 * h] + 0.09 * f[0], d.geom_xpos[3 * h + 1] + 0.09 * f[1], d.geom_xpos[3 * h + 2]], yaw: w.yaw };
  }
  // one ray: where it ended and what it hit (kind ground/road/obstacle/hurdle, or miss)
  cast(ray, o) {
    const { mj, bolt } = this, m = bolt.model, d = bolt.data;
    const a = o.yaw + ray.az, ce = Math.cos(ray.el);
    const v = [ce * Math.cos(a), ce * Math.sin(a), Math.sin(ray.el)];
    let p = o.p.slice(), gone = 0;
    for (let tries = 0; tries < 4; tries++) {
      const t = mj.mj_ray(m, d, p, v, null, 1, bolt.torso, this.gid, this.nrm);
      const g = this.gid.GetView()[0];
      if (t < 0 || g < 0 || gone + t > SCAN.range) return { o: o.p, v, dist: SCAN.range, kind: -1, el: ray.el };
      const hit = [p[0] + t * v[0], p[1] + t * v[1], p[2] + t * v[2]];
      if (this.kind[g] === KIND.robot) { gone += t + 0.02; p = [hit[0] + 0.02 * v[0], hit[1] + 0.02 * v[1], hit[2] + 0.02 * v[2]]; continue; }
      return { o: o.p, v, dist: gone + t, x: hit[0], y: hit[1], z: hit[2], kind: this.kind[g], k: this.hk[g], el: ray.el };
    }
    return { o: o.p, v, dist: SCAN.range, kind: -1, el: ray.el };
  }
  // cast the next n rays of the sweep (spread over control steps); hits go to `onHit`
  step(n, onHit) {
    const o = this.origin(); this.z = o.p[2];
    for (let i = 0; i < n; i++) {
      const h = this.cast(this.rays[this.next], o);
      this.last[this.next] = h; onHit(h);
      if (++this.next >= this.rays.length) { this.next = 0; this.sweeps++; }
    }
  }
}

// ---------------------------------------------------------------- Bolt's own map
export class SenseMap {
  constructor(cell = 0.25, half = 26) {
    this.cell = cell; this.half = half; this.n = Math.round((2 * half) / cell);
    const N = this.n * this.n;
    this.occ = new Float32Array(N);      // log-odds of an obstacle: > 0.5 obstacle, < -0.3 free
    this.seen = new Uint8Array(N);
    this.road = new Uint8Array(N);
    this.top = new Float32Array(N);      // highest point seen on an obstacle in the cell (m)
    this.hk = new Int8Array(N).fill(-1); // a hurdle seen in the cell
    this.hurdles = new Map();            // k -> { pts: [[x, y, z]...] }
    this.version = 0; this.scanZ = 0.45;   // the scanner's height (set by the scanner)
  }
  idx(x, y) {
    const i = Math.floor((x + this.half) / this.cell), j = Math.floor((y + this.half) / this.cell);
    return i < 0 || j < 0 || i >= this.n || j >= this.n ? -1 : j * this.n + i;
  }
  centre(id) { const i = id % this.n, j = (id / this.n) | 0; return [(i + 0.5) * this.cell - this.half, (j + 0.5) * this.cell - this.half]; }
  free(id, by) { if (id >= 0) { this.occ[id] = Math.max(-2, this.occ[id] - by); this.seen[id] = 1; } }
  integrate(h) {
    if (h.kind === -1) return;   // nothing within range: no evidence (a level ray passes over low things)
    const ox = h.o[0], oy = h.o[1], dx = h.x - ox, dy = h.y - oy, L = hyp(dx, dy);
    const ground = h.kind === KIND.ground || h.kind === KIND.road;
    // cells the ray passed over are free: surely for a ray that came down to the ground, a little for the rest
    const by = ground ? 0.35 : 0.12, end = ground ? L : L - 0.3, step = this.cell * 0.5;
    let lastId = -2;
    for (let s = 0.3; s < end; s += step) { const id = this.idx(ox + (dx * s) / L, oy + (dy * s) / L); if (id !== lastId) { this.free(id, by); lastId = id; } }
    const id = this.idx(h.x, h.y);
    if (id < 0) return;
    this.seen[id] = 1;
    if (h.kind === KIND.ground) this.free(id, 0.35);
    else if (h.kind === KIND.road) { this.road[id] = 1; this.free(id, 0.35); }
    else if (h.z > 0.04) {
      this.occ[id] = Math.min(3, this.occ[id] + 1.0); this.top[id] = Math.max(this.top[id], h.z);
      if (h.kind === KIND.hurdle) {
        this.hk[id] = h.k;
        // points kept one per 5 cm voxel (the posts stay known while Bolt looks at the bar)
        let H = this.hurdles.get(h.k); if (!H) this.hurdles.set(h.k, (H = { vox: new Map(), pts: [], seen: 0 }));
        const key = `${Math.round(h.x * 20)},${Math.round(h.y * 20)},${Math.round(h.z * 20)}`;
        if (!H.vox.has(key) && H.vox.size < 4000) { const q = [h.x, h.y, h.z]; H.vox.set(key, q); H.pts.push(q); }
        H.seen++;
      }
    }
    this.version++;
  }
  blocked(id) { return id < 0 || this.occ[id] > 0.5 || this.road[id] === 1; }
  known(id) { return id >= 0 && this.seen[id] === 1; }
  // nothing blocking within r of (x, y); `skipK`: a hurdle that does not count (the one being jumped)
  // `ahead` (optional [ox, oy, fx, fy]): only things ahead of (ox, oy) along (fx, fy) count (moving away from a wall
  // Bolt is touching is fine)
  clear(x, y, r = 0.35, skipK = -1, ahead = null) {
    const c = this.cell, k = Math.ceil(r / c);
    const i0 = Math.floor((x + this.half) / c), j0 = Math.floor((y + this.half) / c);
    for (let dj = -k; dj <= k; dj++) for (let di = -k; di <= k; di++) {
      const i = i0 + di, j = j0 + dj;
      if (i < 0 || j < 0 || i >= this.n || j >= this.n) return false;
      const id = j * this.n + i;
      if ((this.occ[id] > 0.5 && !(skipK >= 0 && this.hk[id] === skipK)) || this.road[id]) {
        const [cx, cy] = this.centre(id);
        if (ahead && (cx - ahead[0]) * ahead[2] + (cy - ahead[1]) * ahead[3] < 0.15) continue;
        if (hyp(cx - x, cy - y) < r + c * 0.5) return false;
      }
    }
    return true;
  }
  // how far Bolt could go along heading a from (x, y) before something blocks (up to `max` m)
  clearance(x, y, a, max = 3.0, r = 0.35, skipK = -1) {
    const fx = Math.cos(a), fy = Math.sin(a), ahead = [x, y, fx, fy];
    for (let s = 0.2; s <= max; s += 0.15) if (!this.clear(x + fx * s, y + fy * s, r, skipK, ahead)) return s - 0.15;
    return max;
  }
  // road seen within `ahead` m in front (a cone of +-cone rad): the kerb
  roadAhead(x, y, yaw, ahead = 1.0, cone = 0.6) {
    for (let s = 0.2; s <= ahead; s += 0.15) for (let a = -cone; a <= cone; a += 0.2) {
      const id = this.idx(x + Math.cos(yaw + a) * s, y + Math.sin(yaw + a) * s);
      if (id >= 0 && this.road[id]) return s;
    }
    return null;
  }
  lineClear(a, b, r = 0.6, needKnown = true, skipK = -1) {
    const n = Math.ceil(hyp(b[0] - a[0], b[1] - a[1]) / 0.25);
    for (let i = 0; i <= n; i++) {
      const x = a[0] + ((b[0] - a[0]) * i) / n, y = a[1] + ((b[1] - a[1]) * i) / n;
      if (!this.clear(x, y, r, skipK)) return false;
      if (needKnown && !this.known(this.idx(x, y))) return false;
    }
    return true;
  }
  // a hurdle from the points seen on it: its line (the bar's direction), centre, width and bar height
  hurdle(k) {
    const H = this.hurdles.get(k);
    if (!H || H.pts.length < 12) return null;
    const P = H.pts;
    let mx = 0, my = 0; for (const p of P) { mx += p[0]; my += p[1]; } mx /= P.length; my /= P.length;
    let sxx = 0, syy = 0, sxy = 0; for (const p of P) { const x = p[0] - mx, y = p[1] - my; sxx += x * x; syy += y * y; sxy += x * y; }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), ux = Math.cos(th), uy = Math.sin(th);
    let lo = Infinity, hi = -Infinity; for (const p of P) { const s = (p[0] - mx) * ux + (p[1] - my) * uy; lo = Math.min(lo, s); hi = Math.max(hi, s); }
    const mid = (lo + hi) / 2, cx = mx + ux * mid, cy = my + uy * mid, width = hi - lo;
    // the bar: points in the middle of the span; else the posts' tops less 10 cm (a standard agility hurdle)
    let bar = -1, post = -1;
    for (const p of P) {
      const s = (p[0] - cx) * ux + (p[1] - cy) * uy;
      if (Math.abs(s) < 0.45) bar = Math.max(bar, p[2]); else if (Math.abs(s) > 0.55 && Math.abs(s) < 0.72) post = Math.max(post, p[2]);
    }
    // a bar hit is on its side or top (radius 1.5 cm); posts only tell if their top (bar + 10 cm) was seen below the
    // scanner's line, else the height stays unknown and Bolt goes closer to look
    const fromBar = bar > 0.12, fromPosts = !fromBar && post > 0.3 && post < this.scanZ - 0.08;
    const h = fromBar ? bar + 0.01 : fromPosts ? post - 0.1 : null;
    return { k, at: [cx, cy], line: th, width, h, from: fromBar ? "bar" : fromPosts ? "posts" : "unknown", n: P.length };
  }
}

// ---------------------------------------------------------------- path planning on the map
// A* on a 0.5 m grid: a cell is closed if anything blocking lies within `inflate` of it; unknown cells are open
// but cost more (Bolt is willing to explore). Returns way points (world) from start to goal, or null.
export function planPath(map, from, to, { inflate = 0.5, skipK = -1, unknownCost = 1.6 } = {}) {
  const c = 0.5, half = map.half, n = Math.round((2 * half) / c), N = n * n;
  const cid = (x, y) => { const i = Math.floor((x + half) / c), j = Math.floor((y + half) / c); return i < 0 || j < 0 || i >= n || j >= n ? -1 : j * n + i; };
  const ctr = (id) => [((id % n) + 0.5) * c - half, (((id / n) | 0) + 0.5) * c - half];
  const s = cid(...from), g = cid(...to);
  if (s < 0 || g < 0) return null;
  const okCache = new Int8Array(N);   // 0 unknown yet, 1 open, 2 closed
  const open = (id) => {
    // near the start a thin margin only, so Bolt can always leave where it stands (next to a wall or the kerb)
    if (!okCache[id]) { const [x, y] = ctr(id), near = hyp(x - from[0], y - from[1]) < 1.0; okCache[id] = id === s || map.clear(x, y, near ? 0.15 : inflate, skipK) ? 1 : 2; }
    return okCache[id] === 1;
  };
  if (!open(g)) return null;
  const gScore = new Float32Array(N).fill(Infinity), came = new Int32Array(N).fill(-1), done = new Uint8Array(N);
  const heap = [], push = (id, f) => { heap.push([f, id]); let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
  const pop = () => { const top = heap[0], last = heap.pop(); if (heap.length) { heap[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } } return top; };
  const [gx, gy] = ctr(g), hcost = (id) => { const [x, y] = ctr(id); return hyp(x - gx, y - gy); };
  gScore[s] = 0; push(s, hcost(s));
  let expanded = 0;
  while (heap.length && expanded < 12000) {
    const [, id] = pop();
    if (done[id]) continue;
    done[id] = 1; expanded++;
    if (id === g) break;
    const i = id % n, j = (id / n) | 0;
    for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      if (!di && !dj) continue;
      const ii = i + di, jj = j + dj; if (ii < 0 || jj < 0 || ii >= n || jj >= n) continue;
      const nb = jj * n + ii; if (done[nb] || !open(nb)) continue;
      const [x, y] = ctr(nb), fine = map.idx(x, y);
      const step = (di && dj ? 1.414 : 1) * c * (map.known(fine) ? 1 : unknownCost);
      const t = gScore[id] + step;
      if (t < gScore[nb]) { gScore[nb] = t; came[nb] = id; push(nb, t + hcost(nb)); }
    }
  }
  if (!done[g]) return null;
  const path = []; for (let id = g; id >= 0; id = came[id]) path.push(ctr(id));
  path.reverse(); path[path.length - 1] = to.slice();
  return path;
}

// the open heading closest to the wanted one (a small vector-field-histogram): Bolt's local steering
export function bestHeading(map, p, want, need = 1.6, skipK = -1) {
  let best = null, bestCost = Infinity, widest = null, wc = -1;
  for (let k = 0; k < 36; k++) {
    const a = want + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 10 * D2R;
    const cl = map.clearance(p.x, p.y, a, 3.0, 0.4, skipK);
    if (cl > wc) { wc = cl; widest = a; }
    if (cl >= need) { const cost = Math.abs(wrap(a - want)); if (cost < bestCost) { bestCost = cost; best = a; } }
  }
  return { heading: best !== null ? best : widest, blocked: best === null, clearance: wc };
}

// ---------------------------------------------------------------- Bolt on its own
const LABEL = {
  explore: "exploring", sniff: "sniffing", trot: "trotting about", zoomies: "zoomies!", sit: "sitting", nap: "napping",
  jump: "going for the jump", pace: "pacing along", lookout: "watching the road", kerb: "stopping at the kerb",
};

export class AutoMind {
  constructor(L, { seed = 1 } = {}) {
    this.L = L; this.bolt = L.bolt; this.body = L.body; this.map = L.map;
    let s = seed >>> 0 || 1; this.rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const r = this.rand;
    this.energy = 0.75 + 0.25 * r(); this.curiosity = 0.4 + 0.5 * r(); this.play = 0.3 + 0.5 * r();
    this.t = 0; this.task = null; this.log = []; this.counts = {}; this.lastName = null;
    this.stats = { jumps: 0, clean: 0, knocked: 0, distance: 0, naps: 0, zoomies: 0, sniffs: 0, kerbStops: 0, replans: 0, blocked: 0 };
    const w = this.bolt.where(); this.prev = [w.x, w.y];
  }
  say(text, kind = "info") { this.log.push({ t: this.t, text, kind }); }
  get label() { return this.task ? this.task.label : "standing"; }
  where() { const w = this.bolt.where(); return { x: w.x, y: w.y, yaw: w.yaw }; }

  // ---- drives (as the Day Out mind)
  drift(dt) {
    const g = this.body.gait, m = this.body.mode;
    const cost = { walk: 0.0012, trot: 0.0035, pace: 0.003, canter: 0.012, jump: 0.03, land: 0.01, bridge: 0.01, fallen: 0.01 }[g] || 0;
    this.energy = clamp(this.energy - cost * dt + (m === "lie" ? 0.012 : m === "stand" || m === "sit" || m === "sniff" ? 0.0015 : 0) * dt, 0, 1);
    this.curiosity = clamp(this.curiosity + 0.004 * dt - (m === "sniff" ? 0.07 * dt : 0), 0, 1);
    this.play = clamp(this.play + 0.003 * dt * (0.5 + this.energy) - (g === "canter" || g === "jump" ? 0.05 * dt : 0), 0, 1);
  }
  choose() {
    const E = this.energy, C = this.curiosity, P = this.play, r = this.rand, M = this.map;
    const unexplored = this.frontier(3, 16, 1).length > 0, road = this.roadEdge(1) !== null, hurdle = this.jumpable().length > 0;
    const opts = {
      explore: unexplored ? 0.35 + 0.8 * C : 0.05,
      sniff: this.sniffTargets(1).length ? 0.15 + 0.6 * C : 0,
      trot: E > 0.35 ? 0.15 + 0.4 * P * E : 0,
      zoomies: E > 0.45 && this.t > 20 ? 0.1 + 0.9 * P * E * E : 0,
      jump: hurdle && E > 0.4 ? 0.25 + 0.8 * P * E : 0,
      pace: 0.1 + 0.12 * E,
      sit: 0.15 + 0.3 * (1 - E),
      nap: E < 0.55 ? 0.3 + 1.6 * (1 - E) - 0.4 : 0.02,
      lookout: road ? 0.08 + 0.15 * C : 0,
    };
    void M;
    if (this.lastName && opts[this.lastName] !== undefined) opts[this.lastName] *= 0.25;
    let best = null, bv = -1;
    for (const [k, v] of Object.entries(opts)) { const s = v * (0.55 + 0.9 * r()); if (s > bv) { bv = s; best = k; } }
    return best;
  }

  // ---- what Bolt knows (all from the map)
  frontier(min, max, limit = 40) {   // seen-free cells next to unseen ones, min..max m away
    const M = this.map, p = this.where(), out = [], n = M.n, c = M.cell;
    const R = Math.ceil(max / c), i0 = Math.floor((p.x + M.half) / c), j0 = Math.floor((p.y + M.half) / c);
    for (let tries = 0; tries < 2500 && out.length < limit; tries++) {
      const i = i0 + Math.floor((this.rand() * 2 - 1) * R), j = j0 + Math.floor((this.rand() * 2 - 1) * R);
      if (i < 1 || j < 1 || i >= n - 1 || j >= n - 1) continue;
      const id = j * n + i;
      if (!M.seen[id] || M.occ[id] > -0.3 || M.road[id]) continue;
      if (M.seen[id + 1] && M.seen[id - 1] && M.seen[id + n] && M.seen[id - n]) continue;
      const [x, y] = M.centre(id), d = hyp(x - p.x, y - p.y);
      if (d < min || d > max || !M.clear(x, y, 1.0)) continue;
      out.push([x, y]);
    }
    return out;
  }
  freeSpot(min, max) {   // a known free spot min..max m away, with room around it
    const M = this.map, p = this.where();
    for (let i = 0; i < 300; i++) {
      const a = this.rand() * 2 * Math.PI, d = min + (max - min) * this.rand(), x = p.x + d * Math.cos(a), y = p.y + d * Math.sin(a);
      const id = M.idx(x, y);
      if (M.known(id) && M.occ[id] < -0.3 && !M.road[id] && M.clear(x, y, 0.7)) return [x, y];
    }
    return null;
  }
  sniffTargets(limit = 20) {   // things Bolt has seen (obstacle cells beside free ground), not the hurdles
    const M = this.map, p = this.where(), out = [], n = M.n;
    for (let tries = 0; tries < 4000 && out.length < limit; tries++) {
      const id = Math.floor(this.rand() * n * n);
      if (M.occ[id] < 1.0 || M.hk[id] >= 0 || M.road[id]) continue;
      const [x, y] = M.centre(id), d = hyp(x - p.x, y - p.y);
      if (d < 1.5 || d > 14) continue;
      // a free spot 0.6-0.9 m from it, facing it
      for (let a = 0; a < 2 * Math.PI; a += Math.PI / 6) {
        const sx = x + 1.05 * Math.cos(a), sy = y + 1.05 * Math.sin(a), sid = M.idx(sx, sy);
        if (M.known(sid) && M.occ[sid] < -0.3 && M.clear(sx, sy, 0.75)) { out.push({ at: [sx, sy], thing: [x, y], top: M.top[id] }); break; }
      }
    }
    return out;
  }
  roadEdge() {   // a free spot ~1.2 m from road Bolt has seen, facing the road
    const M = this.map, p = this.where(), n = M.n;
    let best = null;
    for (let tries = 0; tries < 3000; tries++) {
      const id = Math.floor(this.rand() * n * n);
      if (!M.road[id]) continue;
      const [x, y] = M.centre(id);
      for (let a = 0; a < 2 * Math.PI; a += Math.PI / 4) {
        const sx = x + 1.3 * Math.cos(a), sy = y + 1.3 * Math.sin(a), sid = M.idx(sx, sy);
        if (M.known(sid) && !M.road[sid] && M.occ[sid] < -0.3 && M.clear(sx, sy, 0.5)) {
          const d = hyp(sx - p.x, sy - p.y);
          if (!best || d < best.d) best = { at: [sx, sy], face: Math.atan2(y - sy, x - sx), d };
          break;
        }
      }
      if (best && tries > 400) break;
    }
    return best;
  }
  jumpable() {   // hurdles seen well enough (both ends) with room for a run-up and a landing
    const out = [];
    for (const k of this.map.hurdles.keys()) {
      const e = this.map.hurdle(k);
      if (e && e.width > 1.2 && (e.h === null || (e.h > 0.15 && e.h < 0.6))) out.push(e);
    }
    return out;
  }
  // a line through hurdle e: the approach direction from Bolt's side, flipped if that side has no room
  jumpLine(e) {
    const p = this.where(), nx = -Math.sin(e.line), ny = Math.cos(e.line);   // normal to the bar
    let dir = (p.x - e.at[0]) * nx + (p.y - e.at[1]) * ny > 0 ? Math.atan2(-ny, -nx) : Math.atan2(ny, nx);
    const room = (y) => {
      const run = [e.at[0] - 9.0 * Math.cos(y), e.at[1] - 9.0 * Math.sin(y)], land = [e.at[0] + 5.0 * Math.cos(y), e.at[1] + 5.0 * Math.sin(y)];
      return this.map.lineClear(run, [e.at[0] - 1.0 * Math.cos(y), e.at[1] - 1.0 * Math.sin(y)], 0.8, false, e.k) &&
        this.map.lineClear([e.at[0] + 1.0 * Math.cos(y), e.at[1] + 1.0 * Math.sin(y)], land, 0.55, false, e.k);
    };
    if (!room(dir)) { dir = wrap(dir + Math.PI); if (!room(dir)) return null; }
    return dir;
  }

  // ---- moving: plan on the map, steer round what the scanner sees
  goTo(to, gait, near = 0.6, skipK = -1) {
    const p = this.where(), w = this.body.want, T = this.task, d = hyp(to[0] - p.x, to[1] - p.y);
    if (d < near) { w.gait = "stand"; w.steer = null; return true; }
    if (!T.path || T.pathFor !== to || this.t - T.pathAt > 1.0) {
      T.path = planPath(this.map, [p.x, p.y], to, { skipK }); T.pathFor = to; T.pathAt = this.t;
      if (!T.path) { T.noPath = (T.noPath || 0) + 1; this.stats.replans++; } else T.noPath = 0;
    }
    let tgt = to;
    if (T.path) {   // the first way point at least 1.5 m on (or the goal)
      let best = T.path.length - 1;
      for (let i = 0; i < T.path.length; i++) if (hyp(T.path[i][0] - p.x, T.path[i][1] - p.y) > 1.5) { best = i; break; }
      tgt = T.path[best];
    }
    const want = Math.atan2(tgt[1] - p.y, tgt[0] - p.x);
    const H = bestHeading(this.map, p, want, Math.min(1.4, d), skipK), margin = 0.8 + 0.6 * this.body.speed;
    if (H.blocked && H.clearance < margin) {
      w.gait = "stand"; w.steer = null;
      if (d < near + 1.0) return true;   // as close as it can get
      this.stats.blocked++; T.stuck = (T.stuck || 0) + 1 / 50;
      // boxed in on the way: turn toward the most open heading at a walk (the bumper keeps it off things)
      if (H.clearance >= 0.5) { w.gait = "walk"; w.steer = clamp(1.5 * wrap(H.heading - p.yaw), -1.5, 1.5); }
      return false;
    }
    w.steer = wrap(H.heading - p.yaw);
    const open = H.clearance;
    w.gait = gait === "walk" || d < 3.0 || open < 3.0 ? "walk" : gait;
    return false;
  }
  stop() { const w = this.body.want; w.gait = "stand"; w.steer = null; }

  start(name, extra = {}) {
    const p = this.where(), r = this.rand;
    this.task = { name, label: LABEL[name], phase: "go", pt: 0, t0: this.t, ...extra };
    this.lastName = name; this.counts[name] = (this.counts[name] || 0) + 1;
    const T = this.task;
    switch (name) {
      case "explore": {
        const F = this.frontier(3, 16, 40);
        T.to = F.length ? F[Math.floor(r() * F.length)] : this.freeSpot(4, 10) || [p.x + 2 * Math.cos(p.yaw), p.y + 2 * Math.sin(p.yaw)];
        T.gait = r() < 0.3 && this.energy > 0.45 ? "trot" : "walk";
        this.say(F.length ? "Bolt goes to look at a part of the yard it has not seen" : "Bolt wanders about");
        break;
      }
      case "sniff": {
        const S = this.sniffTargets(10); if (!S.length) { this.task = null; return this.start("explore"); }
        const s = S[Math.floor(r() * S.length)];
        T.to = s.at; T.thing = s.thing; T.hold = 3 + 5 * r();
        T.what = s.top > 1.6 ? "something tall it found" : s.top > 0.6 ? "something it found" : "something low it found";
        T.label = `sniffing ${T.what}`; this.say(`Bolt heads for ${T.what}, nose first`); break;
      }
      case "trot": case "pace": {
        T.to = this.freeSpot(7, 13) || this.freeSpot(3, 8); if (!T.to) { this.task = null; return this.start("sit"); }
        T.gait = name; this.say(name === "trot" ? "Bolt trots across the lawn" : "Bolt ambles along at a pace"); break;
      }
      case "zoomies": {
        const line = this.zoomLine(); if (!line) { this.task = null; return this.start("trot"); }
        [T.from, T.to, T.dir] = line; T.phase = "lineup"; T.canterFor = 2 + 2 * r(); T.label = "getting ready for zoomies";
        this.say("Bolt gets the zoomies!", "good"); this.stats.zoomies++; break;
      }
      case "sit": { T.hold = 4 + 8 * r(); T.phase = "sit"; this.say("Bolt sits down for a moment"); break; }
      case "nap": {   // beside something big it has seen (a wall, the house), else where it is
        const S = this.sniffTargets(20).filter((s) => s.top > 0.8);
        if (S.length) {   // 1.4 m out from it, facing away
          const s = S[Math.floor(r() * S.length)], a = Math.atan2(s.at[1] - s.thing[1], s.at[0] - s.thing[0]);
          T.to = [s.thing[0] + 1.4 * Math.cos(a), s.thing[1] + 1.4 * Math.sin(a)]; T.face = a;
          if (!this.map.clear(T.to[0], T.to[1], 0.7)) T.to = s.at;
        }
        else { T.to = [p.x, p.y]; T.face = p.yaw; }
        T.hold = 18 + 22 * r(); T.label = "off for a nap"; this.say("Bolt is tired and looks for a spot to nap"); break;
      }
      case "lookout": {
        const e = this.roadEdge(); if (!e) { this.task = null; return this.start("explore"); }
        T.to = e.at; T.face = e.face; T.hold = 5 + 6 * r(); this.say("Bolt goes to watch the road"); break;
      }
      case "jump": {
        const J = this.jumpable(); if (!J.length) { this.task = null; return this.start("explore"); }
        const e = extra.k !== undefined ? J.find((q) => q.k === extra.k) || J[0] : J[Math.floor(r() * J.length)];
        const dir = this.jumpLine(e); if (dir === null) { this.say("No room to jump that hurdle from here"); this.task = null; return this.start("explore"); }
        T.k = e.k; T.yaw = dir; T.e = e; T.to = [e.at[0] - 7.0 * Math.cos(dir), e.at[1] - 7.0 * Math.sin(dir)];
        if (e.h === null) { T.phase = "inspect"; T.label = "taking a closer look at a hurdle"; this.say("Bolt goes closer to see how high the bar is"); break; }
        T.label = `lining up for the ${Math.round(e.h * 100)} cm jump`; this.say(`Bolt eyes the hurdle it found (about ${Math.round(e.h * 100)} cm)`); break;
      }
      case "kerb": { this.say("Bolt sees the road and stops at the kerb", "good"); this.stats.kerbStops++; break; }
    }
  }
  zoomLine() {   // a straight line Bolt has seen to be open: ~14 m
    const M = this.map, p = this.where();
    for (let i = 0; i < 80; i++) {
      const here = i < 30, a = here ? [p.x, p.y] : this.freeSpot(1, 6);
      if (!a) continue;
      const ang = here ? p.yaw + (this.rand() - 0.5) * (i < 12 ? 1.2 : 3.0) : this.rand() * 2 * Math.PI;
      const b = [a[0] + 14 * Math.cos(ang), a[1] + 14 * Math.sin(ang)];
      if (M.lineClear(a, b, 0.8, true)) return [a, b, ang];
    }
    return null;
  }
  // the hurdle the jump brain looks at: Bolt's own estimate (the real bar may be a little elsewhere)
  aimEstimate(e, yaw) {
    const b = this.bolt;
    b.aimed = e.k; b.hurdleAt = e.at.slice(); b.hurdleDir = [Math.cos(yaw), Math.sin(yaw)]; b.hurdleH = e.h;
  }

  step(dt) {
    this.t += dt;
    this.drift(dt);
    const body = this.body, w = body.want, p = this.where(), M = this.map;
    if (body.mode !== "fallen") this.stats.distance += hyp(p.x - this.prev[0], p.y - this.prev[1]);
    this.prev = [p.x, p.y];
    if (body.busy) { if (body.mode === "fallen" && this.task && this.task.name === "jump") this.task = null; return; }
    // the kerb: road seen just ahead while moving -> stop (a reflex, before any task)
    const moving = ["walk", "trot", "ctrot", "canter", "pace", "turnclip", "sharp"].includes(body.mode);
    const ra = M.roadAhead(p.x, p.y, p.yaw, 1.0 + 0.8 * body.speed);
    if (moving && ra !== null && !(this.task && this.task.name === "kerb") && !(this.task && this.task.name === "lookout" && this.task.phase !== "go")) {
      this.task = null; body.armed = null; this.start("kerb");
    }
    if (!this.task) this.start(this.choose());
    const T = this.task;
    T.pt += dt;
    const done = (text, kind) => { if (text) this.say(text, kind); this.task = null; if (body.armed !== null) body.armed = null; if (!["sit", "lie"].includes(body.mode)) w.gait = "walk"; };
    if (T.pt > 60 && !["nap", "kerb"].includes(T.name)) return done(null);   // stuck: think again
    if (T.noPath > 3 || (T.stuck || 0) > 6) { this.say("Bolt can't find a way there and changes its mind"); return done(null); }
    switch (T.name) {
      case "kerb": {   // stop, then turn away from the road and walk back
        if (T.phase === "go") {
          this.stop();
          if (T.pt > 1.5) {   // the open direction needing the least turn past a quarter turn away from the road
            T.phase = "away"; T.pt = 0; let best = null;
            for (let a = 0; a < 2 * Math.PI; a += Math.PI / 12) {
              const q = [p.x + 4 * Math.cos(a), p.y + 4 * Math.sin(a)], cost = Math.abs(Math.abs(wrap(a - p.yaw)) - 1.7);
              if (M.roadAhead(p.x, p.y, a, 4, 0.35) === null && M.clearance(p.x, p.y, a, 3, 0.4) >= 2.5 && (!best || cost < best.cost)) best = { a, q, cost };
            }
            T.dir = best ? best.a : wrap(p.yaw + Math.PI); T.to = best ? best.q : [p.x - 3 * Math.cos(p.yaw), p.y - 3 * Math.sin(p.yaw)]; this.say("Bolt turns away from the road");
          }
        } else {
          const e = wrap(T.dir - p.yaw);
          if (Math.abs(e) > 0.35) { w.steer = clamp(2 * e, -1.6, 1.6); w.gait = "walk"; if (M.roadAhead(p.x, p.y, p.yaw, 0.45, 0.2) !== null) this.stop(); }
          else if (this.goTo(T.to, "walk", 0.8)) done(null);
          if (T.pt > 14) done(null);
        }
        break;
      }
      case "explore": case "trot": case "pace":
        if (this.goTo(T.to, T.gait || "walk", 1.0)) done(null);
        break;
      case "sniff":
        if (T.phase === "go") { if (this.goTo(T.to, "walk", 0.5)) { T.phase = "face"; T.pt = 0; } }
        else if (T.phase === "face") { const e = wrap(Math.atan2(T.thing[1] - p.y, T.thing[0] - p.x) - p.yaw), room = M.clearance(p.x, p.y, p.yaw + 0.5 * Math.sign(e), 1.0, 0.35) >= 0.9; w.steer = e; w.gait = Math.abs(e) > 0.5 && T.pt < 4 && room ? "walk" : "sniff"; if (Math.abs(e) <= 0.5 || T.pt >= 4 || !room) { T.phase = "sniff"; T.pt = 0; this.stats.sniffs++; this.say(`Bolt sniffs ${T.what}`); } }
        else { w.steer = null; w.gait = "sniff"; if (T.pt > T.hold) { this.curiosity = Math.max(0, this.curiosity - 0.25); done(null); } }
        break;
      case "lookout":
        if (T.phase === "go") { if (this.goTo(T.to, "walk", 0.6)) { T.phase = "turn"; T.pt = 0; } }
        else if (T.phase === "turn") { const e = wrap(T.face - p.yaw); w.steer = e; w.gait = "walk"; if (Math.abs(e) < 0.4 || T.pt > 6) { T.phase = "sit"; T.pt = 0; this.say("Bolt sits at the kerb and watches the road"); } if (M.roadAhead(p.x, p.y, p.yaw, 0.6, 0.3) !== null) { w.gait = "stand"; T.phase = "sit"; T.pt = 0; } }
        else { w.steer = null; w.gait = T.pt < T.hold ? "sit" : "walk"; if (T.pt > T.hold && body.mode !== "sit") { T.name = "kerb"; T.phase = "go"; T.pt = 2; } }
        break;
      case "sit":
        w.steer = null;
        if (T.pt < T.hold) w.gait = "sit";
        else { w.gait = "stand"; if (!["sit", "rise"].includes(body.mode)) done("Bolt is up again"); }
        break;
      case "nap":
        if (T.phase === "go") { if (this.goTo(T.to, "walk", 0.6) || T.pt > 40) { T.phase = "turn"; T.pt = 0; } }
        else if (T.phase === "turn") {   // lie down facing open ground (stretched out, the nose reaches ~0.6 m)
          const e = wrap(T.face - p.yaw), open = M.clearance(p.x, p.y, p.yaw, 1.0, 0.3) >= 0.9;
          w.steer = e; w.gait = "walk";
          if ((Math.abs(e) < 0.4 && open) || (T.pt > 8 && open)) { T.phase = "down"; T.pt = 0; this.say("Bolt turns round and lies down"); this.stats.naps++; }
          else if (T.pt > 16) done(null);
        }
        else if (T.phase === "down") { w.steer = null; w.gait = "lie"; T.label = "napping"; if ((body.mode === "lie" && this.energy > 0.92 && T.pt > 10) || (T.pt > T.hold + 10 && body.mode === "lie")) T.phase = "wake"; }
        else { w.gait = "stand"; w.steer = null; T.label = "waking up"; if (!T.woke) { T.woke = true; this.say("Bolt wakes up and gets up, front legs first"); } if (!["lie", "getup", "sit", "rise"].includes(body.mode)) done(null); }
        break;
      case "zoomies": {
        if (T.phase === "lineup") { if (this.goTo(T.from, "walk", 1.0)) { T.phase = "face"; T.pt = 0; } }
        else if (T.phase === "face") { const e = wrap(T.dir - p.yaw); w.steer = e; w.gait = "walk"; if (Math.abs(e) < 0.12 || T.pt > 8) { T.phase = "run"; T.pt = 0; T.label = "zoomies!"; } }
        else {
          const look = body.mode === "canter" ? 7 : 5, room = M.clearance(p.x, p.y, p.yaw, look, 0.4) >= look - 0.2 && M.roadAhead(p.x, p.y, p.yaw, look, 0.3) === null;
          w.steer = ["canter", "ctrot", "trot", "pace"].includes(body.mode) ? null : wrap(T.dir - p.yaw);
          if (!room) T.slow = true;
          if (body.mode === "canter") T.cantered = true;
          w.gait = T.slow ? "walk" : body.mode === "canter" && body.mt > T.canterFor ? ((T.slow = true), "walk") : "canter";
          if ((T.cantered && ["walk", "turnclip", "sharp", "stand"].includes(body.mode)) || T.pt > 20) { this.play = Math.max(0, this.play - 0.5); done(T.cantered ? "Zoomies over: Bolt slows to a walk" : null); }
        }
        break;
      }
      case "jump": this.jumpStep(T, p, w, done); break;
    }
  }
  jumpStep(T, p, w, done) {
    const body = this.body, e = this.map.hurdle(T.k) || T.e;
    if (body.mode !== "jump" && body.mode !== "land") T.e = e;   // keep refining the estimate until take-off
    const E = T.e, fx = Math.cos(T.yaw), fy = Math.sin(T.yaw);
    if (T.phase === "inspect") {   // 2.3 m in front of the bar, facing it: the lower rows see the bar
      const spot = [E.at[0] - 2.3 * fx, E.at[1] - 2.3 * fy];
      if (E.h !== null) { T.phase = "go"; T.pt = 0; T.label = `lining up for the ${Math.round(E.h * 100)} cm jump`; this.say(`The bar is about ${Math.round(E.h * 100)} cm: Bolt goes to line up`); return; }
      if (this.goTo(spot, "walk", 0.6)) { const e = wrap(T.yaw - p.yaw); w.steer = Math.abs(e) > 0.3 ? e : null; w.gait = Math.abs(e) > 0.3 ? "walk" : "stand"; }
      if (T.pt > 30) done("Bolt can't make out the bar and leaves the hurdle for now");
      return;
    }
    const along = (p.x - E.at[0]) * fx + (p.y - E.at[1]) * fy, side = -(p.x - E.at[0]) * fy + (p.y - E.at[1]) * fx;
    if (T.phase === "go") {
      T.to = [E.at[0] - 8.5 * fx, E.at[1] - 8.5 * fy];
      if (this.goTo(T.to, "walk", 1.0)) { T.phase = "face"; T.pt = 0; }
    } else if (T.phase === "face") {   // turn to look along the line, at a walk
      const e = wrap(T.yaw - p.yaw); w.steer = e; w.gait = "walk";
      if (Math.abs(e) < 0.3) { T.phase = "line"; T.pt = 0; }
      else if (T.pt > 10 || -along < 4.0) { T.phase = "go"; T.pt = 0; }
    } else if (T.phase === "line") {   // walk the line to the bar, square to it, then trot in
      const aim = [E.at[0] + (along + 2.0) * fx, E.at[1] + (along + 2.0) * fy];
      w.steer = clamp(wrap(Math.atan2(aim[1] - p.y, aim[0] - p.x) - p.yaw), -0.35, 0.35); w.gait = "walk";
      const askew = Math.abs(wrap(p.yaw - T.yaw));
      if (-along < 4.8 && -along > 2.6 && Math.abs(side) < 0.25 && askew < 0.14) {
        T.phase = "commit"; T.pt = 0; this.aimEstimate(E, T.yaw); body.armed = T.k; w.gait = "trot"; T.label = `jumping the ${Math.round(E.h * 100)} cm hurdle`;
      } else if (-along < 2.4 || T.pt > 15) { T.phase = "go"; T.pt = 0; T.tries = (T.tries || 0) + 1; this.say("Bolt was not lined up: it circles round for another go"); if (T.tries > 2) { this.say("Bolt gives up on the jump for now"); done(null); } }
    } else if (T.phase === "commit") {
      w.gait = "trot"; w.steer = 0;
      if (body.armed === T.k) this.aimEstimate(E, T.yaw);
      if (body.jumped === T.k && !body.busy) {
        const J = body.jumpInfo, dropped = this.bolt.barDropped(T.k);
        this.stats.jumps++;
        if (J.cleared && !dropped) { this.stats.clean++; this.say(`Clear jump over the hurdle it found (${Math.round(E.h * 100)} cm)!`, "good"); }
        else if (dropped) { this.stats.knocked++; this.say("Over, but the bar came down", "bad"); }
        else this.say("Bolt ran out of the jump", "bad");
        body.jumped = null; T.phase = "after"; T.pt = 0; this.play = Math.max(0, this.play - 0.35);
      } else if (body.armed === null && !body.busy && body.jumped !== T.k && T.pt > 1.0) { T.phase = "go"; T.pt = 0; }
      if (T.pt > 10) done(null);
    } else {
      w.gait = "walk"; w.steer = null;
      if (T.pt > 1.5) { if (this.bolt.barDropped(T.k)) { this.bolt.resetBar(T.k); this.say("(the bar is put back in its cups)"); } done(null); }
    }
  }
}

// ---------------------------------------------------------------- driving by remote
// A command is absolute state: { mode: "manual" | "auto", vx: -1..1 (joystick forward), turn: -1..1 (+ left),
// skill: { name, seq } (the last skill button pressed; acted on when seq changes), stop: bool, seq, t (sender ms) }.
export const DEADMAN = 0.5;   // s without a command: Bolt stops
export const GAITS = { walk: 0.35, trot: 0.65, canter: 0.95 };

export class Pilot {
  constructor(L) { this.L = L; this.cmd = null; this.at = -Infinity; this.skillSeq = null; this.cruise = null; this.sticky = null; this.guard = null; this.deadman = false; this.events = []; }
  say(text, kind = "info") { this.events.push({ text, kind }); }
  // a command arrived at time `now` (s, the Bolt screen's clock)
  receive(cmd, now) {
    this.cmd = cmd; this.at = now;
    if (this.deadman) { this.deadman = false; this.say("Remote back in touch", "good"); }
    if (cmd.skill && cmd.skill.seq !== this.skillSeq) { const first = this.skillSeq === null; this.skillSeq = cmd.skill.seq; if (!first) this.skill(cmd.skill.name); }
    if (cmd.stop) { this.cruise = null; this.sticky = null; }
  }
  skill(name) {
    const L = this.L;
    if (name === "jump") {
      const p = L.bolt.where(), J = L.auto.jumpable().filter((e) => { const d = hyp(e.at[0] - p.x, e.at[1] - p.y), a = Math.abs(wrap(Math.atan2(e.at[1] - p.y, e.at[0] - p.x) - p.yaw)); return d < 10 && a < 0.8; });
      if (!J.length) { this.say("No hurdle in view ahead: point Bolt at one first", "info"); return; }
      L.auto.task = null; L.auto.start("jump", { k: J[0].k });
      if (L.auto.task && L.auto.task.name === "jump") { this.assist = true; this.say("Bolt takes the jump on its own, then hands back to you", "cmd"); }
      return;
    }
    this.assist = false; if (L.auto.task && L.auto.task.name === "jump") L.auto.task = null;
    if (name in GAITS) { this.cruise = name; this.sticky = null; this.say(`Cruise at a ${name}: steer with the stick, Stop to halt`, "cmd"); }
    else { this.cruise = null; this.sticky = name; this.say({ sit: "Sit", lie: "Lie down", stand: "Stand" }[name] || name, "cmd"); }
  }
  // what Bolt should do this control step in manual mode
  step(dt, now) {
    const L = this.L, body = L.body, w = body.want, M = L.map, p = L.bolt.where();
    if (this.assist) {   // a jump the remote asked for: the auto mind runs it, then hands back
      if (L.auto.task && L.auto.task.name === "jump") { L.auto.step(dt); return; }
      this.assist = false; w.gait = "stand"; w.steer = null;
    }
    const c = this.cmd;
    if (!c || now - this.at > DEADMAN) {
      if (!this.deadman && c) { this.deadman = true; this.say("No signal from the remote for 0.5 s: Bolt stops", "bad"); }
      this.cruise = null; this.sticky = null; w.gait = "stand"; w.steer = null; this.guard = null; return;
    }
    if (c.stop) { w.gait = "stand"; w.steer = null; this.guard = null; return; }
    const vx = clamp(c.vx || 0, -1, 1), turn = clamp(c.turn || 0, -1, 1);
    if (vx > 0.1 || Math.abs(turn) > 0.3) this.sticky = null;
    if (this.sticky && vx <= 0.1) { w.gait = this.sticky; w.steer = null; this.guard = null; return; }
    if (vx < -0.3) this.cruise = null;   // pull back: stop (Bolt cannot walk backwards)
    let level = vx > 0.1 ? vx : this.cruise ? GAITS[this.cruise] : 0;
    // the collision guard: free distance ahead (where Bolt is heading) from its own map
    // the faster Bolt goes the earlier it slows: a canter takes ~3 s and several metres to come down to a stop
    const v = body.speed, head = p.yaw + 0.5 * turn, ahead = M.clearance(p.x, p.y, head, 8.0, 0.3), road = M.roadAhead(p.x, p.y, p.yaw, 1.0 + 0.9 * v, 0.5);
    let cap = 1, why = null;
    if (ahead < 4.0 + 1.5 * v) { cap = 0.65; why = "obstacle ahead: trot at most"; }
    if (ahead < 1.8 + 1.2 * v) { cap = 0.35; why = "obstacle ahead: walking"; }
    if (ahead < 0.7 + 0.8 * v) { cap = 0; why = "obstacle ahead: stopped"; }
    if (road !== null) { cap = 0; why = "kerb ahead: stopped"; }
    if (cap === 0 && level > 0 && Math.abs(turn) > 0.3) {   // turning away is allowed if that side is open
      const side = M.clearance(p.x, p.y, p.yaw + Math.sign(turn) * 0.9, 1.5, 0.3), roadSide = M.roadAhead(p.x, p.y, p.yaw + Math.sign(turn) * 0.9, 1.0, 0.3);
      if (side >= 1.2 && roadSide === null) { cap = 0.3; why = "turning away from what is ahead"; }
    }
    if (why && (!this.guard || this.guard.why !== why) && cap < level) this.say(`Guard: ${why}`, cap === 0 ? "bad" : "info");
    this.guard = cap < level ? { why, cap } : null;
    level = Math.min(level, cap);
    if (level <= 0.1 && Math.abs(turn) <= 0.3) { w.gait = "stand"; w.steer = null; return; }
    w.gait = level > 0.8 && Math.abs(turn) < 0.2 ? "canter" : level > 0.5 ? "trot" : "walk";
    w.steer = Math.abs(turn) < 0.05 ? null : 1.1 * turn;
  }
}

// ---------------------------------------------------------------- everything together
const SHOW = { stand: "standing", sniff: "sniffing", walk: "walking", turnclip: "walking", sharp: "turning", brake: "slowing down", trot: "trotting", ctrot: "trotting", pace: "pacing",
  canter: "cantering", sit: "sitting", lie: "lying down", getup: "getting up", rise: "standing up", jump: "jumping!", land: "landing", bridge: "landing", settle: "settling", fallen: "fell over: getting up" };

export class RemoteLife {
  constructor(live, mj, bundle, brains, clips, seed = 1) {
    const bolt = new live.Bolt(mj, bundle);
    if (brains.jump) { bolt.brains.jump = new live.Brain(brains.jump); bolt.c = { ...bolt.c, jump_steps: brains.jump.jump_steps || 200 }; }
    setupRemoteYard(bolt, bundle);
    this.bolt = bolt; this.bundle = bundle;
    this.gpu = new Imitate(bolt, brains.gpu, clips); this.g18 = new Imitate(bolt, brains.g18, clips);
    const head = (bundle.geoms || []).find((g) => g.part === "head");
    this.head = head ? head.id : 0;
    this.seed = seed; this.reset(seed);
  }
  reset(seed = this.seed) {
    const bolt = this.bolt;
    if (this.body) { setupRemoteYard(bolt, this.bundle); this.gpu.stop(); this.g18.stop(); }
    this.body = new DogBody(bolt, this.gpu, this.g18);
    this.body.jumpLimit = bolt.c.jump_steps || 200;
    const sn = (this.bundle.geoms || []).find((g) => g.part === "snout"); this.body.snout = sn ? sn.id : undefined;
    this.map = new SenseMap(); this.scanner = new Scanner(bolt, this.head);
    this.auto = new AutoMind(this, { seed }); this.pilot = new Pilot(this);
    this.mode = "manual"; this.t = 0; this.log = []; this.seed = seed; this.perStep = Math.ceil(this.scanner.rays.length / (1 / (bolt.dt * SCAN.hz)));
  }
  say(text, kind = "info") { this.log.push({ t: this.t, text, kind }); }
  setMode(mode) {
    if (mode === this.mode || !["manual", "auto"].includes(mode)) return;
    this.mode = mode;
    if (mode === "auto") { this.auto.task = null; this.pilot.assist = false; this.say("Auto self mode: Bolt looks after itself", "cmd"); }
    else { this.auto.task = null; this.body.want.gait = "stand"; this.body.want.steer = null; this.pilot.cruise = null; this.pilot.sticky = null; this.say("Manual: you drive", "cmd"); }
  }
  // one control step (1/50 s); `now`: the Bolt screen's clock in s (for the dead-man stop)
  step(now) {
    const dt = this.bolt.dt;
    this.t += dt;
    this.scanner.step(this.perStep, (h) => this.map.integrate(h)); this.map.scanZ = this.scanner.z;
    if (this.mode === "auto") this.auto.step(dt);
    else { this.pilot.step(dt, now); this.auto.drift(dt); }
    this.bumper();
    this.body.step(dt);
    for (const src of [this.body.events, this.pilot.events, this.auto.log]) while (src.length) { const e = src.shift(); this.log.push({ t: this.t, text: e.text, kind: e.kind }); }
  }
  // the last reflex: never walk into something the scanner sees within half a metre (turning away is allowed
  // where that side is open: Bolt cannot walk backwards)
  bumper() {
    const b = this.body, w = b.want, M = this.map, p = this.bolt.where();
    if (b.busy || !["walk", "trot", "canter", "pace"].includes(w.gait)) { this.bumped = false; return; }
    const skip = b.armed !== null ? b.armed : -1, ahead = M.clearance(p.x, p.y, p.yaw, 1.0, 0.28, skip);
    if (ahead >= 0.5) { this.bumped = false; return; }
    // turn toward the side asked for if it is open, else the more open side, at a walk (a sharp turn)
    const side = (s) => Math.max(...[0.9, 1.6, 2.3].map((a) => M.clearance(p.x, p.y, p.yaw + s * a, 1.2, 0.28, skip)));
    const left = side(1), right = side(-1);
    const s = Math.sign(w.steer || 0), pick = s && Math.abs(w.steer) > 0.1 && (s > 0 ? left : right) >= 0.8 ? s : Math.max(left, right) >= 0.8 ? (left >= right ? 1 : -1) : 0;
    if (pick) { w.gait = "walk"; w.steer = 1.3 * pick; this.bumped = false; return; }
    w.gait = "stand"; w.steer = null;
    this.bumpedFor = this.bumped ? (this.bumpedFor || 0) + this.bolt.dt : 0; this.bumped = true;
    if (this.bumpedFor > 3 && this.mode === "auto" && this.auto.task && this.auto.task.name !== "nap") { this.auto.task = null; this.bumpedFor = 0; }
  }
  doing() {
    const b = this.body;
    if (["jump", "land", "bridge", "fallen", "getup"].includes(b.mode)) return SHOW[b.mode];
    if (this.mode === "auto" || this.pilot.assist) return this.auto.label;
    if (this.pilot.deadman) return "stopped: no signal";
    if (this.pilot.guard) return `guard: ${this.pilot.guard.why}`;
    return SHOW[b.mode] || b.mode;
  }
  // what the Bolt screen sends back to the remote
  status() {
    const b = this.body, S = this.auto.stats, H = [...this.map.hurdles.keys()].map((k) => this.map.hurdle(k)).filter(Boolean);
    return { mode: this.mode, doing: this.doing(), gait: SHOW[b.mode] || b.mode, speed: +b.speed.toFixed(2), falls: b.falls, up: b.recoveries,
      guard: this.pilot.guard ? this.pilot.guard.why : null, deadman: this.pilot.deadman, hurdles: H.length, jumps: S.jumps, clean: S.clean,
      energy: Math.round(100 * this.auto.energy), t: Math.round(this.t) };
  }
}
