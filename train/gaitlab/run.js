// Bolt Run (simulation): an endless chase down a fenced track. Bolt runs with the full MuJoCo physics and its
// real trained brains (the dog-motion brains for walk/trot/canter and turns, the run 12 walk brain for sharp
// turns, the jump brain over the real agility hurdles, the self-righting brain after a fall), the same DogBody
// as Bolt's Day Out. The player only says where to go: which lane, how fast, and when to jump.
// No DOM here: the page (run_template.html) and the headless check (run_test.mjs) both drive RunLife.
//
// The track runs along +x (MuJoCo planes are infinite for contacts, so the floor never ends). Three lanes 1 m
// apart between two walls. Crates, bins and posts are mocap bodies (moving plain world geoms at run time made
// MuJoCo miss collisions), placed in chunks ahead of Bolt and parked again behind it, so the object count is
// fixed. Hurdles are the real agility hurdles (mocap stands, free bars in cups): a knocked bar falls for real.
// The dog-catcher behind is drawn only (kinematic, no body in the physics): it never touches Bolt.
import { DogBody, TAKEOVER, rng } from "./doglife.js";
import { Imitate } from "./imitate.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export const TRACK = {
  lane: 1.0,          // lane centres at -1, 0, +1 m
  wall: 1.85,         // inner face of the side walls (|y|)
  ahead: 34,          // chunks are laid this far ahead of Bolt
  behind: 6,          // and parked once this far behind
  first: 12,          // the first chunk, metres from the start line
};
export const JUMP = { range: 4.5, late: TAKEOVER + 0.1, bonus: 25 };   // Up arms a jump when the bar is 1.2-4.5 m ahead
export const CHASE = { start: 6, maxGap: 9, catch: 1.0, v0: 0.75, ramp: 0.003, vmax: 1.6, stumble: 2.0 };   // stumble: m the catcher gains when Bolt runs into something
export let LOOK = 1.2;   // m: Bolt aims at a point on its lane this far ahead
export function setLook(v) { LOOK = v; }
export let BAND = 0.35, PRED = 2.0;   // m: Bolt trots on while within BAND of its lane now and PRED m ahead
export function setBand(b, p) { BAND = b; PRED = p; }

// the pools of movable solid things: kind, MuJoCo shape, half sizes (cylinder: radius, half height), how many
export const POOL = [
  { kind: "crate", type: "box", size: [0.3, 0.3, 0.3], n: 10 },
  { kind: "bin", type: "cylinder", size: [0.27, 0.45], n: 6 },
  { kind: "post", type: "cylinder", size: [0.07, 0.55], n: 6 },
  { kind: "block", type: "box", size: [0.3, 0.42, 0.35], n: 8 },   // fills the sides beside a hurdle
  { kind: "wall", type: "box", size: [24, 0.1, 0.45], n: 2 },
];

export class RunLife {
  constructor(live, mj, bundle, brains, clips, seed = 1) {
    const bolt = new live.Bolt(mj, bundle);
    if (brains.jump) { bolt.brains.jump = new live.Brain(brains.jump); bolt.c = { ...bolt.c, jump_steps: brains.jump.jump_steps || 200 }; }
    this.items = [];
    for (const p of POOL) for (let j = 0; j < p.n; j++) this.items.push({ kind: p.kind, type: p.type, size: p.size, i: this.items.length, at: null });
    const xml = this.items.map((o) => {
      const z = o.type === "box" ? o.size[2] : o.size[1];
      return `<body name="ob${o.i}" mocap="true" pos="0 ${200 + 2 * o.i} ${z}"><geom name="prop${o.i}" type="${o.type}" size="${o.size.join(" ")}" rgba="0.8 0.8 0.8 1"/></body>`;
    }).join("");
    bolt.build(xml); bolt.props = this.items;
    for (const o of this.items) o.mocap = bolt.model.body_mocapid[bolt.mj.mj_name2id(bolt.model, bolt.mj.mjtObj.mjOBJ_BODY.value, `ob${o.i}`)];
    this.bolt = bolt; this.bundle = bundle;
    this.gpu = new Imitate(bolt, brains.gpu, clips); this.g18 = new Imitate(bolt, brains.g18, clips);
    this.reset(seed);
  }

  reset(seed = this.seed) {
    const b = this.bolt;
    this.gpu.stop(); this.g18.stop();
    b.teleport(this.bundle.start, [0, 0], 0);
    for (const o of this.items) this.park(o);
    for (let k = 0; k < b.real.length; k++) { b.placeReal(k, null, 0, 0); b.real[k].seg = null; }
    this.body = new DogBody(b, this.gpu, this.g18);
    this.body.jumpLimit = b.c.jump_steps || 200;
    this.seed = seed; this.rand = rng(seed * 7919 + 13);
    this.phase = "ready"; this.t = 0; this.runT = 0; this.log = [];
    this.lane = 0; this.gait = "trot"; this.slow = false; this.fast = false;
    this.chaser = -CHASE.start; this.chaserV = 0; this.caughtAt = null;
    this.dist = 0; this.segs = []; this.nextX = TRACK.first; this.segN = 0; this.lastHurdle = 0;
    this.stats = { jumps: 0, clean: 0, knocked: 0, runOuts: 0, bumps: 0, touchS: 0, falls: 0, tooEarly: 0, noBar: 0, notLined: 0, knockedWalking: 0 };
    this.lastTouch = -9; this.jumpNow = null; this.msg = null; this.wallAt = null;
    this.placeWalls(0);
    this.lay();
  }
  say(text, kind = "info") { this.log.push({ t: this.runT, text, kind }); }
  note(text, kind = "info") { this.msg = { text, kind, t: this.t }; }

  // ---- the track: mocap things in and out
  park(o) { this.put(o, [0, 200 + 2 * o.i]); o.at = null; }
  put(o, [x, y], yaw = 0) {
    const d = this.bolt.data, z = o.type === "box" ? o.size[2] : o.size[1];
    d.mocap_pos.set([x, y, z], 3 * o.mocap); d.mocap_quat.set([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 4 * o.mocap);
    o.at = [x, y]; o.yaw = yaw;
  }
  free(kind) { return this.items.find((o) => o.kind === kind && !o.at); }
  freeHurdle() { return this.bolt.real.findIndex((H) => !H.at); }
  // the walls are 48 m long: re-centred on Bolt every 12 m (same shape, so the contacts do not change)
  placeWalls(x) {
    const W = this.items.filter((o) => o.kind === "wall"), c = Math.round(x / 12) * 12;
    if (this.wallAt === c) return;
    this.wallAt = c;
    this.put(W[0], [c + 12, TRACK.wall + 0.1]); this.put(W[1], [c + 12, -TRACK.wall - 0.1]);
  }
  level() { return clamp(this.dist / 400, 0, 1); }   // 0 at the start, 1 from 400 m
  // lay chunks until TRACK.ahead past Bolt; each chunk is one row across the track
  lay() {
    const x = this.bolt.where().x;
    while (this.nextX < x + TRACK.ahead) {
      const seg = this.chunk(this.nextX);
      if (!seg) break;   // the pool is used up: try again once something behind is parked
      this.segs.push(seg);
      const L = this.level(), gap = 9.5 - 3.0 * L + this.rand() * (2.5 - L);
      this.nextX += gap + (seg.kind === "hurdle" ? 2.5 : 0);
    }
  }
  chunk(x) {
    const r = this.rand(), L = this.level(), lanes = [-1, 0, 1], R = this.rand;
    const pick = (a) => a[Math.floor(R() * a.length)];
    const ly = (l) => l * TRACK.lane;
    const need = (spec) => { const got = []; for (const [kind, n] of spec) { const f = this.items.filter((o) => o.kind === kind && !o.at && !got.includes(o)); if (f.length < n) return null; got.push(...f.slice(0, n)); } return got; };
    const seg = { x, kind: null, items: [], hurdle: -1, n: this.segN++ };
    // a hurdle: every few chunks once past the first ones, heights 20 cm at the start up to 50 cm
    const since = seg.n - (this.lastHurdle ?? 0), wantHurdle = seg.n >= 2 && since >= 3 && (since >= 5 || r < 0.35 + 0.15 * L);
    if (wantHurdle) {
      const k = this.freeHurdle(), blocks = need([["block", 2]]);
      if (k >= 0 && blocks) {
        x += 3; seg.x = x;   // a longer run-up: Bolt must be trotting, lined up on the middle lane
        const h = Math.round(100 * (0.2 + (0.1 + 0.2 * L) * R())) / 100;
        this.bolt.placeReal(k, [x, 0], 0, h);   // yaw 0: jumped along +x, the bar across the track
        this.bolt.real[k].seg = seg;
        this.put(blocks[0], [x, 0.94 + 0.42 + 0.02]); this.put(blocks[1], [x, -(0.94 + 0.42 + 0.02)]);
        Object.assign(seg, { kind: "hurdle", hurdle: k, items: blocks, h, lane: 0 }); this.lastHurdle = seg.n;
        return seg;
      }
    }
    if (r < 0.55) {   // one thing in one lane
      const kind = pick(L < 0.3 ? ["crate", "crate", "bin"] : ["crate", "bin", "post"]), o = need([[kind, 1]]);
      if (!o) return null;
      const l = pick(lanes); this.put(o[0], [x, ly(l)], R() * 0.6);
      Object.assign(seg, { kind: "one", items: o, blocked: [l] });
    } else if (r < 0.85 || L < 0.15) {   // two lanes blocked: one way through
      const o = need([["crate", 1], [pick(["bin", "crate"]), 1]]);
      if (!o) return null;
      const open = pick(lanes), [a, c] = lanes.filter((l) => l !== open);
      this.put(o[0], [x, ly(a)], R() * 0.6); this.put(o[1], [x + (R() - 0.5) * 0.6, ly(c)], R() * 0.6);
      Object.assign(seg, { kind: "gap", items: o, blocked: [a, c], open });
    } else {   // a narrowing: posts on the lane lines, then one lane shut a little further on
      const o = need([["post", 2], ["crate", 1]]);
      if (!o) return null;
      const side = R() < 0.5 ? 1 : -1;
      this.put(o[0], [x, 0.5 * side]); this.put(o[1], [x, 1.5 * side]);
      const shut = pick([-side, 0]); this.put(o[2], [x + 3.5, ly(shut)], 0.3);
      Object.assign(seg, { kind: "narrow", items: o, blocked: [side], shut });
    }
    return seg;
  }
  recycle() {
    const x = this.bolt.where().x;
    while (this.segs.length && this.segs[0].x + 5 < x - TRACK.behind) {
      const s = this.segs.shift();
      for (const o of s.items) this.park(o);
      if (s.hurdle >= 0) { this.bolt.placeReal(s.hurdle, null, 0, 0); this.bolt.real[s.hurdle].seg = null; }
    }
  }

  // ---- what the player asks
  start() { if (this.phase === "ready") { this.phase = "run"; this.say("Go! The dog-catcher is after Bolt", "cmd"); } }
  steer(dir) {
    if (this.phase !== "run") return;
    const l = clamp(this.lane + dir, -1, 1);
    if (l === this.lane) { this.note(dir > 0 ? "Already in the left lane" : "Already in the right lane"); return; }
    this.lane = l;
  }
  // the hurdle in front: the nearest real hurdle on the track with its line still ahead of Bolt
  hurdleAhead() {
    const w = this.bolt.where(); let best = null;
    this.bolt.real.forEach((H, k) => { if (!H.at) return; const a = H.at[0] - w.x; if (a > 0.3 && (!best || a < best.ahead)) best = { k, ahead: a, h: H.h }; });
    return best;
  }
  jump() {
    if (this.phase !== "run") return "off";
    const B = this.body, n = this.hurdleAhead();
    if (B.busy) return "busy";
    if (B.armed !== null) return "armed";
    if (!n) { this.stats.noBar++; this.note("No bar ahead", "bad"); return "nobar"; }
    if (n.ahead > JUMP.range) { this.stats.tooEarly++; this.note("Too early", "bad"); return "early"; }
    if (n.ahead < JUMP.late) { this.note("Too late", "bad"); return "late"; }
    const w = this.bolt.where();
    if (Math.abs(w.y + Math.sin(w.yaw) * n.ahead) > 0.3 || Math.abs(w.yaw) > 0.25) { this.stats.notLined++; this.note("Line up in the middle lane first", "bad"); return "notlined"; }
    B.arm(n.k); this.lane = 0; this.note(`Jump! ${Math.round(n.h * 100)} cm`, "good");
    return "armed";
  }
  setSlow(on) { this.slow = !!on; }
  setFast(on) { this.fast = !!on; }

  // ---- one control step (1/50 s)
  step() {
    const b = this.bolt, B = this.body, dt = b.dt, w = b.where();
    this.t += dt;
    if (this.phase === "run") this.runT += dt;
    // steering: aim at a point on the chosen lane a little ahead (a jump lines up on the hurdle's centre)
    const ty = (B.armed !== null ? 0 : this.lane * TRACK.lane), err = w.y - ty;
    const e = wrap(clamp(Math.atan2(-err, LOOK), -0.7, 0.7) - w.yaw);
    if (this.phase === "run") {
      const fast = B.armed !== null ? "trot" : this.slow ? "walk" : this.fast ? "canter" : "trot";
      // at a trot or canter the dog-motion brains cannot steer (turning means slowing to a walk). So Bolt keeps
      // going while it is, and will stay, inside its lane's band; outside it, it slows and turns at a walk, and
      // trots again once it is lined up on the lane (near its centre, heading along the track)
      const fastNow = ["trot", "ctrot", "canter", "pace"].includes(B.mode);
      // before a hurdle Bolt lines up tighter: the jump brain needs it near the bar's centre, heading straight
      const hn = this.hurdleAhead(), near = hn && hn.ahead < 9 && this.lane === 0 && !B.busy;
      const band = near ? 0.18 : BAND;
      const inBand = Math.abs(err) < band && Math.abs(err + Math.sin(w.yaw) * PRED) < band;
      const lined = inBand && Math.abs(err) < (near ? 0.15 : 0.3) && Math.abs(w.yaw) < (near ? 0.1 : 0.2);
      if (B.armed !== null) { B.want.gait = "trot"; B.want.steer = 0; }
      else if (fastNow) { B.want.gait = fast; B.want.steer = inBand ? 0 : Math.sign(e || 1) * Math.max(Math.abs(e), 0.4); }
      else { B.want.gait = lined ? fast : "walk"; B.want.steer = lined ? 0 : e; }
    } else { B.want.gait = "stand"; B.want.steer = null; }
    // an armed jump drifting off line (a trot cannot steer): Bolt runs out rather than hit the wings
    if (B.armed !== null && B.mode !== "jump") {
      const H = b.real[B.armed], ahead = H.at ? H.at[0] - w.x : 0;
      if (ahead > TAKEOVER + 0.05 && Math.abs(w.y + Math.sin(w.yaw) * ahead) > 0.32) { B.armed = null; this.armedWas = null; this.stats.runOuts++; this.note("Off line: Bolt runs out", "bad"); this.say("Bolt drifted off line and ran out of the jump", "bad"); }
    }
    const falls0 = B.falls;
    B.step(dt);
    if (B.falls > falls0) { this.stats.falls++; this.say("Bolt fell: the self-righting brain gets it up", "bad"); }
    // the jump's true outcome: bar still in its cups, all paws past the bar line, no fall
    if (B.mode === "jump" && !this.jumpNow) { this.jumpNow = { k: B.jumpInfo.k, falls: B.falls, from: { speed: +B.jumpInfo.speed.toFixed(2), y: +w.y.toFixed(2), yaw: +w.yaw.toFixed(2), h: b.real[B.jumpInfo.k].h, prev: B.prevMode } }; this.stats.jumps++; const sg = b.real[B.jumpInfo.k].seg; if (sg) sg.jumped = true; }
    if (this.jumpNow && !["jump", "land", "settle", "bridge"].includes(B.mode)) {
      const J = B.jumpInfo, k = this.jumpNow.k, dropped = b.barDropped(k), fell = B.falls > this.jumpNow.falls;
      if (J.cleared && !dropped && !fell) { this.stats.clean++; this.say(`Clean jump over ${Math.round(b.real[k].h * 100)} cm! +${JUMP.bonus}`, "good"); this.note(`Clean! +${JUMP.bonus}`, "good"); }
      else { if (dropped) this.stats.knocked++; this.say(dropped ? "Bar knocked down" : !J.cleared ? "Bolt did not get over" : "Over, but Bolt fell", "bad"); }
      (this.jumpLog = this.jumpLog || []).push({ ...this.jumpNow.from, clean: J.cleared && !dropped && !fell, cleared: J.cleared, dropped, fell, fellIn: fell ? B.fellIn : null });
      this.jumpNow = null;
    }
    if (B.armed === null && this.armedWas !== null && this.armedWas !== undefined && B.mode !== "jump") { this.stats.runOuts++; this.note("Not fast enough: ran out", "bad"); }
    this.armedWas = B.mode === "jump" ? null : B.armed;
    // bumps: Bolt touching any crate, bin, post, block or wall (episodes less than 1 s apart count as one)
    if (b.touchingProps().length) { if (this.t - this.lastTouch > 1.0) { this.stats.bumps++; this.stumble("Bolt ran into something"); } this.lastTouch = this.t; this.stats.touchS += dt; }
    // a hurdle Bolt went through without jumping it: its bar knocked down by a walk or a trot
    for (const [k, H] of b.real.entries()) if (H.at && H.seg && !H.seg.passed && w.x > H.at[0] + 0.5) {
      H.seg.passed = true;
      if (!H.seg.jumped && b.barDropped(k)) { this.stats.knockedWalking++; this.stumble("Bolt ran through the hurdle"); }
    }
    // the track and the chaser
    const p = b.where();
    this.dist = Math.max(this.dist, p.x);
    this.placeWalls(p.x); this.recycle(); this.lay();
    if (this.phase === "run") {
      const v = clamp(CHASE.v0 + CHASE.ramp * this.dist, 0, CHASE.vmax) * Math.min(1, this.runT / 2);
      this.chaserV = v; this.chaser = Math.max(this.chaser + v * dt, p.x - CHASE.maxGap);
      if (p.x - this.chaser < CHASE.catch) { this.phase = "caught"; this.caughtAt = this.runT; this.say("Caught! The dog-catcher got Bolt", "bad"); }
    }
    for (const e2 of B.events.splice(0)) if (!/^Bolt fell/.test(e2.text)) this.say(e2.text, e2.kind);
  }
  gap() { return this.bolt.where().x - this.chaser; }
  // the game's one rule on top of the physics: running into something costs a stumble, the catcher gains ground
  stumble(why) { if (this.phase !== "run") return; this.chaser += CHASE.stumble; this.say(`${why}: the catcher gains ${CHASE.stumble} m`, "bad"); this.note("Bump!", "bad"); }
  score() { return Math.floor(this.dist) + JUMP.bonus * this.stats.clean; }
  status() {
    const B = this.body, n = this.hurdleAhead();
    return { phase: this.phase, t: this.runT, dist: this.dist, score: this.score(), speed: B.speed, gap: this.gap(), mode: B.mode, brain: B.brain(),
      lane: this.lane, bar: n && n.ahead < 12 ? n : null, armed: B.armed !== null, stats: this.stats, chaserV: this.chaserV };
  }
}
