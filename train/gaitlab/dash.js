// Bolt Dash (simulation): an arcade endless run. Three lanes, crates and barrels to dodge, bars to hop, treats to
// grab, a dog-catcher closing in whenever Bolt stumbles. Bolt is the same MuJoCo body as Bolt Run and Bolt's Day
// Out (live.js, doglife.js DogBody): the trained dog-motion brains move every leg at 50 Hz (trot, then the canter
// brain once the pace is high), and the self-righting brain gets Bolt up after a fall.
//
// Arcade assist (said on the page): the brains learned a straight trot and canter, not quick dodges, so the
// player's lane and hop do not go through them. A guide acts on Bolt's body in the physics instead, every
// physics step (500 Hz):
//   lane     a force field on all 13 of Bolt's bodies (in proportion to their mass, so it does not twist the
//            body) pulls the torso onto a smooth path to the chosen lane: visible in under 0.1 s, in lane in
//            about 0.35 s (dash_test.mjs measures both)
//   heading  a torque on the torso holds Bolt along the track, turned a little into a dodge
//   level    a torque on the torso keeps the body level (a hard sideways pull tipped it over without it)
//   pace     a forward force (at most 4 m/s^2) holds the run speed the game asks for, and brakes in a stumble
//   hop      an upward kick (2.4 m/s) and, while in the air, a lift of 45% of gravity: a floaty arcade hop
// The legs stay the brain's: feet still push, slip and land in the physics, and things Bolt hits are solid.
//
// The track runs along +x (MuJoCo planes are infinite for contacts). Lanes 0.8 m apart, +y is left. Crates,
// barrels, barriers and bars are free bodies (real mass, knocked flying when hit) taken from a fixed pool and
// laid in rows ahead of Bolt. Until something touches them they float a hair above the floor with gravity
// compensated (MuJoCo gravcomp), so a still obstacle costs no contacts; once hit, gravity is back and it tumbles.
// Bars rest on nothing at all (the stands are drawn only): a touch drops them, as a bar falls from its cups.
// Treats and the dog-catcher are drawn only: no bodies in the physics. No DOM here: dash_template.html draws it,
// dash_test.mjs plays it headless.
import { DogBody, rng } from "./doglife.js";
import { Imitate } from "./imitate.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, u) => a + (b - a) * u;

export const TRACK = { lane: 0.8, half: 1.4, first: 14, ahead: 42, behind: 8 };
export const ASSIST = {
  w: 16, amax: 30,          // the lane path: a critically damped move (rad/s), at most 30 m/s^2
  kp: 120, kd: 22,          // the pull onto that path (per kg: m/s^2 per m, per m/s)
  yaw: 40, yawD: 4, lean: 0.6,   // heading torque (N m per rad, per rad/s); lean: share of the dodge's angle Bolt faces
  up: 150, upD: 8,          // level torque (N m per unit tilt, per rad/s)
  fwd: 5, fwdMax: 4,        // pace: m/s^2 per m/s short of the asked speed, at most fwdMax
  hop: 2.4, float: 0.45, slam: 1.6,   // hop kick (m/s), lift in the air (share of g), Down in the air: extra pull down (share of g)
};
// the pace: the speed asked rises from v0 to v1 and the clock from clock0 to clock1 over the first `full` m; then the
// clock alone goes on rising to clock2 over the next `more` m (the brains' canter is not asked to go faster than v1)
export const PACE = { v0: 1.7, v1: 2.7, canterAt: 2.05, clock0: 1.0, clock1: 1.35, full: 320, clock2: 1.7, more: 700 };
// the dog-catcher's gap (m): a crash or a knocked bar takes some away at once; it comes back at `regen` m/s (at the
// start, at full pace) once Bolt has run `hold` s without hitting anything; lag: m lost per m/s Bolt is slower than
// the pace (less `slack`), as in a fall
export const CHASE = { start: 6, max: 6, regen: [0.3, 0.1], hold: 2.5, lag: 0.6, slack: 0.35, crash: 2.0, knock: 1.2, caught: 0.5 };
export const SCORE = { metre: 1, treat: 5, bar: 25, streak: 8, maxMult: 5 };   // points times the multiplier: +1 for every `streak` rows passed without a crash or a knocked bar
const RUNNING = ["walk", "trot", "ctrot", "canter", "pace", "turnclip"];

// the pool of solid things: kind, MuJoCo shape, half sizes (cylinder: radius, half height; capsule: radius, half
// length), mass, how many
export const POOL = [
  { kind: "crate", type: "box", size: [0.22, 0.22, 0.22], mass: 3, n: 16 },
  { kind: "barrel", type: "cylinder", size: [0.19, 0.27], mass: 2.5, n: 12 },
  { kind: "barrier", type: "box", size: [0.07, 0.36, 0.2], mass: 4, n: 8 },
  { kind: "bar", type: "capsule", size: [0.022, 1.3], mass: 0.3, n: 4 },
];

export class Dash {
  constructor(live, mj, bundle, brains, clips, seed = 1) {
    const bolt = new live.Bolt(mj, bundle);
    // only Bolt and the floor from the bundle: the agility hurdles (4 stands, 4 loose bars) are not used here
    bolt.baseXml = bundle.xml.replace(/<body name="(hurdle\d*|ice|bar\d+)"[^]*?<\/body>/g, "");
    this.items = [];
    for (const p of POOL) for (let j = 0; j < p.n; j++) this.items.push({ ...p, i: this.items.length });
    const xml = this.items.map((o) => {
      const rot = o.type === "capsule" ? ' euler="1.5708 0 0"' : "";   // a bar lies across the track (along y)
      return `<body name="ob${o.i}" pos="${-50 - o.i} 0 ${30 + o.i}" gravcomp="1"><freejoint/>` +
        `<geom name="ob${o.i}" type="${o.type}" size="${o.size.join(" ")}" mass="${o.mass}"${rot} friction="0.7 0.02 0.01" rgba="0.8 0.8 0.8 1"/></body>`;
    }).join("");
    bolt.build(xml);
    const m = bolt.model, id = (t, n) => mj.mj_name2id(m, t, n);
    for (const o of this.items) {
      o.body = id(mj.mjtObj.mjOBJ_BODY.value, `ob${o.i}`); o.geom = id(mj.mjtObj.mjOBJ_GEOM.value, `ob${o.i}`);
      const j = m.body_jntadr[o.body]; o.q = m.jnt_qposadr[j]; o.v = m.jnt_dofadr[j];
      o.rest = o.type === "box" ? o.size[2] : o.type === "cylinder" ? o.size[1] : 0;
    }
    // Bolt's own bodies (the torso's tree) and their masses, for the guide's force field
    this.dogBodies = []; this.dogMass = [];
    for (let i = 1; i < m.nbody; i++) if (m.body_rootid[i] === m.body_rootid[bolt.torso]) { this.dogBodies.push(i); this.dogMass.push(m.body_mass[i]); }
    // the guide runs inside every physics step; shoves (live.js applyPush) are not used here
    const self = this, M = bolt.c.motor;
    bolt.applyPush = () => {};
    bolt.physics = function (n) {   // live.js Bolt.physics (the real-motor limits) plus the guide
      const { mj: J, model: mm, data: d } = this, fr = mm.actuator_forcerange, v = d.qvel, h = mm.opt.timestep;
      for (let s = 0; s < n; s++) {
        if (M) for (let j = 0; j < 12; j++) {
          const w = Math.abs(v[6 + j]);
          const drive = Math.min(M.torque, M.power / Math.max(w, 1e-6)) * clamp((M.cutoff - w) / (M.cutoff - M.speed), 0, 1);
          fr[2 * j] = v[6 + j] < 0 ? -drive : -M.torque; fr[2 * j + 1] = v[6 + j] >= 0 ? drive : M.torque;
        }
        self.guide(h); J.mj_step(mm, d);
      }
    };
    // the bundle's drawing list numbers Bolt's geoms in the full model; without the hurdles they come earlier
    const first = Math.min(...bundle.geoms.map((g) => g.id));
    this.geoms = bundle.geoms.map((g) => ({ ...g, id: g.id - first + m.body_geomadr[bolt.torso] }));
    this.bolt = bolt; this.bundle = bundle;
    this.gpu = new Imitate(bolt, brains.gpu, clips); this.g18 = new Imitate(bolt, brains.g18, clips);
    this.reset(seed);
  }

  reset(seed = this.seed) {
    const b = this.bolt;
    this.gpu.stop(); this.g18.stop();
    b.teleport(this.bundle.start, [0, 0], 0);
    for (const o of this.items) this.park(o);
    this.body = new DogBody(b, this.gpu, this.g18);
    this.seed = seed; this.rand = rng(seed * 7919 + 17);
    this.phase = "ready"; this.t = 0; this.runT = 0; this.events = [];
    this.lane = 0; this.ref = { y: 0, v: 0 }; this.engaged = false; this.vWant = 0; this.brakeUntil = -1;
    this.hop = null; this.hopWanted = -9; this.lastLand = -9; this.slam = false;
    this.gap = CHASE.start; this.caughtAt = null; this.lastCrash = -9; this.lastKnock = -9;
    this.dist = 0; this.rows = []; this.treats = []; this.nextX = TRACK.first; this.rowN = 0; this.lastBar = -3;
    this.stats = { treats: 0, bars: 0, barsSeen: 0, knocked: 0, crashes: 0, hops: 0, falls: 0, lanes: 0, topSpeed: 0, bestMult: 1 };
    this.points = 0; this.scoredX = 0; this.streak = 0; this.mult = 1;
    this.lay();
  }
  emit(type, extra = {}) { this.events.push({ type, t: this.runT, ...extra }); }

  // ---- the pace: speed and clock rise with the distance run
  level() { return clamp(this.dist / PACE.full, 0, 1); }
  speedWanted() { return lerp(PACE.v0, PACE.v1, this.level()); }
  clock() {   // how much faster than real time the page runs the physics
    return this.dist < PACE.full ? lerp(PACE.clock0, PACE.clock1, this.level()) : lerp(PACE.clock1, PACE.clock2, clamp((this.dist - PACE.full) / PACE.more, 0, 1));
  }
  add(points) { this.points += points * this.mult; }
  // a crash or a knocked bar ends the streak; every SCORE.streak clean rows raise the multiplier
  breakStreak() { if (this.mult > 1) this.emit("mult", { mult: 1 }); this.streak = 0; this.mult = 1; }
  rowPassed() {
    this.streak++;
    const m = Math.min(SCORE.maxMult, 1 + Math.floor(this.streak / SCORE.streak));
    if (m > this.mult) { this.mult = m; this.stats.bestMult = Math.max(this.stats.bestMult, m); this.emit("mult", { mult: m }); }
  }

  // ---- the solid things: in the physics (floating, no contacts) or parked high up out of the way
  park(o) {
    const d = this.bolt.data;
    d.qpos.set([-50 - o.i, 0, 30 + o.i, 1, 0, 0, 0], o.q); for (let k = 0; k < 6; k++) d.qvel[o.v + k] = 0;
    this.bolt.model.body_gravcomp[o.body] = 1; o.at = null; o.hit = false; o.row = null;
  }
  put(o, x, y, yaw = 0, z = null) {
    const d = this.bolt.data;
    d.qpos.set([x, y, (z ?? o.rest) + 0.002, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], o.q); for (let k = 0; k < 6; k++) d.qvel[o.v + k] = 0;
    this.bolt.model.body_gravcomp[o.body] = 1; o.at = [x, y, z ?? o.rest]; o.hit = false;
  }
  take(kind, n = 1) {
    const f = this.items.filter((o) => o.kind === kind && !o.at && !o.taken).slice(0, n);
    if (f.length < n) return null;
    for (const o of f) o.taken = true;
    return f;
  }

  // ---- the track: rows laid ahead of Bolt, one pattern each, spaced in time at the speed asked
  lay() {
    const x = this.bolt.data.qpos[0];
    while (this.nextX < x + TRACK.ahead) {
      const row = this.row(this.nextX);
      if (!row) { this.nextX += 2; continue; }   // the pool is used up: leave a gap, try again further on
      this.rows.push(row);
      const L = this.level(), more = clamp((this.dist - PACE.full) / PACE.more, 0, 1), secs = lerp(1.45, 0.85, L) - 0.15 * more + this.rand() * lerp(0.5, 0.3, L);
      this.nextX = row.end + this.speedWanted() * secs;
    }
  }
  treatLine(x, lane, n, high = false) {
    for (let k = 0; k < n; k++) {
      const u = n > 1 ? k / (n - 1) : 0.5, z = high ? 0.3 + 0.38 * Math.sin(Math.PI * u) : 0.22;
      this.treats.push({ x: x + 0.55 * k, y: lane * TRACK.lane, z, high: high && z > 0.45, got: false });
    }
  }
  row(x) {
    const R = this.rand, L = this.level(), lanes = [-1, 0, 1];
    const pick = (a) => a[Math.floor(R() * a.length)];
    const solid = () => pick(L < 0.25 ? ["crate", "crate", "barrel"] : ["crate", "barrel", "barrel", "barrier"]);
    const row = { x, end: x, kind: null, items: [], blocked: [], n: this.rowN++ };
    const place = (kind, lane, dx = 0) => {
      const o = this.take(kind) || this.take("crate");
      if (!o) return false;
      this.put(o[0], x + dx, lane * TRACK.lane + (R() - 0.5) * 0.06, kind === "barrier" ? 0 : (R() - 0.5) * 0.5);
      o[0].taken = false; o[0].row = row; row.items.push(o[0]); row.end = Math.max(row.end, x + dx);
      return true;
    };
    const fail = () => { for (const o of row.items) this.park(o); return null; };
    const free = (want) => this.items.filter((o) => !o.at && (o.kind === want || o.kind === "crate")).length;
    // a bar across all three lanes: every few rows once past the first ones; 16 cm at the start up to 24 cm
    const since = row.n - this.lastBar;
    if (row.n >= 2 && since >= 3 && R() < 0.3 + 0.2 * L + (since >= 6 ? 1 : 0)) {
      const o = this.take("bar");
      if (o) {
        const h = Math.round(100 * (0.16 + 0.08 * L * R() + 0.02 * L)) / 100;
        this.put(o[0], x, 0, 0, h); o[0].taken = false; o[0].row = row; o[0].h = h;
        Object.assign(row, { kind: "bar", items: [o[0]], h, end: x + 1.2 });
        this.lastBar = row.n; this.stats.barsSeen++;
        if (R() < 0.6) this.treatLine(x - 0.9, pick(lanes), 4, true);   // an arc of treats over the bar: only a hop gets them
        // later on a bar comes with a crate after it in one lane: land, then dodge (or steer in the air)
        if (L > 0.3 && R() < 0.5) { const l = pick(lanes); if (place(solid(), l, 2.2 + 0.4 * R())) { row.blocked2 = [l]; row.at2 = x + 2.2; row.end += 1.6; } }
        return row;
      }
    }
    const r = R();
    if (r < 0.35 - 0.15 * L) {   // one thing in one lane
      const l = pick(lanes);
      if (!place(solid(), l)) return fail();
      Object.assign(row, { kind: "one", blocked: [l] });
      if (R() < 0.4) this.treatLine(x - 1.2, pick(lanes.filter((k) => k !== l)), 3);
    } else if (L > 0.45 && r < 0.55) {   // a zigzag: two gates in quick succession, their open lanes apart
      const a = pick(lanes), c = pick(lanes.filter((k) => k !== a)), dx = this.speedWanted() * lerp(0.9, 0.7, L);
      for (const l of lanes) if (l !== a && !place(solid(), l)) return fail();
      for (const l of lanes) if (l !== c && !place(solid(), l, dx)) return fail();
      Object.assign(row, { kind: "zig", blocked: lanes.filter((l) => l !== a), blocked2: lanes.filter((l) => l !== c), at2: x + dx, open: a });
      if (R() < 0.5) this.treatLine(x + dx - 0.8, c, 3);
    } else if (r < 0.8 || L < 0.2 || free("barrel") < 3) {   // two lanes blocked: one way through, with treats in it
      const open = pick(lanes), shut = lanes.filter((l) => l !== open);
      if (!place(solid(), shut[0]) || !place(solid(), shut[1], (R() - 0.5) * 0.4)) return fail();
      Object.assign(row, { kind: "two", blocked: shut, open });
      if (R() < 0.55) this.treatLine(x - 1.6, open, 4);
    } else {   // a lane lined with barrels (stay out of it), and a crate in another a little further on
      const l = pick(lanes), other = pick(lanes.filter((k) => k !== l));
      for (let k = 0; k < 3; k++) if (!place("barrel", l, 0.75 * k)) return fail();
      if (!place(solid(), other, 2.4)) return fail();
      const open = lanes.find((k) => k !== l && k !== other);
      Object.assign(row, { kind: "line", blocked: [l], blocked2: [l, other], at2: x + 2.4, open });
      if (R() < 0.6) this.treatLine(x, open, 4);
    }
    return row;
  }
  recycle() {
    const x = this.bolt.data.qpos[0];
    while (this.rows.length && this.rows[0].end + 2 < x - TRACK.behind) for (const o of this.rows.shift().items) this.park(o);
    // anything knocked far away (or left behind by a row) goes back to the pool
    for (const o of this.items) if (o.at && !o.row && this.bolt.data.qpos[o.q] < x - TRACK.behind) this.park(o);
    while (this.treats.length && this.treats[0].x < x - TRACK.behind) this.treats.shift();
  }

  // ---- what the player does: change lane, hop, drop
  start() { if (this.phase === "ready") { this.phase = "run"; this.emit("go"); } }
  steer(dir) {   // +1 left, -1 right (+y is left)
    if (this.phase !== "run") return false;
    const l = clamp(this.lane + dir, -1, 1);
    if (l === this.lane) { this.emit("edge"); return false; }
    this.lane = l; this.stats.lanes++; return true;
  }
  left() { return this.steer(1); }
  right() { return this.steer(-1); }
  canHop() { return this.phase === "run" && this.engaged && !this.hop && this.t - this.lastLand > 0.06; }
  jump() {
    if (this.phase !== "run") return false;
    if (!this.canHop()) { this.hopWanted = this.t; return false; }   // pressed a moment early: hop on landing
    const d = this.bolt.data;
    this.hop = { t: this.t, z0: d.qpos[2] }; this.slam = false; this.hopWanted = -9; this.stats.hops++;
    d.qvel[2] += ASSIST.hop;   // the kick is on the root joint: all of Bolt rises together, the legs keep their motion this.emit("hop");
    return true;
  }
  down() { if (this.hop) { this.slam = true; return true; } return false; }

  // ---- the guide (see the top): forces on Bolt's bodies for one physics step of h seconds
  guide(h) {
    const b = this.bolt, d = b.data, f = d.xfrc_applied, q = d.qpos, v = d.qvel, A = ASSIST;
    for (const i of this.dogBodies) f.fill(0, 6 * i, 6 * i + 6);
    if (!this.engaged) return;
    const ty = this.lane * TRACK.lane, r = this.ref;
    const a = clamp(A.w * A.w * (ty - r.y) - 2 * A.w * r.v, -A.amax, A.amax);
    r.v += a * h; r.y += r.v * h;
    const ay = a + A.kp * (r.y - q[1]) + A.kd * (r.v - v[1]);
    const ax = this.phase === "run" && ["trot", "ctrot", "canter"].includes(this.body.mode) ? clamp(A.fwd * (this.vWant - v[0]), -A.fwdMax, A.fwdMax) : 0;
    const az = this.hop ? 9.81 * (this.slam ? -A.slam : A.float) : 0;
    for (let k = 0; k < this.dogBodies.length; k++) {
      const j = 6 * this.dogBodies[k], mk = this.dogMass[k];
      f[j] = mk * ax; f[j + 1] = mk * ay; f[j + 2] = mk * az;
    }
    const R = d.xmat, t = 9 * b.torso, yaw = Math.atan2(R[t + 3], R[t]);
    const want = Math.atan2(r.v, Math.max(1, v[0])) * A.lean, e = Math.atan2(Math.sin(want - yaw), Math.cos(want - yaw));
    const j = 6 * b.torso;
    f[j + 3] = A.up * R[t + 5] - A.upD * v[3];
    f[j + 4] = -A.up * R[t + 2] - A.upD * v[4];
    f[j + 5] = A.yaw * e - A.yawD * v[5];
  }

  // ---- one control step (1/50 s of simulated time)
  step() {
    const b = this.bolt, B = this.body, dt = b.dt, d = b.data;
    this.t += dt;
    if (this.phase === "run") this.runT += dt;
    // the gait the brains run: trot, the canter brain once the pace is high; standing before the start
    const vs = this.speedWanted();
    if (this.phase === "run") { B.want.gait = vs > PACE.canterAt ? "canter" : "trot"; B.want.steer = 0; }
    else { B.want.gait = this.phase === "ready" ? "stand" : "trot"; B.want.steer = 0; }
    this.vWant = this.t < this.brakeUntil ? 0.45 * vs : vs;
    const on = RUNNING.includes(B.mode);
    if (on && !this.engaged) { this.ref.y = d.qpos[1]; this.ref.v = d.qvel[1]; this.lane = clamp(Math.round(d.qpos[1] / TRACK.lane), -1, 1); }
    this.engaged = on;
    if (!on) this.hop = null;
    if (this.t - this.hopWanted < 0.2 && this.canHop()) this.jump();
    const falls0 = B.falls;
    B.step(dt);
    for (const e of B.events.splice(0)) if (/got itself up/.test(e.text)) this.emit("up");
    if (B.falls > falls0) { this.stats.falls++; this.emit("fall"); }
    // the hop ends when Bolt is back down at the height it left from (or after 1.2 s)
    if (this.hop && this.t - this.hop.t > 0.12 && ((d.qvel[2] < 0 && d.qpos[2] < this.hop.z0 + 0.02) || this.t - this.hop.t > 1.2)) { this.hop = null; this.lastLand = this.t; this.emit("land"); }
    const x = d.qpos[0], y = d.qpos[1], z = d.qpos[2];
    this.dist = Math.max(this.dist, x);
    if (this.phase === "run" && x > this.scoredX) { if (this.scoredX > 0) this.add(SCORE.metre * (x - this.scoredX)); this.scoredX = x; }
    if (this.phase === "run" && RUNNING.includes(B.mode)) this.stats.topSpeed = Math.max(this.stats.topSpeed, Math.min(B.speed, d.qvel[0]));
    // hits: an obstacle that moved has been touched; gravity back on, and a stumble if Bolt is the one there
    for (const o of this.items) {
      if (!o.at || o.hit) continue;
      const vv = Math.hypot(d.qvel[o.v], d.qvel[o.v + 1], d.qvel[o.v + 2], 0.2 * d.qvel[o.v + 3], 0.2 * d.qvel[o.v + 4], 0.2 * d.qvel[o.v + 5]);
      if (vv < 0.03) continue;
      o.hit = true; b.model.body_gravcomp[o.body] = 0;
      const near = Math.abs(d.qpos[o.q] - x) < 1.2 && Math.abs(d.qpos[o.q + 1] - y) < 0.9;
      if (!near || this.phase !== "run") continue;
      if (o.kind === "bar") { this.stats.knocked++; this.lastKnock = this.t; this.gap -= CHASE.knock; this.breakStreak(); if (o.row) o.row.spoiled = true; this.emit("knock", { h: o.h }); }
      else if (this.t - this.lastCrash > 0.6) { this.lastCrash = this.t; this.stats.crashes++; this.gap -= CHASE.crash; this.brakeUntil = this.t + 0.5; this.breakStreak(); if (o.row) o.row.spoiled = true; this.emit("crash", { kind: o.kind }); }
    }
    // bars passed with the bar still up: clean
    for (const row of this.rows) if (!row.passed && x > row.end + 0.6 && this.phase === "run") {
      row.passed = true;
      if (row.kind === "bar" && !row.items[0].hit) { this.stats.bars++; this.add(SCORE.bar); this.emit("clean", { h: row.h }); }
      if (!row.spoiled) this.rowPassed();
    }
    // treats: picked up passing within reach (a high one only in the air)
    for (const tr of this.treats) if (!tr.got && Math.abs(tr.x - x - 0.15) < 0.32 && Math.abs(tr.y - y) < 0.36 && (!tr.high || z > 0.5) && this.phase === "run") {
      tr.got = true; this.stats.treats++; this.add(SCORE.treat); this.emit("treat");
    }
    this.recycle(); this.lay();
    // the dog-catcher: falls back toward CHASE.max m while Bolt keeps the pace, closes in while Bolt is slower
    if (this.phase === "run") {
      const lag = Math.max(0, vs - CHASE.slack - B.speed), warm = Math.min(1, this.runT / 3), regen = lerp(CHASE.regen[0], CHASE.regen[1], this.level());
      this.gap += dt * warm * (regen * (this.gap < CHASE.max && this.t - Math.max(this.lastCrash, this.lastKnock) > CHASE.hold ? 1 : 0) - CHASE.lag * lag * (B.mode === "fallen" || B.speed < 0.6 * vs ? 1.6 : 1));
      this.gap = Math.min(this.gap, CHASE.max);
      if (this.gap <= CHASE.caught) { this.phase = "caught"; this.caughtAt = this.runT; this.emit("caught"); }
    }
  }
  score() { return Math.floor(this.points); }
  // the rows still ahead of Bolt (what a player sees): for the autoplayers and the page
  ahead(range = 14) {
    const x = this.bolt.data.qpos[0];
    return this.rows.filter((r) => r.end + 0.5 > x && r.x < x + range);
  }
  status() {
    const B = this.body, d = this.bolt.data;
    return { phase: this.phase, t: this.runT, dist: this.dist, score: this.score(), speed: B.speed, gap: this.gap, mode: B.mode, lane: this.lane,
      air: !!this.hop, mult: this.mult, level: this.level(), clock: this.clock(), stats: this.stats, x: d.qpos[0], y: d.qpos[1], z: d.qpos[2] };
  }
}
