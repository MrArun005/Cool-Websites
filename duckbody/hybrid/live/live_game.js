// The match itself for the LIVE page: match.py's Team / Player / Referee / Game (GameController) on top of
// live_sim.js, with the team decisions from tactics.js (the port of tactics.py). No DOM.
//
// What differs from match.py (the page says so too):
//   - the head cameras are not rendered: a player whose tactics turn its camera on "sees" the ball when it is inside
//     the camera's field of view (the real camera pose from the head joints, 49 deg x 62 deg, 160 x 120), not hidden
//     behind another duck's trunk or head, and big enough to be a blob (area >= 6 px, i.e. < ~2.9 m); the measured
//     position is the truth plus noise growing with distance (1 cm at the feet, ~3 cm at 1.5 m), at 10 Hz. Everything
//     after that (to field coordinates with the duck's own pose estimate, team sharing over "wifi", misses) is
//     match.py's.
//   - ball touches are read from the contacts only while a duck is within 0.35 m of the ball; a leg touch (the ball
//     memory's "touch" fix, match.Game.feel) is any contact of the ball with a foot / shin / thigh (match.py also
//     requires 0.3 N of contact force).
//   - self-occlusion as in match.py: the duck's own parts are the boxes match.py's cameras render (meta.self_boxes, one
//     per body); the ball counts as seen when its centre or one of 4 points on its rim (0.7 radius, across the view)
//     is in the picture and the ray to it misses the own boxes and the other ducks' trunk / head spheres.
//   - the chest camera (match.py CHEST, follow.CHEST_*): a second camera at the front bottom of the trunk, 120 deg wide,
//     looking forward and down; same visibility model. Its fix is used (and wins) when the ball is closer than 0.30 m.
import { World, Runtime, brainsTick, Rng, DT, HZ, clip } from './live_sim.js';
import { TeamTactics, lookAt, wrap, unit, CHEST_BLIND_X } from './tactics.js';

export const HALF_S = 180.0;
const CAM_EVERY = 5, WIFI_DELAY = 0.2, WIFI_NOISE = 0.02, WIFI_EVERY = 0.2;
const LOC_SIGMA = 0.05, LOC_YAW_SIGMA = 0.05, LOC_TAU = 2.0, OPP_NOISE = 0.06, OPP_TAU = 1.0;
const READY_S = 14.0, SET_S = 1.0, GOAL_S = 3.0, FREEKICK_S = 10.0, FREEKICK_D = 0.5, KICKOFF_S = 10.0, NO_PROGRESS_S = 10.0;
const PICKUP_DOWN_S = 12.0, PICKUP_OUT_S = 8.0, PICKUP_S = 10.0, OUT_WHISTLE_S = 0.8;
const ROLES5 = ['GK', 'D1', 'D2', 'A1', 'A2'], ROLES3 = ['GK', 'D1', 'A1'];
const KICKOFF_SPOTS = {
  true: { GK: [-2.25 + 0.3, 0.0], D1: [-1.3, 0.6], D2: [-1.3, -0.6], A1: [-0.22, 0.03], A2: [-0.7, -0.75] },
  false: { GK: [-2.25 + 0.3, 0.0], D1: [-1.3, 0.6], D2: [-1.3, -0.6], A1: [-0.6, 0.0], A2: [-0.75, 0.75] },
};
const CAM_H = 0.21, CAM_DOWN = 0.17;
// ball at the feet (match.py): ball memory, leg touch
const MEM_S = 3.0, MEM_NEAR = 0.4, MEM_TAU = 0.5, SELF_BLIND_X = 0.05, SELF_BLIND_Y = 0.10, TOUCH_SIGMA = 0.02;
const norm2 = (x, y) => Math.hypot(x, y);
const CHEST_RANGE = 0.30;          // chest camera fixes are used (and win) up to this far (match.CHEST_RANGE)

class Team {
  constructor(name, dir) { this.name = name; this.dir = dir; this.msgs = []; this.ball = null; this.ball_t = -99; this.ball_from = null; this.score = 0; this.players = []; }
  send(t, sender, ball, ball_t, rng) { if (ball) this.msgs.push([t + WIFI_DELAY, sender, [ball[0] + rng.normal(0, WIFI_NOISE), ball[1] + rng.normal(0, WIFI_NOISE)], ball_t]); }
  deliver(t) {
    const keep = [];
    for (const m of this.msgs) { if (m[0] <= t) { if (m[3] > this.ball_t) { this.ball = m[2]; this.ball_t = m[3]; this.ball_from = m[1]; } } else keep.push(m); }
    this.msgs = keep;
  }
  announce(ball, t) {
    this.msgs = []; this.ball = [ball[0], ball[1]]; this.ball_t = t; this.ball_from = null;
    for (const p of this.players) { p.ball_f = [ball[0], ball[1]]; p.ball_t = t; p.misses = 0; p.ball_v = [0, 0]; p.fix_rel = null; p.touch_t = -99; p.mem_misses = 0; }
  }
}

class Player {
  constructor(game, duck, rt, team, role, number) {
    Object.assign(this, { g: game, duck, rt, team, role, num: number, name: duck.name });
    this.rng = new Rng(1000 * game.seed + duck.i + 11);
    this.err = [this.rng.normal(0, LOC_SIGMA), this.rng.normal(0, LOC_SIGMA), this.rng.normal(0, LOC_YAW_SIGMA)];
    this.ball_f = null; this.ball_t = -99; this.misses = 0;
    this.ball_v = [0, 0]; this.fix_rel = null; this.mem_misses = 0; this.touch_f = null; this.touch_t = -99;
    this.state = 'search'; this.cmd = null; this.cam_on = false; this.penalized_until = null; this.last_sent = -9;
    this.down_since = null; this.out_since = null; this.pos = [0, 0]; this.psi = 0;
    this.stats = { kicks: 0, connected: 0, shots: 0, passes: 0, passes_completed: 0, clearances: 0, goals: 0, own_goals: 0, touches: 0, pickups: 0, frames_seen: 0, frames: 0, dist: 0, moving: 0, samples: 0 };
    this.hist = [];
  }
  active() { return this.penalized_until === null; }
  update_pose() {
    const a = DT / LOC_TAU, s = Math.sqrt(2 * a), sig = [LOC_SIGMA, LOC_SIGMA, LOC_YAW_SIGMA];
    for (let i = 0; i < 3; i++) this.err[i] += -a * this.err[i] + s * this.rng.normal() * sig[i];
    const [p, yaw] = this.duck.pose();
    this.pos = [p[0] + this.err[0], p[1] + this.err[1]]; this.psi = wrap(yaw + this.err[2]);
  }
  to_field(rel) { const c = Math.cos(this.psi), s = Math.sin(this.psi); return [this.pos[0] + c * rel[0] - s * rel[1], this.pos[1] + s * rel[0] + c * rel[1]]; }
  to_body(p) { const dx = p[0] - this.pos[0], dy = p[1] - this.pos[1], c = Math.cos(this.psi), s = Math.sin(this.psi); return [c * dx + s * dy, -s * dx + c * dy]; }
  // camera projection of a world point: [u, v, depth] (pixels from the image centre), or null when behind
  project(p, ci = this.duck.meta.cam, FY = this.g.FY) {
    const d = this.g.world.d, cp = d.cam_xpos, cm = d.cam_xmat;
    const e = [p[0] - cp[3 * ci], p[1] - cp[3 * ci + 1], p[2] - cp[3 * ci + 2]], R = cm.subarray(9 * ci, 9 * ci + 9);
    const pc = [R[0] * e[0] + R[3] * e[1] + R[6] * e[2], R[1] * e[0] + R[4] * e[1] + R[7] * e[2], R[2] * e[0] + R[5] * e[1] + R[8] * e[2]];
    if (pc[2] > -0.03) return null;
    return [-pc[0] / pc[2] * FY, pc[1] / pc[2] * FY, Math.hypot(pc[0], pc[1], pc[2]), [cp[3 * ci], cp[3 * ci + 1], cp[3 * ci + 2]]];
  }
  expect_visible(ci = this.duck.meta.cam, FY = this.g.FY) {
    if (this.ball_f === null) return false;
    const rel = this.to_body(this.ball_f), [tp, yaw] = this.duck.pose(), c = Math.cos(yaw), s = Math.sin(yaw);
    const pr = this.project([tp[0] + c * rel[0] - s * rel[1], tp[1] + s * rel[0] + c * rel[1], this.g.BALL_R], ci, FY);
    return pr !== null && Math.abs(pr[0]) < 80 - 6 && Math.abs(pr[1]) < 60 - 6;
  }
  // does camera ci see the ball? A blob of >= 6 px whose centre or a rim point (0.7 r, across the view) is in the
  // picture with a clear line of sight (own body boxes, other ducks)
  sees(ci, FY, b) {
    const pr = this.project(b, ci, FY);
    if (pr === null) return false;
    const [u, v, depth, cpos] = pr, rpx = FY * this.g.BALL_R / depth;
    if (!(Math.abs(u) < 80 + rpx * 0.5 && Math.abs(v) < 60 + rpx * 0.5 && Math.PI * rpx * rpx >= 6)) return false;
    const R = this.g.world.d.cam_xmat.subarray(9 * ci, 9 * ci + 9), r = 0.7 * this.g.BALL_R;
    for (const [a, c] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const q = [b[0] + r * (a * R[0] + c * R[1]), b[1] + r * (a * R[3] + c * R[4]), b[2] + r * (a * R[6] + c * R[7])];
      const pq = this.project(q, ci, FY);
      if (pq === null || Math.abs(pq[0]) > 80 || Math.abs(pq[1]) > 60) continue;
      if (!this.g.selfBlocked(cpos, q, this.duck) && !this.g.occluded(cpos, q, this.duck.i)) return true;
    }
    return false;
  }
  see(t) {
    this.stats.frames++;
    const [b] = this.g.world.ball(), m = this.duck.meta;
    let det = this.sees(m.cam, this.g.FY, b);
    if (this.g.chest) {
      const dc = this.sees(m.chest, this.g.CFY, b), [tp] = this.duck.pose();
      this.stats.chest_frames = (this.stats.chest_frames || 0) + 1;
      if (dc && Math.hypot(b[0] - tp[0], b[1] - tp[1]) < CHEST_RANGE) { this.stats.chest_seen = (this.stats.chest_seen || 0) + 1; det = true; }
    }
    if (det) {
      const [tp, yaw] = this.duck.pose(), dx = b[0] - tp[0], dy = b[1] - tp[1], c = Math.cos(yaw), s = Math.sin(yaw);
      const dist = Math.hypot(dx, dy), sig = 0.008 + 0.015 * dist;
      const g = [c * dx + s * dy + this.rng.normal(0, sig), -s * dx + c * dy + this.rng.normal(0, sig)];
      if (Math.hypot(g[0], g[1]) < 3.0) {
        const f = this.to_field(g), age = t - this.ball_t;
        if (this.ball_f !== null && age > 0.05 && age < 0.5) {
          let v = [(f[0] - this.ball_f[0]) / age, (f[1] - this.ball_f[1]) / age];
          const k = Math.min(1.0, 2.0 / Math.max(1e-6, Math.hypot(v[0], v[1]))); v = [v[0] * k, v[1] * k];
          this.ball_v = [0.5 * this.ball_v[0] + 0.5 * v[0], 0.5 * this.ball_v[1] + 0.5 * v[1]];
        } else this.ball_v = [0, 0];
        this.ball_f = f; this.ball_t = t; this.misses = 0; this.mem_misses = 0; this.fix_rel = g; this.stats.frames_seen++;
        return true;
      }
    }
    const chest = this.g.chest;
    if (this.expect_visible() || (chest && this.expect_visible(this.duck.meta.chest, this.g.CFY))) {
      this.misses++;
      if (this.ball_f !== null) { const r = this.to_body(this.mem_pos(t)); if (!(r[0] < (chest ? CHEST_BLIND_X : SELF_BLIND_X) && Math.abs(r[1]) < SELF_BLIND_Y)) this.mem_misses++; }
    }
    return false;
  }
  mem_pos(t) {
    const k = MEM_TAU * (1 - Math.exp(-(t - this.ball_t) / MEM_TAU));
    return [this.ball_f[0] + this.ball_v[0] * k, this.ball_f[1] + this.ball_v[1] * k];
  }
  mem_ok(t) { return this.ball_f !== null && this.fix_rel !== null && t - this.ball_t < MEM_S && Math.hypot(this.fix_rel[0], this.fix_rel[1]) < MEM_NEAR && this.mem_misses < 3; }
  ball_sig(t, src) {
    if (src === 'own') return 0.01 + 0.02 * Math.max(0, t - this.ball_t);
    if (src === 'mem') return 0.02 + 0.12 * (t - this.ball_t);
    if (src === 'touch') return 0.04;
    if (src === 'team') return WIFI_NOISE + 0.1 * Math.max(0, t - this.team.ball_t);
    return 9.0;
  }
  ball(t) {
    if (this.ball_f !== null && t - this.ball_t < 0.7 && this.misses < 4) return [this.ball_f, 'own'];
    const tm = this.team;
    if (this.touch_f !== null && t - this.touch_t < 0.6 && this.touch_t > this.ball_t) return [this.touch_f, 'touch'];
    if (tm.ball !== null && t - tm.ball_t < 1.0 && tm.ball_t > this.ball_t && tm.ball_from !== this.duck.i) return [tm.ball, 'team'];
    if (this.mem_ok(t)) return [this.mem_pos(t), 'mem'];
    if (tm.ball !== null && t - tm.ball_t < 4.0) return [tm.ball, 'team'];
    if (this.ball_f !== null && t - this.ball_t < 4.0 && this.misses < 4) return [this.ball_f, 'own'];
    return [null, null];
  }
  share(t) {
    if (this.ball_f !== null && this.ball_t > this.last_sent && t - this.last_sent >= WIFI_EVERY && t - this.ball_t < 0.3) {
      this.team.send(t, this.duck.i, this.ball_f, this.ball_t, this.rng); this.last_sent = t;
    }
  }
}

class Referee {
  constructor(game, duck, rt) { this.g = game; this.duck = duck; this.rt = rt; this.side = -1.0; }
  act() {
    const rt = this.rt, P = this.g.P;
    if (rt.mode === 'get up' || rt.lying() || rt.mode === 'hop' || rt.mode === 'settle') return [0, 0, 0];
    const [bb] = this.g.world.ball(), b = [bb[0], bb[1]];
    const [p, yaw] = this.duck.pose();
    if (b[1] < -P.hy + 0.9) this.side = 1.0; else if (b[1] > P.hy - 0.9) this.side = -1.0;
    let target = [clip(b[0] - 0.15 * Math.sign(b[0]), -P.hx - 0.1, P.hx + 0.1), clip(b[1] + 0.8 * this.side, -P.hy - 0.35, P.hy + 0.35)];
    for (const dk of this.g.world.ducks) {
      if (dk === this.duck) continue;
      const [q] = dk.pose(), v = [target[0] - q[0], target[1] - q[1]], dd = Math.hypot(v[0], v[1]);
      if (dd < 0.4) { const u = unit(v); target = [target[0] + u[0] * (0.4 - dd), target[1] + u[1] * (0.4 - dd)]; }
    }
    const e = [target[0] - p[0], target[1] - p[1]], c = Math.cos(yaw), s = Math.sin(yaw);
    const eb = [c * e[0] + s * e[1], -s * e[0] + c * e[1]], d = Math.hypot(eb[0], eb[1]), bear = Math.atan2(eb[1], eb[0]);
    const rb = [b[0] - p[0], b[1] - p[1]], rbb = [c * rb[0] + s * rb[1], -s * rb[0] + c * rb[1]];
    const bbr = Math.atan2(rbb[1], rbb[0]), dist = Math.hypot(rbb[0], rbb[1]);
    rt.look(clip(bbr, -1.2, 1.2), Math.atan2(CAM_H - this.g.BALL_R, Math.max(dist, 0.1)) - CAM_DOWN);
    if (d < 0.15) return [0, 0, Math.abs(bbr) < 0.3 ? 0 : Math.sign(bbr) * 0.6];
    let wz = clip(1.6 * bear, -1, 1);
    if (Math.abs(bear) > 0.12 && Math.abs(wz) < 0.6) wz = Math.sign(bear) * 0.6;
    const vx = Math.abs(bear) < 0.5 ? clip(0.3 + 1.2 * d, 0.32, 1.1) : 0;
    return [vx, 0, wz];
  }
}

export class Game {
  // runBrains(name, X Float32Array N*61, N) -> Float32Array N*14 (onnx_mlp.js MlpPolicy.runBatch)
  constructor(mj, xml, meta, moves, runBrains, { seed = 0, halfS = HALF_S, halves = 2, chest = true } = {}) {
    this.seed = seed; this.halfS = halfS; this.halves = halves; this.runBrains = runBrains;
    this.world = new World(mj, xml, meta, { seed, randomize: true });
    this.mj = mj; this.P = meta.pitch; this.BALL_R = meta.pitch.ball_r;
    this.FY = (meta.eye.h / 2) / Math.tan(meta.eye.fovy * Math.PI / 360);
    this.chest = chest && !!meta.chest;                 // the chest camera (needs a model exported with it)
    this.CFY = meta.chest ? (meta.chest.h / 2) / Math.tan(meta.chest.fovy * Math.PI / 360) : 0;
    this.selfBoxes = {};                                // per duck: its own parts as boxes [body, centre, half] (self-occlusion)
    for (const r of meta.robots) this.selfBoxes[r.name] = (meta.self_boxes || []).map(([bn, c, h]) => [r.bodies[r.body_names.indexOf(bn)], c, h]).filter(x => x[0] !== undefined);
    this.rng = new Rng(5000 + seed);
    this.teams = { blue: new Team('blue', 1.0), yellow: new Team('yellow', -1.0) };
    this.players = []; this.rts = [];
    const nPer = meta.robots.filter(r => r.team === 'blue').length, roles = nPer === 3 ? ROLES3 : ROLES5;
    for (const dk of this.world.ducks) {
      const rt = new Runtime(dk, moves, 1000 * seed + dk.i);
      this.rts.push(rt);
      if (dk.team === 'ref') { this.ref = new Referee(this, dk, rt); continue; }
      const tm = this.teams[dk.team], num = +dk.name.slice(1);
      const p = new Player(this, dk, rt, tm, roles[num - 1], num);
      tm.players.push(p); this.players.push(p);
    }
    this.opp_err = {};
    const trng = new Rng(seed * 10 + 1);
    this.tactics = { blue: new TeamTactics(new Rng(seed * 10 + 1), r => new Rng(Math.floor(r.next() * 4294967296))),
      yellow: new TeamTactics(new Rng(seed * 10 + 2), r => new Rng(Math.floor(r.next() * 4294967296))) };
    void trng;
    this.t = 0; this.k = 0; this.half = 1; this.clock = 0; this.phase = 'set'; this.phase_t = 0;
    this.kickoff_team = seed % 2 === 0 ? 'blue' : 'yellow'; this.first_kickoff = this.kickoff_team;
    this.setpiece = null; this.out_pending = null; this.last_touch = null; this.prog = null;
    this.kicks = []; this.events = [];
    this.scratch = new Float32Array(61 * this.rts.length);
    this.brainMs = 0;
    this.place_kickoff(true);
  }

  ev(kind, team = null, text = '', extra = {}) {
    const e = Object.assign({ t: +this.t.toFixed(2), half: this.half, clock: +this.clock.toFixed(1), kind, team, text }, extra);
    this.events.push(e);
    if (this.onEvent) this.onEvent(e);
  }
  other(team) { return team === 'blue' ? 'yellow' : 'blue'; }
  by_index(i) { return this.players.find(p => p.duck.i === i) || null; }
  spot(team, role, kicking) { const s = KICKOFF_SPOTS[kicking][role]; return [s[0] * team.dir, s[1] * team.dir]; }
  selfBlocked(c, q, duck) {         // the segment camera c -> point q crosses one of the duck's own boxes (not one c is in)
    const d = this.world.d, X = d.xpos, M = d.xmat;
    for (const [bi, bc, h] of this.selfBoxes[duck.name] || []) {
      const R = M.subarray(9 * bi, 9 * bi + 9), o = [X[3 * bi] + R[0] * bc[0] + R[1] * bc[1] + R[2] * bc[2],
        X[3 * bi + 1] + R[3] * bc[0] + R[4] * bc[1] + R[5] * bc[2], X[3 * bi + 2] + R[6] * bc[0] + R[7] * bc[1] + R[8] * bc[2]];
      const loc = p => { const e = [p[0] - o[0], p[1] - o[1], p[2] - o[2]]; return [R[0] * e[0] + R[3] * e[1] + R[6] * e[2], R[1] * e[0] + R[4] * e[1] + R[7] * e[2], R[2] * e[0] + R[5] * e[1] + R[8] * e[2]]; };
      const a = loc(c), bb = loc(q);
      if (Math.abs(a[0]) <= h[0] && Math.abs(a[1]) <= h[1] && Math.abs(a[2]) <= h[2]) continue;     // inside: culled
      let t0 = 0, t1 = 1, hit = true;
      for (let k = 0; k < 3 && hit; k++) {
        const dd = bb[k] - a[k];
        if (Math.abs(dd) < 1e-12) { if (Math.abs(a[k]) > h[k]) hit = false; continue; }
        let ta = (-h[k] - a[k]) / dd, tb = (h[k] - a[k]) / dd;
        if (ta > tb) [ta, tb] = [tb, ta];
        t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
        if (t0 > t1) hit = false;
      }
      if (hit) return true;
    }
    return false;
  }
  occluded(c, b, self) {          // the ray camera -> ball passes through another duck's trunk or head (6 cm spheres)
    const d = this.world.d, x = d.xpos, ab = [b[0] - c[0], b[1] - c[1], b[2] - c[2]], L2 = ab[0] ** 2 + ab[1] ** 2 + ab[2] ** 2;
    for (const dk of this.world.ducks) {
      if (dk.i === self) continue;
      for (const bi of [dk.meta.trunk, dk.meta.jaw]) {
        const p = [x[3 * bi] - c[0], x[3 * bi + 1] - c[1], x[3 * bi + 2] - c[2]];
        const s = (p[0] * ab[0] + p[1] * ab[1] + p[2] * ab[2]) / L2;
        if (s <= 0.02 || s >= 0.98) continue;
        const q = [p[0] - s * ab[0], p[1] - s * ab[1], p[2] - s * ab[2]];
        if (q[0] ** 2 + q[1] ** 2 + q[2] ** 2 < 0.06 ** 2) return true;
      }
    }
    return false;
  }

  // ---------------------------------------------------------------- touches, kicks
  feel() {                         // leg touch (match.Game.feel): the touched link's position in the trunk frame + noise
    const w = this.world, x = w.d.xpos;
    for (const [i, b] of w.legTouch) {
      const p = this.by_index(i);
      if (!p) continue;
      const tr = p.duck.meta.trunk, e = [x[3 * b] - x[3 * tr], x[3 * b + 1] - x[3 * tr + 1]], [, yaw] = p.duck.pose(), c = Math.cos(yaw), s = Math.sin(yaw);
      const rel = [c * e[0] + s * e[1] + p.rng.normal(0, TOUCH_SIGMA), -s * e[0] + c * e[1] + p.rng.normal(0, TOUCH_SIGMA)];
      p.touch_f = p.to_field(rel); p.touch_t = this.t;
    }
    w.legTouch.clear();
  }
  process_touches() {
    const tn = Array.from(this.world.touchNow).sort((a, b) => a - b);
    for (const i of tn) {
      const dk = this.world.ducks[i];
      if (dk.team === 'ref') continue;
      const p = this.by_index(i);
      const isNew = this.last_touch === null || this.last_touch[0] !== i || this.t - this.last_touch[2] > 0.5;
      if (isNew) {
        p.stats.touches++;
        for (let k = this.kicks.length - 1; k >= 0; k--) {
          const kc = this.kicks[k];
          if (kc.kind === 'pass' && kc.result == null && kc.who !== i && this.t - kc.t < 6.0) {
            kc.result = dk.team === kc.team ? 'completed' : 'intercepted';
            if (kc.result === 'completed') this.by_index(kc.who).stats.passes_completed++;
            break;
          }
        }
      }
      this.last_touch = [i, dk.team, this.t];
      if (this.setpiece && dk.team === this.setpiece.team) this.setpiece.touched = true;
    }
    this.world.touchNow.clear();
  }
  kick(p, t, foot, kind, mate) {
    const [b] = this.world.ball();
    kind = kind || 'forward';
    const mp = mate ? this.players.find(q => q.name === mate) : null;
    this.kicks.push({ t: +t.toFixed(2), who: p.duck.i, name: p.name, team: p.team.name, foot, kind, mate: mp ? mp.duck.i : null, b0: [b[0], b[1]], vmax: 0, connected: false });
    p.stats.kicks++;
    if (kind === 'shot') p.stats.shots++; else if (kind === 'pass') p.stats.passes++; else if (kind === 'clearance') p.stats.clearances++;
  }

  // ---------------------------------------------------------------- set-ups
  place_kickoff(teleport) {
    const w = this.world;
    w.setBall(0, 0);
    for (const tm of Object.values(this.teams)) tm.announce([0, 0], this.t);
    if (teleport) {
      for (const p of this.players) {
        const s = this.spot(p.team, p.role, p.team.name === this.kickoff_team);
        w.put(p.duck, s[0], s[1], p.team.dir > 0 ? 0 : Math.PI); this.reset_rt(p.rt);
      }
      w.put(this.ref.duck, 0.3, -this.P.hy - 0.25, Math.PI / 2); this.reset_rt(this.ref.rt);
    }
    this.mj.mj_forward(w.m, w.d);
    for (const p of this.players) p.update_pose();
  }
  reset_rt(rt) { rt.reset(); }
  manual_placement() {
    const w = this.world, moved = [];
    for (const p of this.players) {
      if (!p.active()) continue;
      const s = this.spot(p.team, p.role, p.team.name === this.kickoff_team), [q] = p.duck.pose();
      if (Math.hypot(q[0] - s[0], q[1] - s[1]) > 0.35 || p.rt.lying() || p.duck.tiltDeg() > 25) {
        w.put(p.duck, s[0], s[1], p.team.dir > 0 ? 0 : Math.PI); this.reset_rt(p.rt); moved.push(p.name);
      }
    }
    if (moved.length) this.ev('placement', null, 'manual placement: ' + moved.join(', '), { who: moved });
    w.setBall(0, 0);
    this.mj.mj_forward(w.m, w.d);
  }
  penalize(p, why) {
    const tm = p.team, n = this.players.filter(q => !q.active()).length;
    const x = -tm.dir * (0.6 + 0.35 * n), side = this.ref.side > 0 ? -1.0 : 1.0;
    this.world.put(p.duck, x, side * (this.P.hy + 0.25), -side * Math.PI / 2); this.reset_rt(p.rt);
    p.penalized_until = this.t + PICKUP_S; p.stats.pickups++; p.down_since = p.out_since = null;
    this.ev('pickup', tm.name, `${p.name} ${why}: off for ${PICKUP_S} s`, { who: p.name });
  }
  // "kick-off" button: everyone to the kick-off spots now, ball on the centre spot (the clock keeps running)
  restartKickoff() {
    this.setpiece = null; this.out_pending = null;
    for (const p of this.players) { p.penalized_until = null; p.down_since = p.out_since = null; }
    this.place_kickoff(true);
    this.phase = 'set'; this.phase_t = this.t;
    this.ev('ready', this.kickoff_team, 'kick-off (restart)');
  }

  // ---------------------------------------------------------------- tactics glue (match.Game.world_model / apply)
  world_model(tm) {
    const t = this.t, phase = ['playing', 'ready', 'set'].includes(this.phase) ? this.phase : 'other', sp = this.setpiece;
    const players = tm.players.map(p => {
      const [b, src] = p.ball(t);
      // gait phase from the feet (match.Game.world_model), head at home for how long, the walk command
      const m = p.duck.meta, x = this.world.d.xpos;
      if (p.feet === undefined) p.feet = ['ankle_left', 'ankle_right'].map(nm => m.bodies[m.body_names.indexOf(nm)]);
      const fd = x[3 * p.feet[0] + 2] - x[3 * p.feet[1] + 2];
      const swing = fd > 0.004 ? 'left' : (fd < -0.004 ? 'right' : null);
      const rising = swing !== null && Math.abs(fd) > Math.abs(p.prev_fd ?? 0.0); p.prev_fd = fd;
      const at = p.rt.headCmd.every(v => Math.abs(v) < 0.05);
      p.home_since = at ? (p.home_since ?? t) : null;
      return { name: p.name, role: p.role, pos: p.pos, psi: p.psi, active: p.active(), down: p.rt.mode === 'get up' || p.rt.lying(),
        busy: ['kick left', 'kick right', 'hop', 'settle'].includes(p.rt.mode), ball: b, ball_src: src, ball_age: t - p.ball_t,
        ball_sig: p.ball_sig(t, src), swing, rising, head_home: p.home_since === null ? 0.0 : t - p.home_since,
        moving: Math.hypot(p.rt.vel[0], p.rt.vel[1]) };
    });
    const a = DT / OPP_TAU, opps = [];
    for (const q of this.teams[this.other(tm.name)].players) {
      const key = tm.name + q.name;
      const e = this.opp_err[key] || (this.opp_err[key] = [this.rng.normal(0, OPP_NOISE), this.rng.normal(0, OPP_NOISE)]);
      for (let i = 0; i < 2; i++) e[i] += -a * e[i] + Math.sqrt(2 * a) * this.rng.normal(0, OPP_NOISE);
      const [qp] = q.duck.pose();
      opps.push({ pos: [qp[0] + e[0], qp[1] + e[1]], role: q.role, active: q.active() });
    }
    const spots = {};
    for (const p of tm.players) spots[p.name] = this.spot(tm, p.role, tm.name === this.kickoff_team);
    return { t, phase, dir: tm.dir, ball: tm.ball === null || t - tm.ball_t > 4.0 ? null : tm.ball,
      setpiece: sp === null ? null : { kind: sp.kind, ours: sp.team === tm.name, ball: sp.ball, radius: sp.radius }, spots, players, opps, chest: this.chest };
  }
  apply(p, c, t) {
    p.cmd = c; p.state = c.state;
    p.cam_on = !!c.cam && p.active();
    if (c.head !== null) p.rt.look(c.head[0], c.head[1], c.head[2]);
    if (c.kick) { this.kick(p, t, c.kick, c.kind, c.mate); p.rt.command(`kick ${c.kick}`); return; }
    p.rt.drive(c.vx, c.vy, c.wz);
  }

  // ---------------------------------------------------------------- the GameController (match.Game.rules)
  award(kind, team, pos) {
    const w = this.world;
    w.setBall(pos[0], pos[1]); this.mj.mj_forward(w.m, w.d);
    for (const tm of Object.values(this.teams)) tm.announce(pos, this.t);
    this.setpiece = { kind, team, ball: [pos[0], pos[1]], until: this.t + FREEKICK_S, radius: FREEKICK_D, touched: false };
    this.ev(kind, team, `${kind} ${team || ''}`.trim());
  }
  rules() {
    const w = this.world, t = this.t, P = this.P, [b] = w.ball();
    if (this.phase === 'playing') {
      this.clock += DT;
      const sp = this.setpiece;
      if (sp !== null) {
        const moved = Math.hypot(b[0] - sp.ball[0], b[1] - sp.ball[1]) > 0.15;
        if (t > sp.until || moved || (sp.touched && sp.kind !== 'kick-off')) this.setpiece = null;
      }
      if (this.prog === null) this.prog = [t, [b[0], b[1]]];
      if (Math.hypot(b[0] - this.prog[1][0], b[1] - this.prog[1][1]) > 0.15 || this.setpiece !== null) this.prog = [t, [b[0], b[1]]];
      else if (t - this.prog[0] > NO_PROGRESS_S && this.out_pending === null) {
        const pos = [clip(b[0], -P.hx + 0.3, P.hx - 0.3), clip(b[1], -P.hy + 0.3, P.hy - 0.3)];
        this.award('dropped ball', null, pos); this.setpiece.until = t + 3.0; this.prog = [t, pos];
      }
      if (this.out_pending === null) {
        const gx = P.hx + P.line_w / 2 + P.ball_r;
        if (Math.abs(b[0]) > gx && Math.abs(b[1]) < P.goal_w / 2 && b[2] < P.goal_h) {
          return this.goal((b[0] > 0) === (this.teams.blue.dir > 0) ? 'blue' : 'yellow');
        }
        if (Math.abs(b[1]) > P.hy + P.line_w / 2 + P.ball_r) {
          const lt = this.last_touch ? this.last_touch[1] : null;
          const team = (lt === 'blue' || lt === 'yellow') ? this.other(lt) : (b[0] * this.teams.blue.dir < 0 ? 'blue' : 'yellow');
          this.out_pending = [t, 'throw-in', team, [clip(b[0], -P.hx + 0.05, P.hx - 0.05), Math.sign(b[1]) * (P.hy - 0.02)]];
          this.ev('whistle', null, 'ball out over the touch line');
        } else if (Math.abs(b[0]) > gx) {
          const defending = (b[0] < 0) === (this.teams.blue.dir > 0) ? 'blue' : 'yellow';
          const lt = this.last_touch ? this.last_touch[1] : null, sx = Math.sign(b[0]) || 1, sy = Math.sign(b[1]) || 1;
          if (lt === defending) this.out_pending = [t, 'corner kick', this.other(defending), [sx * (P.hx - 0.03), sy * (P.hy - 0.03)]];
          else this.out_pending = [t, 'goal kick', defending, [sx * (P.hx - P.goal_area[0]), sy * P.goal_area[1] / 2]];
          this.ev('whistle', null, 'ball out over the goal line');
        }
      } else if (t - this.out_pending[0] > OUT_WHISTLE_S) {
        const [, kind, team, pos] = this.out_pending; this.out_pending = null; this.award(kind, team, pos);
      }
      for (const p of this.players) {
        if (!p.active()) {
          if (t >= p.penalized_until) { p.penalized_until = null; this.ev('return', p.team.name, `${p.name} back in`, { who: p.name }); }
          continue;
        }
        const [q] = p.duck.pose(), down = p.rt.lying() || p.duck.tiltDeg() > 25;
        p.down_since = down ? (p.down_since ?? t) : null;
        const out = Math.abs(q[0]) > P.hx + 0.35 || Math.abs(q[1]) > P.hy + 0.35;
        p.out_since = out ? (p.out_since ?? t) : null;
        if (p.down_since !== null && t - p.down_since > PICKUP_DOWN_S) this.penalize(p, 'down too long');
        else if (p.out_since !== null && t - p.out_since > PICKUP_OUT_S) this.penalize(p, 'left the field');
      }
      if (this.clock >= this.halfS) this.end_half();
    } else if (this.phase === 'goal') {
      if (t - this.phase_t > GOAL_S) {
        this.phase = 'ready'; this.phase_t = t; w.setBall(0, 0);
        for (const tm of Object.values(this.teams)) tm.announce([0, 0], t);
        this.ev('ready', this.kickoff_team, 'players to kick-off positions');
      }
    } else if (this.phase === 'ready') {
      const done = this.players.filter(p => p.active()).every(p => {
        const [q] = p.duck.pose(), s = this.spot(p.team, p.role, p.team.name === this.kickoff_team);
        return Math.hypot(q[0] - s[0], q[1] - s[1]) < 0.25;
      });
      if (t - this.phase_t > READY_S || (done && t - this.phase_t > 2.0)) {
        this.phase = 'set'; this.phase_t = t; this.manual_placement();
        for (const tm of Object.values(this.teams)) tm.announce([0, 0], t);
      }
    } else if (this.phase === 'set') {
      if (t - this.phase_t > SET_S) {
        this.phase = 'playing'; this.phase_t = t;
        this.setpiece = { kind: 'kick-off', team: this.kickoff_team, ball: [0, 0], until: t + KICKOFF_S, radius: P.circle_r, touched: false };
        this.ev('kick-off', this.kickoff_team, `kick-off ${this.kickoff_team}`);
      }
    }
  }
  goal(team) {
    const lt = this.last_touch, scorer = lt !== null ? this.by_index(lt[0]) : null, tm = this.teams[team];
    tm.score++;
    const own = scorer !== null && scorer.team !== tm;
    if (scorer !== null) scorer.stats[own ? 'own_goals' : 'goals']++;
    const sc = [this.teams.blue.score, this.teams.yellow.score];
    this.ev('goal', team, `GOAL ${team}` + (scorer ? ` (${scorer.name}${own ? ', own goal' : ''})` : '') + `, ${sc[0]}:${sc[1]}`,
      { scorer: scorer ? scorer.name : null, own_goal: own, score: sc });
    this.phase = 'goal'; this.phase_t = this.t; this.setpiece = null; this.out_pending = null;
    this.kickoff_team = this.other(team);
    if (scorer !== null && !own) {
      const [sp] = scorer.duck.pose();
      for (const p of tm.players) { const [q] = p.duck.pose(); if (p.active() && Math.hypot(q[0] - sp[0], q[1] - sp[1]) < 0.8 && !p.rt.lying()) p.rt.command('hop'); }
    }
    if (!this.ref.rt.lying()) this.ref.rt.command('hop');
  }
  end_half() {
    this.ev('whistle', null, `end of half ${this.half}`);
    if (this.half >= this.halves) {
      this.phase = 'end';
      this.ev('full-time', null, `full time ${this.teams.blue.score}:${this.teams.yellow.score}`);
      return;
    }
    this.half++; this.clock = 0;
    for (const tm of Object.values(this.teams)) tm.dir = -tm.dir;
    this.kickoff_team = this.other(this.first_kickoff);
    for (const p of this.players) { p.penalized_until = null; p.down_since = p.out_since = null; }
    this.setpiece = null; this.out_pending = null;
    this.place_kickoff(true);
    this.phase = 'set'; this.phase_t = this.t;
    this.ev('half-time', null, 'half-time: teams change ends');
  }

  // ---------------------------------------------------------------- the 50 Hz loop (match.Game.tick)
  tick() {
    const t = this.t, w = this.world;
    for (const p of this.players) p.update_pose();
    for (const tm of Object.values(this.teams)) tm.deliver(t);
    const playing = this.phase === 'playing';
    const cmds = {};
    for (const tm of Object.values(this.teams)) Object.assign(cmds, this.tactics[tm.name].step(this.world_model(tm)));
    for (const p of this.players) p.cam_on = !!cmds[p.name].cam && p.active() && playing;
    if (this.k % CAM_EVERY === 0 && playing) for (const p of this.players) if (p.cam_on) p.see(t);
    for (const p of this.players) if (p.cam_on) p.share(t);
    for (const p of this.players) {
      const c = cmds[p.name];
      if (!(p.active() && (this.phase === 'playing' || this.phase === 'ready'))) { c.vx = c.vy = c.wz = 0; c.kick = null; }
      this.apply(p, c, t);
    }
    const [rv, rvy, rwz] = this.ref.act();
    this.ref.rt.drive(rv, rvy, rwz);
    const tb = performance.now();
    brainsTick(this.rts, this.runBrains, this.scratch);
    this.brainMs += performance.now() - tb;
    const [b0] = w.ball();
    const near = w.ducks.some(dk => { const [q] = dk.pose(); return Math.hypot(q[0] - b0[0], q[1] - b0[1]) < 0.35; });
    w.physics(near);
    for (const rt of this.rts) rt.after();
    this.feel();
    this.process_touches();
    const [, bv] = w.ball(), spd = Math.hypot(bv[0], bv[1]);
    for (const kc of this.kicks.slice(-3)) {
      if (t - kc.t < 1.2) {
        kc.vmax = Math.max(kc.vmax, spd);
        if (!kc.connected && kc.vmax > 0.3) { kc.connected = true; this.by_index(kc.who).stats.connected++; }
      }
    }
    this.rules();
    this.metrics();
    this.k++; this.t = this.k * DT;
  }
  // measurement only (match.Metrics, simplified): distance walked and share of playing time moving (> 3 cm/s over 0.5 s)
  metrics() {
    if (this.phase !== 'playing') { for (const p of this.players) p.hist = []; return; }
    // ball lost at the feet (match.Diag, simplified): a player's trunk within 15 cm of the ball, no own fix for 0.3 s
    const [bb] = this.world.ball(), L = this.lost || (this.lost = { near: 0, lost: 0, search: 0, standing: 0 });
    for (const p of this.players) {
      if (!p.active()) continue;
      const [q] = p.duck.pose();
      if (Math.hypot(q[0] - bb[0], q[1] - bb[1]) >= 0.15) continue;
      L.near++;
      if (this.t - p.ball_t <= 0.3) continue;
      L.lost++;
      const c = p.cmd, busy = ['kick left', 'kick right', 'get up', 'hop'].includes(p.rt.mode);
      if ((p.state || '').startsWith('search')) L.search++;
      else if (c && !busy && c.vx === 0 && c.vy === 0 && c.wz === 0) L.standing++;
    }
    if (this.k % 5) return;
    for (const p of this.players) {
      if (!p.active()) { p.hist = []; continue; }
      const [q] = p.duck.pose(); p.hist.push(q); if (p.hist.length > 6) p.hist.shift();
      if (p.hist.length === 6) {
        const disp = Math.hypot(p.hist[5][0] - p.hist[0][0], p.hist[5][1] - p.hist[0][1]);
        p.stats.samples++; if (disp / 0.5 > 0.03) p.stats.moving++;
        if (this.k % 25 === 0) p.stats.dist += disp;
      }
    }
  }
  summary() {
    const per = {};
    for (const p of this.players) per[p.name] = { team: p.team.name, role: p.role, falls: p.rt.falls, unsticks: p.rt.unsticks, ...p.stats };
    per.ref = { team: 'ref', falls: this.ref.rt.falls };
    return { seed: this.seed, score: [this.teams.blue.score, this.teams.yellow.score], sim_s: +this.t.toFixed(1), half: this.half, clock: +this.clock.toFixed(1), phase: this.phase,
      kicks: this.kicks.length, connected: this.kicks.filter(k => k.connected).length, falls: this.rts.reduce((a, r) => a + r.falls, 0),
      goals: this.events.filter(e => e.kind === 'goal').map(e => e.text), events: this.events.length, players: per,
      dropped: this.events.filter(e => e.kind === 'dropped ball').length,
      lost_near_s: this.lost ? Object.fromEntries(Object.entries(this.lost).map(([k, v]) => [k, +(v * DT).toFixed(1)])) : null };
  }
}
