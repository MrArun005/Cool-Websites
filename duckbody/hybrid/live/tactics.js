// Team tactics for the duck match: a line-by-line JavaScript port of ../tactics.py (pure decisions, no physics).
//   const tac = new TeamTactics(rng); const cmds = tac.step(wm);   // every 50 Hz tick, per team
// wm and the returned Cmd per player name are exactly tactics.py's (see its docstring). Checked against the Python
// on recorded world models with tactics_check.mjs / tactics_check.py (same commands, rng stubbed on both sides).

export const HX = 2.25, HY = 1.5, GOAL_W = 1.3, PEN_AREA = [1.0, 2.5], BORDER = 0.5, BALL_R = 0.035, DT = 0.02;
export const KICK_X = 0.08, KICK_Y = 0.032, TOL_X = 0.02, TOL_Y = 0.02, TOL_H = 0.15, BEHIND = 0.24, AROUND = 0.28;
export const READY_S = 1.0, QUICK_READY_S = 0.8, QUICK_X = [-0.03, 0.035], QUICK_Y = 0.03, QUICK_H = 0.45;
export const FOOT_VIEW = [-0.6, 1.1], CAM_H = 0.21, CAM_DOWN = 0.17;
const DUEL_D = 0.35, ALIGN_MAX_S = 3.0, SHOOT_D = 2.4, LANE_R = 0.18, PASS_MIN = 0.5, PASS_MAX = 2.2, PASS_LEAD = 0.2;
const RECEIVE_S = 4.0, FOLLOW_S = 3.0, ROLE_EVERY = 15, SUPPORT_EVERY = 2.5, COVER_D = 0.6, CONTEST_D = 0.45, MARK_D = 0.3;
const SHUFFLE_A = 0.12, SHUFFLE_T = 5.0;
const AIM_Y = 0.35, DRIBBLE_D = [1.3, 3.2], DRIBBLE_FREE = 0.55, DRIBBLE_MAX_S = 5.0;
const GAP = 0.4, CROWD_VX = 0.6, CROWD_WZ = 0.75, CROWD_VX_BALL = 0.75, PINNED_S = 1.5;

// ---------------------------------------------------------------- 2-D helpers ([x, y] arrays)
export const wrap = a => { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; };
const clip = (x, a, b) => Math.min(b, Math.max(a, x));
const add = (a, b) => [a[0] + b[0], a[1] + b[1]], sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const mul = (a, s) => [a[0] * s, a[1] * s], dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const norm = a => Math.hypot(a[0], a[1]);
const sign = x => (x < 0 || Object.is(x, -0) ? -1 : 1);
const copysign = (m, s) => Math.abs(m) * sign(s);
export const unit = v => { const n = norm(v); return n > 1e-9 ? [v[0] / n, v[1] / n] : [1.0, 0.0]; };
export function segDist(a, b, p) {
  const ab = sub(b, a), L2 = dot(ab, ab);
  const s = L2 > 1e-9 ? clip(dot(sub(p, a), ab) / L2, 0, 1) : 0.0;
  return [norm(sub(add(a, mul(ab, s)), p)), s];
}
const clipLim = (p, lim) => [clip(p[0], -lim[0], lim[0]), clip(p[1], -lim[1], lim[1])];
const minOf = (arr, dflt) => arr.length ? Math.min(...arr, dflt) : dflt;

export class Cmd {
  constructor(vx = 0, vy = 0, wz = 0, o = {}) {
    this.vx = vx; this.vy = vy; this.wz = wz;
    this.head = o.head ?? null; this.kick = o.kick ?? null; this.kind = o.kind ?? null; this.mate = o.mate ?? null;
    this.cam = o.cam ?? false; this.state = o.state ?? ''; this.role = o.role ?? '';
  }
}

class Mem {
  constructor(name, rng) { this.name = name; this.rng = rng; this.reset(0.0); this.phase = rng.uniform(0, 2 * Math.PI); }
  reset(t) {
    this.state = 'search'; this.state_t = t;
    this.aim = null; this.aim_kind = null; this.aim_mate = null;
    this.foot = null; this.ok_t = 0.0; this.kick_at = null;
    this.search_dir = 1.0;
    this.turn_hist = []; this.backup_until = -1.0;
    this.duel_hist = []; this.detour_until = -1.0; this.detour_sign = 1.0;
    this.run_target = null; this.run_t = -99.0;
    this.dribble_block = -1.0;
  }
  set_state(s, t) { if (s !== this.state) { this.state = s; this.state_t = t; } }
}

class Me {
  constructor(d) { this.d = d; this.name = d.name; this.role = d.role; this.pos = [d.pos[0], d.pos[1]]; this.psi = d.psi; }
  to_body(p) { const e = sub(p, this.pos), c = Math.cos(this.psi), s = Math.sin(this.psi); return [c * e[0] + s * e[1], -s * e[0] + c * e[1]]; }
}

// ---------------------------------------------------------------- motion primitives
export function goTo(me, target, vmax = 0.9, face = null, strafe_r = 0.0) {
  const e = me.to_body(target), d = Math.hypot(e[0], e[1]), b = Math.atan2(e[1], e[0]);
  if (face !== null && d < Math.max(strafe_r, 0.08)) {
    const eh = wrap(face - me.psi);
    const wz = Math.abs(eh) < 0.15 ? 0.0 : copysign(clip(1.4 * Math.abs(eh), 0.6, 1.0), eh);
    if (d < 0.05 || (strafe_r === 0.0 && d < 0.08)) return [0.0, 0.0, wz, d];
    const vx = e[0] > 0.03 ? 0.45 : (e[0] < -0.03 ? -0.35 : 0.0);
    const vy = Math.abs(e[1]) > 0.03 ? copysign(0.45, e[1]) : 0.0;
    return [vx, vy, wz, d];
  }
  let wz = clip(1.6 * b, -1.0, 1.0);
  if (Math.abs(b) > 0.12 && Math.abs(wz) < 0.6) wz = copysign(0.6, b);
  let vx = 0.0;
  if (Math.abs(b) < 0.6 && d > 0.04) vx = clip(0.3 + 1.2 * d, 0.32, vmax);
  if (Math.abs(b) > 0.9 && d < 0.3) vx = 0.0;
  if (Math.abs(b) > 2.4 && d < 0.25) return [-0.4, 0.0, 0.0, d];
  return [vx, 0.0, wz, d];
}
export function lookAt(me, p) {
  const rel = me.to_body(p), d = Math.hypot(rel[0], rel[1]), bear = Math.atan2(rel[1], rel[0]);
  if (d < 0.24) return [clip(bear, -0.8, 0.8), FOOT_VIEW[1], FOOT_VIEW[0]];
  const elev = Math.atan2(CAM_H - BALL_R, Math.max(d - 0.04, 0.05));
  return [bear, elev - CAM_DOWN, null];
}
const headingTo = (a, b) => Math.atan2(b[1] - a[1], b[0] - a[0]);
const clipField = (p, mx = 0.15, my = 0.15) => [clip(p[0], -HX + mx, HX - mx), clip(p[1], -HY + my, HY - my)];

// ---------------------------------------------------------------- the team brain
export class TeamTactics {
  // rng: an object with uniform(a, b) and next() (live_sim.js Rng); Mems get their own child rngs
  constructor(rng, childRng = r => r) {
    this.rng = rng; this.childRng = childRng;
    this.mem = {}; this.striker = null; this.second = null; this.k = 0;
    this.ball_hist = []; this.pass_to = null; this.last_kick = null; this.prev_phase = null;
  }
  m(name) { if (!(name in this.mem)) this.mem[name] = new Mem(name, this.childRng(this.rng)); return this.mem[name]; }
  static own_goal(d) { return [-d * HX, 0.0]; }
  opp_pos(wm, gk = true) { return wm.opps.filter(o => o.active && (gk || o.role !== 'GK')).map(o => [o.pos[0], o.pos[1]]); }
  nearest_opp(wm, p) { const ds = this.opp_pos(wm).map(q => norm(sub(q, p))); return ds.length ? Math.min(...ds) : 9.0; }
  ball_vel() {
    const h = this.ball_hist;
    if (h.length < 2 || h[h.length - 1][0] - h[0][0] < 0.2) return [0, 0];
    return mul(sub(h[h.length - 1][1], h[0][1]), 1 / (h[h.length - 1][0] - h[0][0]));
  }

  step(wm) {
    const t = wm.t, d = wm.dir;
    this._wm = wm;
    this.k += 1;
    const players = wm.players.slice();
    const out = {};
    if (wm.phase !== this.prev_phase) {
      for (const p of players) this.m(p.name).reset(t);
      this.pass_to = null; this.last_kick = null; this.ball_hist = [];
      this.prev_phase = wm.phase;
    }
    let b = wm.ball;
    for (const p of players) if (p.name === this.striker && p.ball_src === 'own') b = p.ball;
    b = b === null || b === undefined ? null : [b[0], b[1]];
    if (b !== null) this.ball_hist = this.ball_hist.filter(h => t - h[0] <= 0.4).concat([[t, b.slice()]]);
    const bv = this.ball_vel();
    if (wm.phase !== 'playing') {
      for (const p of players) out[p.name] = this.stopped(wm, p, b);
      return out;
    }
    const sp = wm.setpiece;
    const restricted = sp !== null && sp !== undefined && !sp.ours;
    const field = players.filter(p => p.role !== 'GK' && p.active);
    if (this.pass_to !== null && (t - this.pass_to[2] > RECEIVE_S || !players.some(p => p.name === this.pass_to[0] && p.active)))
      this.pass_to = null;
    if (b !== null && field.length && (this.k % ROLE_EVERY === 1 || !field.some(p => p.name === this.striker))) {
      const costs = field.map((p, i) => [this.cost(p, b, d, t), i, p]).sort((x, y) => x[0] - y[0] || x[1] - y[1]).map(x => x[2]);
      this.striker = costs[0].name;
      this.second = costs.length > 1 ? costs[1].name : null;
    }
    for (const p of players) {
      const me = new Me(p), mem = this.m(me.name);
      if (!p.active) { out[me.name] = new Cmd(0, 0, 0, { head: b !== null ? lookAt(me, b) : null, state: 'off', role: 'off' }); continue; }
      if (p.down) { mem.set_state('get up', t); out[me.name] = new Cmd(0, 0, 0, { state: 'get up', role: this.role_of(me.name, p) }); continue; }
      if (p.busy) { out[me.name] = new Cmd(0, 0, 0, { state: mem.state, role: this.role_of(me.name, p) }); continue; }
      if (mem.state === 'get up' || mem.state === 'kick') mem.set_state('approach', t);
      const role = this.role_of(me.name, p);
      let c;
      if (b === null) c = this.no_ball(wm, me, mem, p);
      else if (mem.state === 'ready' && !restricted) c = this.ball_play(wm, me, mem, p, b);
      else if (role === 'GK') c = this.keeper(wm, me, mem, p, b, bv, restricted, sp);
      else if (role === 'striker') {
        if (restricted) {
          const u = unit(sub(TeamTactics.own_goal(d), sp.ball));
          const tgt = add(sp.ball, mul(u, sp.radius + 0.12));
          c = this.shadow(me, mem, t, tgt, b, 0.9, 0.4);
        } else if (this.pass_to !== null && this.pass_to[0] === me.name && norm(bv) > 0.2 && norm(sub(b, me.pos)) > 0.35) {
          c = this.receive(me, mem, t, b, bv);
        } else c = this.ball_play(wm, me, mem, p, b);
      } else if (this.last_kick !== null && this.last_kick[0] === me.name && this.last_kick[1] === 'pass' && t - this.last_kick[2] < FOLLOW_S) {
        c = this.support_run(wm, me, mem, b, true);
      } else if (role === 'second') c = this.second_play(wm, me, mem, p, b, restricted, sp);
      else if (p.role.startsWith('A')) c = this.support_run(wm, me, mem, b);
      else c = this.defend(wm, me, mem, p, b, restricted, sp);
      c.role = role;
      if (restricted && role !== 'striker') c = this.respect(me, mem, c, sp, b);
      c = this.give_way(wm, me, c, role);
      c = this.caution(wm, me, c, role === 'striker' || c.state.startsWith('contest'));
      out[me.name] = this.reflex(me, mem, t, c);
      if (c.kick) {
        this.last_kick = [me.name, c.kind, t];
        if (c.kind === 'pass' && c.mate !== null) { this.pass_to = [c.mate, me.name, t]; this.striker = c.mate; }
      }
    }
    return out;
  }

  role_of(name, p) {
    if (p.role === 'GK') return 'GK';
    if (name === this.striker) return 'striker';
    if (name === this.second) return 'second';
    return p.role.startsWith('A') ? 'support' : 'defend';
  }
  cost(p, b, d, t) {
    const me = new Me(p), e = me.to_body(b);
    let c = Math.hypot(e[0], e[1]) / 0.3 + Math.abs(Math.atan2(e[1], e[0])) / 1.2;
    if (p.down) c += 8.0;
    if (p.role.startsWith('D') && b[0] * d > 0.6) c += 2.5;
    if (p.role.startsWith('A') && b[0] * d < -1.2) c += 1.5;
    if (p.name === this.striker) c -= 1.5;
    if (this.pass_to !== null && this.pass_to[0] === p.name) c -= 6.0;
    if (this.last_kick !== null && this.last_kick[0] === p.name && this.last_kick[1] !== 'pass' && t - this.last_kick[2] < FOLLOW_S) c -= 2.0;
    return c;
  }

  // ------------------------------------------------------------ outside play
  stopped(wm, p, b) {
    const me = new Me(p), mem = this.m(p.name);
    const head = b !== null ? lookAt(me, b) : null;
    if (wm.phase === 'ready' && p.active && !p.down && !p.busy) {
      const s = wm.spots[me.name], face = wm.dir > 0 ? 0.0 : Math.PI;
      const [vx, vy, wz] = goTo(me, s, 1.0, face, 0.25);
      return this.reflex(me, mem, wm.t, new Cmd(vx, vy, wz, { head, state: 'position', role: 'ready' }));
    }
    return new Cmd(0, 0, 0, { head, state: 'wait', role: 'wait' });
  }
  no_ball(wm, me, mem, p) {
    const t = wm.t;
    if (me.name === this.striker || p.role === 'GK') return this.search(me, mem, t, true);
    const s = this.formation(p.role, [0, 0], wm.dir, t, mem);
    const [vx, vy, wz] = goTo(me, s, 0.8, headingTo(s, [0, 0]), 0.35);
    const ph = t - mem.state_t;
    return new Cmd(vx, vy, wz, { head: [0.9 * Math.sin(0.8 * ph), 0.1, null], state: 'hold' });
  }
  search(me, mem, t, cam = true) {
    mem.set_state('search', t);
    const ph = t - mem.state_t, nod = 0.5 + 0.5 * Math.sin(2 * Math.PI * 0.5 * ph);
    return new Cmd(0.0, 0.0, 0.9 * mem.search_dir, { head: [0.6 * mem.search_dir, 0.1 + 0.8 * nod, -0.3 * nod], cam, state: 'search' });
  }
  give_way(wm, me, c, role) {
    if (c.kick !== null || c.vx <= 0.0) return c;
    const rank = { striker: 0, GK: 1 }, rk = r => (r in rank ? rank[r] : 2);
    for (const x of wm.players) {
      if (x.name === me.name || !x.active) continue;
      if (rk(role) < rk(this.role_of(x.name, x))) continue;
      const rel = me.to_body(x.pos);
      if (0.0 < rel[0] && rel[0] < 0.35 && Math.abs(rel[1]) < 0.2) {
        c.vx = 0.0; c.vy = -copysign(0.45, rel[1]); c.state += '/give way'; break;
      }
    }
    return c;
  }
  caution(wm, me, c, carrier = false) {
    if (c.kick !== null) return c;
    const others = wm.players.filter(x => x.name !== me.name && x.active).map(x => [x.pos[0], x.pos[1]]).concat(this.opp_pos(wm));
    for (const q of others) {
      const rel = me.to_body(q), d = Math.hypot(rel[0], rel[1]);
      if (d < 0.4 && rel[0] > 0.0 && Math.abs(rel[1]) < 0.25) c.vx = Math.min(c.vx, carrier ? CROWD_VX_BALL : CROWD_VX);
      if (d < 0.3) c.wz = clip(c.wz, -CROWD_WZ, CROWD_WZ);
    }
    return c;
  }
  spread(wm, me, target, r = GAP) {
    let t = [target[0], target[1]];
    for (const x of wm.players) {
      if (x.name === me.name || !x.active) continue;
      const v = sub(t, x.pos), dd = Math.hypot(v[0], v[1]);
      if (dd < r) t = add(t, mul(unit(dd > 1e-6 ? v : sub(me.pos, t)), r - dd));
    }
    return t;
  }
  reflex(me, mem, t, c) {
    const turning = Math.abs(c.wz) >= 0.5 && c.vx <= 0.0 && !c.kick;
    mem.turn_hist = mem.turn_hist.concat([[turning, me.psi]]).slice(-75);
    if (mem.turn_hist.length === 75 && mem.turn_hist.every(a => a[0]) && Math.abs(wrap(mem.turn_hist[74][1] - mem.turn_hist[0][1])) < 0.25) {
      mem.backup_until = t + 1.0; mem.turn_hist = [];
    }
    if (t < mem.backup_until && !c.kick) { c.vx = -0.5; c.vy = 0.0; c.wz = 0.0; }
    return c;
  }
  respect(me, mem, c, sp, b) {
    const bb = sp.ball, v = sub(me.pos, bb);
    if (norm(v) < sp.radius + 0.05) {
      const tgt = add(bb, mul(unit(v), sp.radius + 0.15));
      const [vx, vy, wz] = goTo(me, tgt, 0.9);
      c.vx = vx; c.vy = vy; c.wz = wz; c.kick = null;
    }
    return c;
  }
  shadow(me, mem, t, target, b, vmax = 0.9, strafe = 0.4, cam = false, state = 'position') {
    target = clipField(this.spread(this._wm, me, target), 0.12, 0.12);
    const [vx, vy, wz] = goTo(me, target, vmax, headingTo(target, b), strafe);
    mem.set_state(state, t);
    return new Cmd(vx, vy, wz, { head: lookAt(me, b), cam, state });
  }

  // ------------------------------------------------------------ formation and off-ball play
  formation(role, b, d, t, mem) {
    const bt = [b[0] * d, b[1] * d];
    const sh = SHUFFLE_A * Math.sin(2 * Math.PI * t / SHUFFLE_T + mem.phase);
    let s;
    if (role === 'D1') s = [clip(0.45 * bt[0] - 1.2, -HX + 0.45, 0.2), clip(0.35 * bt[1] + 0.45 + sh, -HY + 0.3, HY - 0.3)];
    else if (role === 'D2') s = [clip(0.45 * bt[0] - 1.2, -HX + 0.45, 0.2), clip(0.35 * bt[1] - 0.45 + sh, -HY + 0.3, HY - 0.3)];
    else if (role === 'A1') s = [clip(bt[0] + 0.4, -0.6, HX - 0.5), clip(0.4 * bt[1] + 0.7 + sh, -HY + 0.25, HY - 0.25)];
    else if (role === 'A2') s = [clip(bt[0] + 0.4, -0.6, HX - 0.5), clip(0.4 * bt[1] - 0.7 + sh, -HY + 0.25, HY - 0.25)];
    else s = [-HX + 0.3, 0.0];
    return [s[0] * d, s[1] * d];
  }
  mates_pos(wm, me) { return wm.players.filter(p => p.name !== me.name && p.active).map(p => [p.pos[0], p.pos[1]]); }
  support_run(wm, me, mem, b, give_and_go = false) {
    const t = wm.t, d = wm.dir, opps = this.opp_pos(wm), mates = this.mates_pos(wm, me);
    if (mem.run_target === null || t - mem.run_t > SUPPORT_EVERY || norm(sub(mem.run_target, me.pos)) < 0.12) {
      let best = null, bs = -1e9;
      for (const dist of [0.7, 1.0, 1.3]) for (const ang of [-1.2, -0.75, -0.35, 0.0, 0.35, 0.75, 1.2]) {
        const a = (d > 0 ? 0.0 : Math.PI) + ang;
        const c = clipField(add(b, [dist * Math.cos(a), dist * Math.sin(a)]), 0.3, 0.25);
        const fwd = (c[0] - b[0]) * d;
        const open_ = minOf(opps.map(q => norm(sub(q, c))), 1.0);
        const lane = minOf(opps.map(q => segDist(b, c, q)[0]), 1.0);
        const crowd = minOf(mates.map(q => norm(sub(q, c))), 1.0);
        const goal_d = norm(sub([d * HX, 0.0], c));
        let s = 0.6 * fwd + 1.2 * Math.min(open_, 0.8) + 1.0 * Math.min(lane, 0.4) + 0.8 * Math.min(crowd, 0.6) - 0.2 * goal_d;
        s -= 0.6 * norm(sub(c, me.pos));
        if (norm(sub(c, me.pos)) < 0.25) s -= 1.0;
        if (give_and_go) s += 0.8 * fwd;
        s += mem.rng.uniform(0, 0.15);
        if (s > bs) { best = c; bs = s; }
      }
      mem.run_target = best; mem.run_t = t;
    }
    const st = give_and_go ? 'give-and-go' : 'support';
    return this.shadow(me, mem, t, mem.run_target, b, 1.0, 0.3, false, st);
  }
  defend(wm, me, mem, p, b, restricted, sp) {
    const t = wm.t, d = wm.dir, own = TeamTactics.own_goal(d);
    const threats = this.opp_pos(wm, false).filter(q => q[0] * d < 0.4 && norm(sub(q, b)) > 0.3);
    const spot = this.formation(p.role, b, d, t, mem);
    if (threats.length) {
      const others = wm.players.filter(x => x.role.startsWith('D') && x.name !== me.name).map(x => [x.pos[0], x.pos[1]]);
      const key = q => norm(sub(q, spot)) + norm(sub(q, own)) * 0.3;
      let q = threats.reduce((a, x) => key(x) < key(a) ? x : a);
      if (others.length && threats.length > 1) {
        const o = others[0];
        if (norm(sub(o, q)) < norm(sub(me.pos, q)) - 0.2) {
          const rest = threats.filter(x => x !== q);
          q = rest.reduce((a, x) => norm(sub(x, spot)) < norm(sub(a, spot)) ? x : a);
        }
      }
      const tgt = add(add(q, mul(unit(sub(own, q)), MARK_D)), mul(unit(sub(b, q)), 0.1));
      return this.shadow(me, mem, t, tgt, b, 1.0, 0.35, false, 'mark');
    }
    return this.shadow(me, mem, t, spot, b, 0.9, 0.4, false, 'block');
  }
  second_play(wm, me, mem, p, b, restricted, sp) {
    const t = wm.t, d = wm.dir, own = TeamTactics.own_goal(d);
    const opp_d = this.nearest_opp(wm, b), db = norm(sub(b, me.pos));
    const st = wm.players.find(x => x.name === this.striker);
    const st_near = st !== undefined && norm(sub(st.pos, me.pos)) < 0.3;
    const st_db = st !== undefined ? norm(sub(st.pos, b)) : 9.0;
    if (!restricted && db < CONTEST_D && opp_d < Math.min(0.45, st_db) && !st_near) {
      if (!['approach', 'chase', 'duel', 'align', 'ready', 'dribble'].includes(mem.state)) mem.set_state('duel', t);
      const c = this.ball_play(wm, me, mem, p, b);
      c.state = 'contest:' + c.state;
      return c;
    }
    if (['duel', 'align', 'ready', 'approach', 'chase', 'dribble'].includes(mem.state)) mem.set_state('cover', t);
    let side = 1.0;
    const u = unit(sub(own, b)), n = [-u[1], u[0]];
    if (st !== undefined) side = dot(sub(st.pos, b), n) > 0 ? -1.0 : 1.0;
    let tgt, state;
    if (opp_d < 0.8 || b[0] * d < 0.0) { tgt = add(add(b, mul(u, COVER_D)), mul(n, side * 0.15)); state = 'press'; }
    else { tgt = add(add(b, mul(u, 0.55)), mul(n, side * 0.45)); state = 'cover'; }
    return this.shadow(me, mem, t, tgt, b, 1.1, 0.3, db < 1.0, state);
  }
  receive(me, mem, t, b, bv) {
    mem.set_state('receive', t);
    const u = unit(bv), s = Math.max(0.0, dot(sub(me.pos, b), u));
    const meet = add(b, mul(u, Math.max(0.0, s - 0.15)));
    const [vx, vy, wz] = goTo(me, meet, 0.9, headingTo(me.pos, b), 0.5);
    return new Cmd(vx, vy, wz, { head: lookAt(me, b), cam: true, state: 'receive' });
  }

  // ------------------------------------------------------------ the goalkeeper
  keeper(wm, me, mem, p, b, bv, restricted, sp) {
    const d = wm.dir, t = wm.t, own = TeamTactics.own_goal(d);
    const bt = [b[0] * d, b[1] * d];
    const in_box = bt[0] < -HX + PEN_AREA[0] + 0.1 && Math.abs(bt[1]) < PEN_AREA[1] / 2 + 0.1;
    const my_d = norm(sub(b, me.pos)), opp_d = this.nearest_opp(wm, b);
    const mate_d = minOf(wm.players.filter(x => x.name !== me.name && x.active && !x.down).map(x => norm(sub(x.pos, b))), 9.0);
    const claim = in_box && (my_d < 0.35 || (my_d < 0.7 && my_d < mate_d) || my_d < Math.min(opp_d, mate_d) - 0.1) && !restricted;
    if (['approach', 'chase', 'align', 'ready', 'duel', 'dribble'].includes(mem.state)) {
      if (!in_box && bt[0] > -HX + PEN_AREA[0] + 0.35) mem.set_state('keeper', t);
      else { const c = this.ball_play(wm, me, mem, p, b); c.cam = true; return c; }
    } else if (claim) {
      mem.set_state('approach', t);
      const c = this.ball_play(wm, me, mem, p, b); c.cam = true; return c;
    }
    mem.set_state('keeper', t);
    const far = clip((bt[0] + HX) / (2 * HX), 0, 1), r = 0.25 + 0.2 * far;
    let tgt = add(own, mul(unit(sub(b, own)), r));
    const tt = [tgt[0] * d, tgt[1] * d];
    tt[0] = Math.max(tt[0], -HX + 0.18);
    tt[1] = clip(tt[1], -GOAL_W / 2 + 0.1, GOAL_W / 2 - 0.1);
    tgt = [tt[0] * d, tt[1] * d];
    const [vx, vy, wz] = goTo(me, tgt, 0.8, headingTo(tgt, b), 0.3);
    return new Cmd(vx, vy, wz, { head: lookAt(me, b), cam: true, state: 'keeper' });
  }

  // ------------------------------------------------------------ aim
  choose_aim(wm, me, b) {
    const d = wm.dir;
    const gk = wm.opps.find(o => o.role === 'GK' && o.active);
    const gy = gk !== undefined ? gk.pos[1] : 0.0;
    const side = Math.abs(gy) > 0.08 ? -copysign(1.0, gy) : copysign(1.0, Math.abs(b[1]) > 0.05 ? b[1] : 1.0);
    const goal = [d * HX, side * AIM_Y];
    const opps = this.opp_pos(wm, false);
    const dg = norm(sub(goal, b));
    const lane_open = (a, c, r = LANE_R) => {
      for (const q of opps) { const [dd, s] = segDist(a, c, q); if (dd < r && 0.03 < s && norm(sub(q, a)) < 1.8) return false; }
      return true;
    };
    const shot_open = lane_open(b, goal);
    let best = null, bs = -9.0;
    for (const m of wm.players) {
      if (m.name === me.name || m.role === 'GK' || !m.active || m.down) continue;
      const mp = [m.pos[0], m.pos[1]], fwd = (mp[0] - b[0]) * d, dist = norm(sub(mp, b));
      if (fwd > 0.15 && PASS_MIN < dist && dist < PASS_MAX && lane_open(b, mp, 0.2)) {
        const free = minOf(opps.map(q => norm(sub(q, mp))), 1.0);
        if (free < 0.35) continue;
        const mg = norm(sub([d * HX, 0.0], mp));
        const sc = fwd - 0.3 * dist + 0.5 * Math.min(free, 0.8) + (mg < dg - 0.6 ? 0.6 : 0.0);
        if (sc > bs) { best = [m, mp, mg]; bs = sc; }
      }
    }
    if (shot_open && dg < 1.6) return [goal, 'shot', null];
    if (best !== null && (!shot_open || dg > SHOOT_D || best[2] < dg - 0.9)) {
      const [m, mp] = best;
      return [add(mp, [d * PASS_LEAD, 0.0]), 'pass', m.name];
    }
    if (dg < SHOOT_D) return [goal, 'shot', null];
    if (b[0] * d < -0.75 || me.role === 'GK') return [goal, 'clearance', null];
    return [goal, 'forward', null];
  }

  // ------------------------------------------------------------ ball play
  ball_play(wm, me, mem, p, b) {
    const t = wm.t, src = p.ball_src;
    const fresh_age = p.ball_age ?? 9.0;
    if (p.ball_src === 'own' && p.ball !== null && p.ball !== undefined) b = [p.ball[0], p.ball[1]];
    const rel = me.to_body(b), dball = Math.hypot(rel[0], rel[1]);
    if (['search', 'position', 'keeper', 'support', 'block', 'mark', 'press', 'cover', 'receive', 'give-and-go', 'hold', 'wait'].includes(mem.state)) {
      mem.set_state('approach', t);
      mem.search_dir = Math.atan2(rel[1], rel[0]) >= 0 ? 1.0 : -1.0;
    }
    const opp_d = this.nearest_opp(wm, b);
    if (['approach', 'chase', 'duel', 'dribble'].includes(mem.state) || mem.aim === null) [mem.aim, mem.aim_kind, mem.aim_mate] = this.choose_aim(wm, me, b);
    const k = unit(sub(mem.aim, b)), n = [-k[1], k[0]];
    const r = sub(me.pos, b), along = dot(r, k), lat = dot(r, n);
    const phi_d = Math.atan2(r[1], r[0]), phi_p = Math.atan2(-k[1], -k[0]);
    let dphi = wrap(phi_p - phi_d);
    const lim = [HX + BORDER - 0.12, HY + BORDER - 0.12];
    const own_fresh = src === 'own' && fresh_age < 0.3;
    const eh = wrap(Math.atan2(k[1], k[0]) - me.psi);
    const cmd = (vx = 0, vy = 0, wz = 0, head = null, state = null) =>
      new Cmd(vx, vy, wz, { head: head !== null ? head : lookAt(me, b), cam: true, state: state || mem.state });

    if (['approach', 'chase', 'duel', 'align', 'dribble'].includes(mem.state) && own_fresh && Math.abs(eh) < QUICK_H
        && !(mem.state === 'dribble' && norm(sub([wm.dir * HX, 0.0], b)) > SHOOT_D)) {
      for (const [foot, ty] of [['left', KICK_Y], ['right', -KICK_Y]]) {
        const ex = rel[0] - KICK_X, ey = rel[1] - ty;
        if (QUICK_X[0] < ex && ex < QUICK_X[1] && Math.abs(ey) < QUICK_Y) {
          mem.foot = foot;
          mem.kick_at = { foot, kind: mem.aim_kind, mate: mem.aim_mate, quick: true };
          mem.set_state('ready', t);
          return cmd(0, 0, 0, [0.0, 0.0, null]);
        }
      }
    }
    if (['approach', 'chase'].includes(mem.state) && opp_d < DUEL_D && dball < 0.7) mem.set_state('duel', t);
    if (mem.state === 'duel') {
      if (opp_d > DUEL_D + 0.2 && t - mem.state_t > 0.6) mem.set_state('approach', t);
      const nh = Math.trunc(PINNED_S / DT);
      mem.duel_hist = mem.duel_hist.concat([b.slice()]).slice(-nh);
      if (mem.duel_hist.length === nh && norm(sub(mem.duel_hist[0], b)) < 0.05 && t > mem.detour_until) {
        mem.detour_until = t + 1.2; mem.detour_sign = mem.rng.choice([-1.0, 1.0]); mem.duel_hist = [];
      }
      if (t < mem.detour_until) {
        const a = phi_d + mem.detour_sign * 1.0;
        const target = clipLim(add(b, [0.22 * Math.cos(a), 0.22 * Math.sin(a)]), lim);
        const [vx, vy, wz] = goTo(me, target, 1.0);
        return cmd(vx, vy, wz);
      }
      let vx, vy, wz;
      if (Math.abs(dphi) > 1.0) {
        const a = phi_d + clip(dphi, -0.9, 0.9);
        const target = clipLim(add(b, [0.2 * Math.cos(a), 0.2 * Math.sin(a)]), lim);
        [vx, vy, wz] = goTo(me, target, 1.0);
      } else {
        const sg = rel[1] > 0 ? 1.0 : -1.0;
        [vx, vy, wz] = goTo(me, sub(add(b, mul(k, 0.15)), mul(n, sg * KICK_Y)), 1.0);
      }
      return cmd(vx, vy, wz);
    }
    if (mem.state === 'approach' || mem.state === 'chase') {
      mem.set_state(dball > 0.6 ? 'chase' : 'approach', t);
      let vx, vy, wz;
      if (Math.abs(dphi) > 0.6) {
        if (Math.abs(dphi) > 2.6) {
          const room = [1.0, -1.0].map(sg => { const q = add(b, mul(n, sg * AROUND)); return Math.min(HY + 0.3 - Math.abs(q[1]), HX + 0.3 - Math.abs(q[0])); });
          dphi = room[0] >= room[1] ? Math.abs(dphi) : -Math.abs(dphi);
        }
        const a = phi_d + clip(dphi, -0.9, 0.9), rad = Math.max(AROUND, Math.min(dball, 0.5));
        const target = clipLim(add(b, [rad * Math.cos(a), rad * Math.sin(a)]), lim);
        [vx, vy, wz] = goTo(me, target, dball > 0.8 ? 1.1 : 0.9);
      } else {
        const dg = norm(sub([wm.dir * HX, 0.0], b));
        if (['shot', 'forward'].includes(mem.aim_kind) && DRIBBLE_D[0] < dg && dg < DRIBBLE_D[1] && opp_d > DRIBBLE_FREE
            && Math.abs(dphi) < 0.35 && dball < 0.3 && t > mem.dribble_block) {
          mem.set_state('dribble', t);
          return cmd();
        }
        const P = clipLim(sub(b, mul(k, BEHIND)), lim);
        let dd;
        [vx, vy, wz, dd] = goTo(me, P, dball > 0.8 ? 1.1 : 0.9, Math.atan2(k[1], k[0]));
        if (src === 'own' && (dd < 0.08 || (along < -0.12 && Math.abs(lat) < 0.07 && dball < BEHIND + 0.08))) { mem.set_state('align', t); mem.foot = null; }
      }
      return cmd(vx, vy, wz);
    }
    if (mem.state === 'dribble') {
      const goal = [wm.dir * HX, 0.0], dg = norm(sub(goal, b));
      if (dg < DRIBBLE_D[0] || opp_d < DUEL_D || t - mem.state_t > DRIBBLE_MAX_S || dball > 0.35 || Math.abs(dphi) > 1.0) {
        mem.dribble_block = t + 2.0;
        mem.set_state('approach', t);
        return cmd();
      }
      const kd = unit(sub(goal, b)), nd = [-kd[1], kd[0]], sg = rel[1] > 0 ? 1.0 : -1.0;
      const [vx, vy, wz] = goTo(me, sub(add(b, mul(kd, 0.15)), mul(nd, sg * KICK_Y)), 0.75);
      return cmd(vx, vy, wz);
    }
    if (mem.state === 'align') {
      const head = [0.0, FOOT_VIEW[1], FOOT_VIEW[0]];
      if (dball > BEHIND + 0.15 || along > 0.0) { mem.set_state('approach', t); return cmd(0, 0, 0, head); }
      if (opp_d < DUEL_D - 0.1 || t - mem.state_t > ALIGN_MAX_S) { mem.set_state('duel', t); return cmd(0, 0, 0, head); }
      if (mem.foot === null) {
        mem.foot = rel[1] > 0 ? 'left' : 'right';
        if (b[1] > HY - 0.15) mem.foot = wm.dir > 0 ? 'left' : 'right';
        else if (b[1] < -HY + 0.15) mem.foot = wm.dir > 0 ? 'right' : 'left';
      }
      const ty = mem.foot === 'left' ? KICK_Y : -KICK_Y;
      const ex = rel[0] - KICK_X, ey = rel[1] - ty;
      const fresh = src === 'own' && fresh_age < 0.25;
      let vx = 0.0, vy = 0.0, wz = 0.0;
      if (Math.abs(eh) > 0.3) { wz = copysign(0.65, eh); if (ex > 0.08) vx = 0.33; }
      else {
        if (Math.abs(ex) > TOL_X) vx = ex > 0 ? 0.33 : -0.4;
        if (Math.abs(ey) > TOL_Y) vy = ey > 0 ? 0.45 : -0.45;
        if (Math.abs(eh) > TOL_H) wz = copysign(0.6, eh);
      }
      mem.ok_t = (vx === 0 && vy === 0 && wz === 0 && fresh) ? mem.ok_t + DT : 0.0;
      if (mem.ok_t > 0.2) {
        mem.set_state('ready', t); mem.ok_t = 0.0;
        mem.kick_at = { foot: mem.foot, kind: mem.aim_kind, mate: mem.aim_mate };
      }
      return cmd(vx, vy, wz, head);
    }
    if (mem.state === 'ready') {
      const quick = (mem.kick_at || {}).quick;
      if (t - mem.state_t >= (quick ? QUICK_READY_S : READY_S)) {
        mem.set_state('kick', t);
        const ka = mem.kick_at || {};
        return new Cmd(0, 0, 0, { head: [0.0, 0.0, null], kick: mem.foot, kind: ka.kind || 'forward', mate: ka.mate ?? null, cam: true, state: 'kick' });
      }
      return cmd(0, 0, 0, [0.0, 0.0, null]);
    }
    mem.set_state('approach', t);
    return cmd();
  }
}
