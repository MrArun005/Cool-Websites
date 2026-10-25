// The hybrid ducks' simulation for the LIVE match page, in the viewer's browser (also runs in node for tests).
// A port of match.py's World / MultiBam / Duck and runtime_hybrid.py's HybridRuntime (+ match.MatchRuntime's split
// tick), with moves_hybrid.py's Hop and Keyframes. No DOM here: live.html draws, live_game.js plays.
//
//   physics   MuJoCo WASM (@mujoco/mujoco, Google DeepMind's official build), the model from export_live.py
//   servos    BAM XL330 "m6" per servo (Rhoban/bam MujocoController.update, vectorised like match.MultiBam:
//             firmware P loop -> duty -> volts, DC motor with back-EMF, load-dependent Stribeck friction written into
//             dof_frictionloss / dof_damping every 5 ms, supply voltage drop per robot, the friction-row index quirk)
//   brains    Microduck's ONNX policies (onnx_mlp.js), batched per brain (the caller supplies runBrains)
//   delays    3-6 physics step command delay per duck, observation noise, domain randomization per duck

export const TIMESTEP = 0.005, DECIMATION = 4, HZ = 50, DT = 0.02;
export const HOME_TRUNK_Z = 0.125;
const NOISE = [0.03, 0.03, 0.03, 0.01, 0.01, 0.01].concat(Array(14).fill(0.001), Array(14).fill(0.25), Array(14).fill(0)); // 48 (last_action: none)
const DELAY_LAG = [3, 6];
// runtime_hybrid.py
const LYING_COS = 0.6, HEAD_RATE = 2.0, DRIVE_ACC = [1.5, 1.0, 3.0], GETUP_SETTLE_S = 0.5, UNSTICK_AFTER_S = 2.0;
const UNSTICK_SEQ = [['lie_down', 'done'], ['roulade', 0.8], ['roulade', 1.5]];
const KICK_S = 0.5, HOP_LAND_S = 0.35;
const PLANS = {
  stand: [['velstand', null]], drive: [['velstand', null]], 'get up': [['velstand', null]],
  settle: [['velstand', 0.8], ['velstand', null]],
  hop: [['velstand', 0.4], ['hop', 'done'], ['velstand', null]],
  'kick left': [['kick_left', KICK_S], ['velstand', null]], 'kick right': [['kick_right', KICK_S], ['velstand', null]],
};
export const ONNX = { velstand: 'velstand', roulade: 'roulade', kick_left: 'ball_kick_left', kick_right: 'ball_kick_right' };

// ------------------------------------------------------------------------------------------------ small helpers
export class Rng {                       // mulberry32 + Box-Muller: seeded, so a match can be replayed
  constructor(seed) { this.s = (seed >>> 0) || 1; this.spare = null; }
  next() { let t = (this.s = (this.s + 0x6D2B79F5) >>> 0); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
  uniform(a = 0, b = 1) { return a + (b - a) * this.next(); }
  int(a, b) { return a + Math.floor(this.next() * (b - a + 1)); }       // inclusive
  normal(mu = 0, s = 1) {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return mu + s * v; }
    let u = 0, v = 0; while (u === 0) u = this.next(); v = this.next();
    const r = Math.sqrt(-2 * Math.log(u)); this.spare = r * Math.sin(2 * Math.PI * v); return mu + s * r * Math.cos(2 * Math.PI * v);
  }
  choice(a) { return a[Math.floor(this.next() * a.length)]; }
}
export const clip = (x, a, b) => Math.min(b, Math.max(a, x));
const smooth = s => { s = clip(s, 0, 1); return 0.5 - 0.5 * Math.cos(Math.PI * s); };
const interp = (x, xs, ys) => {
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[ys.length - 1];
};

// ------------------------------------------------------------------------------------------------ the world
export class World {
  // mj: the loaded MuJoCo module, xml: live_model_*.xml, meta: live_meta_*.json
  constructor(mj, xml, meta, { seed = 0, randomize = true } = {}) {
    this.mj = mj; this.meta = meta; this.seed = seed;
    this.m = mj.MjModel.from_xml_string(xml);
    this.m.opt.timestep = TIMESTEP;
    this.d = new mj.MjData(this.m);
    this.DP = Float64Array.from(meta.default_pose);
    this.n = meta.robots.length;
    const B = meta.bam;
    this.bam = B;
    this.rng = new Rng(seed * 7919 + 17);
    // MultiBam's index tables
    const N = 14 * this.n;
    this.act = new Int32Array(N); this.qadr = new Int32Array(N); this.dof = new Int32Array(N); this.quirk = new Int32Array(N);
    meta.robots.forEach((r, k) => { for (let j = 0; j < 14; j++) { const i = 14 * k + j; this.act[i] = r.act[j]; this.qadr[i] = r.qadr[j]; this.dof[i] = r.dof[j]; this.quirk[i] = r.quirk[j]; } });
    this.qTarget = new Float64Array(N); this.prev = new Float64Array(N);
    this.kp = new Float64Array(this.n).fill(B.kp_fw); this.vin = new Float64Array(this.n).fill(B.vin);
    this.fr = new Float64Array(this.m.nv);
    const arm = this.m.dof_armature; for (let i = 0; i < N; i++) arm[this.dof[i]] = B.armature;
    this.mass0 = Float64Array.from(this.m.body_mass); this.inertia0 = Float64Array.from(this.m.body_inertia);
    this.ipos0 = Float64Array.from(this.m.body_ipos); this.fric0 = Float64Array.from(this.m.geom_friction);
    mj.mj_setConst(this.m, this.d);
    this.ducks = meta.robots.map((r, i) => new Duck(this, i, r));
    this.bq = meta.ball_q; this.bv = meta.ball_v; this.ballGeom = meta.ball_geom;
    this.geomOwner = new Int16Array(this.m.ngeom).fill(-1);
    meta.robots.forEach((r, i) => { for (let g = r.geom0; g <= r.geom1; g++) this.geomOwner[g] = i; });
    if (randomize) this.randomize();
    this.ducks.forEach((dk, i) => this.put(dk, -2.0 + 0.4 * i, 0.0, 0.0));
    mj.mj_forward(this.m, this.d);
    this.touchNow = new Set();
    this.stepMs = 0; this.bamMs = 0;
  }

  // harness.MicroduckSim.randomize_body per duck (match.World.randomize)
  randomize() {
    const m = this.m, r = this.rng;
    const mass = m.body_mass, inertia = m.body_inertia, ipos = m.body_ipos, fric = m.geom_friction;
    this.ducks.forEach((dk, k) => {
      let sum = 0;
      for (const b of dk.meta.bodies) {
        const s = r.uniform(0.95, 1.05); sum += s;
        mass[b] = this.mass0[b] * s;
        for (let a = 0; a < 3; a++) { inertia[3 * b + a] = this.inertia0[3 * b + a] * s; ipos[3 * b + a] = this.ipos0[3 * b + a]; }
      }
      for (const b of [dk.meta.trunk, dk.meta.jaw]) for (let a = 0; a < 3; a++) ipos[3 * b + a] += r.uniform(-0.003, 0.003);
      const f = r.uniform(0.7, 1.3);
      for (let g = dk.meta.geom0; g <= dk.meta.geom1; g++) fric[3 * g] = this.fric0[3 * g] * f;
      this.kp[k] = this.bam.kp_fw * r.uniform(0.85, 1.15);
      this.vin[k] = r.uniform(6.5, 8.2);
      dk.dr = { mass: +(sum / dk.meta.bodies.length).toFixed(3), friction: +f.toFixed(3), kp: +this.kp[k].toFixed(1), vin: +this.vin[k].toFixed(2), lag: dk.lag };
    });
    this.mj.mj_setConst(m, this.d);
  }

  put(duck, x, y, yaw, joints = null) {
    const d = this.d, qpos = d.qpos, qvel = d.qvel, fq = duck.meta.free_q, fv = duck.meta.free_v;
    const q = joints || this.DP;
    qpos.set([x, y, HOME_TRUNK_Z, Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], fq);
    for (let a = 0; a < 6; a++) qvel[fv + a] = 0;
    for (let j = 0; j < 14; j++) { qpos[duck.meta.qadr[j]] = q[j]; qvel[duck.meta.dof[j]] = 0; this.qTarget[14 * duck.i + j] = q[j]; }
    duck.pending.set(q); duck.hist = [];
  }

  // BAM MujocoController.update() for all 14 x n servos (match.MultiBam.update)
  bamUpdate() {
    const B = this.bam, m = this.m, d = this.d, N = 14 * this.n;
    const qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl, bias = d.qfrc_bias, cons = d.qfrc_constraint, qact = d.qfrc_actuator;
    const kt = B.kt, R = B.R, eg = B.error_gain, mp = B.max_pwm;
    for (let k = 0; k < this.n; k++) {
      let load = 0; for (let j = 0; j < 14; j++) load += Math.abs(this.prev[14 * k + j]);
      const vin = Math.max(this.vin[k] - B.vin_drop_gain * load, B.vin_min), kp = this.kp[k];
      for (let j = 0; j < 14; j++) {
        const i = 14 * k + j, dq = qvel[this.dof[i]];
        const duty = clip((this.qTarget[i] - qpos[this.qadr[i]]) * kp * eg, -mp, mp);   // max_current = None
        const torque = kt * (vin * duty) / R - kt * kt * dq / R;
        this.prev[i] = torque; ctrl[this.act[i]] = torque;
      }
    }
    // friction rows (efc type FRICTION_DOF) summed per dof; BAM reads them at the servo's *joint* index (quirk)
    const fr = this.fr; fr.fill(0);
    const nefc = d.nefc;
    if (nefc > 0) {
      const et = d.efc_type, eid = d.efc_id, ef = d.efc_force, FD = this.meta.frictiondof;
      for (let e = 0; e < nefc; e++) if (et[e] === FD) fr[eid[e]] += ef[e];
    }
    const fl = m.dof_frictionloss, damp = m.dof_damping;
    const lfe = B.load_friction_external, lfm = B.load_friction_motor, lfes = B.load_friction_external_stribeck,
      lfms = B.load_friction_motor_stribeck, lfeq = B.load_friction_external_quad, lfmq = B.load_friction_motor_quad;
    for (let i = 0; i < N; i++) {
      const v = this.dof[i], dq = qvel[v];
      const ext = -bias[v] + cons[v] - fr[this.quirk[i]], mot = qact[v];
      const gb = Math.abs(ext * lfe - mot * lfm), gbs = Math.abs(ext * lfes - mot * lfms);
      const st = Math.exp(-Math.pow(Math.abs(dq / B.dtheta_stribeck), B.alpha));
      let f = B.friction_base + gb + st * B.friction_stribeck + gbs * st;
      const ae = Math.abs(ext), am = Math.abs(mot);
      if (Math.sign(ext) !== Math.sign(mot)) f += st * ((ae < am ? lfeq * ae * ae : 0) + (ae > am ? lfmq * am * am : 0));
      fl[v] = f; damp[v] = B.friction_viscous;
    }
  }

  // 4 physics steps of 5 ms with every robot's latest targets through its own command delay (match.World.physics)
  physics(watchBall = false) {
    const mj = this.mj;
    for (let s = 0; s < DECIMATION; s++) {
      for (const dk of this.ducks) {
        dk.hist.push(dk.pending);
        if (dk.hist.length > dk.lag + 1) dk.hist.shift();
        this.qTarget.set(dk.hist[0], 14 * dk.i);
      }
      const t0 = performance.now();
      this.bamUpdate();
      const t1 = performance.now();
      mj.mj_step(this.m, this.d);
      const t2 = performance.now();
      this.bamMs += t1 - t0; this.stepMs += t2 - t1;
      if (watchBall) this.touches();
    }
  }

  ball() { const q = this.d.qpos, v = this.d.qvel, a = this.bq, b = this.bv; return [[q[a], q[a + 1], q[a + 2]], [v[b], v[b + 1], v[b + 2]]]; }
  setBall(x, y, vx = 0, vy = 0) {
    this.d.qpos.set([x, y, this.meta.pitch.ball_r, 1, 0, 0, 0], this.bq);
    this.d.qvel.set([vx, vy, 0, 0, 0, 0], this.bv);
  }
  // ducks touching the ball (contacts of the ball geom); only called when a duck is near the ball (data.contact copies)
  touches() {
    const d = this.d, n = d.ncon;
    if (!n) return;
    const cs = d.contact, bg = this.ballGeom;
    for (let c = 0; c < n; c++) {
      const ct = cs.get(c);
      const g1 = ct.geom1, g2 = ct.geom2; ct.delete && ct.delete();
      if (g1 === bg || g2 === bg) { const o = this.geomOwner[g1 === bg ? g2 : g1]; if (o >= 0) this.touchNow.add(o); }
    }
    cs.delete();
  }
}

// ------------------------------------------------------------------------------------------------ one duck
export class Duck {
  constructor(world, i, meta) {
    this.world = world; this.i = i; this.meta = meta; this.name = meta.name; this.team = meta.team;
    this.rng = new Rng(world.seed * 100 + i + 1);
    this.obsNoise = true; this.delay = true;
    this.pending = Float64Array.from(world.DP);
    this.lastAction = new Float32Array(14); this.command = new Float32Array(13);
    this.hist = []; this.dr = {};
    this.resetState();
  }
  resetState() {
    this.lastAction.fill(0); this.command.fill(0);
    this.lag = this.delay ? this.rng.int(DELAY_LAG[0], DELAY_LAG[1]) : 0;
    this.hist = []; this.pending = Float64Array.from(this.world.DP);
    this.dr.lag = this.lag;
  }
  xmat() { const x = this.world.d.xmat, o = 9 * this.meta.trunk; return x.subarray(o, o + 9); }
  gravity() { const R = this.xmat(); return [-R[6], -R[7], -R[8]]; }
  upZ() { return this.xmat()[8]; }
  tiltDeg() { return Math.acos(clip(this.upZ(), -1, 1)) * 180 / Math.PI; }
  trunkZ() { return this.world.d.xpos[3 * this.meta.trunk + 2]; }
  gyro() { const s = this.world.d.sensordata, a = this.meta.gyro; return [s[a], s[a + 1], s[a + 2]]; }
  pose() { const p = this.world.d.xpos, o = 3 * this.meta.trunk, R = this.xmat(); return [[p[o], p[o + 1]], Math.atan2(R[3], R[0])]; }
  obs(out) {                                   // the 61-D observation (harness.MicroduckSim.obs), into out (Float32Array)
    const d = this.world.d, qpos = d.qpos, qvel = d.qvel, DP = this.world.DP, g = this.gyro(), gr = this.gravity();
    out[0] = g[0]; out[1] = g[1]; out[2] = g[2]; out[3] = gr[0]; out[4] = gr[1]; out[5] = gr[2];
    for (let j = 0; j < 14; j++) { out[6 + j] = qpos[this.meta.qadr[j]] - DP[j]; out[20 + j] = qvel[this.meta.dof[j]]; out[34 + j] = this.lastAction[j]; }
    for (let j = 0; j < 13; j++) out[48 + j] = this.command[j];
    if (this.obsNoise) for (let k = 0; k < 48; k++) out[k] += this.rng.uniform(-1, 1) * NOISE[k];
    return out;
  }
  setAction(a) {                               // the brain's action for this tick -> servo targets (after the delay)
    const DP = this.world.DP;
    for (let j = 0; j < 14; j++) { this.lastAction[j] = a[j]; }
    const p = new Float64Array(14); for (let j = 0; j < 14; j++) p[j] = DP[j] + this.lastAction[j];
    this.pending = p;
  }
}

// ------------------------------------------------------------------------------------------------ scripted moves
const L_HIP = 2, L_KNEE = 3, L_ANK = 4, R_HIP = 11, R_KNEE = 12, R_ANK = 13, NECK = 5, HEAD_ROLL = 8;
const LEG_IK_H = [-0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06];
const LEG_IK = [[-0.3913, -0.7696, -0.3783], [-0.1434, -0.2799, -0.1365], [0.0, 0.0, 0.0], [0.115, 0.222, 0.1071],
  [0.2153, 0.4137, 0.1983], [0.3069, 0.586, 0.2791], [0.3926, 0.7449, 0.3523], [0.4748, 0.8939, 0.4191],
  [0.5548, 1.0353, 0.4804], [0.6343, 1.1707, 0.5364], [0.7149, 1.3013, 0.5865], [0.7987, 1.4281, 0.6294],
  [0.8894, 1.5515, 0.6621], [0.9936, 1.672, 0.6784], [1.1265, 1.7893, 0.6628]];
const ik = h => { h = clip(h, LEG_IK_H[0], LEG_IK_H[LEG_IK_H.length - 1]); return [0, 1, 2].map(i => interp(h, LEG_IK_H, LEG_IK.map(r => r[i]))); };
function legs(q, h, dank = 0, dhip = 0, dknee = 0) {
  let [dh, dk, da] = ik(h); dh += dhip; dk += dknee; da += dank;
  q[L_HIP] += dh; q[L_KNEE] += dk; q[L_ANK] += da; q[R_HIP] -= dh; q[R_KNEE] -= dk; q[R_ANK] -= da; return q;
}
const pitchOf = obs => Math.atan2(obs[3], -obs[5]);
const balance = (obs, kp = 3.0, kd = 0.1) => kp * pitchOf(obs) + kd * obs[1];

class Script {
  constructor() { this.rt = null; this.k = 0; this.quiet = false; }
  begin(rt) { this.rt = rt; this.k = 0; const DP = rt.sim.world.DP; this.q0 = Float64Array.from(DP, (v, j) => v + rt.sim.lastAction[j]); }
  get t() { return this.k * 0.02; }
  head(q) { if (this.rt && !this.quiet) for (let j = 0; j < 4; j++) q[NECK + j] += this.rt.headCmd[j]; return q; }
  act(obs) { const q = this.target(obs, this.t); this.k++; const DP = this.rt.sim.world.DP; return Float32Array.from(q, (v, j) => v - DP[j]); }
}
class Hop extends Script {           // moves_hybrid.Hop (parameters: moves_hybrid.json "hop", Tland HOP_LAND_S)
  constructor(P) { super(); this.P = P; this.startH = 0; this.tRel = null; this.quiet = true; }
  begin(rt) { super.begin(rt); this.tRel = null; }
  get duration() { return this.P.Tc + this.P.Tp_max + this.P.Tland; }
  done() { return this.tRel !== null && this.t >= this.tRel + this.P.Tland; }
  target(obs, t) {
    const P = this.P, bal = balance(obs, P.kp, P.kd), q = Float64Array.from(this.rt.sim.world.DP);
    if (t < P.Tc) { const s = smooth(t / P.Tc); q[NECK] += P.neck_c * s; return legs(q, this.startH + (P.hc - this.startH) * s, bal + P.lean * s); }
    const knee = obs[6 + L_KNEE];
    if (this.tRel === null && (knee < P.release || t >= P.Tc + P.Tp_max)) this.tRel = t;
    if (this.tRel === null) { q[NECK] += P.neck_p; return legs(q, -0.01, P.push_ank + bal, P.push_hip); }
    const tr = t - this.tRel;
    if (tr < P.Ta) { q[NECK] += P.neck_p; return legs(q, P.h_air, P.air_ank); }
    const s = smooth((tr - P.Ta) / 0.1); q[NECK] += P.neck_p * (1 - s);
    return legs(q, P.h_air + (P.ht - P.h_air) * s, bal);
  }
}
class Keyframes extends Script {     // moves_hybrid.Keyframes (the lie-down keys: the first unstick step)
  constructor(K) { super(); this.keys = K.keys; this.durs = K.durations; this.bal = K.bal || K.keys.map(() => 0); this.quiet = !(K.head ?? true); }
  get duration() { return this.durs.reduce((a, b) => a + b, 0); }
  target(obs, t) {
    let prev = this.q0;
    for (let i = 0; i < this.keys.length; i++) {
      const key = this.keys[i], dur = this.durs[i];
      if (t < dur || i === this.keys.length - 1) {
        const s = smooth(t / dur), q = Float64Array.from(prev, (v, j) => v + (key[j] - v) * s);
        const b = this.bal[i] * balance(obs); q[L_ANK] += b; q[R_ANK] -= b;
        return this.head(q);
      }
      t -= dur; prev = key;
    }
    return Float64Array.from(this.keys[this.keys.length - 1]);
  }
}

// ------------------------------------------------------------------------------------------------ the runtime
// runtime_hybrid.HybridRuntime for what the match uses (stand, drive, get up + unstick, kicks, celebration hop), split
// like match.MatchRuntime: brainInput() (personality, command, which brain) / setAction() / after().
export class Runtime {
  constructor(sim, moves, seed = 0) {
    this.sim = sim; this.moves = moves; this.rng = new Rng(seed + 3);
    this.steps = 0; this.mode = 'stand'; this.skill = null; this.plan = []; this.planT = 0; this.cur = null;
    this.vel = [0, 0, 0]; this.velGoal = [0, 0, 0]; this.headCmd = [0, 0, 0, 0]; this.headGoal = [0, 0, 0, 0];
    this.falls = 0; this.upTicks = 0; this.downTicks = 0; this.unsticks = 0; this.unstickK = 0;
    this.events = [];
    this.reset();
  }
  reset() {
    this.sim.resetState();
    this.steps = 0; this.headCmd = [0, 0, 0, 0]; this.headGoal = [0, 0, 0, 0];
    this.mode = 'stand'; this.plan = [['velstand', null]]; this.vel = [0, 0, 0]; this.velGoal = [0, 0, 0];
    this.start('velstand');
  }
  log(msg) { this.events.push([+(this.steps * DT).toFixed(2), msg]); if (this.events.length > 40) this.events.shift(); }
  lying() { return this.sim.upZ() < LYING_COS; }
  command(name) {
    const prev = this.mode;
    if (this.lying() && name !== 'lie down') name = 'get up';
    if (['drive'].includes(prev) && name === 'hop') { this.vel = [0, 0, 0]; return this.chain('settle', name); }
    if (!PLANS[name]) return this.log(`unknown command ${name}`);
    this.mode = name;
    this.vel = [0, 0, 0];
    if (name === 'drive') this.velGoal = [0, 0, 0];
    this.headGoal = [0, 0, 0, 0];
    this.plan = PLANS[name].map(p => p.slice());
    this.next();
  }
  chain(first, then) { this.mode = first; this.plan = PLANS[first].map(p => p.slice()); this.plan[this.plan.length - 1] = ['then:' + then, null]; this.next(); }
  drive(vx, vy = 0, wz = 0) {
    this.velGoal = [vx, vy, wz];
    if (this.mode === 'drive' && this.skill === 'velstand') return;
    if (this.mode === 'get up' && (this.lying() || this.upTicks < GETUP_SETTLE_S * HZ)) return;
    if (['hop', 'kick left', 'kick right', 'settle'].includes(this.mode) &&
      (this.plan.length > 1 || this.skill !== 'velstand' || this.planT < GETUP_SETTLE_S * HZ)) return;
    this.mode = 'drive'; this.vel = [0, 0, 0]; this.velGoal = [vx, vy, wz]; this.headGoal = [0, 0, 0, 0];
    this.plan = [['velstand', null]]; this.next();
  }
  look(yaw = 0, pitch = 0, neck = null) {
    if (neck === null) this.headGoal = [0, clip(pitch, -0.6, 0.6), clip(yaw, -1.2, 1.2), 0];
    else this.headGoal = [clip(neck, -1.1, 1.1), clip(pitch, -1.1, 1.1), clip(yaw, -1.2, 1.2), 0];
  }
  brain(skill) {
    if (ONNX[skill]) return { onnx: ONNX[skill] };
    if (skill === 'hop') return new Hop(Object.assign({}, this.moves.hop.params, { Tland: HOP_LAND_S }));
    if (skill === 'lie_down') return new Keyframes(this.moves.lie_down);
    return null;
  }
  next() {
    const [skill, end] = this.plan[0];
    if (skill.startsWith('then:')) return this.command(skill.slice(5));
    this.start(skill, end);
  }
  start(skill) {
    let b = this.brain(skill);
    if (!b) { skill = 'velstand'; b = this.brain(skill); this.plan = [['velstand', null]]; }
    if (b.begin) b.begin(this);
    this.skill = skill; this.cur = b; this.planT = 0;
    this.log(`${this.mode}: ${skill}`);
  }
  stepDone(end) {
    if (end === null) return false;
    if (end === 'done') return (this.cur.done ? this.cur.done() : false) || this.planT * DT >= (this.cur.duration || 0);
    return this.planT * DT >= end;
  }
  commandVector() {
    const c = this.sim.command; c.fill(0);
    if (this.skill === 'velstand') {
      if (this.mode === 'drive') for (let a = 0; a < 3; a++) this.vel[a] += clip(this.velGoal[a] - this.vel[a], -DRIVE_ACC[a] * DT, DRIVE_ACC[a] * DT);
      c[0] = this.vel[0]; c[1] = this.vel[1]; c[2] = this.vel[2];
      for (let j = 0; j < 4; j++) c[3 + j] = this.headCmd[j];
    }
    return c;   // kicks, roulade and scripts: all zero (as trained)
  }
  personality() {
    const goal = this.headGoal.slice();
    const quiet = (this.cur && this.cur.quiet) || this.skill === 'roulade' || ['get up', 'hop', 'kick left', 'kick right'].includes(this.mode) || this.lying();
    if (quiet) goal.fill(0);
    const st = HEAD_RATE * DT;
    for (let j = 0; j < 4; j++) this.headCmd[j] += clip(goal[j] - this.headCmd[j], -st, st);
    this.sim.world.d.ctrl[this.sim.meta.mouth] = 0;     // the beak stays shut (nobody talks in the match)
  }
  // tick, part 1: returns {onnx: name} (obs already in obsOut) or applies a scripted action itself
  brainInput(obsOut) {
    this.personality();
    this.commandVector();
    this.sim.obs(obsOut);
    if (this.cur.onnx) return this.cur.onnx;
    this.sim.setAction(this.cur.act(obsOut));
    return null;
  }
  // tick, part 3 (after the physics): match.MatchRuntime.tick_after
  after() {
    this.steps++; this.planT++;
    const up = this.sim.tiltDeg() < 25;
    this.upTicks = up ? this.upTicks + 1 : 0; this.downTicks = up ? 0 : this.downTicks + 1;
    if (this.upTicks > HZ) this.unstickK = 0;
    if (this.lying() && !['lie down', 'roll', 'get up'].includes(this.mode)) {
      this.log('fell over: getting up'); this.falls++;
      this.vel = [0, 0, 0]; this.velGoal = [0, 0, 0];
      this.mode = 'get up'; this.plan = [['velstand', null]]; this.start('velstand');
      return;
    }
    if (this.mode === 'get up' && this.skill === 'velstand' && Math.min(this.downTicks, this.planT) > UNSTICK_AFTER_S * HZ) {
      const step = UNSTICK_SEQ[this.unstickK % UNSTICK_SEQ.length];
      this.log(`stuck getting up: ${step[0]} ${step[1]}, then get up`);
      this.unsticks++; this.unstickK++; this.downTicks = 0;
      this.plan = [step.slice(), ['velstand', null]]; this.start(step[0]);
      return;
    }
    if (this.plan.length && this.stepDone(this.plan[0][1])) { this.plan.shift(); if (this.plan.length) this.next(); }
  }
}

// ------------------------------------------------------------------------------------------------ one 50 Hz tick
// The brains of every duck, batched per ONNX brain. runBrains(name, Float32Array N*61, N) -> Float32Array N*14
// (synchronous: onnx_mlp.js MlpPolicy.runBatch)
export function brainsTick(rts, runBrains, scratch) {
  const groups = {};
  rts.forEach((rt, i) => {
    const o = scratch.subarray(61 * i, 61 * i + 61);
    const name = rt.brainInput(o);
    if (name) (groups[name] = groups[name] || []).push(i);
  });
  for (const n of Object.keys(groups)) {
    const idx = groups[n], X = new Float32Array(61 * idx.length);
    idx.forEach((i, k) => X.set(scratch.subarray(61 * i, 61 * i + 61), 61 * k));
    const Y = runBrains(n, X, idx.length);
    idx.forEach((i, k) => rts[i].sim.setAction(Y.subarray(14 * k, 14 * k + 14)));
  }
}
