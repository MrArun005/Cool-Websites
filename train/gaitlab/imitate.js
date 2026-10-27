// Bolt moving like a real dog in the browser: the dog-imitation brain (train/imitate_env.py) on the browser
// physics (live.js Bolt). A port of ImitateEnv's control loop: a clip clock, the brain's inputs exactly as
// ImitateEnv._obs, and motor targets clip(q_ref(next step) + ACTION_SCALE * action). Checked against Python
// step by step with imitate_test.mjs. The brain and clips come from train/export_imitate.py
// (models/<name>/brain.json and clips.json).
//
//   const dog = new Imitate(bolt, brainSpec, clipsJson);
//   dog.play("walk");   // a gait joins at the frame whose pose is closest to Bolt's; sit and lie play from their start
//   every 1/50 s: dog.tick();
//   dog.stop();         // hand the motors back (tick() then does nothing)
import { Brain, floats } from "./live.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
// tracking() and fell() read what the reward and early termination read in ImitateEnv; the brain needs neither
const FIELDS = { joints: 12, height: 1, v_body: 3, w_body: 3, joint_vel: 12, tilt: 9, gravity: 3, paws: 12 };

// A retargeted clip (imitate_env.Clip): per-frame body-relative quantities, blended between frames.
export class Clip {
  constructor(name, spec) {
    Object.assign(this, { name, fps: spec.fps, loop: spec.loop, n: spec.n, length: spec.length });
    this.f = {};
    for (const [k, w] of Object.entries(FIELDS)) if (spec[k]) {
      const a = floats(spec[k]);
      this.f[k] = Array.from({ length: this.n }, (_, t) => Array.from(a.subarray(t * w, (t + 1) * w)));
    }
  }
  // [index, next index, blend] at clip time tau (s): loops wrap, sit/lie hold their last frame
  frame(tau) {
    let u = tau * this.fps;
    if (this.loop) {
      u = ((u % this.n) + this.n) % this.n;   // Python's % (never negative)
      const i = Math.floor(u);
      return [i, (i + 1) % this.n, u - i];
    }
    u = Math.min(Math.max(u, 0), this.n - 1);
    const i = Math.floor(u);
    return [i, Math.min(i + 1, this.n - 1), u - i];
  }
  at(tau, key) {
    const [i, j, b] = this.frame(tau), a = this.f[key];
    return a[i].map((v, k) => (1 - b) * v + b * a[j][k]);
  }
  phase(tau) { return this.loop ? (((tau / this.length) % 1) + 1) % 1 : Math.min(tau / this.length, 1); }
  // the frame whose joint angles are closest to `q` (ImitateEnv._maybe_switch)
  closest(q) {
    let best = 0, bestE = Infinity;
    this.f.joints.forEach((row, t) => {
      let e = 0;
      for (let i = 0; i < 12; i++) e += (row[i] - q[i]) ** 2;
      if (e < bestE) { bestE = e; best = t; }
    });
    return best;
  }
}

// 3x3 helpers (row-major arrays of 9)
const mul3 = (A, B) => Array.from({ length: 9 }, (_, k) => A[3 * (k / 3 | 0)] * B[k % 3] + A[3 * (k / 3 | 0) + 1] * B[3 + k % 3] + A[3 * (k / 3 | 0) + 2] * B[6 + k % 3]);
const rotz = (a) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
// the nearest rotation to M (its polar factor, = u @ vt of the SVD), by Newton's iteration R <- (R + R^-T) / 2
function orthonormalize(M) {
  let R = M.slice();
  for (let it = 0; it < 30; it++) {
    const [a, b, c, d, e, f, g, h, i] = R, det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const invT = [(e * i - f * h) / det, -(d * i - f * g) / det, (d * h - e * g) / det, -(b * i - c * h) / det, (a * i - c * g) / det,
      -(a * h - b * g) / det, (b * f - c * e) / det, -(a * f - c * d) / det, (a * e - b * d) / det];
    R = R.map((v, k) => 0.5 * (v + invT[k]));
  }
  return R;
}
function mat2quat(R) {   // (w, x, y, z), w >= 0
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = R, tr = m00 + m11 + m22;
  let q;
  if (tr > 0) { const s = 2 * Math.sqrt(1 + tr); q = [s / 4, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s]; }
  else if (m00 > m11 && m00 > m22) { const s = 2 * Math.sqrt(1 + m00 - m11 - m22); q = [(m21 - m12) / s, s / 4, (m01 + m10) / s, (m02 + m20) / s]; }
  else if (m11 > m22) { const s = 2 * Math.sqrt(1 + m11 - m00 - m22); q = [(m02 - m20) / s, (m01 + m10) / s, s / 4, (m12 + m21) / s]; }
  else { const s = 2 * Math.sqrt(1 + m22 - m00 - m11); q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, s / 4]; }
  const n = Math.hypot(...q) * (q[0] < 0 ? -1 : 1);
  return q.map((v) => v / n);
}

export class Imitate {
  // bolt: a live.js Bolt (its model, data, motor model and joint limits); brain: brain.json; clips: clips.json
  constructor(bolt, brain, clips) {
    this.bolt = bolt; this.c = clips.constants;
    this.brain = brain instanceof Brain ? brain : new Brain(brain);
    this.slots = this.brain.inputs - 66;   // 8: a brain from before the getup clip
    this.clips = {};
    for (const [name, spec] of Object.entries(clips.clips)) this.clips[name] = new Clip(name, spec);
    this.dt = this.c.dt;
    const { mj, model: m } = bolt;
    this.feet = this.c.feet.map((f) => mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, f));
    this.bodyGeoms = new Set();   // torso, head, snout (ImitateEnv.body_geoms: not a leg, not the floor)
    for (let g = 0; g < m.ngeom; g++) if (m.geom_bodyid[g] === bolt.torso) this.bodyGeoms.add(g);
    this.clip = null; this.t = 0; this.tau0 = 0; this.lastAction = new Float64Array(12);
  }
  playing() { return this.clip !== null; }
  tau(ahead = 0) { return this.tau0 + (this.t + ahead) * this.dt; }
  ref(key, ahead = 0) { return this.clip.at(this.tau(ahead), key); }
  // seconds since a sit/lie clip ended (0 while it plays, always 0 for loops)
  finished() { return this.clip.loop ? 0 : Math.max(0, this.tau() - this.clip.length); }
  joints() { return Array.from(this.bolt.data.qpos.slice(7, 19)); }
  // Start or switch clip, as ImitateEnv._maybe_switch: a gait joins at the frame whose pose is closest to Bolt's
  // now; sit and lie play from their start. The previous action carries over (from stopped: zeros, as reset).
  play(name) {
    const c = this.clips[name];
    if (!c) throw new Error(`clip ${name} unknown to this brain (it knows ${Object.keys(this.clips).join(", ")})`);
    if (!this.clip) { this.t = 0; this.lastAction = new Float64Array(12); }
    const start = c.loop ? c.closest(this.joints()) / c.fps : 0;
    this.clip = c; this.tau0 = start - this.t * this.dt;
  }
  stop() { this.clip = null; }
  // Put Bolt inside a clip at time tau, moving as the dog was, facing `yaw` at (x, y) (ImitateEnv.reset without
  // randomization: reference state initialization). Then tick() as usual.
  spawn(name, tau = 0, yaw = 0, x = 0, y = 0) {
    const { mj, model: m, data: d, lo, hi } = this.bolt;
    mj.mj_resetData(m, d);
    this.clip = this.clips[name]; this.tau0 = tau; this.t = 0; this.lastAction = new Float64Array(12);
    const Rw = mul3(rotz(yaw), orthonormalize(this.ref("tilt")));
    d.qpos.set([x, y, this.ref("height")[0] + 0.005, ...mat2quat(Rw)], 0);
    const q = this.ref("joints").map((v, i) => clamp(v, lo[i], hi[i]));
    d.qpos.set(q, 7);
    const moving = this.finished() > 0 ? 0 : 1, v = this.ref("v_body");
    d.qvel.set([0, 1, 2].map((r) => (Rw[3 * r] * v[0] + Rw[3 * r + 1] * v[1] + Rw[3 * r + 2] * v[2]) * moving), 0);
    d.qvel.set(this.ref("w_body").map((w) => w * moving), 3);
    d.qvel.set(this.ref("joint_vel").map((w) => w * moving), 6);
    d.ctrl.set(q);
    mj.mj_forward(m, d);
    let pen = 0;   // never start with a paw or the body in the floor: lift by the deepest penetration
    const cs = d.contact;
    for (let k = 0; k < d.ncon; k++) { const ct = cs.get(k); pen = Math.max(pen, -ct.dist); ct.delete(); }
    cs.delete();
    if (pen > 0) { d.qpos[2] += pen + 0.002; mj.mj_forward(m, d); }
  }
  // the 66 + slots inputs (ImitateEnv._obs with randomize=False)
  obs() {
    const d = this.bolt.data, R = this.bolt.rotation(), q = d.qpos, v = d.qvel, D = this.c.dog_pose;
    const o = [-R[6], -R[7], -R[8], v[3] * 0.25, v[4] * 0.25, v[5] * 0.25];
    for (let j = 0; j < 3; j++) o.push(R[j] * v[0] + R[3 + j] * v[1] + R[6 + j] * v[2]);
    o.push(q[2]);
    for (let i = 0; i < 12; i++) o.push(q[7 + i] - D[i]);
    for (let i = 0; i < 12; i++) o.push(v[6 + i] * 0.1);
    o.push(...this.lastAction);
    const ph = 2 * Math.PI * this.clip.phase(this.tau());
    o.push(Math.sin(ph), Math.cos(ph));
    const hot = this.c.all_clips.indexOf(this.clip.name);
    for (let k = 0; k < this.slots; k++) o.push(k === hot ? 1 : 0);
    o.push(...this.ref("joints", 1).map((x, i) => x - D[i]), this.ref("height", 1)[0], ...this.ref("v_body", 1), this.ref("w_body", 1)[2]);
    o.push(this.finished());
    return o;
  }
  // one control step (1/50 s): the clip's next pose plus the brain's correction -> 10 physics steps
  step(action) {
    const b = this.bolt, a = Array.from(action, (x) => clamp(x, -1, 1)), S = this.c.action_scale;
    b.data.ctrl.set(this.ref("joints", 1).map((x, i) => clamp(x + S[i] * a[i], b.lo[i], b.hi[i])));
    b.applyPush(b.dt);
    b.physics(this.c.frame_skip);
    this.t += 1;
    this.lastAction = Float64Array.from(a);
  }
  tick() { if (this.clip) this.step(this.brain.act(this.obs())); }
  // ---------- how well it follows the dog (ImitateEnv.tracking and its early termination), for tests ----------
  headingPaws() {
    const d = this.bolt.data, R = this.bolt.rotation(), yaw = Math.atan2(R[3], R[0]), c = Math.cos(yaw), s = Math.sin(yaw);
    return this.feet.flatMap((g) => {
      const x = d.geom_xpos[3 * g] - d.qpos[0], y = d.geom_xpos[3 * g + 1] - d.qpos[1], z = d.geom_xpos[3 * g + 2] - d.qpos[2];
      return [c * x + s * y, -s * x + c * y, z];   // rel @ Rz(yaw)
    });
  }
  tracking() {
    const d = this.bolt.data, R = this.bolt.rotation(), q = d.qpos, v = d.qvel;
    const sq = (a, b) => a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0);
    const ePose = sq(this.ref("joints"), this.joints()), eVel = sq(this.ref("joint_vel"), Array.from(v.slice(6, 18)));
    const ePaw = sq(this.ref("paws"), this.headingPaws()), eH = this.ref("height")[0] - q[2];
    const g = [-R[6], -R[7], -R[8]], gr = this.ref("gravity"), n = Math.hypot(...gr);
    const tilt = Math.acos(clamp((g[0] * gr[0] + g[1] * gr[1] + g[2] * gr[2]) / n, -1, 1));
    const vb = [0, 1, 2].map((j) => R[j] * v[0] + R[3 + j] * v[1] + R[6 + j] * v[2]);
    const eV = sq(this.ref("v_body"), vb), eW = sq(this.ref("w_body"), Array.from(v.slice(3, 6)));
    const terms = { pose: Math.exp(-5 * ePose), vel: Math.exp(-0.1 * eVel), paw: Math.exp(-40 * ePaw),
      root: Math.exp(-20 * eH * eH - 10 * tilt * tilt), root_vel: Math.exp(-2 * eV - 0.2 * eW) };
    const reward = 0.5 * terms.pose + 0.05 * terms.vel + 0.2 * terms.paw + 0.15 * terms.root + 0.1 * terms.root_vel;
    return { reward, terms, heightErr: eH, tilt, poseRms: Math.sqrt(ePose / 12) };
  }
  bodyOnFloor() {
    const d = this.bolt.data, cs = d.contact, fl = this.bolt.floor;
    let hit = false;
    for (let k = 0; k < d.ncon && !hit; k++) {
      const c = cs.get(k);
      hit = (c.geom1 === fl && this.bodyGeoms.has(c.geom2)) || (c.geom2 === fl && this.bodyGeoms.has(c.geom1));
      c.delete();
    }
    cs.delete(); return hit;
  }
  // lost the motion as ImitateEnv.step judges it: 15 cm off the clip's height, tilted 0.8 rad from it, or the
  // body on the floor while the dog's was not (sitting and lying may rest on it)
  fell(tr = this.tracking()) {
    return Math.abs(tr.heightErr) > 0.15 || tr.tilt > 0.8 || (this.bodyOnFloor() && !(this.ref("height")[0] < 0.3));
  }
}
