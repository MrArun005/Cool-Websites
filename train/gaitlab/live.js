// Bolt running live: MuJoCo physics (official WebAssembly build) plus the trained brains, in the
// browser. A line-for-line port of BoltEnv's control loop for the "dog" gait with turning and speed
// control (train/bolt_env.py, reference_gait.py), checked against Python step by step with
// live_check.mjs. The bundle comes from train/export_live.py.

const LINK = 0.2;               // thigh and shin length (m)
const PAIR = [0, 1, 1, 0];      // FL, FR, RL, RR: which half of the trot cycle each leg is on

export function floats(b64, Type = Float32Array) {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Type(bytes.buffer);
}

// A trained brain: observation normalizer, two 256-unit ELU layers, linear output (the mean action).
export class Brain {
  constructor(spec) {
    this.inputs = spec.inputs;
    this.mean = floats(spec.mean, Float64Array); this.var = floats(spec.var, Float64Array); this.eps = spec.eps; this.clip = spec.clip;
    this.layers = spec.layers.map((l) => ({ w: floats(l.w), b: floats(l.b), inp: l.inp, out: l.out }));
  }
  act(obs) {
    let x = new Float64Array(this.inputs);
    for (let i = 0; i < this.inputs; i++) {
      const v = (Math.fround(obs[i]) - this.mean[i]) / Math.sqrt(this.var[i] + this.eps);
      x[i] = Math.max(-this.clip, Math.min(this.clip, v));
    }
    this.layers.forEach((l, k) => {
      const y = new Float64Array(l.out);
      for (let o = 0; o < l.out; o++) {
        let s = l.b[o];
        const row = o * l.inp;
        for (let i = 0; i < l.inp; i++) s += l.w[row + i] * x[i];
        y[o] = k < this.layers.length - 1 ? (s > 0 ? s : Math.exp(s) - 1) : Math.max(-1, Math.min(1, s));
      }
      x = y;
    });
    return x;
  }
}

function legIK(x, z, front) {
  const reach = Math.min(Math.hypot(x, z), 2 * LINK - 1e-6);
  const bend = 2 * Math.acos(reach / (2 * LINK));
  const beta = Math.atan2(-x, -z);
  return front ? [beta + bend / 2, -bend] : [beta - bend / 2, bend];
}

// Joint targets of the paw-path trot at a phase (reference_gait.paw_trot). spin (rad turned per
// stance) is the sharp turn: paws step sideways along the arc the turn sweeps at each hip.
function pawTrot(phase, step, lift, height, turn, spin, hipX, hipY) {
  const q = new Float64Array(12);
  for (let leg = 0; leg < 4; leg++) {
    const u = ((((phase + Math.PI * PAIR[leg] + Math.PI / 2) / (2 * Math.PI)) % 1) + 1) % 1;
    const right = leg === 1 || leg === 3;
    const stride = step + (right ? turn : -turn) - spin * hipY[leg], side = spin * hipX[leg];
    let x, y, z;
    if (u < 0.5) { const k = u / 0.5; x = stride / 2 - stride * k; y = side / 2 - side * k; z = -height; }
    else {
      const w = (u - 0.5) / 0.5, smooth = w * w * (3 - 2 * w);
      x = -stride / 2 + stride * smooth; y = -side / 2 + side * smooth; z = -height + lift * Math.sin(Math.PI * w);
    }
    const [hip, knee] = legIK(x, -Math.hypot(z, y), leg < 2);
    q[3 * leg] = Math.atan2(y, -z); q[3 * leg + 1] = hip; q[3 * leg + 2] = knee;
  }
  return q;
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class Bolt {
  constructor(mj, bundle) {
    this.mj = mj; this.c = bundle.constants;
    this.model = mj.MjModel.from_xml_string(bundle.xml);
    this.data = new mj.MjData(this.model);
    this.brains = { posture: new Brain(bundle.brains.posture), walk: new Brain(bundle.brains.walk) };
    const r = this.model.actuator_ctrlrange;
    this.lo = Array.from({ length: 12 }, (_, i) => r[2 * i]); this.hi = Array.from({ length: 12 }, (_, i) => r[2 * i + 1]);
    this.torso = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_BODY.value, "torso");
    this.dt = this.c.dt;
  }
  reset(start) {
    const { mj, model: m, data: d } = this;
    mj.mj_resetData(m, d);
    d.qpos.set(start.qpos); d.qvel.set(start.qvel); d.ctrl.set(start.ctrl);
    mj.mj_forward(m, d);
    this.t = 0; this.command = "stand"; this.walkSince = 0; this.walkFrom = this.c.stand_pose.slice();
    this.lastAction = new Float64Array(12); this.yawCmd = 0; this.speedCmd = 0.4; this.phase = 0;
  }
  setCommand(name) {
    if (name !== this.command && name === "walk") {
      this.walkSince = this.t;
      this.walkFrom = (this.t > 0 ? this.c.dog_pose : this.c.stand_pose).slice();
    }
    this.command = name;
  }
  setTurn(rate) { this.yawCmd = rate; }
  setSpeed(v) { this.speedCmd = v; }
  freq() { return 2.0 + clamp((this.speedCmd - 0.5) / 0.7, 0, 1); }
  stride() { return clamp(this.speedCmd / (2.15 * this.freq()), 0, 0.2); }
  turn() {
    if (this.c.sharp_turns) return 0;   // sharp-turn brains steer with spin instead
    const limit = Math.min(0.05, Math.max(0, 0.22 - this.stride()));
    return clamp(this.c.turn_gain * this.yawCmd, -limit, limit);
  }
  spin() { return this.c.sharp_turns ? clamp(this.yawCmd, -this.c.sharp_turn, this.c.sharp_turn) * 0.5 / this.freq() : 0; }
  // the fastest turn this brain practised at speed v (rad/s)
  maxTurn(v) { return this.c.sharp_turns ? clamp(this.c.sharp_turn - 1.25 * Math.max(0, v - 0.4), 1, this.c.sharp_turn) : this.c.max_turn; }
  basePose() {
    if (this.command !== "walk") return this.command === "stand" ? this.c.dog_pose : this.c.stand_pose;
    const g = this.c.paw_gait, q = pawTrot(this.phase, this.stride(), g.lift, g.height, this.turn(), this.spin(), this.c.hip_x, this.c.hip_y);
    const a = Math.min(1, (this.t - this.walkSince) / 25);
    return Array.from(q, (v, i) => this.walkFrom[i] + a * (v - this.walkFrom[i]));
  }
  rotation() { const x = this.data.xmat, k = 9 * this.torso; return Array.from({ length: 9 }, (_, i) => x[k + i]); }
  up() { return this.rotation()[8]; }
  // what the brains feel (BoltEnv._obs with randomize=False, full size)
  obs() {
    const d = this.data, R = this.rotation(), qpos = d.qpos, qvel = d.qvel, c = this.c;
    const o = [];
    o.push(-R[6], -R[7], -R[8]);                                         // gravity in the body frame: R^T (0,0,-1)
    o.push(qvel[3] * 0.25, qvel[4] * 0.25, qvel[5] * 0.25);
    const v = [0, 1, 2].map((j) => R[j] * qvel[0] + R[3 + j] * qvel[1] + R[6 + j] * qvel[2]);  // R^T v
    if (this.command === "walk") v[0] -= this.speedCmd - c.speed_ref;      // speed felt relative to the asked speed
    o.push(...v, qpos[2]);
    for (let i = 0; i < 12; i++) o.push(qpos[7 + i] - c.stand_pose[i]);
    for (let i = 0; i < 12; i++) o.push(qvel[6 + i] * 0.1);
    o.push(...this.lastAction);
    for (const name of c.commands) o.push(name === this.command ? 1 : 0);
    o.push(Math.sin(this.phase), Math.cos(this.phase), this.yawCmd, this.speedCmd);
    return o;
  }
  // one control step (1/50 s): brain -> motor targets -> 10 physics steps
  step(action) {
    const { mj, model: m, data: d, c } = this;
    const a = Array.from(action, (v) => clamp(v, -1, 1));
    const base = this.basePose(), k = this.command === "walk" ? c.residual_scale : 1;
    const ctrl = base.map((b, i) => clamp(b + c.action_scale[i] * k * a[i], this.lo[i], this.hi[i]));
    d.ctrl.set(ctrl);
    for (let s = 0; s < c.frame_skip; s++) mj.mj_step(m, d);
    this.t += 1;
    this.phase += 2 * Math.PI * this.freq() * this.dt;
    this.lastAction = Float64Array.from(a);
  }
  tick() {
    const brain = this.command === "walk" ? this.brains.walk : this.brains.posture;
    this.step(brain.act(this.obs()));
  }
  fell() { return this.up() < 0.2; }
  // where Bolt is and which way it faces
  where() {
    const R = this.rotation(), q = this.data.qpos;
    return { x: q[0], y: q[1], z: q[2], yaw: Math.atan2(R[3], R[0]), R };
  }
}

// From the person's position relative to Bolt to commands (train/follow.py FollowController), plus
// two things learned from the browser check: stopping dead from a run tips Bolt over (fell 2-5 in 8
// from 0.7-1.2 m/s; easing the speed down first: 0 in 8 from 0.7), and 1.2 m/s is not steady yet.
// So the asked speed changes gradually, tops out at 1.0 m/s, and Bolt stands only after trotting
// slowly for half a second. Checked headless with live_follow_test.mjs.
export class Follow {
  // maxTurn(v): the fastest turn the walk brain practised at speed v (Bolt.maxTurn). Sharp-turn brains
  // turn harder toward the person (gain 2.5 instead of 1.5).
  constructor(maxTurn = () => 0.6, sharp = false) {
    this.KEEP = 1.0; this.TOP = 1.0; this.UP = 0.6; this.DOWN = 1.0;  // m, m/s, m/s per s
    this.GAIN = sharp ? 2.5 : 1.5; this.maxTurn = maxTurn;
    this.standing = true; this.speed = 0.2; this.slowSteps = 0;
  }
  decide(dist, bearing, dt) {
    if (this.standing && dist > this.KEEP + 0.4) { this.standing = false; this.speed = 0.2; this.slowSteps = 0; }
    if (this.standing) return { command: "stand", turn: 0, speed: this.speed };
    const stopping = dist < this.KEEP - 0.1;
    // Turn first, then run: with the person off to the side or behind, it trots slowly round to face
    // them (speeding up mid-turn tipped it over twice in 80 simulated minutes).
    let want = (0.2 + 1.0 * (dist - this.KEEP)) * Math.max(0, Math.cos(bearing)) ** 2;
    want = stopping ? 0.2 : clamp(want, 0.2, this.TOP);
    this.speed = want > this.speed ? Math.min(want, this.speed + this.UP * dt) : Math.max(want, this.speed - this.DOWN * dt);
    this.slowSteps = stopping && this.speed <= 0.2 ? this.slowSteps + 1 : 0;
    if (this.slowSteps >= 25) { this.standing = true; return { command: "stand", turn: 0, speed: this.speed }; }
    const top = this.maxTurn(this.speed);
    return { command: "walk", turn: clamp(this.GAIN * bearing, -top, top), speed: this.speed };
  }
}

// Five people wandering an open floor: each walks to a random spot, sometimes jogs, sometimes stops
// for a while. Seeded, so a run can be repeated.
export class Crowd {
  constructor(n = 5, seed = 1, half = 7) {
    let s = seed >>> 0;
    this.rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    this.half = half;
    this.people = Array.from({ length: n }, (_, i) => {
      const a = (2 * Math.PI * i) / n;
      return { x: 3 * Math.cos(a), y: 3 * Math.sin(a), heading: a + Math.PI / 2, v: 0, goal: null, pause: 1 + 2 * this.rand(), pace: 0.6 };
    });
  }
  step(dt) {
    for (const p of this.people) {
      if (p.pause > 0) { p.pause -= dt; p.v = 0; continue; }
      if (!p.goal) {
        p.goal = [(2 * this.rand() - 1) * this.half, (2 * this.rand() - 1) * this.half];
        const r = this.rand();
        p.pace = r < 0.2 ? 1.0 : 0.45 + 0.4 * this.rand();  // a jog now and then, else a walk
      }
      const dx = p.goal[0] - p.x, dy = p.goal[1] - p.y, d = Math.hypot(dx, dy);
      if (d < 0.3) { p.goal = null; p.v = 0; if (this.rand() < 0.5) p.pause = 1 + 4 * this.rand(); continue; }
      let turn = Math.atan2(dy, dx) - p.heading;
      turn = Math.atan2(Math.sin(turn), Math.cos(turn));
      p.heading += clamp(turn, -1.2 * dt, 1.2 * dt);
      p.v = p.pace * Math.max(0.2, Math.cos(turn));
      p.x += p.v * Math.cos(p.heading) * dt; p.y += p.v * Math.sin(p.heading) * dt;
    }
  }
}

// Where a person is as Bolt's camera would see it: distance and bearing (+ is to Bolt's left).
export function sight(bolt, p) {
  const w = bolt.where(), dx = p.x - w.x, dy = p.y - w.y;
  const fx = Math.cos(w.yaw) * dx + Math.sin(w.yaw) * dy, fy = -Math.sin(w.yaw) * dx + Math.cos(w.yaw) * dy;
  return { dist: Math.hypot(fx, fy), bearing: Math.atan2(fy, fx) };
}

// One step of the live page: people move, Bolt looks at the chosen one, decides, and moves. Bolt
// stands for its first 2 s, as in follow.py (the walk practised taking over from a standing dog).
export function followStep(bolt, crowd, follow, target) {
  crowd.step(bolt.dt);
  const s = sight(bolt, crowd.people[target]);
  const c = bolt.t < 100 ? { command: "stand", turn: 0, speed: 0.2 } : follow.decide(s.dist, s.bearing, bolt.dt);
  bolt.setCommand(c.command);
  if (c.command === "walk") { bolt.setTurn(c.turn); bolt.setSpeed(c.speed); }
  bolt.tick();
  return { ...c, ...s };
}
