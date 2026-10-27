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
// a prop's orientation (w, x, y, z): turned `yaw` about the vertical, then tilted `roll` about its own x
export function propQuat(p) {
  const cz = Math.cos((p.yaw || 0) / 2), sz = Math.sin((p.yaw || 0) / 2), cx = Math.cos((p.roll || 0) / 2), sx = Math.sin((p.roll || 0) / 2);
  return [cz * cx, cz * sx, sz * sx, sz * cx];
}

export class Bolt {
  constructor(mj, bundle) {
    this.mj = mj; this.c = bundle.constants; this.baseXml = bundle.xml;
    this.brains = { posture: new Brain(bundle.brains.posture), walk: new Brain(bundle.brains.walk) };
    if (bundle.brains.jump) this.brains.jump = new Brain(bundle.brains.jump);   // jump_env.py: a real jump
    if (bundle.brains.getup) this.brains.getup = new Brain(bundle.brains.getup);   // getup_env.py: up after a fall
    this.build("");
    this.dt = this.c.dt;
  }
  // Solid course objects (tunnel walls, the pause table, poles, posts, towers): static geoms in one body
  // after the robot, so the robot's geom ids and qpos stay where the brains and the drawing expect them.
  // props: [{ kind: "box" | "cyl", size: [half sizes] (cyl: [radius, half height]), pos: [x, y, z], yaw }]
  // in world coordinates. The model is rebuilt; call before reset() for a new course.
  setProps(props) {
    this.props = props;
    const geoms = props.map((p, i) => {
      const q = propQuat(p);
      return `<geom name="prop${i}" type="${p.kind === "cyl" ? "cylinder" : "box"}" size="${p.size.join(" ")}" ` +
        `pos="${p.pos.join(" ")}" quat="${q.join(" ")}" rgba="0.8 0.8 0.8 1"/>`;
    });
    this.build(geoms.length ? `<body name="props">${geoms.join("")}</body>` : "");
  }
  build(propsXml) {
    const mj = this.mj;
    let xml = this.baseXml;
    if (propsXml) xml = xml.includes('<body name="bar0"') ? xml.replace('<body name="bar0"', propsXml + '<body name="bar0"') : xml.replace("</worldbody>", propsXml + "</worldbody>");
    if (this.data) { this.data.delete(); this.model.delete(); }
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
    const r = this.model.actuator_ctrlrange;
    this.lo = Array.from({ length: 12 }, (_, i) => r[2 * i]); this.hi = Array.from({ length: 12 }, (_, i) => r[2 * i + 1]);
    this.torso = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_BODY.value, "torso");
    this.hurdle = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_GEOM.value, "hurdle");   // -1 in bundles without one
    this.ice = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_GEOM.value, "ice");
    // the bar and the ice are mocap bodies: moved through data.mocap_pos (moving plain world geoms at
    // run time made MuJoCo miss collisions: legs went through bars)
    const mocap = (geom) => (geom < 0 ? -1 : this.model.body_mocapid[this.model.geom_bodyid[geom]]);
    this.hurdleMocap = mocap(this.hurdle); this.iceMocap = mocap(this.ice);
    // real agility hurdles (hurdle.py): solid stands (mocap bodies) and loose bars (free bodies) in cups
    this.real = [];
    for (let k = 0; ; k++) {
      const body = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_BODY.value, `hurdle${k}`);
      if (body < 0) break;
      const bar = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_BODY.value, `bar${k}`);
      const joint = this.model.body_jntadr[bar], geoms = new Set();
      for (let g = 0; g < this.model.ngeom; g++) if (this.model.geom_bodyid[g] === body || this.model.geom_bodyid[g] === bar) geoms.add(g);
      this.real.push({ mocap: this.model.body_mocapid[body], body, bar, q: this.model.jnt_qposadr[joint], v: this.model.jnt_dofadr[joint], geoms, rest: null, at: null });
    }
    this.floor = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_GEOM.value, "floor");
    this.prop0 = mj.mj_name2id(this.model, mj.mjtObj.mjOBJ_GEOM.value, "prop0");   // -1 with no props
    this.aimed = -1;
  }
  // swap in another walk brain (agility competitors), with the gait it was trained on
  useWalk(spec, sharp) {
    this.brains.walk = spec instanceof Brain ? spec : new Brain(spec);
    this.c = { ...this.c, sharp_turns: sharp };
  }
  reset(start) {
    const { mj, model: m, data: d } = this;
    mj.mj_resetData(m, d);
    d.qpos.set(start.qpos); d.qvel.set(start.qvel); d.ctrl.set(start.ctrl);
    mj.mj_forward(m, d);
    this.t = 0; this.command = "stand"; this.walkSince = 0; this.walkFrom = this.c.stand_pose.slice();
    this.lastAction = new Float64Array(12); this.yawCmd = 0; this.speedCmd = 0.4; this.phase = 0;
    this.hurdleAt = [100, 0]; this.hurdleDir = [1, 0]; this.hurdleH = 0; this.aimed = -1;
    // mj_resetData put the bar and the ice back where the model parks them
    this.push = null;
  }
  // Put Bolt standing at a spot, facing `yaw` (a game's restart after a fall)
  teleport(start, [x, y], yaw) {
    this.reset(start);
    const q = this.data.qpos;
    q[0] = x; q[1] = y; q.set([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 3);
    this.data.qvel.fill(0);
    this.mj.mj_forward(this.model, this.data);
  }
  // Game hazards. Ice: a 2 m x 1.4 m slab centred at a spot, along `yaw`, with friction `mu` (paws: 1.0).
  setIce([x, y], yaw, mu, on = true) {
    if (this.iceMocap < 0) return;
    this.data.mocap_pos.set([x, y, on ? 0.005 : -1], 3 * this.iceMocap);
    this.data.mocap_quat.set([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 4 * this.iceMocap);
    this.model.geom_friction[3 * this.ice] = mu;
    this.mj.mj_forward(this.model, this.data);
  }
  // A shove: force (N) on the body, sideways (+ to Bolt's left) and forward, for `seconds`.
  shove(side, forward, seconds) { this.push = { side, forward, left: seconds }; }
  applyPush(dt) {
    const f = this.data.xfrc_applied, k = 6 * this.torso;
    if (!this.push || this.push.left <= 0) { f[k] = f[k + 1] = 0; this.push = null; return; }
    const w = this.where(), c = Math.cos(w.yaw), s = Math.sin(w.yaw);
    f[k] = c * this.push.forward - s * this.push.side; f[k + 1] = s * this.push.forward + c * this.push.side;
    this.push.left -= dt;
  }
  // Hurdles (BoltEnv.place_hurdle): a bar `ahead` m in front, across Bolt's path, `height` m tall. The
  // trot steps over it, lifting the paws to the bar's height plus a margin while near it.
  placeHurdle(ahead, height) {
    const w = this.where(), f = [Math.cos(w.yaw), Math.sin(w.yaw)];
    this.setHurdle([w.x + f[0] * ahead, w.y + f[1] * ahead], w.yaw, height);
  }
  setHurdle([x, y], yaw, height) {   // put the bar at a spot on the course, across the direction `yaw`
    const n = Math.hypot(Math.cos(yaw), Math.sin(yaw));
    this.hurdleAt = [x, y]; this.hurdleDir = [Math.cos(yaw) / n, Math.sin(yaw) / n]; this.hurdleH = height;
    this.data.mocap_pos.set([x, y, height - (this.c.hurdle_half || 0.15)], 3 * this.hurdleMocap);
    this.data.mocap_quat.set([Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)], 4 * this.hurdleMocap);
    this.mj.mj_forward(this.model, this.data);
  }
  // Real hurdles: put hurdle k at a spot, across the direction `yaw`, its bar `height` m up, resting in
  // its cups (null: park it). The stands are solid; the bar falls if knocked hard (1.5-2 m/s).
  placeReal(k, spot, yaw, height) {
    const H = this.real[k], g = this.c.real_hurdle, d = this.data;
    const [x, y] = spot || [100, 3 * k], h = spot ? height : 0.15, qt = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
    d.mocap_pos.set([x, y, h], 3 * H.mocap); d.mocap_quat.set(qt, 4 * H.mocap);
    H.rest = [x, y, h - g.bar_r]; H.at = spot ? [x, y] : null; H.yaw = yaw; H.h = h;
    d.qpos.set([...H.rest, ...qt], H.q); for (let i = 0; i < 6; i++) d.qvel[H.v + i] = 0;
    this.mj.mj_forward(this.model, d);
  }
  resetBar(k) { const H = this.real[k]; if (H.at) this.placeReal(k, H.at, H.yaw, H.h); }
  // the hurdle the jump brain looks at (its distance, height and line)
  aim(k) {
    const H = this.real[k]; this.aimed = k;
    this.hurdleAt = H.at.slice(); this.hurdleDir = [Math.cos(H.yaw), Math.sin(H.yaw)]; this.hurdleH = H.h;
  }
  // the bar of hurdle k knocked out of its cups (moved 3 cm or more): the only fault in agility
  barDropped(k) {
    const H = this.real[k], q = this.data.qpos;
    return Math.hypot(q[H.q] - H.rest[0], q[H.q + 1] - H.rest[1], q[H.q + 2] - H.rest[2]) > 0.03;
  }
  // which solid course objects (indexes into this.props) Bolt is touching now (not counting the floor)
  touchingProps() {
    if (this.prop0 < 0) return [];
    const n = this.props.length, out = new Set(), cs = this.data.contact;
    for (let k = 0; k < this.data.ncon; k++) {
      const c = cs.get(k);
      for (const g of [c.geom1, c.geom2]) if (g >= this.prop0 && g < this.prop0 + n) out.add(g - this.prop0);
      c.delete();
    }
    cs.delete(); return [...out];
  }
  // where hurdle k's bar is now (position and quaternion), for drawing
  barPose(k) { const q = this.data.qpos, H = this.real[k]; return [q.slice(H.q, H.q + 3), q.slice(H.q + 3, H.q + 7)]; }
  // angle from Bolt's heading to the aimed hurdle's line (+: to the left), and how far left of the line Bolt is
  headingError() {
    const R = this.rotation(), [ux, uy] = this.hurdleDir;
    return Math.atan2(R[0] * uy - R[3] * ux, R[0] * ux + R[3] * uy);
  }
  offLine() {
    const q = this.data.qpos, [ux, uy] = this.hurdleDir, rx = q[0] - this.hurdleAt[0], ry = q[1] - this.hurdleAt[1];
    return -rx * uy + ry * ux;
  }
  hurdleAhead() {
    const q = this.data.qpos;
    return (this.hurdleAt[0] - q[0]) * this.hurdleDir[0] + (this.hurdleAt[1] - q[1]) * this.hurdleDir[1];
  }
  hurdleHits() {
    if (this.aimed >= 0) {   // Bolt touching any real hurdle's stands or bar (not a bar in its cups or on the floor)
      const all = new Set(); for (const H of this.real) for (const g of H.geoms) all.add(g);
      const cs = this.data.contact; let n = 0;
      for (let k = 0; k < this.data.ncon; k++) {
        const c = cs.get(k);
        if (all.has(c.geom1) !== all.has(c.geom2) && c.geom1 !== this.floor && c.geom2 !== this.floor) n++;
        c.delete();
      }
      cs.delete(); return n;
    }
    if (this.hurdle < 0) return 0;
    const cs = this.data.contact; let n = 0;
    for (let k = 0; k < this.data.ncon; k++) { const c = cs.get(k); if (c.geom1 === this.hurdle || c.geom2 === this.hurdle) n++; c.delete(); }
    cs.delete(); return n;
  }
  lift() {
    const z = this.c.hurdle_zone, a = this.hurdleAhead();
    return this.hurdle >= 0 && z && a > z[0] && a < z[1] ? this.hurdleH + this.c.hurdle_margin : this.c.paw_gait.lift;
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
    const g = this.c.paw_gait, q = pawTrot(this.phase, this.stride(), this.lift(), g.height, this.turn(), this.spin(), this.c.hip_x, this.c.hip_y);
    const a = Math.min(1, (this.t - this.walkSince) / 25);
    return Array.from(q, (v, i) => this.walkFrom[i] + a * (v - this.walkFrom[i]));
  }
  rotation() { const x = this.data.xmat, k = 9 * this.torso; return Array.from({ length: 9 }, (_, i) => x[k + i]); }
  up() { return this.rotation()[8]; }
  // n physics steps (bolt_env.physics): with c.motor, real-motor limits at every step: a motor driving its
  // joint gives at most `power` W (less torque the faster it spins) and nothing past `speed` rad/s (fading
  // to zero by `cutoff`); braking against the motion keeps the full torque
  physics(n) {
    const { mj, model: m, data: d } = this, M = this.c.motor, fr = m.actuator_forcerange, v = d.qvel;
    for (let s = 0; s < n; s++) {
      if (M) for (let j = 0; j < 12; j++) {
        const w = Math.abs(v[6 + j]);
        const drive = Math.min(M.torque, M.power / Math.max(w, 1e-6)) * clamp((M.cutoff - w) / (M.cutoff - M.speed), 0, 1);
        fr[2 * j] = v[6 + j] < 0 ? -drive : -M.torque; fr[2 * j + 1] = v[6 + j] >= 0 ? drive : M.torque;
      }
      mj.mj_step(m, d);
    }
  }
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
    this.applyPush(this.dt);
    this.physics(c.frame_skip);
    this.t += 1;
    this.phase += 2 * Math.PI * this.freq() * this.dt;
    this.lastAction = Float64Array.from(a);
  }
  tick() {
    const brain = this.command === "walk" ? this.brains.walk : this.brains.posture;
    this.step(brain.act(this.obs()));
  }
  fell() { return this.up() < 0.2; }
  // The jump brain (JumpEnv in jump_env.py): from standing in front of the bar, 2 s of full-range motor
  // control around the dog stance. startJump(), then jumpTick() 100 times.
  startJump() { this.jt = 0; this.jumpLast = new Float64Array(12); }
  // what the full-range brains (jump, get-up) feel of the body: gravity and spin in the body frame, body-frame
  // velocity, height, joints around the dog stance, joint speeds, and their own last action (46 numbers)
  bodyObs(last) {
    const d = this.data, R = this.rotation(), q = d.qpos, v = d.qvel, c = this.c, o = [-R[6], -R[7], -R[8]];
    o.push(v[3] * 0.25, v[4] * 0.25, v[5] * 0.25);
    for (let j = 0; j < 3; j++) o.push(R[j] * v[0] + R[3 + j] * v[1] + R[6 + j] * v[2]);
    o.push(q[2]);
    for (let i = 0; i < 12; i++) o.push(q[7 + i] - c.dog_pose[i]);
    for (let i = 0; i < 12; i++) o.push(v[6 + i] * 0.1);
    o.push(...last);
    return o;
  }
  jumpObs() {
    const o = this.bodyObs(this.jumpLast);
    o.push(this.hurdleAhead(), this.hurdleH, this.jt / this.c.jump_steps);
    if (this.brains.jump.inputs === 51) o.push(Math.sin(this.headingError()), this.offLine());   // run 16 on
    return o;
  }
  // full-range motor control around the dog stance (JumpEnv.step, GetUpEnv.step): 1/50 s
  fullStep(a, scale = this.c.action_scale, lo = this.lo, hi = this.hi) {
    const { mj, model: m, data: d, c } = this;
    d.ctrl.set(c.dog_pose.map((b, i) => clamp(b + scale[i] * a[i], lo[i], hi[i])));
    this.applyPush(this.dt);
    this.physics(c.frame_skip);
  }
  jumpStep(action) {
    const a = Array.from(action, (v) => clamp(v, -1, 1));
    this.fullStep(a);
    this.jt += 1; this.jumpLast = Float64Array.from(a);
  }
  jumpTick() { this.jumpStep(this.brains.jump.act(this.jumpObs())); }
  // The get-up brain (GetUpEnv in getup_env.py): from lying on its side, back or half-tipped, roll over.
  // Run 17c's self-righting brain (goal "selfright", the ANYmal recipe) stops belly down, legs tucked, and
  // lies still (righted()); the posture brain's "stand" then gets up from there. A brain trained with goal
  // "stand" goes all the way to standing(). Trained on 4 s tries (200 steps), each starting after 1 s lying
  // still with the motors holding the pose it fell in. startGetUp(), then getUpTick().
  startGetUp() { this.gt = 0; this.getUpLast = new Float64Array(12); }
  getUpObs() { return this.bodyObs(this.getUpLast); }   // GetUpEnv._obs with randomize=False
  getUpStep(action) {
    const a = Array.from(action, (v) => clamp(v, -1, 1));
    this.fullStep(a, this.c.getup_scale || this.c.action_scale);   // run 17d on: each joint's whole range
    this.gt += 1; this.getUpLast = Float64Array.from(a);
  }
  getUpTick() { this.getUpStep(this.brains.getup.act(this.getUpObs())); }
  // up as the get-up brain was taught it (GetUpEnv: upright within ~25° and the body 38 cm or more up)
  standing(ground = 0) { return this.up() > 0.9 && this.data.qpos[2] - ground > 0.38; }
  // belly down and nearly still, as the self-righting brain was taught it (GetUpEnv goal "selfright")
  righted() { const v = this.data.qvel; return this.up() > 0.9 && Math.hypot(v[0], v[1], v[2]) < 0.2; }
  // lie still after a fall: the motors keep their last targets (how every get-up try started)
  restTick() {
    const { mj, model: m, data: d, c } = this;
    this.applyPush(this.dt);
    this.physics(c.frame_skip);
  }
  // Tuck the legs under the body (the lying pose), moving the motor targets there from where the joints are
  // over 0.7 s (35 steps), then holding (the main session's fold, Python: 15-17 of 20 stood). The
  // self-righting brain ends belly down with its legs anywhere (17d 7M: front hips bent backwards, knees
  // straight, hips out 0.9 rad); the posture brain standing up from there rolled Bolt onto its back.
  // In the browser (no random physics) the posture brain's own "lie" tucked better (game.js GETUP_TUCK).
  startTuck() { this.tt = 0; this.tuckFrom = Array.from(this.data.qpos.slice(7, 19)); }
  tuckTick() {
    const a = Math.min(1, ++this.tt / 35), L = this.c.lie_pose;
    this.data.ctrl.set(this.tuckFrom.map((v, i) => clamp(v + a * (L[i] - v), this.lo[i], this.hi[i])));
    this.restTick();
  }
  // hand the body from the get-up brain to the standing brain. The stand starts from zeros (from belly down
  // 20/20 either way); carry: from the get-up brain's last action (same mapping only with action_scale)
  afterGetUp(carry = false) { this.lastAction = carry ? Float64Array.from(this.getUpLast) : new Float64Array(12); this.command = "stand"; }
  // the motors hold the dog stance (how every training jump started: 0.5 s of this, then the jump)
  holdTick() {
    const { mj, model: m, data: d, c } = this;
    d.ctrl.set(c.dog_pose); this.applyPush(this.dt);
    this.physics(c.frame_skip);
  }
  // run one game command: walk/stand/sit/lie use the trained brains; "hold" and "jump" the jump sequence;
  // "rest" and "getup" getting up after a fall
  run(cmd) {
    if (cmd.command === "hold") return this.holdTick();
    if (cmd.command === "jump") return this.jumpTick();
    if (cmd.command === "getup") return this.getUpTick();
    if (cmd.command === "rest") return this.restTick();
    if (cmd.command === "tuck") return this.tuckTick();
    this.setCommand(cmd.command);
    if (cmd.command === "walk") { this.setTurn(cmd.turn); this.setSpeed(cmd.speed); }
    this.tick();
  }
  jumping() { return this.jt !== undefined && this.jt < this.c.jump_steps; }
  canGetUp() { return !!this.brains.getup; }
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
  // turn harder toward the person (gain 2.5 instead of 1.5), but at most 1.5 rad/s: spinning round at
  // 2 rad/s tipped run 12's 10.5M brain over 3 times in 80 simulated minutes, 1.5 rad/s none. The 11.5M
  // brain fell at neither, and followed just as closely at both, so the margin stays.
  constructor(maxTurn = () => 0.6, sharp = false) {
    this.KEEP = 1.0; this.TOP = 1.0; this.UP = 0.6; this.DOWN = 1.0;  // m, m/s, m/s per s
    this.GAIN = sharp ? 2.5 : 1.5; this.CAP = sharp ? 1.5 : Infinity; this.maxTurn = maxTurn;
    if (sharp) this.TOP = 1.2;   // the sharp-turn brain runs 1.2 m/s steadily (0 falls in 8; 0 in 80 min following)
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
    const top = Math.min(this.CAP, this.maxTurn(this.speed));
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
