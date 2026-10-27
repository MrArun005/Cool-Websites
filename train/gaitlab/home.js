// Bolt at home: Bolt as a pet. It has drives that change over time (energy, play, attention, curiosity,
// rest), a mood that follows from them, and picks its own behaviour by scoring every behaviour it knows
// (utility AI, with hysteresis so it doesn't flicker). It reacts to the person in the room, understands
// spoken and typed commands, and remembers the person (name, what it was praised and scolded for).
//
// Every move is Bolt's real trained brains in real physics (live.js): behaviours only choose where to
// walk, how fast, and which posture (stand, sit, lie) to take. The motion layer (Body) keeps to what the
// games taught: slow to a trot before stopping, stand before walking on from a sit or lie, turn before
// speeding up, never stop dead from a run. The ball is a real free body Bolt bumps and pushes; when Bolt
// "carries" it in its mouth (Bolt has no jaw) the ball is pinned in front of the snout with its
// collisions off, and the person's throw sets its start position and speed, after which physics flies it.
//
// Pure logic, no DOM: home_test.mjs runs ten minutes of pet life headless; home_template.html draws it.

import { Driver, Jumper, posture, rng } from "./game.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// ---------- the room ----------
// x to the right (east), y away from the door (north); the door is in the middle of the south wall.
export const ROOM = {
  half: [5, 4],
  door: [0, -4],
  bed: [-3.9, -2.9],                     // a cushion (flat: painted, not solid)
  rug: [0, 0.2, 2.2, 1.5],               // centre and half sizes (painted)
  sofa: [-2.2, 3.45],
  bowl: [4.2, 3.2],
  plant: [4.35, -3.35],
  hurdle: { at: [2.2, -2.4], yaw: 0, h: 0.15 },   // one real agility hurdle (live.js placeReal), jumped toward +x
  ball: { r: 0.09, mass: 0.12, start: [1.2, 1.4] },
};

// Solid things, one list for the physics (live.js setProps) and the drawing (props3d.js).
export function roomProps(room = ROOM) {
  const [W, H] = room.half, T = 0.05, Z = 0.25, wall = { color: "#d9d4ca" }, out = [];
  out.push({ kind: "box", size: [W + 2 * T, T, Z], pos: [0, H + T, Z], look: wall });
  out.push({ kind: "box", size: [W + 2 * T, T, Z], pos: [0, -H - T, Z], look: wall });
  out.push({ kind: "box", size: [T, H, Z], pos: [W + T, 0, Z], look: wall });
  out.push({ kind: "box", size: [T, H, Z], pos: [-W - T, 0, Z], look: wall });
  const [sx, sy] = room.sofa, fabric = { color: "#5b7fa6" }, dark = { color: "#4a6a8e" };
  out.push({ kind: "box", size: [1.0, 0.5, 0.2], pos: [sx, sy, 0.2], look: fabric });          // seat
  out.push({ kind: "box", size: [1.15, 0.12, 0.4], pos: [sx, sy + 0.42, 0.4], look: dark });   // back
  for (const s of [1, -1]) out.push({ kind: "box", size: [0.12, 0.5, 0.3], pos: [sx + s * 1.08, sy, 0.3], look: dark });   // arms
  out.push({ kind: "cyl", size: [0.16, 0.045], pos: [...room.bowl, 0.045], look: { color: "#c4372f" } });
  out.push({ kind: "cyl", size: [0.24, 0.22], pos: [...room.plant, 0.22], look: { color: "#a0703c" } });
  return out;
}

// Rectangles Bolt's centre keeps out of (each solid thing grown by Bolt's half length), for planning.
export function obstacles(room = ROOM) {
  const [sx, sy] = room.sofa, [bx, by] = room.bowl, [px, py] = room.plant, h = room.hurdle;
  const c = Math.cos(h.yaw), s = Math.sin(h.yaw);
  const ends = [-0.96, 0.96].map((k) => [h.at[0] - s * k, h.at[1] + c * k]);
  return [
    { name: "sofa", x0: sx - 1.2, x1: sx + 1.2, y0: sy - 0.5, y1: sy + 0.6 },
    { name: "bowl", x0: bx - 0.16, x1: bx + 0.16, y0: by - 0.16, y1: by + 0.16 },
    { name: "plant", x0: px - 0.24, x1: px + 0.24, y0: py - 0.24, y1: py + 0.24 },
    { name: "hurdle", x0: Math.min(ends[0][0], ends[1][0]) - 0.05, x1: Math.max(ends[0][0], ends[1][0]) + 0.05, y0: Math.min(ends[0][1], ends[1][1]), y1: Math.max(ends[0][1], ends[1][1]) },
  ];
}

// Places worth a sniff: where to stand and what to face.
export function interests(room = ROOM) {
  const [sx, sy] = room.sofa, [bx, by] = room.bowl, [px, py] = room.plant, [dx, dy] = room.door;
  return [
    { name: "sofa", spot: [sx + 0.5, sy - 1.05], face: [sx + 0.5, sy] },
    { name: "other end of the sofa", spot: [sx - 0.6, sy - 1.05], face: [sx - 0.6, sy] },
    { name: "food bowl", spot: [bx - 0.55, by - 0.4], face: [bx, by] },
    { name: "plant", spot: [px - 0.75, py + 0.35], face: [px, py] },
    { name: "door", spot: [dx, dy + 0.85], face: [dx, dy] },
    { name: "bed", spot: [room.bed[0] + 0.9, room.bed[1] + 0.5], face: room.bed },
    { name: "corner by the window", spot: [-4.1, 2.0], face: [-5, 2.5] },
    { name: "rug", spot: [1.2, -0.6], face: [0, 0.2] },
  ];
}

const BALL_XML = (b) => `<body name="ball" pos="${b.start[0]} ${b.start[1]} ${b.r}"><freejoint name="ball"/>` +
  `<geom name="ball" type="sphere" size="${b.r}" mass="${b.mass}" condim="6" friction="0.8 0.02 0.03" rgba="0.95 0.75 0.1 1"/></body>`;

// Build the room into Bolt's physics: the walls and furniture as static props, the ball as a free body
// (added after everything else, so the robot's and the hurdles' qpos stay where the brains expect them),
// and one real hurdle. Call once; then reset/teleport keep it (the hurdle and ball are put back by Pet).
export function setupRoom(bolt, start, room = ROOM) {
  if (!bolt.baseXml.includes('<body name="ball"')) bolt.baseXml = bolt.baseXml.replace("</worldbody>", BALL_XML(room.ball) + "</worldbody>");
  bolt.setProps(roomProps(room));
  bolt.reset(start);
  const mj = bolt.mj, m = bolt.model;
  const body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY.value, "ball"), joint = m.body_jntadr[body];
  bolt.ball = { body, geom: mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, "ball"), q: m.jnt_qposadr[joint], v: m.jnt_dofadr[joint], r: room.ball.r };
  bolt.snout = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, "snout");
  if (bolt.snout < 0) bolt.snout = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, "head");
  placeHurdle(bolt, room);
  return bolt.ball;
}
export function placeHurdle(bolt, room = ROOM) {
  if (!bolt.real || !bolt.real.length) return false;
  bolt.placeReal(0, room.hurdle.at, room.hurdle.yaw, room.hurdle.h);
  for (let k = 1; k < bolt.real.length; k++) bolt.placeReal(k, null, 0, 0);
  return true;
}
export const ballPos = (bolt) => { const q = bolt.data.qpos, k = bolt.ball.q; return [q[k], q[k + 1], q[k + 2]]; };
export const ballSpeed = (bolt) => { const v = bolt.data.qvel, k = bolt.ball.v; return Math.hypot(v[k], v[k + 1], v[k + 2]); };
function setBall(bolt, pos, vel = [0, 0, 0]) {
  const q = bolt.data.qpos, v = bolt.data.qvel, B = bolt.ball;
  q.set([...pos, 1, 0, 0, 0], B.q);
  for (let i = 0; i < 6; i++) v[B.v + i] = i < 3 ? vel[i] : 0;
}
function ballSolid(bolt, on) { const g = bolt.ball.geom; bolt.model.geom_contype[g] = on ? 1 : 0; bolt.model.geom_conaffinity[g] = on ? 1 : 0; }
// the point just in front of Bolt's snout
function mouth(bolt) {
  const x = bolt.data.geom_xpos, i = bolt.snout, w = bolt.where();
  return [x[3 * i] + 0.08 * Math.cos(w.yaw), x[3 * i + 1] + 0.08 * Math.sin(w.yaw), Math.max(bolt.ball.r, x[3 * i + 2] - 0.03)];
}

// ---------- paths: around the furniture, inside the walls ----------
function segHits(a, b, r) {   // does segment a-b cross rectangle r (Liang-Barsky)
  let t0 = 0, t1 = 1;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  for (const [p, q] of [[-dx, a[0] - r.x0], [dx, r.x1 - a[0]], [-dy, a[1] - r.y0], [dy, r.y1 - a[1]]]) {
    if (p === 0) { if (q < 0) return false; continue; }
    const t = q / p;
    if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; }
  }
  return t0 <= t1;
}
const inside = (p, r) => p[0] > r.x0 && p[0] < r.x1 && p[1] > r.y0 && p[1] < r.y1;
const grow = (r, m) => ({ ...r, x0: r.x0 - m, x1: r.x1 + m, y0: r.y0 - m, y1: r.y1 + m });
export class Planner {
  constructor(room = ROOM, margin = 0.32) {
    this.room = room; this.margin = margin; this.fixed = obstacles(room).map((r) => grow(r, margin));
    this.edge = 0.5;   // Bolt's centre stays this far from the walls
    this.dynamic = [];   // things that move: the ball, when Bolt shouldn't step on it
  }
  get rects() { return this.dynamic.length ? this.fixed.concat(this.dynamic) : this.fixed; }
  avoid(p, half) { this.dynamic = p ? [grow({ name: "ball", x0: p[0] - half, x1: p[0] + half, y0: p[1] - half, y1: p[1] + half }, this.margin)] : []; }
  clampIn(p) {
    const [W, H] = this.room.half, e = this.edge;
    let q = [clamp(p[0], -W + e, W - e), clamp(p[1], -H + e, H - e)];
    for (const r of this.rects) if (inside(q, r)) {   // push a target out of furniture, to the nearest side
      const opts = [[r.x0 - 0.02, q[1]], [r.x1 + 0.02, q[1]], [q[0], r.y0 - 0.02], [q[0], r.y1 + 0.02]];
      q = opts.sort((a, b) => dist2(a, q) - dist2(b, q))[0];
    }
    return q;
  }
  free(p) { return !this.rects.some((r) => inside(p, r)); }
  clear(a, b, skip) { return this.rects.every((r) => r === skip || inside(a, r) || !segHits(a, b, r)); }
  // the next point to head for on a shortest path round the furniture (corners of the grown rectangles)
  next(from, to) {
    if (this.clear(from, to)) return to;
    const [W, H] = this.room.half, e = this.edge;
    const nodes = [from, to];
    for (const r of this.rects) for (const c of [[r.x0 - 0.06, r.y0 - 0.06], [r.x1 + 0.06, r.y0 - 0.06], [r.x0 - 0.06, r.y1 + 0.06], [r.x1 + 0.06, r.y1 + 0.06]]) {
      if (Math.abs(c[0]) <= W - e + 0.05 && Math.abs(c[1]) <= H - e + 0.05 && this.free(c)) nodes.push(c);
    }
    const n = nodes.length, d = new Array(n).fill(Infinity), prev = new Array(n).fill(-1), done = new Array(n).fill(false);
    d[0] = 0;
    for (let it = 0; it < n; it++) {
      let u = -1; for (let i = 0; i < n; i++) if (!done[i] && (u < 0 || d[i] < d[u])) u = i;
      if (u < 0 || d[u] === Infinity) break;
      done[u] = true; if (u === 1) break;
      for (let v = 0; v < n; v++) if (!done[v] && this.clear(nodes[u], nodes[v])) {
        const nd = d[u] + dist2(nodes[u], nodes[v]);
        if (nd < d[v]) { d[v] = nd; prev[v] = u; }
      }
    }
    if (prev[1] < 0) return to;
    let k = 1; while (prev[k] !== 0) k = prev[k];
    return nodes[k];
  }
}

// ---------- the body: what Bolt's trained brains can do, kept safe ----------
// desire (from a behaviour, every step):
//   { kind: "go", to, cruise, near }  walk there round the furniture, slow down, stop within `near`
//   { kind: "steer", to, cruise, bend }  keep moving toward a point (laps, pushing the ball)
//   { kind: "pose", pose }  stand, sit or lie where it is
//   { kind: "face", to, pose }  turn on the spot (a slow tight trot) to face a point, then pose
//   { kind: "jump", k }  the jump brain over real hurdle k (game.js Jumper)
export class Body {
  constructor(bolt, room = ROOM) {
    this.bolt = bolt; this.planner = new Planner(room); this.reset();
  }
  reset() {
    this.driver = new Driver(this.bolt);
    this.state = "still"; this.pose = "stand"; this.timer = 0; this.held = 0; this.stillFor = 0;
    this.cmd = { command: "stand", turn: 0, speed: 0.2 }; this.goal = null; this.arrived = false; this.wp = null; this.replan = 0;
    this.best = Infinity; this.noProgress = 0; this.stuck = false; this.facing = false; this.jumper = null; this.jumpResult = null;
  }
  where() { return this.bolt.where(); }
  bearingTo(p) { const w = this.where(); return wrap(Math.atan2(p[1] - w.y, p[0] - w.x) - w.yaw); }
  distTo(p) { const w = this.where(); return Math.hypot(p[0] - w.x, p[1] - w.y); }
  // does this desire need Bolt to move now?
  moving(d) {
    if (!d) return false;
    if (d.kind === "steer") return true;
    if (d.kind === "go") {
      if (!this.goal || dist2(this.goal, d.to) > 0.35) { this.goal = d.to.slice(); this.arrived = false; this.best = Infinity; this.noProgress = 0; this.stuck = false; }
      if (!this.arrived && this.distTo(d.to) < (d.near || 0.3)) this.arrived = true;
      // re-arrive hysteresis: only set off again when clearly away from the spot
      if (this.arrived && this.distTo(d.to) > (d.near || 0.3) + 0.45) this.arrived = false;
      return !this.arrived && !this.stuck;
    }
    if (d.kind === "face") {
      const b = Math.abs(this.bearingTo(d.to));
      // sitting or lying, Bolt follows you with its eyes; it only gets up to turn if you go right round it
      const slack = this.state === "still" && this.pose !== "stand" ? 1.2 : 0.35;
      if (this.facing && b > (d.tol || 0.3) + slack) this.facing = false;
      if (!this.facing && b < (d.tol || 0.3)) this.facing = true;
      return !this.facing;
    }
    return false;
  }
  drive(d, dt) {
    const w = this.where(), here = [w.x, w.y];
    if (d.kind === "face") {   // a slow, tight trot round toward the point
      const c = this.driver.toward(d.to, 0.2, false, dt); c.speed = this.driver.speed = 0.2;
      return c;
    }
    const [W, H] = this.planner.room.half, e = this.planner.edge;
    const to = d.kind === "go" ? this.planner.clampIn(d.to) : [clamp(d.to[0], -W + e, W - e), clamp(d.to[1], -H + e, H - e)];
    if ((this.replan -= dt) <= 0 || !this.wp) { this.wp = d.kind === "go" ? this.planner.next(here, to) : to; this.replan = 0.25; }
    if (d.kind === "steer") this.wp = to;
    if (d.kind === "steer" && dist2(here, to) < 0.25) return { command: "walk", turn: 0, speed: (this.driver.speed = Math.max(0.2, this.driver.speed - 1.0 * dt)) };
    const final = this.wp === to;
    // slow down coming into a corner of the path: the next leg may need a sharp turn
    const cruise = final ? d.cruise || 0.6 : Math.min(d.cruise || 0.6, 0.3 + 0.6 * dist2(here, this.wp));
    const c = this.driver.toward(this.wp, cruise, d.kind === "go" && final, dt, d.bend || 2);
    if (d.kind === "go") {   // blocked (pushing on a wall or a sofa)? give up after a while
      const left = dist2(here, to);
      if (left < this.best - 0.05) { this.best = left; this.noProgress = 0; } else if ((this.noProgress += dt) > 6) this.stuck = true;
    }
    return c;
  }
  update(d, dt) {
    const b = this.bolt;
    if (d && d.kind === "jump" && this.state !== "jump" && this.state === "still" && this.pose === "stand" && this.stillFor > 0.4) {
      this.jumper = new Jumper(b, this.driver, d.k); this.state = "jump"; this.jumpResult = null;
    }
    const move = this.moving(d);
    let cmd;
    switch (this.state) {
      case "jump": {
        cmd = this.jumper.step(dt);
        if (this.jumper.done) {
          this.jumpResult = { over: this.jumper.over, touched: this.jumper.touched, dropped: this.jumper.dropped };
          this.state = "still"; this.pose = "stand"; this.stillFor = 0; this.jumper = null; this.driver.speed = 0.2;
        }
        break;
      }
      case "walk":
        if (move) { cmd = this.drive(d, dt); break; }
        this.state = "brake"; this.timer = 0;
      // falls through
      case "brake":   // trot slowly for half a second before stopping (stopping dead from a run tips Bolt over)
        if (move) { this.state = "walk"; cmd = this.drive(d, dt); break; }
        this.driver.speed = Math.max(0.2, this.driver.speed - 1.0 * dt);
        if (this.driver.speed <= 0.2) this.timer += dt;
        cmd = { command: "walk", turn: 0.5 * (this.cmd.turn || 0), speed: this.driver.speed };
        if (this.timer > 0.5) { this.state = "settle"; this.timer = 0; }
        break;
      case "settle":   // stand still a moment
        if (move && this.timer > 0.2) { this.state = "walk"; this.driver.speed = 0.2; cmd = this.drive(d, dt); break; }
        cmd = { command: "stand", turn: 0, speed: 0.2 };
        if ((this.timer += dt) > 0.6) { this.state = "still"; this.pose = "stand"; this.stillFor = 0; }
        break;
      case "rise":   // stand for a second before walking on (or changing posture)
        cmd = { command: "stand", turn: 0, speed: 0.2 };
        if ((this.timer += dt) > 1.0) { this.state = "still"; this.pose = "stand"; this.stillFor = 0; }
        break;
      default: {   // still
        const want = (d && (d.kind === "pose" || d.kind === "face") && d.pose) || "stand";
        // never break off halfway into a sit or a lie: finish it (1.5 s), then get up (standing up from a
        // half-sit tipped Bolt over)
        const settling = this.pose !== "stand" && this.stillFor < 1.5 && this.held < 0.3;
        if (settling) { cmd = { command: this.pose, turn: 0, speed: 0.2 }; this.stillFor += dt; break; }
        if (move) {
          if (this.pose === "stand") { this.state = "walk"; this.driver.speed = 0.2; cmd = this.drive(d, dt); break; }
          this.state = "rise"; this.timer = 0; cmd = { command: "stand", turn: 0, speed: 0.2 }; break;
        }
        if (want !== this.pose) {
          if (this.pose === "stand") { this.pose = want; this.held = 0; this.stillFor = 0; }
          else { this.state = "rise"; this.timer = 0; cmd = { command: "stand", turn: 0, speed: 0.2 }; break; }
        }
        cmd = { command: this.pose, turn: 0, speed: 0.2 };
        this.stillFor += dt;
        const p = posture(b);
        this.held = (this.pose === "stand" ? p === "still" : p === this.pose) ? this.held + dt : 0;
      }
    }
    if (this.state !== "still") this.held = 0;
    // change the turn rate gradually (a turn flipped from +0.7 to -1.5 rad/s in one step tipped Bolt over)
    if (cmd.command === "walk") {
      const last = this.cmd.command === "walk" ? this.cmd.turn || 0 : 0;
      cmd = { ...cmd, turn: last + clamp(cmd.turn - last, -3.5 * dt, 3.5 * dt) };
    }
    this.cmd = cmd;
    return cmd;
  }
  // a summary for behaviours
  get still() { return this.state === "still"; }
  // arrived at this spot (the last "go" was to it), stuck on the way there, facing a point
  at(to) { return !!this.goal && dist2(this.goal, to) < 0.35 && this.arrived; }
  stuckAt(to) { return !!this.goal && dist2(this.goal, to) < 0.35 && this.stuck; }
  faces(p, tol = 0.35) { return Math.abs(this.bearingTo(p)) < tol; }
  posed(p) { return this.state === "still" && this.pose === p && this.held > 0.3; }
}

// ---------- understanding what people say ----------
// understand(text, state) -> [{ intent, ... }], one per clause ("good boy, now sit" -> praise, sit).
// A plain local parser: synonyms, phrases, the dog's name, polite and indirect phrasing. Pet.hear()
// calls whatever understander is set (setUnderstander), so a language model can replace this later
// (same input: the text and Bolt's state; same output: the intent list; it may return a promise).
export const INTENTS = ["sit", "lie", "stand", "come", "stay", "follow", "play", "fetch", "jump", "spin", "zoomies", "explore", "nap", "bowl",
  "praise", "pet", "scold", "cancel", "greet", "bye", "attend", "status", "speak", "name_set", "name_ask", "unknown"];
const RULES = [
  // order matters: the first rule that matches a clause wins
  ["lie", /\bplay dead\b/],
  ["name_set", /\b(?:my name is|my name's|call me|i am called)\s+([a-z][a-z'-]+)/],
  ["name_ask", /\b(what'?s my name|what is my name|who am i|do you (know|remember) me|remember me)\b/],
  ["cancel", /^(don'?t|do not|stop)\s+(sit|lie|lay|stand|come|follow|play|run|jump|fetch|spin|bark|do that|doing that)/],
  ["stay", /\b(stay|wait( there| here)?|hold (it|on|still)|freeze|don'?t move|stop there|halt|stand still|stop moving|calm down|easy)\b|^stop$/],
  ["scold", /^(no+|nope|bad( boy| dog| girl| robot)?|stop it|stop that|naughty|leave it|quit it|enough|drop it|don'?t( do that)?|uh[- ]?oh)\b|\bbad (boy|dog|girl)\b|\bthat'?s (bad|naughty|not ok)/],
  ["pet", /\b(pets?|pat|pats|stroke|scratch|rub|cuddle|cuddles|hug|belly rub|love you|i love you|snuggle)\b/],
  ["praise", /\b(good (boy|dog|girl|job|bolt|one|pup|puppy|robot|lad)|well done|nice( one| job| work)?|clever|attaboy|that'?s (it|right)|great job|awesome|amazing|you'?re (a |the )?(good|great|best|clever|smart)|so good|perfect|bravo|yes+|yay|good)\b/],
  ["nap", /\b(go to (bed|sleep)|bed ?time|sleep|nap|good ?night|night night|have a rest|take a rest|get some rest|lie in your bed|your bed|look(s)? (tired|sleepy)|(are )?you (tired|sleepy)|time for bed|rest)\b/],
  ["bowl", /\b(hungry|thirsty|eat|drink|food|dinner|breakfast|lunch|water|bowl|treat|snack)\b/],
  ["fetch", /\b(fetch|get (the|your|it|that)|bring (it|the ball|me|back)|go get|catch|ball|throw)\b/],
  ["jump", /\b(jump|hop|leap|hurdle|agility|over the bar)\b/],
  ["spin", /\b(spin|twirl|turn around|chase your tail|go round|circle|dance)\b/],
  ["zoomies", /\b(zoomies|run around|go run|go fast|race|sprint|run)\b/],
  ["explore", /\b(go explore|explore|sniff|look around|have a look|go and look)\b/],
  ["follow", /\b(follow( me)?|walk with me|heel|let'?s (go )?(for a )?walk|walkies|go for a walk|with me|this way|let'?s go)\b/],
  ["come", /\b(come( here| back| on| to me| over)?|here (boy|bolt|buddy|girl|pup)|over here|to me|recall|where are you|here)\b/],
  ["play", /\b(go play|let'?s play|play( time)?|wanna play|want to play|have fun|go have fun|you'?re free|off you go|free|release|at ease|go on|ok(ay)?|carry on|as you were)\b/],
  ["sit", /\b(sit|sit down|sitz|take a seat|seat|park it|bottom down|sitting)\b/],
  ["lie", /\b(lie|lay|lie down|lay down|down|play dead|belly down|flat|drop|settle)\b/],
  ["stand", /\b(stand( up)?|get up|on your feet|rise|up)\b/],
  ["greet", /^(hi|hello|hey|hiya|howdy|good (morning|evening|afternoon)|i'?m (home|back)|yo|morning|evening)\b/],
  ["bye", /\b(bye|goodbye|see you|see ya|i'?m (leaving|going out|off)|gotta go)\b/],
  ["status", /\b(how are you|how do you feel|how'?re you|are you ok(ay)?|you ok(ay)?|what'?s up|how'?s it going|are you (happy|sad|bored|ok)|what are you doing|how is bolt|how'?s bolt)\b/],
  ["speak", /\b(speak|bark|say something|woof|talk|sing)\b/],
];
export function understand(text, state = {}) {
  const name = (state.name || "bolt").toLowerCase();
  let t = String(text || "").toLowerCase().replace(/[’`]/g, "'").replace(/\s+/g, " ").trim();
  const addressed = new RegExp(`\\b${name}\\b`).test(t);
  t = t.replace(/\b(please|pls|plz|now|just|can you|could you|would you|will you|i want you to|go ahead and|buddy|mate|pal|little one|doggy|doggo)\b/g, " ");
  const clauses = t.split(/[,.;!?]+|\bthen\b|\bafter that\b|\band\b(?= (?:sit|lie|lay|stay|come|follow|fetch|jump|spin|stand|go|play|wait|get))/)
    .map((c) => (c || "").replace(new RegExp(`\\b${name}\\b`, "g"), " ").replace(/\s+/g, " ").trim()).filter(Boolean);
  const out = [];
  for (const c of clauses) {
    let hit = null;
    for (const [intent, re] of RULES) {
      const m = c.match(re);
      if (!m) continue;
      if (intent === "name_set") { if (/^(home|back|tired|here|leaving|going|off|sorry|ok|okay|fine|good)$/.test(m[1])) continue; hit = { intent, name: m[1][0].toUpperCase() + m[1].slice(1) }; break; }
      if (intent === "nap" && /\b(look|looks|are you|you) (tired|sleepy)\b|\btired\?/.test(c)) { hit = { intent, soft: true }; break; }
      if (intent === "praise" && /^(good (morning|evening|afternoon|night))/.test(c)) continue;
      if (intent === "come" && /\bhere\b/.test(c) && !/\b(come|over here|here (boy|bolt|buddy|girl|pup))\b/.test(c) && c.split(" ").length > 2) continue;
      hit = { intent }; break;
    }
    if (hit) out.push(hit);
  }
  if (!out.length) out.push({ intent: addressed && t.replace(new RegExp(`\\b${name}\\b`, "g"), "").trim().length < 3 ? "attend" : "unknown" });
  // drop repeats ("sit, sit!") but keep order
  return out.filter((x, i) => i === 0 || x.intent !== out[i - 1].intent).map((x) => ({ ...x, addressed }));
}

// ---------- memory: who the person is, what Bolt was praised and scolded for ----------
function browserStore() {
  try {
    const ls = globalThis.localStorage;
    if (!ls) return null;
    ls.getItem("bolt-home");
    return { get: (k) => { try { return ls.getItem(k); } catch { return null; } }, set: (k, v) => { try { ls.setItem(k, v); } catch { /* full or blocked: fine */ } } };
  } catch { return null; }
}
export function memoryStore() { const m = new Map(); return { get: (k) => m.get(k) ?? null, set: (k, v) => m.set(k, v) }; }
export class Memory {
  constructor(store = browserStore() || memoryStore(), key = "bolt-home-v1") {
    this.store = store; this.key = key;
    this.data = { name: null, affection: 0.5, prefs: {}, praised: {}, scolded: {}, lastSeen: null, visits: 0, heard: 0 };
    try { const s = store.get(key); if (s) Object.assign(this.data, JSON.parse(s)); } catch { /* corrupt or blocked: start fresh */ }
    this.recent = [];   // short-term memory: the last things that happened (not saved)
  }
  save() { try { this.store.set(this.key, JSON.stringify(this.data)); } catch { /* ignore */ } }
  note(text, t) { this.recent.unshift({ t, text }); this.recent.length = Math.min(this.recent.length, 12); }
  pref(name) { return this.data.prefs[name] || 0; }
  credit(name, amount) {
    const p = this.data.prefs; p[name] = clamp((p[name] || 0) + amount, -1, 1);
    const tally = amount > 0 ? this.data.praised : this.data.scolded; tally[name] = (tally[name] || 0) + 1;
    this.save();
  }
  forget() { this.data = { name: null, affection: 0.5, prefs: {}, praised: {}, scolded: {}, lastSeen: null, visits: 0, heard: 0 }; this.recent = []; this.save(); }
}

// ---------- behaviours ----------
// Each: score(pet) -> how much Bolt wants it now (0 to ~1.3); begin(pet) -> the log line; tick(pet, dt)
// -> a desire for the Body, or null when finished. Pet runs the best-scoring one, with hysteresis.
const LABEL = {
  chill: "chilling", explore: "exploring", play: "playing with the ball", zoomies: "zoomies!", fetch: "fetching",
  greet: "greeting you", follow: "following you", beg: "asking for attention", nap: "napping", sulk: "sulking",
  wait: "waiting by the door", obey: "obeying", spin: "spinning", bowl: "at the food bowl", jump: "jumping",
};
export const BEHAVIOURS = Object.keys(LABEL).filter((k) => k !== "obey");

const near = (pet, d) => pet.person.present && pet.personDist() < d;

const B = {
  chill: {
    score: () => 0.3,
    begin(p) {
      this.t = 0; this.spot = null;
      if (p.person.present && p.personDist() > 3 && p.drives.attention > 0.3) { this.spot = p.spotNear(p.personPos(), 1.6); return `${p.why()} wanders over to lie near you.`; }
      return `${p.why()} settles down for a bit.`;
    },
    tick(p, dt) {
      this.t += dt;
      if (this.spot && !p.body.at(this.spot) && !p.body.stuckAt(this.spot) && this.t < 25) return { kind: "go", to: this.spot, cruise: 0.5, near: 0.4 };
      if (this.spot) { this.spot = null; this.t = 0; }
      if (this.t > 40) return null;
      return { kind: "pose", pose: this.t < 6 ? "stand" : this.t < 18 ? "sit" : "lie" };
    },
  },
  explore: {
    score: (p) => 0.12 + 0.75 * p.drives.curiosity * p.en(),
    begin(p) {
      const all = interests(p.room).filter((i) => i.name !== p.lastSniff);
      this.poi = all[Math.floor(p.rand() * all.length)]; p.lastSniff = this.poi.name; this.t = 0; this.phase = "go"; this.sniff = 0;
      return `${p.why()} goes to sniff the ${this.poi.name}.`;
    },
    tick(p, dt) {
      this.t += dt;
      if (this.t > 45) return null;
      if (this.phase === "go") {
        if (!p.body.at(this.poi.spot) && !p.body.stuckAt(this.poi.spot)) return { kind: "go", to: this.poi.spot, cruise: 0.55 + 0.3 * p.drives.curiosity, near: 0.3 };
        this.phase = "face";
      }
      if (this.phase === "face") {
        if (!(p.body.still && p.body.faces(this.poi.face, 0.4))) return { kind: "face", to: this.poi.face, pose: "stand" };
        this.phase = "sniff";
      }
      if ((this.sniff += dt) > 3) { p.drives.curiosity = Math.max(0, p.drives.curiosity - 0.25); return null; }
      return { kind: "pose", pose: "stand" };   // sniffing: nose to it, still
    },
  },
  play: {   // push the ball round the room
    score: (p) => (p.carrying || p.ballNearPerson() ? 0 : 0.08 + 0.7 * p.drives.play * p.en()),
    begin(p) { this.t = 0; this.pushes = 0; this.phase = "to"; return `${p.why()} goes to bat the ball around.`; },
    tick(p, dt) {
      this.t += dt;
      if (this.t > 30 || this.pushes >= 3) return null;
      const b = ballPos(p.bolt), w = p.bolt.where(), d = Math.hypot(b[0] - w.x, b[1] - w.y);
      if (this.phase === "to") {   // trot up to the ball, then on through it
        if (d < 0.9) { this.phase = "push"; this.dir = [(b[0] - w.x) / d, (b[1] - w.y) / d]; }
        return { kind: "go", to: [b[0], b[1]], cruise: 0.7, near: 0.5 };
      }
      const aim = [b[0] + 0.6 * this.dir[0], b[1] + 0.6 * this.dir[1]];
      if (d > 1.6 || !p.planner.free(aim) || Math.abs(aim[0]) > p.room.half[0] - 0.6 || Math.abs(aim[1]) > p.room.half[1] - 0.6) {
        this.pushes += 1; this.phase = "to"; p.drives.play = Math.max(0, p.drives.play - 0.12); p.excite(2);
        return { kind: "pose", pose: "stand" };
      }
      return { kind: "steer", to: aim, cruise: 0.55, bend: 2 };
    },
  },
  zoomies: {
    score: (p) => (p.drives.play > 0.68 && p.drives.energy > 0.55 && p.cool("zoomies") ? 0.55 + 0.6 * (p.drives.play - 0.68) / 0.32 + (p.t < p.excitedUntil ? 0.3 : 0) : 0),
    begin(p) { this.t = 0; this.turned = 0; this.last = null; this.dir = p.rand() < 0.5 ? 1 : -1; return `${p.why()} gets the zoomies and races round the room!`; },
    tick(p, dt) {
      this.t += dt;
      const c = [0, 0.1], R = 1.9, w = p.bolt.where(), a = Math.atan2(w.y - c[1], w.x - c[0]);
      if (this.last !== null && Math.hypot(w.x - c[0], w.y - c[1]) > 1.2) this.turned += Math.abs(wrap(a - this.last));
      this.last = a;
      if (this.turned > 4.5 * Math.PI || this.t > 30) { p.drives.play = Math.max(0, p.drives.play - 0.5); return null; }
      const to = [c[0] + R * Math.cos(a + this.dir * 0.75), c[1] + R * Math.sin(a + this.dir * 0.75)];
      // turn onto the circle at a trot first, then speed up (turning hard while speeding up tips Bolt over)
      const on = Math.abs(Math.hypot(w.x - c[0], w.y - c[1]) - R) < 0.4 && Math.abs(p.body.bearingTo(to)) < 0.6;
      this.onFor = on ? (this.onFor || 0) + dt : 0;
      return { kind: "steer", to, cruise: this.onFor > 1 ? 1.05 : 0.5, bend: this.onFor > 1 ? 1 : 2 };
    },
  },
  fetch: {   // a thrown ball: chase it, pick it up, bring it to the person, drop it at their feet
    score: (p) => (p.carrying && p.person.present ? 1.2 : p.ballThrown && p.person.present && p.drives.energy > 0.12 ? 1.05 : 0),
    begin(p) { this.t = 0; this.phase = "chase"; return `${p.why("fetch")} chases the ball!`; },
    tick(p, dt) {
      this.t += dt;
      if (this.t > 60 || !p.person.present) { p.ballThrown = false; p.dropBall(); return null; }
      if (this.phase === "chase") {
        const b = ballPos(p.bolt), m = mouth(p.bolt), w = p.bolt.where(), d = Math.hypot(b[0] - w.x, b[1] - w.y);
        const reach = Math.hypot(b[0] - m[0], b[1] - m[1]) < 0.24 || (d < 0.65 && Math.abs(p.body.bearingTo(b)) < 0.5);
        if (reach && b[2] < 0.3) { p.pickUp(); this.phase = "bring"; p.log("Got it! Bolt brings the ball back.", "good"); }
        else if (this.t > 30) { p.ballThrown = false; p.log("Bolt can't get at the ball and gives up.", "bad"); return null; }
        else {
          // by a wall or the furniture: walk to the nearest spot Bolt fits, then reach with its nose (Bolt
          // can't walk backwards, so it must never push its nose into a corner)
          const spot = p.planner.clampIn([b[0], b[1]]);
          if (dist2(spot, b) > 0.05) {
            if (!p.body.at(spot) && !p.body.stuckAt(spot)) return { kind: "go", to: spot, cruise: d < 1.2 ? 0.3 : 1.0, near: 0.2 };
            return { kind: "face", to: [b[0], b[1]], pose: "stand", tol: 0.3 };
          }
          // slow right down for the last metre so the ball isn't kicked away (only the snout should reach it)
          return d < 1.0 ? { kind: "steer", to: [b[0], b[1]], cruise: 0.25, bend: 2 } : { kind: "go", to: [b[0], b[1]], cruise: 1.0, near: 0.2 };
        }
      }
      if (this.phase === "bring") {
        if (p.personDist() > 1.3) return { kind: "go", to: p.spotNear(p.personPos(), 0.85), cruise: 0.8, near: 0.3 };
        if (!p.body.still) return { kind: "pose", pose: "stand" };
        p.dropBall(); p.ballThrown = false; p.delivered += 1; this.phase = "sit"; this.t2 = 0; this.lock = true;
        p.drives.play = Math.max(0, p.drives.play - 0.15); p.drives.attention = Math.max(0, p.drives.attention - 0.2);
        p.log("Bolt drops the ball at your feet and sits. Throw it again?", "good");
        p.fetched();
      }
      if ((this.t2 += dt) > 6) return null;
      return { kind: "face", to: p.personPos(), pose: "sit", tol: 0.4 };
    },
  },
  greet: {
    score: (p) => (p.person.present && p.arrivedAt !== null && !p.greeted && p.t - p.arrivedAt < 30 ? 1.3 : 0),
    begin(p) {
      this.t = 0; this.done = false; p.greeted = true;
      const n = p.memory.data.name, missed = p.missedFor;
      return `${n ? n + " is home!" : "Someone's home!"} Bolt trots over to say hello${missed > 3600 ? ": it's been a while" : ""}.`;
    },
    tick(p, dt) {
      this.t += dt;
      if (this.t > 30) return null;
      if (!this.done) {
        if (p.personDist() > 1.3) return { kind: "go", to: p.spotNear(p.personPos(), 0.9), cruise: 0.95, near: 0.3 };
        this.done = true; this.t = 24; this.lock = true; p.drives.attention = Math.max(0, p.drives.attention - 0.35); p.excite(10);
        p.affection(0.03); p.log("Bolt sits in front of you, tail going like mad.", "good");
      }
      return { kind: "face", to: p.personPos(), pose: "sit", tol: 0.4 };
    },
  },
  follow: {
    score: (p) => (!p.person.present ? 0 : p.followUntil > p.t ? 1.1 : p.personMoving() ? 0.25 + 0.6 * p.drives.attention : 0),
    begin(p) { this.t = 0; this.quiet = 0; this.moving = true; return p.followUntil > p.t ? "Bolt follows you." : `${p.why()} tags along after you.`; },
    tick(p, dt) {
      this.t += dt;
      if (!p.person.present) return null;
      this.quiet = p.personMoving() ? 0 : this.quiet + dt;
      if (p.followUntil > p.t && p.personDist() < 2.0) p.markDone("follow");
      if (p.followUntil < p.t && (this.quiet > 8 || this.t > 90)) return null;
      const d = p.personDist();
      if (d > 1.7) this.moving = true;
      if (this.moving && d < 1.25) this.moving = false;
      if (this.moving) return { kind: "go", to: p.spotNear(p.personPos(), 1.0), cruise: clamp(0.4 + 0.5 * (d - 1), 0.4, 1.1), near: 0.3 };
      return { kind: "face", to: p.personPos(), pose: this.quiet > 3 ? "sit" : "stand", tol: 0.5 };
    },
  },
  beg: {
    score: (p) => (p.person.present && p.drives.attention > 0.45 && p.cool("beg") ? 0.15 + 0.75 * p.drives.attention : 0),
    begin(p) { this.t = 0; this.moving = true; return `${p.why("beg")} comes to sit in front of you and stares, hoping for attention.`; },
    tick(p, dt) {
      this.t += dt;
      if (this.t > 45 || p.drives.attention < 0.25) return null;
      const d = p.personDist();
      if (d > 1.6) this.moving = true;
      if (this.moving && d < 1.2) this.moving = false;
      if (this.moving) return { kind: "go", to: p.spotNear(p.personPos(), 0.9), cruise: 0.7, near: 0.3 };
      return { kind: "face", to: p.personPos(), pose: "sit", tol: 0.4 };
    },
  },
  nap: {
    score: (p) => (p.napping ? 1.0 : clamp((Math.max(1 - p.drives.energy, p.drives.rest) - 0.5) * 2.5, 0, 1.15) + (p.napAsked > p.t ? 1 : 0)),
    begin(p) { this.t = 0; this.phase = "go"; return `${p.why("nap")} heads to the bed for a nap.`; },
    tick(p, dt) {
      this.t += dt;
      const bed = [p.room.bed[0] + 0.15, p.room.bed[1] + 0.1];
      if (this.phase === "go") {
        if (this.t < 40 && !p.body.at(bed) && !p.body.stuckAt(bed)) return { kind: "go", to: bed, cruise: 0.5, near: 0.35 };
        this.phase = "sleep";
      }
      p.napping = p.body.posed("lie");
      if (p.drives.energy > 0.92 && p.drives.rest < 0.2) { p.napping = false; p.log("Bolt wakes up, stretches, and looks around.", "info"); return null; }
      return { kind: "pose", pose: "lie" };
    },
    end(p) { p.napping = false; },
  },
  sulk: {
    score: (p) => (p.person.present && p.ignored > 100 && p.drives.attention > 0.6 && p.cool("sulk") ? 1.0 : 0),
    begin(p) {
      this.t = 0; this.phase = 0;
      const pp = p.personPos(), opts = [[-3.8, 1.5], [3.6, 0.8], [-3.5, -1.2], [3.6, -1.0], [-1.0, 2.2], [1.0, -3.0], [0, 2.2]].filter((s) => p.planner.free(s));
      this.spot = opts.sort((a, b) => dist2(b, pp) - dist2(a, pp))[0];
      const away = Math.atan2(this.spot[1] - pp[1], this.spot[0] - pp[0]);
      this.pre = p.planner.clampIn([this.spot[0] - 0.8 * Math.cos(away), this.spot[1] - 0.8 * Math.sin(away)]);
      this.away = [this.spot[0] + 3 * Math.cos(away), this.spot[1] + 3 * Math.sin(away)];
      return `${p.why("sulk")} walks off and lies down with its back to you.`;
    },
    tick(p, dt) {
      this.t += dt;
      if (this.t > 70) { p.ignored = 40; return null; }
      if (this.phase === 0) { if (p.body.at(this.pre) || p.body.stuckAt(this.pre)) this.phase = 1; else return { kind: "go", to: this.pre, cruise: 0.45, near: 0.35 }; }
      if (this.phase === 1) { if (p.body.at(this.spot) || p.body.stuckAt(this.spot)) this.phase = 2; else return { kind: "go", to: this.spot, cruise: 0.35, near: 0.3 }; }
      if (this.phase === 2) { if (p.body.still && p.body.faces(this.away, 0.6)) this.phase = 3; else return { kind: "face", to: this.away, pose: "stand", tol: 0.5 }; }
      return { kind: "pose", pose: "lie" };
    },
  },
  wait: {   // the person is out: wait by the door
    score: (p) => (!p.person.present && p.t - p.leftAt > 15 ? 0.3 + 0.45 * p.drives.attention : 0),
    begin(p) { this.t = 0; this.phase = "go"; return `${p.why("wait")} goes to wait by the door.`; },
    tick(p, dt) {
      this.t += dt;
      if (p.person.present) return null;
      const d = p.room.door, spot = [d[0], d[1] + 0.95];
      if (this.phase === "go") {
        if (!p.body.at(spot) && !p.body.stuckAt(spot) && this.t < 40) return { kind: "go", to: spot, cruise: 0.5, near: 0.3 };
        this.phase = "face";
      }
      if (this.phase === "face") { if (p.body.still && p.body.faces(d, 0.6)) this.phase = "wait"; else return { kind: "face", to: d, pose: "stand", tol: 0.5 }; }
      if (this.t > 90 && p.drives.rest > 0.6) return null;
      return { kind: "pose", pose: this.t < 25 ? "sit" : "lie" };
    },
  },
  spin: {   // one tight circle: a slow trot turning hard
    score: () => 0,
    begin() { this.t = 0; this.turned = 0; this.last = null; return "Bolt spins round in a circle."; },
    tick(p, dt) {
      this.t += dt;
      const w = p.bolt.where();
      if (this.last !== null) this.turned += wrap(w.yaw - this.last);
      this.last = w.yaw;
      if (Math.abs(this.turned) > 2 * Math.PI || this.t > 15) return null;
      const ahead = w.yaw + 1.2;
      return { kind: "steer", to: [w.x + Math.cos(ahead), w.y + Math.sin(ahead)], cruise: 0.3, bend: 0 };
    },
  },
  bowl: {
    score: () => 0,
    begin() { this.t = 0; this.eat = 0; this.phase = "go"; return "Bolt trots to the food bowl."; },
    tick(p, dt) {
      this.t += dt;
      const [bx, by] = p.room.bowl, spot = [bx - 0.55, by - 0.4];
      if (this.t > 40) return null;
      if (this.phase === "go") { if (!p.body.at(spot) && !p.body.stuckAt(spot)) return { kind: "go", to: spot, cruise: 0.7, near: 0.25 }; this.phase = "face"; }
      if (this.phase === "face") { if (p.body.still && p.body.faces([bx, by], 0.35)) this.phase = "eat"; else return { kind: "face", to: [bx, by], pose: "stand", tol: 0.25 }; }
      if ((this.eat += dt) > 4) { p.drives.energy = Math.min(1, p.drives.energy + 0.15); p.log("Bolt licks the bowl clean.", "info"); return null; }
      return { kind: "pose", pose: "stand" };
    },
  },
  jump: {
    score: () => 0,
    begin() { this.t = 0; this.phase = "runup"; return "Bolt lines up for the hurdle."; },
    tick(p, dt) {
      this.t += dt;
      const h = p.room.hurdle, dir = [Math.cos(h.yaw), Math.sin(h.yaw)], run = [h.at[0] - 2.0 * dir[0], h.at[1] - 2.0 * dir[1]];
      const ahead = [h.at[0] + 3 * dir[0], h.at[1] + 3 * dir[1]];
      if (this.t > 60) return null;
      if (this.phase === "runup") {
        if (!p.body.at(run) && !p.body.stuckAt(run)) return { kind: "go", to: run, cruise: 0.7, near: 0.25 };
        this.phase = "line";
      }
      if (this.phase === "line") {
        if (!(p.body.still && p.body.faces(ahead, 0.2))) return { kind: "face", to: ahead, pose: "stand", tol: 0.12 };
        this.phase = "jump"; p.body.jumpResult = null;
      }
      if (this.phase === "jump") {
        if (p.body.jumpResult) {
          const r = p.body.jumpResult; this.result = r; this.phase = "land"; this.t2 = 0;
          p.log(r.over && !r.dropped ? "Clean jump over the bar!" : r.dropped ? "Over, but the bar came down." : "Bolt didn't make it over.", r.over && !r.dropped ? "good" : "bad");
          if (r.dropped) p.bolt.resetBar(0);
          return { kind: "pose", pose: "stand" };
        }
        return { kind: "jump", k: 0 };
      }
      if ((this.t2 += dt) > 2.5) return null;   // stand a while after landing before anything else
      return { kind: "pose", pose: "stand" };
    },
  },
};
for (const k of Object.keys(B)) B[k].name = k;

// What Bolt does when told: these run in place of the drive-picked behaviours, until done or released.
const COMMAND_TEXT = {
  sit: "sits", lie: "lies down", stand: "stands up", stay: "stays put", come: "comes to you", follow: "follows you",
  fetch: "goes for the ball", jump: "goes to jump the hurdle", spin: "spins round", nap: "goes to bed", bowl: "goes to the food bowl",
  zoomies: "races round the room", explore: "goes off to sniff around", attend: "looks at you",
};
class Obey {
  constructor(p, intent) {
    this.name = "obey"; this.intent = intent; this.t = 0; this.done = false; this.since = p.t; this.hold = 0;
    this.sub = null;
    const sub = { fetch: "fetch", jump: "jump", spin: "spin", nap: "nap", bowl: "bowl", zoomies: "zoomies", explore: "explore" }[intent];
    if (sub) { this.sub = Object.create(B[sub]); this.subText = this.sub.begin(p); this.credit = sub; }
    else this.credit = intent;
  }
  tick(p, dt) {
    this.t += dt;
    const i = this.intent, body = p.body;
    const ok = () => { if (!this.done) { this.done = true; p.obeyed(this); } };
    if (this.sub) {
      const d = this.sub.tick(p, dt);
      if (i === "fetch" && p.delivered > this.delivered0) ok();
      if (i === "jump" && this.sub.result) ok();
      if (i === "spin" && Math.abs(this.sub.turned) > 2 * Math.PI) ok();
      if (i === "nap" && body.posed("lie")) ok();
      if (i === "bowl" && this.sub.eat > 0) ok();
      if ((i === "zoomies" || i === "explore") && this.t > 2) ok();
      if (!d && this.sub.end) this.sub.end(p);
      return d;
    }
    if (i === "sit" || i === "lie" || i === "stand" || i === "stay") {
      if (i === "stay" && !this.pose) this.pose = body.still ? body.pose : "stand";
      const pose = i === "stay" ? this.pose : i;
      if (body.posed(pose) || (i === "stay" && body.still && this.t > 0.5)) ok();
      if (this.done && (this.hold += dt) > this.holdFor(p)) return null;
      if (this.t > 20 && !this.done) return null;
      return { kind: "pose", pose };
    }
    if (i === "come") {
      const d = p.personDist();
      if (!p.person.present) return null;
      if (d > 1.8) this.arrived = false;
      if (!this.arrived) {
        if (d > 1.25) return { kind: "go", to: p.spotNear(p.personPos(), 0.85), cruise: 1.0, near: 0.3 };
        this.arrived = true;
      }
      if (body.still) ok();
      if (this.done && (this.hold += dt) > this.holdFor(p)) return null;
      return { kind: "face", to: p.personPos(), pose: "sit", tol: 0.4 };
    }
    if (i === "follow") {
      p.followUntil = p.t + 90;
      if (p.personDist() < 2.0) ok();
      return null;   // the follow behaviour takes it from here
    }
    if (i === "attend") {
      if (body.still && body.faces(p.personPos(), 0.45)) ok();
      if (this.done && (this.hold += dt) > 3) return null;
      if (!p.person.present || this.t > 8) return null;
      return { kind: "face", to: p.personPos(), pose: body.pose === "lie" ? "lie" : body.pose, tol: 0.35 };
    }
    return null;
  }
  end(p) { if (this.sub && this.sub.end) this.sub.end(p); }
  holdFor(p) {
    const base = this.intent === "stay" || p.stayUntil > p.t ? 60 : this.intent === "come" || this.intent === "stand" ? 8 : 25;
    return base * (0.6 + 0.8 * p.memory.data.affection);
  }
}

// ---------- the pet ----------
export class Pet {
  constructor(bolt, start, { room = ROOM, memory = new Memory(), seed = 1, understander = understand, wallClock = () => Date.now() } = {}) {
    this.bolt = bolt; this.start = start; this.room = room; this.memory = memory; this.rand = rng(seed); this.understander = understander;
    this.clock = wallClock;
    setupRoom(bolt, start, room);
    this.hasHurdle = !!(bolt.brains.jump && bolt.real && bolt.real.length);
    this.planner = new Planner(room);
    this.body = new Body(bolt, room);
    this.t = 0; this.events = []; this.falls = 0; this.fallen = 0;
    this.drives = { energy: 0.8, play: 0.45, attention: 0.35, curiosity: 0.6, rest: 0.15 };
    this.person = { x: room.door[0], y: room.door[1] - 1, present: false, goal: null, speed: 1.2, trail: [] };
    this.arrivedAt = null; this.greeted = true; this.leftAt = -1e9; this.missedFor = 0;
    this.ignored = 0; this.lastTalk = -1e9; this.excitedUntil = 0; this.followUntil = 0; this.stayUntil = 0; this.napAsked = 0;
    this.current = null; this.since = 0; this.cooldown = {}; this.lastCredit = null; this.lastSniff = null;
    this.carrying = false; this.ballThrown = false; this.delivered = 0; this.napping = false;
    this.stats = { time: {}, runs: {}, commands: [], moods: {} };
    this.pending = [];   // commands heard, waiting for a sulky/sleepy Bolt to get round to them
    bolt.teleport(start, [-1.5, -0.5], 0.4); placeHurdle(bolt, room);
    setBall(bolt, [...room.ball.start, room.ball.r]);
    this.log(this.memory.data.name ? `Bolt is home. It remembers ${this.memory.data.name}.` : "Bolt is home alone.", "info");
  }
  // ----- helpers for behaviours -----
  log(text, kind = "info") { this.events.push({ t: this.t, text, kind }); this.memory.note(text, this.t); }
  en() { return clamp((this.drives.energy - 0.15) / 0.35, 0, 1); }
  cool(name) { return !(this.cooldown[name] > this.t); }
  excite(s) { this.excitedUntil = Math.max(this.excitedUntil, this.t + s); }
  affection(d) { const m = this.memory.data; m.affection = clamp(m.affection + d, 0, 1); this.memory.save(); }
  personPos() { return [this.person.x, this.person.y]; }
  personDist() { const w = this.bolt.where(); return Math.hypot(this.person.x - w.x, this.person.y - w.y); }
  personMoving() { const tr = this.person.trail; return tr.length > 1 && dist2(tr[0], tr[tr.length - 1]) > 0.6; }
  ballNearPerson() { const b = ballPos(this.bolt); return this.person.present && Math.hypot(b[0] - this.person.x, b[1] - this.person.y) < 1.3; }
  // a free spot `r` m from p, on Bolt's side of it
  spotNear(p, r) {
    const w = this.bolt.where();
    let a = Math.atan2(w.y - p[1], w.x - p[0]);
    for (const da of [0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, Math.PI]) {
      const s = [p[0] + r * Math.cos(a + da), p[1] + r * Math.sin(a + da)], c = this.planner.clampIn(s);
      if (dist2(c, s) < 0.05) return s;
    }
    return this.planner.clampIn([p[0] + r * Math.cos(a), p[1] + r * Math.sin(a)]);
  }
  // the drive behind a choice, for the log ("Bolt is bored, ...")
  why(kind) {
    const m = this.mood();
    if (kind === "fetch") return "Ball! Bolt";
    if (kind === "nap") return this.drives.energy < 0.3 ? "Worn out, Bolt" : "Sleepy, Bolt";
    if (kind === "beg") return "Wanting a fuss, Bolt";
    if (kind === "sulk") return "Ignored for too long, Bolt";
    if (kind === "wait") return "Missing you, Bolt";
    return { happy: "Happy, Bolt", excited: "Excited, Bolt", sleepy: "Drowsy, Bolt", bored: "Bored, Bolt", sulky: "Grumpy, Bolt" }[m];
  }
  mood() {
    const d = this.drives;
    const doing = this.current ? this.current.name : "";
    if (doing === "sulk" || (this.person.present && this.ignored > 90 && d.attention > 0.6)) return "sulky";
    if (d.energy < 0.3 || d.rest > 0.72 || this.napping) return "sleepy";
    if ((d.play > 0.85 && d.energy > 0.5) || this.t < this.excitedUntil) return "excited";
    if (d.curiosity + d.play > 0.8 && (["", "chill", "wait", "beg"].includes(doing) || (doing === "explore" && d.curiosity > 0.7))) return "bored";
    return "happy";
  }
  label() { return this.current ? (this.current.name === "obey" ? (COMMAND_TEXT[this.current.intent] ? `told: ${this.current.intent === "lie" ? "lie down" : this.current.intent}` : "obeying") : LABEL[this.current.name]) : "–"; }
  // how the tail should go: wag rate (Hz) and size (rad), from the mood
  tail() {
    const m = this.mood();
    return { happy: [2.2, 0.35], excited: [4.5, 0.5], sleepy: [0.6, 0.12], bored: [1.0, 0.15], sulky: [0.3, 0.05] }[m];
  }
  // ----- the person -----
  arrive() {
    if (this.person.present) return;
    const [dx, dy] = this.room.door;
    this.person.present = true; this.person.x = dx; this.person.y = dy + 0.3; this.person.goal = [dx + 0.3, dy + 1.8]; this.person.trail = [];
    this.arrivedAt = this.t; this.greeted = false;
    const m = this.memory.data, now = this.clock();
    this.missedFor = m.lastSeen ? (now - m.lastSeen) / 1000 : 0;
    m.lastSeen = now; m.visits += 1; this.memory.save();
    this.drives.attention = Math.min(1, this.drives.attention + 0.3); this.excite(15); this.ignored = 0;
    if (this.napping) this.log("Bolt hears the door and wakes up!", "info");
    this.interrupt();
    this.log(`${m.name || "You"} came home.`, "you");
  }
  leave() {
    if (!this.person.present) return;
    this.person.present = false; this.leftAt = this.t; this.memory.data.lastSeen = this.clock(); this.memory.save();
    this.supersede();
    this.followUntil = 0;
    if (this.carrying) this.dropBall();
    this.log(`${this.memory.data.name || "You"} went out.`, "you");
  }
  walkPerson(p) { this.person.goal = this.planner.clampIn(p); }
  movePerson(p) { this.person.x = clamp(p[0], -this.room.half[0] + 0.3, this.room.half[0] - 0.3); this.person.y = clamp(p[1], -this.room.half[1] + 0.3, this.room.half[1] - 0.3); this.person.goal = null; }
  personStep(dt) {
    const P = this.person;
    if (P.goal) {
      const dx = P.goal[0] - P.x, dy = P.goal[1] - P.y, d = Math.hypot(dx, dy);
      if (d < 0.05) P.goal = null; else { const s = Math.min(d, P.speed * dt); P.x += (dx / d) * s; P.y += (dy / d) * s; }
    }
    if ((this.trailT = (this.trailT || 0) + dt) > 0.25) { this.trailT = 0; P.trail.push([P.x, P.y]); if (P.trail.length > 12) P.trail.shift(); }
  }
  // the person throws the ball toward a point (default: a random open spot 2.5-4 m away)
  throwBall(to) {
    if (!this.person.present) return false;
    if (this.carrying) this.dropBall();
    const P = this.person;
    if (!to) {
      for (let k = 0; k < 30; k++) {
        const a = this.rand() * 2 * Math.PI, r = 2 + 1.5 * this.rand(), s = [P.x + r * Math.cos(a), P.y + r * Math.sin(a)];
        if (Math.abs(s[0]) < this.room.half[0] - 1.3 && Math.abs(s[1]) < this.room.half[1] - 1.3 && this.planner.free(s)) { to = s; break; }
      }
      to = to || [0, 0];
    }
    const from = [P.x, P.y, 1.1], T = 1.2, g = 9.81;   // a gentle lob
    const v = [(to[0] - from[0]) / T, (to[1] - from[1]) / T, (this.room.ball.r - from[2] + 0.5 * g * T * T) / T];
    ballSolid(this.bolt, true); setBall(this.bolt, from, v);
    this.ballThrown = true; this.ballThrowAt = this.t;
    // a throw releases a sit or a come (not a "stay": a good dog waits for the word)
    if (this.current && this.current.name === "obey" && this.current.intent !== "stay" && !(this.stayUntil > this.t) && this.current.done) this.interrupt();
    if (this.current && this.current.name !== "obey" && this.current.name !== "fetch") this.interrupt();
    this.drives.play = Math.min(1, this.drives.play + 0.2); this.drives.curiosity = Math.min(1, this.drives.curiosity + 0.1);
    this.ignored = 0; this.lastTalk = this.t; this.excite(8);
    this.log("You throw the ball.", "you");
    return true;
  }
  pickUp() { this.carrying = true; ballSolid(this.bolt, false); }
  dropBall() {
    if (!this.carrying) return;
    this.carrying = false;
    // put it down a little ahead of the nose, clear of the front paws (a ball under the paws trips Bolt)
    const m = mouth(this.bolt), w = this.bolt.where(), r = this.room.ball.r;
    setBall(this.bolt, [m[0] + 0.12 * Math.cos(w.yaw), m[1] + 0.12 * Math.sin(w.yaw), r + 0.01]); ballSolid(this.bolt, true);
  }
  holdBall() { if (this.carrying) { const m = mouth(this.bolt), v = this.bolt.data.qvel; setBall(this.bolt, m, [v[0], v[1], 0]); } }
  fetched() { this.lastCredit = { name: "fetch", t: this.t }; }
  // ----- hearing -----
  snapshot() { return { name: "Bolt", person: this.memory.data.name, mood: this.mood(), drives: { ...this.drives }, doing: this.label(), present: this.person.present }; }
  // what the person said -> what Bolt does, and a line describing its reaction
  hear(text) {
    const r = this.understander(text, this.snapshot());
    const go = (intents) => this.act(intents, text);
    return r && typeof r.then === "function" ? r.then(go) : go(r);
  }
  act(intents, text = "") {
    this.memory.data.heard += 1;
    if (text) this.log(`You: “${text}”`, "you");
    this.pending = []; this.fresh = true;   // a new instruction replaces any not started yet
    const replies = [];
    for (const it of intents) replies.push(this.command(it));
    const reply = replies.filter(Boolean).join(" ");
    if (reply) this.log(reply, "bolt");
    return { intents, reply };
  }
  // one intent; returns how Bolt reacts (a sentence)
  command(it) {
    const i = it.intent, m = this.memory.data, d = this.drives;
    this.lastTalk = this.t;
    const social = !["unknown", "bye"].includes(i);
    if (social) { this.ignored = 0; d.attention = Math.max(0, d.attention - 0.08); }
    if (this.current && this.current.name === "sulk" && social && !["scold"].includes(i)) { this.interrupt(); this.cooldown.sulk = this.t + 60; }
    const credit = () => (this.current && this.current.name !== "obey" ? this.current.name : this.current ? this.current.credit : this.lastCredit && this.t - this.lastCredit.t < 10 ? this.lastCredit.name : null);
    switch (i) {
      case "praise": case "pet": {
        const c = credit();
        if (c) this.memory.credit(c, 0.2);
        this.affection(i === "pet" ? 0.06 : 0.04); d.attention = Math.max(0, d.attention - (i === "pet" ? 0.45 : 0.3)); this.excite(2);
        return i === "pet" ? "Bolt leans into your hand, tail wagging." : `Bolt wags happily${c && c !== "obey" ? ` (it'll ${c === "fetch" ? "fetch" : "do that"} more often)` : ""}.`;
      }
      case "scold": {
        const c = credit();
        if (c) { this.memory.credit(c, -0.25); this.cooldown[c] = this.t + 60; }
        this.affection(-0.03);
        const stop = this.current && this.current.name !== "obey" && !["nap", "chill"].includes(this.current.name);
        if (stop) this.interrupt();
        this.cooldown.chill = 0;
        return `Bolt's ears droop${stop ? " and it stops" : ""}.`;
      }
      case "cancel": this.interrupt(); this.stayUntil = 0; return "Bolt stops.";
      case "name_set": m.name = it.name; this.memory.save(); this.affection(0.02); return `Bolt tilts its head. It will remember you as ${it.name}.`;
      case "name_ask": return m.name ? `Bolt knows you: you're ${m.name}${m.visits > 1 ? `, home ${m.visits} times so far` : ""}.` : "Bolt doesn't know your name yet. Tell it: “my name is …”.";
      case "greet":
        if (!this.person.present) { this.arrive(); return ""; }
        this.excite(3);
        return this.current && this.current.name === "greet" ? "Bolt's tail goes even faster." : this.obey("attend", it);
      case "bye": return "Bolt watches you with big eyes.";
      case "status": {
        const mood = this.mood(), top = Object.entries(d).sort((a, b) => b[1] - a[1])[0][0];
        return { happy: "Bolt seems happy and content.", excited: "Bolt is bouncing with excitement!", sleepy: "Bolt yawns: it's sleepy.", bored: "Bolt looks bored: it wants something to do.", sulky: "Bolt is sulking: it wants attention." }[mood] +
          ` (energy ${Math.round(100 * d.energy)}%, strongest wish: ${top})`;
      }
      case "speak": this.excite(3); return "Woof! (Bolt has no speaker yet, so it wags instead.)";
      case "unknown": d.curiosity = Math.min(1, d.curiosity + 0.03); return "Bolt tilts its head. It didn't understand.";
      case "play":
        this.interrupt(); this.stayUntil = 0; this.followUntil = 0;
        if (d.energy < 0.2) return "Bolt is too tired to play.";
        d.play = Math.min(1, d.play + 0.35); this.excite(10);
        return "Bolt bounces off, ready to play!";
      case "nap":
        if (it.soft && d.energy > 0.55 && d.rest < 0.5) return "Bolt isn't tired at all, and wags to prove it.";
        this.napAsked = this.t + 20;
        return this.obey("nap", it) || "Bolt heads for its bed.";
      case "jump":
        if (!this.hasHurdle) return "There's no hurdle here.";
        if (d.energy < 0.15) return "Bolt is too tired to jump.";
        return this.obey("jump", it);
      case "fetch":
        if (!this.person.present) return "There's nobody to throw the ball.";
        if (d.energy < 0.12) return "Bolt is too tired to fetch.";
        if (this.ballNearPerson() && !this.carrying) this.throwBall();
        else if (!this.ballThrown && !this.carrying) this.ballThrown = true;
        return this.obey("fetch", it);
      case "follow": this.stayUntil = 0; return this.obey("follow", it);
      case "stay": this.stayUntil = this.t + 60; return this.obey("stay", it);
      case "sit": case "lie": case "stand": case "come": case "spin": case "bowl": case "zoomies": case "explore": case "attend":
        if ((i === "come" || i === "attend") && !this.person.present) return "Bolt can't see you.";
        if (i !== "stand") this.followUntil = 0;
        return this.obey(i, it);
      default: return "";
    }
  }
  obey(intent) {
    const m = this.mood(), delay = m === "sulky" ? 2.0 : m === "sleepy" ? 1.2 : 0;
    if (this.fresh) { this.supersede(); if (this.current && this.current.name === "obey") this.interrupt(); }
    this.fresh = false;
    this.pending.push({ intent, at: this.t + delay, heard: this.t });
    if (intent === "follow") this.followUntil = this.t + 90;
    this.stats.commands.push({ intent, t: this.t, done: null });
    const n = this.memory.data.name;
    return delay ? `Bolt ${m === "sulky" ? "huffs" : "yawns"}, then ${COMMAND_TEXT[intent] || "does it"}.` : `Bolt ${COMMAND_TEXT[intent] || "does it"}${n && this.rand() < 0.2 ? ` for ${n}` : ""}.`;
  }
  obeyed(o) {
    this.markDone(o.intent);
    this.lastCredit = { name: o.credit, t: this.t };
  }
  // commands replaced by a newer one (or cut short by the person leaving) before Bolt finished them
  supersede() { for (const c of this.stats.commands) if (c.done === null) c.superseded = true; }
  markDone(intent) {
    const rec = [...this.stats.commands].reverse().find((c) => c.intent === intent && c.done === null && !c.superseded);
    if (rec) { rec.done = this.t; rec.latency = this.t - rec.t; }
  }
  interrupt() { if (this.current) this.finish(true); }
  // ----- choosing what to do -----
  choose() {
    const busy = this.current && this.current.name === "obey" && !this.current.done;
    if (this.pending.length && this.t >= this.pending[0].at && !busy) {
      const c = this.pending.shift();
      this.interrupt();
      this.current = new Obey(this, c.intent); this.current.delivered0 = this.delivered; this.since = this.t;
      if (c.intent === "follow") { this.current = null; this.followUntil = this.t + 90; }
      else return;
    }
    const cur = this.current;
    if (cur && cur.name === "obey") return;
    const scores = {};
    for (const name of BEHAVIOURS) {
      let s = B[name].score(this);
      if (s <= 0) continue;
      s += 0.15 * this.memory.pref(name) + 0.04 * (this.rand() - 0.5);
      if (this.stayUntil > this.t && !["greet"].includes(name)) s = 0;
      scores[name] = s;
    }
    this.scores = scores;
    const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
    if (!best) return;
    if (cur) {
      if (cur.lock && best[1] < 1.1) return;   // finishing something (sitting after a delivery or a greeting)
      const mine = (scores[cur.name] || 0) + 0.15, dwell = this.t - this.since;
      if (best[0] === cur.name || best[1] < mine || (dwell < 5 && best[1] < 1.0)) return;
      this.interrupt();
    }
    this.launch(best[0]);
  }
  launch(name) {
    if (globalThis.HOME_DEBUG) console.log("launch", name, this.t.toFixed(2), JSON.stringify(this.scores));
    if (this.carrying && name !== "fetch") this.dropBall();
    const b = Object.create(B[name]);
    const line = b.begin(this);
    this.current = b; this.since = this.t;
    this.stats.runs[name] = (this.stats.runs[name] || 0) + 1;
    if (line) this.log(line, "doing");
  }
  finish(cut = false) {
    const c = this.current;
    if (!c) return;
    if (globalThis.HOME_DEBUG) console.log("finish", c.name, cut, this.t.toFixed(2));
    if (c.end) c.end(this);
    if (c.name !== "obey") {
      if (!cut) this.lastCredit = { name: c.name, t: this.t };
      const cd = { explore: 4, play: 25, zoomies: 120, nap: 90, sulk: 60, beg: 40, chill: 2, wait: 10, greet: 5, follow: 10 }[c.name] || 5;
      this.cooldown[c.name] = this.t + cd;
    } else if (["stay", "sit", "lie", "come"].includes(c.intent)) this.stayUntil = 0;
    this.current = null;
  }
  // ----- drives -----
  drift(dt) {
    const d = this.drives, w = this.bolt.data.qvel, speed = Math.hypot(w[0], w[1]), pose = this.body.pose, still = this.body.still;
    const lying = still && pose === "lie", sitting = still && pose === "sit";
    const doing = this.current ? this.current.name : "";
    d.energy += dt * (lying ? (this.napping ? 0.012 : 0.004) : sitting ? 0.0015 : -(0.0016 + 0.0045 * speed));
    d.rest += dt * (this.napping ? -0.02 : lying ? -0.003 : 0.0011);
    d.play += dt * (["play", "zoomies", "fetch"].includes(doing) ? -0.012 : this.napping ? 0 : 0.0025);
    d.curiosity += dt * (doing === "explore" ? -0.004 : this.napping ? 0 : 0.0045);
    // attention is a need for the person: talking, praise, petting and games satisfy it; being near
    // them only a little (a dog ignored at your feet still wants you)
    const close = this.person.present && this.personDist() < 1.6;
    d.attention += dt * (this.person.present ? (close ? 0.004 : 0.006) : 0.0015) * (this.napping ? 0.2 : 1);
    if (doing === "follow" && this.personMoving()) d.attention -= dt * 0.008;
    for (const k of Object.keys(d)) d[k] = clamp(d[k], 0, 1);
    this.ignored = this.person.present && !this.napping ? this.ignored + dt : 0;
  }
  // ----- one control step (1/50 s) -----
  step() {
    const dt = this.bolt.dt, b = this.bolt;
    this.t += dt;
    this.personStep(dt);
    if (this.fallen > 0) return this.recovering(dt);
    this.drift(dt);
    // walk round the ball unless playing with it (stepping on it trips Bolt)
    const withBall = this.carrying || (this.current && (["fetch", "play"].includes(this.current.name) || this.current.credit === "fetch"));
    this.planner.avoid(withBall ? null : ballPos(this.bolt), this.room.ball.r);
    if (this.t < 2) this.cmd = this.body.update({ kind: "pose", pose: "stand" }, dt);   // stand for 2 s first
    else {
      if (!this.current || (this.decide = (this.decide || 0) - dt) <= 0) { this.choose(); this.decide = 0.5; }
      let desire = this.current ? this.current.tick(this, dt) : null;
      if (this.current && !desire) { this.finish(); this.choose(); desire = this.current ? this.current.tick(this, dt) : null; }
      this.cmd = this.body.update(desire || { kind: "pose", pose: this.body.pose }, dt);
    }
    if (this.ballThrown && !this.carrying && this.t - this.ballThrowAt > 1 && ballSpeed(b) < 0.02 && !(this.current && (this.current.name === "fetch" || this.current.intent === "fetch"))) {
      if (this.drives.energy <= 0.12 || this.t - this.ballThrowAt > 20) this.ballThrown = false;
    }
    this.holdBall();
    b.run(this.cmd);
    this.holdBall();
    const name = this.current ? (this.current.name === "obey" ? "obey" : this.current.name) : "idle";
    this.stats.time[name] = (this.stats.time[name] || 0) + dt;
    const mood = this.mood(); this.stats.moods[mood] = (this.stats.moods[mood] || 0) + dt;
    if (b.fell()) { this.falls += 1; this.log("Bolt tumbles over!", "bad"); this.startFall(); }
  }
  // after a fall: lie still a second, then get up with the get-up brain (live.js getUpTick) when the
  // bundle has one (4 s per try, as trained; two tries), then stand a second before carrying on. Without
  // a get-up brain (or if both tries fail) Bolt is set back on its feet where it lies.
  recovering(dt) {
    const b = this.bolt, R = this.rec;
    R.t += dt;
    if (R.stage === "rest") {
      b.run({ command: b.restTick ? "rest" : "lie", turn: 0, speed: 0.2 });
      if (R.t > 1.0) {
        if (b.canGetUp && b.canGetUp() && R.tries < 2) { R.stage = "getup"; R.t = 0; R.tries += 1; b.startGetUp(); }
        else { this.setUpright(); R.stage = "stand"; R.t = 0; }
      }
      return;
    }
    if (R.stage === "getup") {
      b.run({ command: "getup" });
      if (b.standing()) { b.afterGetUp(); R.stage = "stand"; R.t = 0; this.log("Bolt rolls over and gets back on its feet.", "info"); }
      else if (R.t > 4) { R.stage = "rest"; R.t = 0; }
      return;
    }
    b.run({ command: "stand", turn: 0, speed: 0.2 });
    if (R.t > 1.0) {
      this.fallen = 0; this.rec = null; this.body.reset(); this.pending = [];
      if (b.fell()) { this.falls += 1; this.startFall(); }
    }
  }
  startFall() {
    this.fallen = 1; this.rec = { stage: "rest", t: 0, tries: 0 };
    this.interrupt();
    if (this.carrying) this.dropBall();
  }
  setUpright() {
    const b = this.bolt, w = b.where(), ball = ballPos(b);
    b.teleport(this.start, [w.x, w.y], w.yaw);
    placeHurdle(b, this.room); setBall(b, ball);
    this.log("Bolt is helped back on its feet.", "info");
  }
  // for the page: drives, mood and what Bolt remembers
  summary() {
    const m = this.memory.data, likes = Object.entries(m.prefs).filter(([, v]) => Math.abs(v) > 0.05).sort((a, b) => b[1] - a[1]);
    return { mood: this.mood(), doing: this.label(), drives: { ...this.drives }, affection: m.affection, name: m.name, likes, visits: m.visits, lastSeen: m.lastSeen };
  }
}
