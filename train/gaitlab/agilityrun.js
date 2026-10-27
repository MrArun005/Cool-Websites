// Bolt's Agility Run: you are the handler. Bolt runs an agility course on the live simulator (live.js), with its
// real trained brains, and you call the commands: walk, trot, run (canter), jump, sit, down, turn.
//
// Which brain moves the body for each command (one robot, one physics world, the A1 motor model):
//   walk, trot, run    the dog-imitation brain (run 18b, imitate.js): each gait is a clip of a real dog
//   small steering     its turn_left / turn_right clips (they turn slowly: ~0.2-0.3 rad/s in the browser)
//   sharp turns        the run 12 walk brain (live.js), which turns up to ~1.4 rad/s by stepping sideways
//   jump               the real-hurdle jump brain (run 16g, live.js jumpObs, 51 inputs)
//   stand, sit, down   the posture brain (posture-getup 3M)
//   after a fall       the get-up chain (game.js GetUp: run 17e self-righting, then the posture brain)
//
// Every hand-over between brains below was measured headless (agilityrun_test.mjs and the notes next to
// each rule), and they are where Bolt is weak. Measured in the browser physics, 8 tries each:
//   walk -> stand (posture brain) straight away fell 4 of 8; braking with the run 12 brain first: 0 of 8
//   walk -> sit or down (posture brain) straight away: 0 of 8 fell, 8 of 8 sat / lay down
//   trot -> walk straight away fell 2 of 8; trot -> pace for 0.6-1 s -> walk: 0 of 16 (pace is a dog's
//     gait in between); canter -> pace -> walk 0 of 8
//   trot -> a turn clip fell 2 of 8, trot -> the run 12 brain fell: Bolt slows to a walk before it turns
//   walk <-> the run 12 brain: 0 of 8
//   jump: from a trot the jump brain takes over 1.1 m before the bar and clears it; it lands at ~3.8 m/s,
//     faster than any gait, and that is where it falls: handing the landing to a canter, pace, trot or walk
//     fell 4-7 of 8. Best found: the jump brain keeps control until Bolt is slow and upright (it learned to
//     stop after landing), then the posture brain sits it down for a moment and stands it up (3 of 8 fell).
//     A fall is not hidden: Bolt gets up by itself with the get-up chain, which costs time.
import { GetUp, posture } from "./game.js";
import { Imitate } from "./imitate.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
export const FAULT_SECONDS = 5;   // each fault adds 5 s (time-plus-faults scoring)
export const TAKEOVER = { walk: 0.9, trot: 1.1, canter: 1.25 };   // m before the bar the jump brain takes over
export const GAITS = ["walk", "trot", "canter"];

// ---------------------------------------------------------------- the courses
// A course is a handler's line (straights and arcs, course coordinates: x ahead of Bolt at the start, y to its
// left) with obstacles at distances s along it. Each hurdle sits on a straight with at least 3 m before it.
// After each hurdle the line runs straight on for 9 m or more: Bolt lands at ~3.8 m/s and the jump brain needs
// 5-8 m to stop it.
export const LEVELS = [
  { n: 1, name: "Novice", blurb: "A straight line: two 20 cm hurdles and the pause table (sit for 3 s).",
    pieces: [["S", 34]],
    obstacles: [{ kind: "hurdle", s: 6.5, h: 0.20 }, { kind: "hurdle", s: 17, h: 0.20 }, { kind: "table", s: 28, pose: "sit", hold: 3 }],
    finish: 32 },
  { n: 2, name: "Intermediate", blurb: "A curve, three hurdles of 20-30 cm, slalom markers and a down-stay on the table.",
    pieces: [["S", 17], ["A", Math.PI / 2, 3.5], ["S", 28], ["A", -Math.PI / 2, 3.5], ["S", 18]],
    obstacles: [{ kind: "hurdle", s: 6, h: 0.20 }, { kind: "slalom", s: 24, n: 5, gap: 2.4 }, { kind: "hurdle", s: 41, h: 0.25 },
      { kind: "table", s: 59, pose: "lie", hold: 3 }, { kind: "hurdle", s: 66, h: 0.30 }],
    finish: 72 },
  { n: 3, name: "Championship", blurb: "The full course: four hurdles of 30-40 cm, three curves, slalom, a sit and a down.",
    pieces: [["S", 17], ["A", -Math.PI / 2, 3.2], ["S", 32], ["A", Math.PI / 2, 3.2], ["S", 22], ["A", Math.PI / 2, 3.5], ["S", 22]],
    obstacles: [{ kind: "hurdle", s: 6, h: 0.30 }, { kind: "slalom", s: 24.5, n: 6, gap: 2.4 }, { kind: "hurdle", s: 43, h: 0.35 },
      { kind: "table", s: 62, pose: "sit", hold: 3 }, { kind: "hurdle", s: 68, h: 0.40 }, { kind: "table", s: 89, pose: "lie", hold: 3 },
      { kind: "hurdle", s: 96, h: 0.35 }],
    finish: 105 },
];
export const NAMES = { hurdle: "Hurdle", slalom: "Slalom", table: "Pause table" };
export const SLALOM_SWING = 0.6;   // m: the handler's line swings this far either side of the markers
export const TABLE = { half: 0.6 };   // the pause table area: 1.2 m square, painted on the floor
const START_LINE = 1.0;            // m along the line: the clock starts as Bolt crosses it

export function buildCourse(levelIndex) {
  const L = LEVELS[levelIndex], step = 0.05, pts = [];
  let x = 0, y = 0, h = 0, s = 0;
  pts.push({ x, y, h, s });
  for (const [kind, a, r] of L.pieces) {
    const len = kind === "S" ? a : Math.abs(a) * r, n = Math.round(len / step);
    for (let i = 0; i < n; i++) {
      if (kind === "A") h += a / n;
      x += Math.cos(h) * (len / n); y += Math.sin(h) * (len / n); s += len / n;
      pts.push({ x, y, h, s });
    }
  }
  const at = (d) => { const i = clamp(Math.round(d / step), 0, pts.length - 1); return pts[i]; };
  let slot = 0, n = 0;
  const obstacles = L.obstacles.map((o) => {
    const p = at(o.s), out = { ...o, n: ++n, x: p.x, y: p.y, hd: p.h, title: NAMES[o.kind] };
    if (o.kind === "hurdle") { out.slot = slot++; out.title = `Hurdle ${Math.round(o.h * 100)} cm`; }
    if (o.kind === "slalom") {
      out.markers = Array.from({ length: o.n }, (_, k) => { const q = at(o.s + k * o.gap); return { x: q.x, y: q.y, h: q.h, s: o.s + k * o.gap }; });
      out.end = o.s + (o.n - 1) * o.gap;
      out.title = `Slalom, ${o.n} markers`;
    }
    if (o.kind === "table") out.title = `Pause table: ${o.pose === "sit" ? "sit" : "down"} ${o.hold} s`;
    return out;
  });
  const f = at(L.finish), st = at(START_LINE);
  // the standard course time: the line at a brisk trot-and-walk pace (1.0 m/s), plus each jump's steadying
  // (~4 s: the jump brain stops Bolt, then it sits a moment and stands) and each table's stay plus getting
  // into position and up again (~5 s)
  const jumps = obstacles.filter((o) => o.kind === "hurdle").length, tables = obstacles.filter((o) => o.kind === "table");
  const sct = Math.round((L.finish - START_LINE) / 1.0 + 4 * jumps + tables.reduce((t, o) => t + o.hold + 5, 0));
  return { level: L, index: levelIndex, pts, step, at, obstacles, start: { x: st.x, y: st.y, h: st.h, s: START_LINE },
    finish: { x: f.x, y: f.y, h: f.h, s: L.finish }, length: L.finish, sct };
}

// The solid course (live.js setProps): the gate posts, the slalom markers and the judge's number posts are
// real poles Bolt can bump into. In world coordinates for Bolt standing at the start (w = bolt.where()).
export function courseProps(course, w) {
  const c0 = Math.cos(w.yaw), s0 = Math.sin(w.yaw), out = [];
  const put = (x, y, z, size, kind, look, yaw = 0) => out.push({ kind, size, pos: [w.x + c0 * x - s0 * y, w.y + s0 * x + c0 * y, z], yaw: w.yaw + yaw, look });
  const side = (p, d) => [p.x - Math.sin(p.h) * d, p.y + Math.cos(p.h) * d];
  for (const [g, color] of [[course.start, "#2f8a4c"], [course.finish, "#c4372f"]]) for (const d of [1.0, -1.0]) { const [x, y] = side(g, d); put(x, y, 0.65, [0.035, 0.65], "cyl", { color }); }
  for (const o of course.obstacles) {
    if (o.kind === "slalom") o.markers.forEach((m, k) => put(m.x, m.y, 0.5, [0.025, 0.5], "cyl", { color: "#f4f4f0", bands: k % 2 ? "#1f4f86" : "#e3620b" }));
    // the judge's number post: beside the obstacle, on the side away from the line's next bend
    const p = { x: o.x, y: o.y, h: o.hd }, d = o.kind === "slalom" ? 1.5 : 1.3;
    let best = null;
    for (const sgn of [1, -1]) {
      const [x, y] = side(p, sgn * d);
      let clear = Infinity;
      for (const q of course.pts) clear = Math.min(clear, Math.hypot(q.x - x, q.y - y));
      if (!best || clear > best.clear) best = { x, y, clear };
    }
    o.post = [best.x, best.y];
    put(best.x, best.y, 0.45, [0.012, 0.45], "cyl", { color: "#9aa19c" });
  }
  return out;
}

// A new course in the simulator: its solid objects built in, Bolt at the start, the hurdles in place.
export function setupCourse(bolt, course, start) {
  bolt.reset(start);
  const w = bolt.where();
  course.origin = [w.x, w.y]; course.yaw0 = w.yaw;
  bolt.setProps(courseProps(course, w));
  bolt.reset(start);
  placeHurdles(bolt, course);
}
// every hurdle in place, its bar in the cups (after bolt.reset(), which parks them)
export function placeHurdles(bolt, course) {
  const [ox, oy] = course.origin, yaw = course.yaw0, c = Math.cos(yaw), s = Math.sin(yaw), w = { x: ox, y: oy, yaw };
  for (let k = 0; k < bolt.real.length; k++) bolt.placeReal(k, null, 0, 0);
  for (const o of course.obstacles) if (o.kind === "hurdle") bolt.placeReal(o.slot, [w.x + c * o.x - s * o.y, w.y + s * o.x + c * o.y], w.yaw + o.hd, o.h);
}

// ---------------------------------------------------------------- Bolt's body: which brain is in charge
// want: { gait: stand | walk | trot | canter | sit | lie, turn: -1 | 0 | 1 (the handler's turn keys), steer: heading
// error to the handler's line (rad, + left) or null }
export class Body {
  constructor(bolt, dog) {
    this.bolt = bolt; this.dog = dog;
    this.mode = "stand"; this.mt = 0; this.t = 0; this.next = null; this.armed = null; this.speed = 0;
    this.want = { gait: "stand", turn: 0, steer: null };
    this.jumped = null; this.events = [];
  }
  // a fresh start, standing (the bundle's start pose is the posture brain standing)
  reset() {
    this.dog.stop(); this.bolt.afterGetUp(false);
    this.mode = "stand"; this.mt = 0; this.next = null; this.armed = null; this.jumped = null; this.getup = null; this.speed = 0;
  }
  say(text, kind = "info") { this.events.push({ text, kind }); }
  // the gait Bolt is in, as the handler sees it
  get gait() {
    const m = this.mode;
    if (m === "turnclip" || m === "sharp" || m === "brake") return "walk";
    if (m === "pace") return "pace";
    return m;
  }
  get busy() { return ["jump", "land", "settle", "getup"].includes(this.mode); }
  // a gait clip: join it at the moment closest to Bolt's own pose and joint speeds (skills.closest_moment: speeds
  // weigh a little, so it picks the right half of the stride)
  clip(name) {
    const d = this.dog, fresh = !d.playing();
    if (fresh) this.bolt.lastAction = new Float64Array(12);
    d.play(name);
    const c = d.clips[name];
    if (c.loop) {
      const q = d.joints(), qd = Array.from(this.bolt.data.qvel.slice(6, 18));
      let best = 0, bestE = Infinity;
      for (let t = 0; t < c.n; t++) {
        let e = 0;
        for (let i = 0; i < 12; i++) e += (c.f.joints[t][i] - q[i]) ** 2 + 0.002 * (c.f.joint_vel[t][i] - qd[i]) ** 2;
        if (e < bestE) { bestE = e; best = t; }
      }
      d.tau0 = best / c.fps - d.t * d.dt;
    }
  }
  go(mode, extra = {}) {
    const b = this.bolt, was = this.mode, P = ["stand", "sit", "lie", "settle"];
    if (GAITS.includes(mode) || mode === "pace" || mode === "turnclip") {
      if (mode !== "turnclip") this.clip(mode);
    } else if (P.includes(mode) && !P.includes(was)) { this.dog.stop(); b.afterGetUp(false); }
    else if ((mode === "sharp" || mode === "brake") && was !== "sharp" && was !== "brake") {
      // the run 12 walk brain takes over from a moving dog (as from a standing one: zeros, its walk blend)
      this.dog.stop(); b.lastAction = new Float64Array(12); b.command = "stand"; b.speedCmd = 0.5;
    }
    this.mode = mode; this.mt = 0; Object.assign(this, extra);
  }
  // arm a jump over real hurdle k: the jump brain takes over when the bar is `from` m ahead
  arm(k) { this.armed = k; this.bolt.aim(k); }
  disarm() { this.armed = null; }
  fallen() { return this.bolt.up() < 0.5; }

  step(dt) {
    const b = this.bolt, w = this.want, v = b.data.qvel;
    this.t += dt; this.mt += dt;
    this.speed += (Math.hypot(v[0], v[1]) - this.speed) * 0.1;
    if (this.mode !== "getup" && this.fallen()) {
      this.say("Bolt fell", "bad");
      this.dog.stop(); this.armed = null; this.getup = new GetUp(b); this.go("getup");
      this.falls = (this.falls || 0) + 1;
    }
    const m = this.mode;
    // ---- the jump and the landing
    if (m === "getup") {
      const cmd = this.getup.step(dt);
      if (this.getup.failed) { this.say("Bolt could not get up by itself", "bad"); this.stuck = true; return b.run({ command: "rest", turn: 0, speed: 0.2 }); }
      b.run(cmd);
      if (this.getup.done) { this.say(`Bolt got up by itself (${this.getup.secs.toFixed(1)} s)`, "good"); this.mode = "stand"; this.mt = 0; }
      return;
    }
    if (m === "jump") {
      b.jumpTick();
      if (b.hurdleHits()) this.jumpInfo.touched = true;
      const past = this.pawsPast(), J = this.jumpInfo;
      if (!J.cleared && Math.min(...past) > 0.08) { J.cleared = true; J.at = b.jt; }
      // JumpEnv's "landed": on its feet half a second after all four paws passed the bar
      if ((J.cleared && b.jt - J.at >= 25 && b.up() > 0.85 && b.data.qpos[2] > 0.3) || b.jt >= 150) { J.landed = J.cleared; this.go("land"); }
      return;
    }
    if (m === "land") {   // the jump brain stops Bolt (it learned to stand tall and still after landing)
      b.jumpTick();
      if ((Math.hypot(v[0], v[1]) < 1.0 && b.up() > 0.95 && Math.hypot(v[3], v[4]) < 0.5) || b.jt >= 220) this.go("settle");
      return;
    }
    if (m === "settle") {   // a moment sitting, then standing: the posture brain steadies Bolt after the jump
      b.run({ command: this.mt < 1.0 ? "sit" : "stand", turn: 0, speed: 0.2 });
      if (this.mt > 2.0) this.go("stand");
      return;
    }
    // ---- an armed jump: take over at the right distance
    if (this.armed !== null && ["walk", "trot", "canter", "turnclip", "sharp", "pace"].includes(m)) {
      const ahead = b.hurdleAhead(), from = TAKEOVER[m === "trot" || m === "canter" ? m : "walk"];
      if (ahead <= from && ahead > 0.3) {
        this.dog.stop(); b.startJump();
        this.jumpInfo = { k: this.armed, cleared: false, touched: false, landed: false, from: ahead, gait: m };
        this.jumped = this.armed; this.armed = null;
        this.go("jump");
        return;
      }
    }
    // ---- gaits and postures: what the handler wants
    const moving = GAITS.includes(w.gait), steer = w.turn ? w.turn * 1.0 : w.steer;
    const sharp = w.turn !== 0 || (steer !== null && Math.abs(steer) > 0.4);
    switch (m) {
      case "stand":
        if (w.gait === "sit" || w.gait === "lie") this.go(w.gait);
        else if (moving && this.mt > 0.3) {
          if (sharp) this.go("sharp");
          else this.go(w.gait === "canter" ? "trot" : w.gait === "trot" ? "walk" : "walk");
        }
        break;
      case "sit": case "lie":
        if (w.gait === "sit" || w.gait === "lie") { if (w.gait !== m) this.go(w.gait); }
        else this.go("rise");
        break;
      case "rise":   // stand up for 1.2 s before walking on (posture brain)
        if (w.gait === "sit" || w.gait === "lie") this.go(w.gait);
        else if (this.mt > 1.2) this.go("stand");
        break;
      case "walk": case "turnclip":
        if (w.gait === "sit" || w.gait === "lie") this.go(w.gait);           // 0 of 8 fell from a walk
        else if (w.gait === "stand") this.go("brake");
        else if (sharp) this.go("sharp");
        else if ((w.gait === "trot" || w.gait === "canter") && this.mt > 0.8 && (steer === null || Math.abs(steer) < 0.2)) this.go("trot");
        else if (m === "walk" && steer !== null && Math.abs(steer) > 0.2 && this.mt > 0.4) { this.go("turnclip"); this.clip(steer > 0 ? "turn_left" : "turn_right"); this.turnSign = Math.sign(steer); }
        else if (m === "turnclip" && this.mt > 0.4 && (steer === null || Math.abs(steer) < 0.08 || Math.sign(steer) !== this.turnSign)) this.go("walk");
        break;
      case "sharp":
        if (w.gait === "sit" || w.gait === "lie") this.go(w.gait);
        else if (w.gait === "stand") this.go("brake");
        else if (this.mt > 0.5 && !w.turn && (steer === null || Math.abs(steer) < 0.12)) this.go("walk");
        break;
      case "brake":
        if (moving && this.mt < 0.6) this.go("walk");
        else if (this.mt > 1.3) this.go("stand");
        break;
      case "trot":
        if (w.gait === "canter" && this.mt > 1.0 && this.armed === null) this.go("canter");
        else if (w.gait !== "trot" && w.gait !== "canter") this.go("pace", { next: "walk", paceFor: 0.8 });
        else if (sharp && this.armed === null) this.go("pace", { next: "walk", paceFor: 0.8 });   // slow down to turn
        break;
      case "canter":
        if (w.gait !== "canter" || (sharp && this.armed === null)) this.go("pace", { next: w.gait === "trot" && !sharp ? "trot" : "walk", paceFor: 1.0 });
        break;
      case "pace":
        if (this.mt > this.paceFor) this.go(this.next || "walk");
        break;
    }
    this.drive();
  }
  // one control step with the brain in charge
  drive() {
    const b = this.bolt, m = this.mode, w = this.want;
    if (["stand", "sit", "lie", "rise"].includes(m)) return b.run({ command: m === "rise" ? "stand" : m, turn: 0, speed: 0.2 });
    if (m === "sharp") {
      const steer = w.turn ? w.turn * 1.2 : w.steer || 0, top = Math.min(1.4, b.maxTurn(0.5));
      return b.run({ command: "walk", turn: clamp(w.turn ? w.turn * 1.1 : 2.5 * steer, -top, top), speed: clamp(0.25 + 0.6 * Math.cos(steer), 0.3, 0.6) });
    }
    if (m === "brake") return b.run({ command: "walk", turn: 0, speed: this.mt < 0.6 ? 0.4 : 0.2 });
    this.dog.tick();
  }
  // how far each paw is past the aimed bar, along the jump
  pawsPast() {
    const b = this.bolt, d = b.data, [ux, uy] = b.hurdleDir, [hx, hy] = b.hurdleAt;
    return this.dog.feet.map((g) => (d.geom_xpos[3 * g] - hx) * ux + (d.geom_xpos[3 * g + 1] - hy) * uy);
  }
}

// ---------------------------------------------------------------- one judged run
export class Run {
  constructor(bolt, dog, course, start) {
    this.bolt = bolt; this.course = course; this.start = start;
    setupCourse(bolt, course, start);
    this.body = new Body(bolt, dog); this.body.reset();
    this.t = 0; this.clock = null; this.index = 0; this.faults = []; this.events = []; this.over = false;
    this.cmd = "stand"; this.turn = 0; this.autoJump = true; this.autoSteer = true;
    this.fallTime = 0; this.task = this.newTask(); this.lineS = 0; this.lineOff = 0;
    this.stats = { commands: {}, falls: 0, jumps: 0, cleanJumps: 0 };
  }
  // course <-> world
  world(x, y) { const c = Math.cos(this.course.yaw0), s = Math.sin(this.course.yaw0), o = this.course.origin; return [o[0] + c * x - s * y, o[1] + s * x + c * y]; }
  local() {
    const w = this.bolt.where(), o = this.course.origin, c = Math.cos(this.course.yaw0), s = Math.sin(this.course.yaw0), dx = w.x - o[0], dy = w.y - o[1];
    return { x: c * dx + s * dy, y: -s * dx + c * dy, h: wrap(w.yaw - this.course.yaw0) };
  }
  rel(p, hd) { const q = this.local(), dx = q.x - p.x, dy = q.y - p.y; return [Math.cos(hd) * dx + Math.sin(hd) * dy, -Math.sin(hd) * dx + Math.cos(hd) * dy]; }
  get obstacle() { return this.course.obstacles[this.index]; }
  say(text, kind = "info") { this.events.push({ t: this.clock || 0, text, kind }); }
  fault(why) { this.faults.push(why); this.say(`${why}: +${FAULT_SECONDS} s`, "bad"); }
  get time() { return this.clock || 0; }
  get total() { return this.time + FAULT_SECONDS * this.faults.length; }
  newTask() { return { held: 0, marker: 0, missed: false, done: false, count: null }; }
  // the handler's command (W T R S D X, J)
  command(c) {
    if (this.over) return;
    this.stats.commands[c] = (this.stats.commands[c] || 0) + 1;
    if (c === "jump") return this.jumpNow();
    this.cmd = c;
  }
  jumpNow() {   // J: jump the next hurdle if it is ahead and Bolt is lined up for it; else nothing to jump
    const o = this.obstacle, b = this.body;
    if (!o || o.kind !== "hurdle" || b.busy) { this.say("Nothing to jump here", "info"); return false; }
    const [along, side] = this.rel(o, o.hd), ahead = -along;
    if (ahead < 0.4 || ahead > 5 || Math.abs(side) > 0.6) { this.say("No hurdle in front of Bolt", "info"); return false; }
    b.arm(o.slot); if (!GAITS.includes(this.cmd)) this.cmd = "walk";
    this.say("Hup!", "cmd"); return true;
  }
  // the heading error to the handler's line: a point ~1-1.5 m ahead on it (the slalom swings either side)
  steer() {
    const q = this.local(), C = this.course, pts = C.pts;
    let i0 = Math.max(0, Math.round(this.lineS / C.step) - 40), best = i0, bd = Infinity;
    for (let i = i0; i < Math.min(pts.length, i0 + 120); i++) { const d = Math.hypot(pts[i].x - q.x, pts[i].y - q.y); if (d < bd) { bd = d; best = i; } }
    this.lineS = pts[best].s; this.lineOff = bd;
    const look = this.body.mode === "trot" || this.body.mode === "canter" ? 2.5 : 2.0;
    const target = (s) => {
      const p = C.at(s);
      let off = 0;
      for (const o of C.obstacles) if (o.kind === "slalom" && s > o.s - 1.2 && s < o.end + 1.2) {
        // first marker on Bolt's left: the line passes it on the right, then alternates
        const u = clamp((s - o.s) / o.gap, -0.5, o.n - 0.5);
        off = -SLALOM_SWING * Math.cos(Math.PI * u) * Math.min(1, (s - (o.s - 1.2)) / 1.2, (o.end + 1.2 - s) / 1.2);
      }
      return [p.x - Math.sin(p.h) * off, p.y + Math.cos(p.h) * off];
    };
    // inside the slalom aim less far ahead (follow the swing)
    const inSlalom = C.obstacles.some((o) => o.kind === "slalom" && this.lineS > o.s - 1.5 && this.lineS < o.end + 0.5);
    const [tx, ty] = target(Math.min(C.length + 3, this.lineS + (inSlalom ? 0.7 : look)));
    return wrap(Math.atan2(ty - q.y, tx - q.x) - q.h);
  }

  step() {
    if (this.over) return;
    const dt = this.bolt.dt, b = this.body;
    this.t += dt;
    if (this.clock !== null) this.clock += dt;
    // what the handler asks, with auto-steer and auto-jump
    const want = b.want;
    want.gait = this.cmd; want.turn = this.turn;
    const e = this.steer();
    want.steer = this.autoSteer && GAITS.includes(this.cmd) ? e : null;
    const o = this.obstacle;
    if (this.autoJump && o && o.kind === "hurdle" && b.armed === null && !b.busy && b.jumped !== o.slot && GAITS.includes(this.cmd)) {
      const [along, side] = this.rel(o, o.hd);
      if (-along < 6 && -along > 0.5 && Math.abs(side) < 0.6) b.arm(o.slot);
    }
    const wasMode = b.mode;
    b.step(dt);
    while (b.events.length) { const e = b.events.shift(); this.say(e.text, e.kind); if (e.text === "Bolt fell") this.stats.falls++; }
    if (b.mode === "getup" && wasMode !== "getup") this.fellAt = this.t;
    if (b.stuck) this.rescue();
    this.judge();
    // the maximum course time: three times the standard
    if (this.clock !== null && this.clock > 3 * this.course.sct) { this.over = true; this.phase = "time"; this.say("Over the maximum course time", "bad"); }
  }
  // Bolt could not get up in 2 tries: the handler stands it up where it fell (a rare case), 10 s lost
  rescue() {
    const q = this.local(), b = this.body;
    const p = this.course.at(this.lineS);
    const bars = this.course.obstacles.filter((o) => o.kind === "hurdle").map((o) => this.bolt.barPose(o.slot)[0].slice());
    this.bolt.teleport(this.start, this.world(p.x, p.y), this.course.yaw0 + p.h);
    placeHurdles(this.bolt, this.course);   // the teleport parked them; knocked bars stay knocked for the judge
    this.course.obstacles.filter((o) => o.kind === "hurdle").forEach((o, i) => { const H = this.bolt.real[o.slot]; if (Math.hypot(bars[i][0] - H.rest[0], bars[i][1] - H.rest[1], bars[i][2] - H.rest[2]) > 0.03) { this.bolt.data.qpos.set(bars[i], H.q); } });
    this.bolt.mj.mj_forward(this.bolt.model, this.bolt.data);
    if (this.clock !== null) this.clock += 10;
    b.reset(); b.stuck = false; this.cmd = "stand"; this.say("The handler helped Bolt up: +10 s", "bad");
    void q;
  }
  judge() {
    const C = this.course, q = this.local(), b = this.body;
    if (this.clock === null) {   // the clock starts as Bolt crosses the start line
      const [along] = this.rel(C.start, C.start.h);
      if (along > 0) { this.clock = 0; this.say("Timing started", "cmd"); }
    }
    const o = this.obstacle;
    if (!o) {
      const [along] = this.rel(C.finish, C.finish.h);
      if (along > 0 && !b.busy) { this.over = true; this.phase = "finished"; this.say(`Finish: ${this.time.toFixed(2)} s, ${this.faults.length} faults`, "good"); }
      return;
    }
    const k = this.task, [along, side] = this.rel(o, o.hd);
    if (o.kind === "hurdle") {
      if (b.busy && b.mode !== "getup") return;   // judged once Bolt is through the jump
      if (along > 0.4) {
        const jumped = b.jumped === o.slot, dropped = this.bolt.barDropped(o.slot);
        if (jumped) { this.stats.jumps++; if (!dropped && b.jumpInfo && b.jumpInfo.landed) this.stats.cleanJumps++; }
        if (!jumped && Math.abs(side) > 0.7) this.fault("Missed the hurdle");
        else if (!jumped) this.fault("Refusal: ran through the hurdle");
        if (dropped) this.fault("Knocked the bar down");
        if (jumped && !dropped) this.say(b.jumpInfo && b.jumpInfo.touched ? "Brushed the bar, it stayed up" : "Clear jump!", "good");
        b.disarm(); return this.next();
      }
      // fell in front of the bar, or ran past it to the side without jumping: the bar goes back up, try again
      if (b.jumped === o.slot && along < -0.2 && !b.busy) { if (this.bolt.barDropped(o.slot)) { this.fault("Knocked the bar down"); this.bolt.resetBar(o.slot); } b.jumped = null; }
      if (along > -0.3 && Math.abs(side) > 0.7 && this.lineS > C.obstacles[this.index].s + 1.5) { this.fault("Missed the hurdle"); b.disarm(); return this.next(); }
      return;
    }
    if (o.kind === "slalom") {
      while (k.marker < o.n) {
        const m = o.markers[k.marker], [a, sd] = this.rel(m, m.h);
        if (a < 0) break;
        const want = k.marker % 2 ? 1 : -1;   // marker 1 on Bolt's left (Bolt passes to its right), then alternate
        if (Math.sign(sd) !== want || Math.abs(sd) > 1.6) k.missed = true;
        k.marker += 1;
      }
      if (k.marker >= o.n) { if (k.missed) this.fault("Missed a slalom marker"); else this.say("Slalom: clean", "good"); return this.next(); }
      return;
    }
    if (o.kind === "table") {
      const inside = Math.abs(along) < TABLE.half && Math.abs(side) < TABLE.half;
      const pose = posture(this.bolt);
      if (inside && pose === o.pose && (b.mode === "sit" || b.mode === "lie")) {
        k.held += this.bolt.dt;
        const c = Math.max(0, Math.ceil(o.hold - k.held));
        if (c !== k.count) { k.count = c; if (c > 0) this.say(`Judge: ${c}…`, "count"); }
        if (k.held >= o.hold) { this.say("Judge: go!", "good"); return this.next(); }
      } else {
        if (k.held > 0.2) this.say("Broke the stay: the count starts again", "info");
        k.held = 0; k.count = null;
      }
      if (along > TABLE.half + 1.5 && b.mode !== "sit" && b.mode !== "lie") { this.fault("Missed the pause table"); return this.next(); }
    }
  }
  next() { this.index += 1; this.task = this.newTask(); }
  // the result: Clear round (no faults, inside the standard time), Qualified (finished with a total within
  // 1.5 x the standard time), else Keep practising
  result() {
    const C = this.course, finished = this.phase === "finished", total = this.total;
    const rank = !finished ? "Keep practising" : this.faults.length === 0 && this.time <= C.sct ? "Clear round" : total <= 1.5 * C.sct ? "Qualified" : "Keep practising";
    return { finished, time: this.time, faults: this.faults.slice(), total, sct: C.sct, rank, falls: this.stats.falls, jumps: this.stats.jumps, cleanJumps: this.stats.cleanJumps };
  }
}

// ---------------------------------------------------------------- a scripted handler (tests, and the demo)
// Walks into curves and the slalom, trots the straights and into each hurdle, stops on the table with the
// right command and starts again once the judge says go.
export class Bot {
  // rand: a random number generator (game.js rng) for the handler's reaction times
  // style: "walk" (jumps from a walk), "trot" (trots the last 4 m into each jump), "canter" (canters the straights)
  constructor(run, { style = "walk", rand = Math.random } = {}) { this.run = run; this.style = style; this.canter = style === "canter"; this.rand = rand; this.wait = 0.2 + rand(); }
  step() {
    const r = this.run, C = r.course, o = r.obstacle, s = r.lineS, b = r.body;
    if (r.over || b.busy) return;
    if ((this.wait -= r.bolt.dt) > 0) return;
    const curving = (d) => { const a = C.at(s).h, z = C.at(Math.min(C.length, s + d)).h; return Math.abs(wrap(z - a)) > 0.12; };
    let want = this.canter ? "canter" : "trot";
    if (curving(3.5) || r.lineOff > 0.5) want = "walk";
    if (o && o.kind === "slalom" && s > o.s - 3 && s < o.end + 1.2) want = "walk";
    if (o && o.kind === "table") {
      const [along, side] = r.rel(o, o.hd);
      if (along > -4) want = "walk";
      if (along > -0.45 && along < 0.3 && Math.abs(side) < 0.4) want = o.pose;
    }
    if (o && o.kind === "hurdle") {
      // walk in and jump from the walk: the walk can steer (turn clips, the run 12 brain), the trot cannot
      // (each trot starts with a 0.1-0.2 rad swerve, and turning out of a trot fell 5-7 times in 12), and the jump
      // brain wants Bolt square to the bar; or (style "trot") line up at a walk, then trot the last 4 m
      const [along, side] = r.rel(o, o.hd), askew = Math.abs(wrap(r.local().h - o.hd));
      if (along > -8 && along < -0.5) {
        if (this.style !== "trot" || along < -4.5) want = "walk";
        else if (this.committed !== o.n) { if (Math.abs(side) < 0.15 && askew < 0.08) this.committed = o.n; else want = "walk"; }
        else if (Math.abs(side) > 0.5) want = "walk";
      }
    }
    if (want !== r.cmd) { r.command(want); this.wait = 0.2 + 0.3 * this.rand(); }
  }
}

// Create everything for a page or a test: the imitation brain on the same physics.
export function makeDog(bolt, brain, clips) { return new Imitate(bolt, brain, clips); }
