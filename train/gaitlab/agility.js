// Bolt's Agility Cup: a dog-agility course for Bolt's trained brains, on the live simulator (live.js).
//
// Like a real agility competition: a numbered course is set out, each dog runs it in turn, and the
// judge counts faults (5 for a missed weave pole, 5 for a refusal or a wrong tunnel entry, 5 for a
// knocked bar) plus time faults (1 per second over the standard course time). A fall eliminates the
// dog. The best round is the fewest faults, then the fastest time. Each competitor is a different
// trained walk brain, steered round the course by the same handler (the course runner below).
// agility_test.mjs runs many rounds headless.

import { Driver, Jumper, posture, rng } from "./game.js";

export const OBSTACLES = {
  weave: { title: "Weave poles", how: "Weave through every pole, first pole on the left shoulder" },
  tunnel: { title: "Tunnel", how: "Run through the tunnel, in at the near end" },
  table: { title: "Pause table", how: "Lie down on the table while the judge counts 5" },
  jump: { title: "Hurdle", how: "Jump the hurdle without knocking the bar down" },
};
// m: Bolt's body is rigid, so the poles are wider apart than a real 60 cm weave. Bolt follows a smooth
// wave through them (side: how far it swings out at each pole), at most ~1.4 rad/s of turning.
// side 0.5: with solid poles, a 0.4 m swing jammed Ace and Blaze in 2 Championship weaves of 10 (1.0 m
// apart); 0.5 m: 10 of 10 through, no pole missed
export const WEAVE = { poles: 8, spacing: 1.2, side: 0.5, lookahead: 0.5 };
export let WEAVE_SPEED = 0.8;   // m/s through the weaves
export function setWeaveSpeed(v) { WEAVE_SPEED = v; }

// The course: obstacles along a winding path from the start line (course coordinates: x ahead of
// Bolt at the start, y to its left). Each obstacle has an entry point, a heading and a length.
// Classes: Standard, and Championship (12 tighter weave poles, three tunnels, sharper turns, a higher
// bar, a stricter course time). Hurdles need the jump brain (jump_env.py): with solid bars, the trot
// touched even a 5 cm bar every time. Every dog jumps with the same jump brain: the dog stops ~0.6 m
// before the bar, steadies, and jumps from standing (game.js Jumper), ~4.5 s per hurdle.
export const CLASSES = {
  standard: { name: "Standard", poles: 8, spacing: 1.2, tunnels: 2, turn: 0.9, pace: 0.55, bar: 0.15 },
  championship: { name: "Championship", poles: 12, spacing: 1.0, tunnels: 3, turn: 1.2, pace: 0.65, bar: 0.20 },
};
const JUMP_TIME = 4.5;   // s: stop, steady, jump and land, added to the course time per hurdle
export function buildAgility(seed, { jumps = false, cls = "standard" } = {}) {
  const r = rng(seed), C = CLASSES[cls];
  WEAVE.poles = C.poles; WEAVE.spacing = C.spacing;
  const kinds = ["weave", "table"].concat(Array(C.tunnels).fill("tunnel"), jumps ? ["jump", "jump"] : []);
  for (let i = kinds.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
  if (kinds[0] === "table") kinds.push(kinds.shift());              // a course never opens with the table
  let x = 0.5, y = 0, h = 0;
  const obstacles = kinds.map((kind, n) => {
    // up to ~50° between obstacles; a hurdle gets a straighter, longer run-up to line up for the jump
    h += n === 0 ? (r() * 2 - 1) * 0.3 : (r() * 2 - 1) * (kind === "jump" ? 0.5 : C.turn);
    const gap = kind === "weave" ? 2.6 : kind === "jump" ? 3.0 : 2.2, L = kind === "weave" ? WEAVE.spacing * (WEAVE.poles - 1) : kind === "tunnel" ? 3.0 : 0;
    const entry = [x + gap * Math.cos(h), y + gap * Math.sin(h)];
    const exit = [entry[0] + L * Math.cos(h), entry[1] + L * Math.sin(h)];
    x = exit[0]; y = exit[1];
    return { n: n + 1, kind, entry, exit, h, L, ...OBSTACLES[kind], ...(kind === "jump" ? { bar: C.bar, how: `Jump the ${Math.round(C.bar * 100)} cm hurdle without knocking the bar down` } : {}) };
  });
  const last = obstacles[obstacles.length - 1];
  const finish = [last.exit[0] + 2.5 * Math.cos(last.h), last.exit[1] + 2.5 * Math.sin(last.h)];
  // standard course time: the path at the class pace, plus the 5 s on the table and getting on and off it,
  // plus the time each jump takes
  let length = 0, p = [0, 0];
  // the weave counted at the pace dogs really go through solid poles (~0.45 m/s: 12 poles took ~26 s)
  for (const o of obstacles) { length += Math.hypot(o.entry[0] - p[0], o.entry[1] - p[1]) + (o.kind === "weave" ? o.L * C.pace / 0.45 : o.L); p = o.exit; }
  length += Math.hypot(finish[0] - p[0], finish[1] - p[1]);
  const jumpCount = obstacles.filter((o) => o.kind === "jump").length;
  return { seed, cls: C.name, obstacles, finish, finishH: last.h, sct: Math.round(length / C.pace + 9 + JUMP_TIME * jumpCount) };
}

// The solid course (live.js setProps): everything that stands up is a real object Bolt collides with.
// The tunnel is an arch (flat sides, a round roof of 8 panels, 0.86 m wide, 0.85 m high; Bolt is 0.24 m
// wide and ~0.55 m tall), the pause table a 1 m square platform 4 cm high, the weave poles, the gates' and
// the judge's number posts are poles. In world coordinates for Bolt standing at the start (w = bolt.where()).
export const TUNNEL = { r: 0.43, wall: 0.012, panels: 8 };
export const TABLE = { half: 0.5, h: 0.04 };
// the judge's number post for each obstacle: beside it, off the line dogs come in on (at -0.6 m, 0.75 m
// to the side it stood on that line: Ace pushed into the tunnel's post until time ran out)
export const SIGN = [0.2, 1.4];
export function courseProps(course, w) {
  const c0 = Math.cos(w.yaw), s0 = Math.sin(w.yaw), out = [];
  const put = (o, along, side, z, size, kind, look, roll = 0) => {   // a prop in obstacle o's frame
    const h = o ? o.h : 0, e = o ? o.entry : [0, 0], x = e[0] + Math.cos(h) * along - Math.sin(h) * side, y = e[1] + Math.sin(h) * along + Math.cos(h) * side;
    out.push({ kind, size, pos: [w.x + c0 * x - s0 * y, w.y + s0 * x + c0 * y, z], yaw: w.yaw + h, roll, look });
  };
  const gate = (at, h, color) => { for (const y of [0.9, -0.9]) put({ h, entry: at }, 0, y, 0.65, [0.035, 0.65], "cyl", { color }); };
  gate([0.3, 0], 0, "#2f8a4c"); gate(course.finish, course.finishH, "#c4372f");
  for (const o of course.obstacles) {
    put(o, SIGN[0], SIGN[1], 0.45, [0.012, 0.45], "cyl", { color: "#9aa19c" });   // the judge's number post
    if (o.kind === "weave") for (let k = 0; k < WEAVE.poles; k++) put(o, k * WEAVE.spacing, 0, 0.5, [0.02, 0.5], "cyl", { color: "#f4f4f0", bands: k % 2 ? "#1f4f86" : "#c4372f" });
    if (o.kind === "tunnel") {
      const { r, wall, panels } = TUNNEL, look = { color: "#2f6fb5", opacity: 0.55, tunnel: true };
      for (const s of [1, -1]) put(o, o.L / 2, s * (r + wall), r / 2, [o.L / 2, wall, r / 2], "box", look);
      for (let i = 0; i < panels; i++) {
        const th = (i + 0.5) * Math.PI / panels, chord = 2 * (r + wall) * Math.sin(Math.PI / (2 * panels)) + 0.004;
        put(o, o.L / 2, (r + wall) * Math.cos(th), r + (r + wall) * Math.sin(th), [o.L / 2, chord / 2, wall], "box", look, th - Math.PI / 2);
      }
    }
    if (o.kind === "table") put(o, 0, 0, TABLE.h / 2, [TABLE.half, TABLE.half, TABLE.h / 2], "box", { color: "#e3620b" });
  }
  return out;
}
// A new course in the simulator: its solid objects built in, Bolt at the start, the hurdles in place.
export function setupCourse(bolt, course, start) {
  bolt.reset(start);
  bolt.setProps(courseProps(course, bolt.where()));
  bolt.reset(start);
  placeJumps(bolt, course);
}

// Every hurdle on the course is real and solid from the start: solid stands, a bar in its cups (live.js
// placeReal). Bolt stands at the start line (course coordinates begin where it stands).
export function placeJumps(bolt, course) {
  const w = bolt.where(), c = Math.cos(w.yaw), s = Math.sin(w.yaw);
  let slot = 0;
  for (const o of course.obstacles) if (o.kind === "jump") {
    o.slot = slot;
    bolt.placeReal(slot++, [w.x + c * o.entry[0] - s * o.entry[1], w.y + s * o.entry[0] + c * o.entry[1]], w.yaw + o.h, o.bar);
  }
}

// One competitor's round. Call step() 50 times a second.
export class Round {
  constructor(bolt, course, start, name) {
    this.bolt = bolt; this.course = course; this.name = name;
    bolt.reset(start);
    const w = bolt.where();
    this.origin = [w.x, w.y]; this.yaw0 = w.yaw;
    placeJumps(bolt, course);
    this.driver = new Driver(bolt);
    this.t = 0; this.clock = 0; this.index = 0; this.faults = 0; this.notes = [];
    this.phase = "ready"; this.task = null; this.events = []; this.over = false;
    this.cmd = { command: "stand", turn: 0, speed: 0.2 };
  }
  world([x, y]) { const c = Math.cos(this.yaw0), s = Math.sin(this.yaw0); return [this.origin[0] + c * x - s * y, this.origin[1] + s * x + c * y]; }
  local() {
    const w = this.bolt.where(), dx = w.x - this.origin[0], dy = w.y - this.origin[1], c = Math.cos(this.yaw0), s = Math.sin(this.yaw0);
    return [c * dx + s * dy, -s * dx + c * dy];
  }
  // position relative to an obstacle: along its heading from the entry, and to its left
  rel(o) { const [x, y] = this.local(), dx = x - o.entry[0], dy = y - o.entry[1]; return [Math.cos(o.h) * dx + Math.sin(o.h) * dy, -Math.sin(o.h) * dx + Math.cos(o.h) * dy]; }
  at(o, along, side) { return [o.entry[0] + Math.cos(o.h) * along - Math.sin(o.h) * side, o.entry[1] + Math.sin(o.h) * along + Math.cos(o.h) * side]; }
  get obstacle() { return this.course.obstacles[this.index]; }
  say(text, kind = "info") { this.events.push({ t: this.clock, text, kind }); }
  fault(n, why) { this.faults += n; this.notes.push(why); this.say(`${why}: ${n} faults`, "bad"); }
  get timeFaults() { return Math.max(0, Math.ceil(this.clock - this.course.sct)); }
  get total() { return this.faults + this.timeFaults; }

  step() {
    if (this.over) return;
    const dt = this.bolt.dt;
    this.t += dt;
    if (this.phase === "ready") {
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if (this.t >= 2) { this.phase = "run"; this.say(`${this.name}: go!`, "station"); this.begin(); }
    } else {
      this.clock += dt;
      this.play(dt);
      this.unstick(dt);
      // the maximum course time, twice the standard: a dog stuck at an obstacle is eliminated
      if (!this.over && this.clock > 2 * this.course.sct) { this.over = true; this.phase = "eliminated"; this.say(`${this.name}: over the maximum course time, eliminated`, "bad"); }
    }
    const b = this.bolt;
    b.run(this.cmd);
    if (b.fell()) { this.over = true; this.phase = "eliminated"; this.say(`${this.name} fell: eliminated`, "bad"); }
  }
  // Objects are solid: a dog pressed against a pole or a wall, going nowhere, is turned away from it and
  // walked clear for 1.5 s before the handler sends it on again (pushing on into it never got past)
  unstick(dt) {
    const b = this.bolt, w = b.where();
    if (this.free && this.t < this.free.until) { this.cmd = { command: "walk", turn: this.free.turn, speed: 0.3 }; return; }
    if (this.cmd.command !== "walk") { this.lastPos = null; return; }
    if (!this.lastPos || this.t - this.lastPos.t > 1.5) {
      const moved = this.lastPos ? Math.hypot(w.x - this.lastPos.x, w.y - this.lastPos.y) : 1, touching = b.touchingProps();
      if (moved < 0.1 && touching.length) {
        const p = b.props[touching[0]].pos, bearing = Math.atan2(p[1] - w.y, p[0] - w.x) - w.yaw;
        this.free = { until: this.t + 1.5, turn: Math.sin(bearing) > 0 ? -1.2 : 1.2 };   // turn away from it
      }
      this.lastPos = { x: w.x, y: w.y, t: this.t };
    }
  }
  begin() {
    const o = this.obstacle;
    this.task = o ? { stage: "approach", wp: 0, held: 0, settle: 0, checked: 0, missed: false } : { stage: "finish" };
    if (o) this.say(`${o.n}. ${o.title}`, "station");
  }
  next(note) {
    if (note) this.say(note, "good");
    this.index += 1; this.begin();
  }
  waypoints(o) {   // the line the handler sends Bolt along through this obstacle
    if (o.kind === "weave") {
      const s = WEAVE.spacing, w = WEAVE.side;
      // first pole on Bolt's left shoulder: Bolt passes it on the right (negative side), then alternates
      return [[-1.4, -w * 0.5], ...Array.from({ length: WEAVE.poles }, (_, k) => [k * s, k % 2 ? w : -w]), [(WEAVE.poles - 1) * s + 1.2, 0]];
    }
    if (o.kind === "tunnel") return [[-1.2, 0], [0, 0], [o.L, 0], [o.L + 1.0, 0]];
    if (o.kind === "jump") return [[-2.0, 0]];   // line up straight; the jumper does the rest
    return [[0, 0]];
  }

  play(dt) {
    const o = this.obstacle, k = this.task;
    if (!o) {   // run for the finish line
      const f = this.course.finish;
      this.cmd = this.driver.toward(this.world(f), 1.0, false, dt);
      const [x, y] = this.local(), along = Math.cos(this.course.finishH) * (x - f[0]) + Math.sin(this.course.finishH) * (y - f[1]);
      if (along > 0) {
        this.over = true; this.phase = "finished";
        this.say(`${this.name}: ${this.clock.toFixed(2)} s, ${this.total} faults${this.total === 0 ? ", clear round!" : ""}`, this.total === 0 ? "good" : "info");
      }
      return;
    }
    if (o.kind === "table") return this.table(o, dt);
    if (o.kind === "jump" && k.wp > 0) return this.jump(o, dt);
    const wps = this.waypoints(o), [along, side] = this.rel(o);
    let target = wps[k.wp];
    const last = k.wp === wps.length - 1, weaving = o.kind === "weave" && k.wp > 0;
    const cruise = weaving ? WEAVE_SPEED : o.kind === "tunnel" ? 0.7 : 0.8;
    if (weaving) {   // aim a little way ahead on a smooth wave through the poles, like a handler's line
      const a = Math.min(along + WEAVE.lookahead, (WEAVE.poles - 1) * WEAVE.spacing + 1.2);
      const end = (WEAVE.poles - 1) * WEAVE.spacing;
      // keep swinging until past the last pole (straightening at the pole cut into it)
      target = [a, a > end + 0.4 ? 0 : -WEAVE.side * Math.cos(Math.PI * Math.max(0, a) / WEAVE.spacing)];
    }
    this.cmd = this.driver.toward(this.world(this.at(o, target[0], target[1])), k.wp === 0 ? 0.9 : cruise, false, dt, weaving ? 1 : 2);
    // move on to the next waypoint once level with this one
    if (along >= target[0] - (o.kind === "weave" ? 0.05 : 0.2) && (k.wp > 0 || Math.abs(side) < 0.6)) k.wp += 1;
    if (o.kind === "weave") {      // judge each pole as Bolt passes it: pole k must be on the right side
      while (k.checked < WEAVE.poles && along >= k.checked * WEAVE.spacing) {
        const want = k.checked % 2 ? 1 : -1;
        if (Math.sign(side) !== want || Math.abs(side) < 0.05) k.missed = true;
        k.checked += 1;
      }
    }
    if (o.kind === "tunnel" && k.wp === 2 && !k.entered) {   // just went in: through the near end?
      k.entered = true;
      if (Math.abs(side) > 0.4) this.fault(5, "Missed the tunnel entry");
    }
    if (o.kind === "jump") return;   // lined up: the jump starts next step
    if (k.wp >= wps.length || (last && along > target[0])) {
      if (o.kind === "weave" && k.missed) this.fault(5, "Missed a weave pole");
      this.next(o.kind === "weave" && !k.missed ? "Weaves: clean" : o.kind === "tunnel" ? "Tunnel: through" : "");
    }
  }
  jump(o, dt) {   // a real jump from standing (game.js Jumper); a bar knocked down: 5 faults
    const k = this.task;
    if (!k.jumper) k.jumper = new Jumper(this.bolt, this.driver, o.slot);
    this.cmd = k.jumper.step(dt);
    if (k.jumper.done) {
      if (k.jumper.dropped) { this.fault(5, "Knocked the bar down"); this.next(""); }
      else this.next(k.jumper.touched ? "Brushed the bar, it stayed up" : "Clear jump");
    }
  }
  table(o, dt) {   // run up, stop, lie down for the judge's count of 5, get up, go on
    const k = this.task;
    if (k.stage === "approach") {
      this.cmd = this.driver.toward(this.world(o.entry), 0.9, true, dt);
      if (this.cmd.dist < 0.25 || (this.cmd.dist < 0.45 && this.driver.speed <= 0.25)) { k.stage = "brake"; k.settle = 0; }
      return;
    }
    if (k.stage === "brake") {    // a slow trot for 0.5 s before stopping (stopping dead tips Bolt over)
      this.cmd = { ...this.driver.toward(this.world(o.entry), 0.2, true, dt), speed: 0.2 };
      if ((k.settle += dt) > 0.5) { k.stage = "settle"; k.settle = 0; }
      return;
    }
    if (k.stage === "settle") {
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if ((k.settle += dt) > 0.6) { k.stage = "down"; k.count = 6; }   // the judge starts counting at 5
      return;
    }
    if (k.stage === "down") {
      this.cmd = { command: "lie", turn: 0, speed: 0.2 };
      k.held = posture(this.bolt, TABLE.h) === "lie" ? k.held + dt : 0;   // lying on the table, 4 cm up
      const count = Math.max(0, 5 - Math.floor(k.held));
      if (count !== k.count) { k.count = count; if (count > 0) this.say(`Judge: ${count}…`, "count"); }
      if (k.held >= 5) { k.stage = "up"; k.settle = 0; this.say("Judge: go!", "count"); }
      return;
    }
    if (k.stage === "up") {
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if ((k.settle += dt) > 1.0) {
        const [along, side] = this.rel(o);
        if (Math.hypot(along, side) > 0.6) this.fault(5, "Off the table");
        this.driver.speed = 0.2; this.next("Table: done");
      }
    }
  }
}

// The leaderboard: clear rounds first, then fewest faults, then fastest.
export function rank(results) {
  return results.slice().sort((a, b) => (a.eliminated - b.eliminated) || (a.total - b.total) || (a.time - b.time));
}
