// Bolt's Castle: a Takeshi's Castle-style course for Bolt, run on the live simulator (live.js).
//
// Each game builds a new course from shuffled stations: a sprint, a slalom between poles, a narrow
// bridge to cross slowly, a box to sit in, a mat to lie on, from level 3 an ice rink to stop and sit on,
// and the castle gate at the end. The guard shouts surprise commands at any moment ("SIT!", "LIE DOWN!",
// "FREEZE!"), and so can the viewer; from level 3 his cannon shoves Bolt sideways. Five levels, each
// harder (LEVELS); 3 lives: a failed station or a fall costs one (after a fall Bolt gets up by itself
// where it fell, with the get-up brain, and carries on; if it cannot, it restarts at that station).
// Bolt plays on its own: a course runner steers its trained brains from station to station.
// Every hazard is real physics (live.js): the shoves are forces, the ice is low friction.
// game_test.mjs plays many games headless.

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// What each posture looks like in the simulator (measured with the posture brain: stand 43 cm level,
// sit 20.7 cm nose up 30°, lie 12.5 cm level; each settles in under a second).
// ground: the height of what Bolt stands on (the agility pause table is 4 cm up)
export function posture(bolt, ground = 0) {
  const R = bolt.rotation(), q = bolt.data.qpos, v = bolt.data.qvel;
  const h = q[2] - ground, pitch = Math.asin(clamp(R[6], -1, 1)), speed = Math.hypot(v[0], v[1]);
  if (h > 0.18 && h < 0.24 && pitch > 0.35 && pitch < 0.7 && speed < 0.15) return "sit";
  if (h < 0.16 && Math.abs(pitch) < 0.2 && speed < 0.15) return "lie";
  if (h > 0.38 && Math.abs(pitch) < 0.2 && speed < 0.1) return "still";
  return "moving";
}

// How hard each level is. Measured limits (12M walk brain, live.js): shoves up to 60 N sideways never
// toppled Bolt, 80 N toppled it 6 times in 10; on ice of friction 0.06, braking from 1 m/s and sitting
// toppled it 4 times in 6, at 0.12 never.
export const LEVELS = [
  { n: 1, name: "Rookie", bar: 0.12, sprint: 7.0, bridge: 0.30, gates: 4, gateSide: 0.9, gateR: 0.45, shouts: 2, obey: 5.0, cannon: 0, shots: 0, ice: 0 },
  { n: 2, name: "Contender", bar: 0.15, sprint: 6.4, bridge: 0.24, gates: 5, gateSide: 1.0, gateR: 0.40, shouts: 3, obey: 4.0, cannon: 0, shots: 0, ice: 0 },
  { n: 3, name: "Challenger", bar: 0.17, sprint: 6.0, bridge: 0.20, gates: 6, gateSide: 1.1, gateR: 0.38, shouts: 4, obey: 3.5, cannon: 50, shots: 2, ice: 0.12 },
  { n: 4, name: "Warrior", bar: 0.20, sprint: 5.6, bridge: 0.17, gates: 6, gateSide: 1.2, gateR: 0.35, shouts: 5, obey: 3.0, cannon: 65, shots: 3, ice: 0.08 },
  { n: 5, name: "Takeshi", bar: 0.22, sprint: 5.3, bridge: 0.15, gates: 7, gateSide: 1.3, gateR: 0.32, shouts: 6, obey: 2.5, cannon: 75, shots: 4, ice: 0.06 },
];

export const STATIONS = {
  sprint: { title: "Sprint", say: "RUN!", how: (L) => `Cover 6 m in under ${L.sprint} s` },
  slalom: { title: "Slalom", say: "WEAVE!", how: (L) => `Pass between all ${L.gates} pairs of poles` },
  bridge: { title: "Narrow bridge", say: "SLOWLY…", how: (L) => `Cross the 4 m plank (${Math.round(L.bridge * 200)} cm wide) without stepping off or rushing` },
  ice: { title: "Ice rink", say: "SIT ON THE ICE!", how: () => "Run onto the ice, stop on the spot and sit for 2 s without slipping off" },
  sit: { title: "Sit box", say: "SIT!", how: () => "Stop in the box and sit for 2 s" },
  lie: { title: "Lie-down mat", say: "LIE DOWN!", how: () => "Stop on the mat and lie down for 2 s" },
  hurdles: { title: "Hurdles", say: "JUMP!", how: (L) => `Jump 3 hurdles, bars ${Math.round(L.bar * 100)} cm high, without knocking a bar down` },
  gate: { title: "Castle gate", say: "HOME!", how: () => "Reach the castle" },
};
export const SHOUTS = { sit: "SIT!", lie: "LIE DOWN!", still: "FREEZE!" };

// Getting up: stand for 1 s before walking on. (The 2.5M posture brain fell 4 times in 12 standing
// straight up from a sit; run 13's brain practised getting up and fell in none of 48 tries, and
// finished 40 of 40 headless games.)
const rise = () => "stand";
const riseTime = () => 1.0;

// A seeded random number generator, so a castle can be replayed.
export function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

// The course, in course coordinates: x forward from Bolt's start, y to its left.
// Hurdles need the jump brain: with solid bars, stepping over touched even a 5 cm bar every time.
export function buildCastle(seed, L = LEVELS[0], { hurdles = false } = {}) {
  const r = rng(seed), kinds = ["sprint", "slalom", "bridge", "sit", "lie"].concat(L.ice ? ["ice"] : [], hurdles ? ["hurdles"] : []);
  for (let i = kinds.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
  kinds.push("gate");
  let x = 2.5, y = 0;
  const stations = kinds.map((kind) => {
    y = clamp(y + (r() * 2 - 1) * 1.2, -2, 2);   // each station a little to one side: Bolt must steer
    let s;
    if (kind === "sprint") s = { kind, start: [x, y], end: [x + 6, y], limit: L.sprint };
    else if (kind === "bridge") s = { kind, start: [x, y], end: [x + 4, y], halfWidth: L.bridge, maxSpeed: 0.6 };
    else if (kind === "ice") s = { kind, start: [x, y], end: [x + 0.9, y], rink: [x + 0.9, y], radius: 0.4, mu: L.ice };
    else if (kind === "hurdles") {   // three bars 3.5 m apart, at this level's height (jumped by the jump brain)
      s = { kind, bars: [0, 1, 2].map((k) => [x + 2.5 + 3.5 * k, y, L.bar]) };
      s.start = [x, y]; s.end = [x + 10.0, y];
    }
    else if (kind === "slalom") {
      const side = r() < 0.5 ? 1 : -1;
      s = { kind, gates: Array.from({ length: L.gates }, (_, k) => [x + 1 + 2.5 * k, y + side * (k % 2 ? -L.gateSide : L.gateSide)]), radius: L.gateR };
      s.start = s.gates[0]; s.end = s.gates[L.gates - 1];
    } else s = { kind, start: [x, y], end: [x, y], radius: 0.5 };
    x = s.end[0] + (kind === "slalom" ? 3.0 : 2.5);
    return { ...s, title: STATIONS[kind].title, say: STATIONS[kind].say, how: STATIONS[kind].how(L) };
  });
  // surprise shouts from the guard, and cannon shots from level 3, at random moments of the run
  const moments = (n, from, to) => Array.from({ length: n }, (_, i) => from + ((i + r()) / Math.max(1, n)) * (to - from));
  const shouts = moments(L.shouts, 6, 70).map((t) => ({ t, cmd: ["sit", "lie", "still"][Math.floor(r() * 3)] }));
  const shots = moments(L.shots, 10, 75).map((t) => ({ t, side: r() < 0.5 ? 1 : -1 }));
  return { seed, level: L, stations, shouts, shots };
}

// Drives Bolt to points: turn toward the point first, then speed up; brake before stopping.
export class Driver {
  constructor(bolt) { this.bolt = bolt; this.speed = 0.2; this.slow = 0; }
  // returns {command, turn, speed}; stop: come to a halt at the point (true) or drive through it;
  // bend: how much to slow for a turn (2: turn first, then go; 1: keep moving through constant turns)
  toward(p, cruise, stop, dt, bend = 2) {
    const w = this.bolt.where(), dx = p[0] - w.x, dy = p[1] - w.y, dist = Math.hypot(dx, dy);
    const bearing = wrap(Math.atan2(dy, dx) - w.yaw);
    let want = Math.min(cruise, stop ? 0.2 + 1.0 * Math.max(0, dist - 0.3) : cruise) * Math.max(0, Math.cos(bearing)) ** bend;
    want = clamp(want, 0.2, 1.2);
    this.speed = want > this.speed ? Math.min(want, this.speed + 0.6 * dt) : Math.max(want, this.speed - 1.0 * dt);
    const top = Math.min(1.5, this.bolt.maxTurn(this.speed));
    return { command: "walk", turn: clamp(2.5 * bearing, -top, top), speed: this.speed, dist, bearing };
  }
}

// One hurdle, jumped for real by the jump brain (jump_env.py): walk up to ~0.65 m before the bar, stop,
// let the motors hold the dog stance 0.5 s (how every training jump started), jump (2 s), stand 0.6 s.
// Measured with the 6M jump brain in live.js: 15 cm bars 6/6 clean from 0.55-0.65 m, 20 cm bars 6/6 clean
// from 0.65 m; from 0.45 m it falls, and after a posture-brain stand without the hold it fell 1 in 3.
export let HOLD = 1.0;   // s the motors hold the dog stance before a jump (10M brain: 0.8-1.2 s all good, 0.5 s fell more)
export function setHold(s) { HOLD = s; }
export let SETTLE = "hold";   // how Bolt comes to rest before the jump: "hold" (motors: take-off 1-5° off square) or "stand" (the standing brain twisted it 5-13°)
export function setSettle(s) { SETTLE = s; }
export let LAND_HOLD = 0;   // s the motors hold the stance after landing, before the standing brain
export function setLandHold(s) { LAND_HOLD = s; }
// k: which real hurdle (live.js placeReal: solid stands, a bar in cups that drops if knocked).
// touched: Bolt brushed the stands or bar; dropped: the bar came down (the agility fault).
export class Jumper {
  constructor(bolt, driver, k) {
    const H = bolt.real[k], yaw = H.yaw, bar = H.at;
    this.bolt = bolt; this.driver = driver; this.k = k; this.dir = [Math.cos(yaw), Math.sin(yaw)]; this.bar = bar;
    this.spot = [bar[0] - 0.6 * this.dir[0], bar[1] - 0.6 * this.dir[1]];   // where to stop
    this.stage = "approach"; this.t = 0; this.touched = false; this.dropped = false; this.done = false;
    bolt.aim(k);
  }
  get busy() { return this.stage !== "approach"; }   // no shouts or cannon from stopping until landed
  // the run-up follows the line through the middle of the bar, aiming a little ahead along it, so Bolt
  // arrives square to the bar (aiming straight at the stopping spot left it up to 31° askew, and the
  // jump brain only practised within ~7°: it jumped diagonally)
  line(cruise, dt) {
    const w = this.bolt.where(), [dx, dy] = this.dir, rx = w.x - this.bar[0], ry = w.y - this.bar[1];
    const before = -(rx * dx + ry * dy), side = -rx * dy + ry * dx;
    // aim at a point this far before the bar; near the bar it lies beyond it (aiming at the stopping spot
    // itself, almost under Bolt's nose while braking, swung it ~10° off)
    const t = before - Math.max(0.5, 1.5 * Math.abs(side));
    return this.driver.toward([this.bar[0] - t * dx, this.bar[1] - t * dy], cruise, false, dt);
  }
  step(dt) {
    const b = this.bolt, ahead = b.hurdleAhead();
    if (this.stage === "approach") {
      const cmd = this.line(Math.min(0.8, Math.max(0.25, 0.25 + 0.6 * (ahead - 0.8))), dt);
      if (ahead < 0.8) { this.stage = "brake"; this.t = 0; }
      return cmd;
    }
    if (this.stage === "brake") {   // a slow trot before stopping (stopping dead tips Bolt over)
      const cmd = { ...this.line(0.2, dt), speed: 0.2 };
      if ((this.t += dt) > 3.0 || ahead < 0.64) { this.stage = "settle"; this.t = 0; }
      return cmd;
    }
    if (this.stage === "settle") {
      if ((this.t += dt) > 0.8) {
        // some walk brains drift back a few cm as they stand (run 11b: 0.68-0.71 m, and fell 3 jumps in 9):
        // creep up once more, so the jump starts within the 0.45-0.65 m it was trained from
        if (ahead > 0.67 && !this.crept) { this.crept = true; this.stage = "brake"; this.t = 0; }
        else { this.stage = "hold"; this.t = 0; this.from = ahead; }
      }
      return { command: SETTLE, turn: 0, speed: 0.2 };
    }
    if (this.stage === "hold") {
      if ((this.t += dt) > HOLD) { this.stage = "jump"; b.startJump(); }
      return { command: "hold", turn: 0, speed: 0.2 };
    }
    if (this.stage === "jump") {
      if (b.hurdleHits() > 0) this.touched = true;
      if (!b.jumping()) { this.stage = "land"; this.t = 0; }
      return { command: "jump", turn: 0, speed: 0.2 };
    }
    if ((this.t += dt) > 0.6 + LAND_HOLD) {
      this.done = true; this.over = ahead < -0.3; this.driver.speed = 0.2; this.dropped = b.barDropped(this.k);
    }
    return { command: this.t < LAND_HOLD ? "hold" : "stand", turn: 0, speed: 0.2 };
  }
}

// Getting up after a fall, by itself, where it fell, the way real robot dogs recover (ANYmal: Lee, Hwangbo
// & Hutter 2019): a small selector over skills. Fallen: the self-righting brain (getup_env.py, run 17c) rolls
// Bolt onto its belly, legs tucked, until it lies still (righted: up > 0.9, body under 0.2 m/s, for 0.3 s;
// at most 4 s, its training tries); then the posture brain's "stand" gets up from there (from the lying
// pose 20/20) and holds it 1.5 s; toppled again (up < 0.7): another try. Not standing after 2 tries or 10 s:
// failed, and the game falls back to a restart. A get-up brain trained to stand (goal "stand") hands over
// once standing instead.
// The brain practised from rest (each try began after 1 s lying still), so by default Bolt first lies still
// until it has stopped rolling (at most 1.5 s); "now" hands over the moment it falls.
export let GETUP_WAIT = "rest";
export function setGetUpWait(w) { GETUP_WAIT = w; }
export let GETUP_CARRY = false;   // the standing brain starts from the get-up brain's last action (else zeros)
export function setGetUpCarry(c) { GETUP_CARRY = c; }
export const GETUP = { right: 4, still: 0.3, stand: 1.5, tries: 2, limit: 10 };
export class GetUp {
  constructor(bolt) {
    this.bolt = bolt; this.stage = "rest"; this.t = 0; this.st = 0; this.tries = 0; this.done = false; this.failed = false; this.secs = null;
    this.goal = (bolt.brains.getup && bolt.brains.getup.goal) || "selfright";
  }
  // the command for this step; then read done (standing after 1.5 s on the standing brain) or failed
  step(dt) {
    const b = this.bolt, v = b.data.qvel, rest = { command: "rest", turn: 0, speed: 0.2 }, stand = { command: "stand", turn: 0, speed: 0.2 };
    this.t += dt; this.st += dt;
    const next = () => {   // another try, or give up
      if (this.tries >= GETUP.tries) { this.failed = true; return rest; }
      this.stage = "rest"; this.st = 0; return rest;
    };
    if (this.t > GETUP.limit) { this.failed = true; return rest; }
    if (this.stage === "stand") {
      if (b.up() < 0.7) return next();
      if (this.st >= GETUP.stand) {
        if (b.standing()) { this.done = true; this.secs = this.t; return stand; }
        return next();
      }
      return stand;
    }
    if (this.stage === "rest") {
      const still = Math.hypot(v[3], v[4], v[5]) < 0.5 && Math.hypot(v[0], v[1], v[2]) < 0.1;
      if (!(GETUP_WAIT === "now" || this.st > 1.5 || (this.st >= 0.2 && still))) return rest;
      this.tries += 1; this.stage = "right"; this.st = 0; this.held = 0; b.startGetUp();
    }
    // stage "right": the self-righting brain until belly down and still (or a stand brain until standing)
    const ok = this.goal === "stand" ? b.standing() : b.righted();
    this.held = ok ? this.held + dt : 0;
    if (this.held >= (this.goal === "stand" ? dt : GETUP.still)) { b.afterGetUp(GETUP_CARRY); this.stage = "stand"; this.st = 0; return stand; }
    if (this.st > GETUP.right) return next();
    return { command: "getup", turn: 0, speed: 0.2 };
  }
}

// The solid castle (live.js setProps): every post, pole, tower and wall Bolt could run into is a real
// object. Flat things (lines, mats, the bridge plank, the ice) stay paint; the hurdles are their own
// physics (placeReal). In world coordinates for Bolt standing at the start (w = bolt.where()).
export function castleProps(castle, w) {
  const c = Math.cos(w.yaw), s = Math.sin(w.yaw), out = [];
  const put = (x, y, z, size, kind, look) => out.push({ kind, size, pos: [w.x + c * x - s * y, w.y + s * x + c * y, z], yaw: w.yaw, look });
  for (const st of castle.stations) {
    if (st.kind === "sprint") for (const y of [0.5, -0.5]) put(st.end[0], st.end[1] + y, 0.35, [0.025, 0.35], "cyl", { color: "#e3620b" });
    if (st.kind === "slalom") st.gates.forEach(([x, y], k) => { for (const side of [0.55, -0.55]) put(x, y + side, 0.55, [0.03, 0.55], "cyl", { color: k % 2 ? "#f4f4f0" : "#e3620b", cap: k % 2 ? "#e3620b" : "#f4f4f0" }); });
    if (st.kind === "gate") {   // the castle: two towers, the walls either side of the doorway, the lintel
      const [x, y] = st.end, X = x + 1.0, stone = { color: "#d8d2c4" };
      for (const side of [1.25, -1.25]) put(X, y + side, 1.3, [0.42, 1.3], "cyl", stone);
      for (const side of [0.75, -0.75]) put(X, y + side, 1.0, [0.2, 0.45, 1.0], "box", stone);
      put(X, y, 1.65, [0.2, 0.3, 0.35], "box", stone);
    }
  }
  return out;
}

// One game. Call step() 50 times a second; read state for the screen.
export class Game {
  // level: 1-5; lives and score carry over from the level before
  // hurdles: on when the bundle has a jump brain
  constructor(bolt, seed, start, { level = 1, lives = 3, score = 0, hurdles = !!bolt.brains.jump } = {}) {
    this.L = LEVELS[level - 1];
    this.bolt = bolt; this.castle = buildCastle(seed, this.L, { hurdles }); this.start = start;
    bolt.reset(start);
    bolt.setProps(castleProps(this.castle, bolt.where()));   // the course's solid objects, built into the physics
    bolt.reset(start);
    const w = bolt.where();
    this.origin = [w.x, w.y]; this.yaw0 = w.yaw;       // course coordinates start where Bolt stands
    this.placeHurdles();
    this.driver = new Driver(bolt);
    this.t = 0; this.index = 0; this.score = score; this.lives = lives; this.results = [];
    this.phase = "ready"; this.task = null; this.shout = null; this.events = []; this.over = false; this.readyFor = 2;
    this.cmd = { command: "stand", turn: 0, speed: 0.2 };
    this.pendingShouts = this.castle.shouts.slice();
    this.pendingShots = this.castle.shots.slice();
    this.falls = 0; this.getUps = 0; this.restarts = 0;   // falls Bolt got up from by itself, and restarts
  }
  loseLife(why) {
    this.lives -= 1;
    this.say(`${why} ${this.lives > 0 ? `${this.lives} ${this.lives === 1 ? "life" : "lives"} left.` : "No lives left."}`, "bad");
    if (this.lives <= 0) { this.over = true; this.phase = "eliminated"; this.say("ELIMINATED!", "bad"); }
    return this.lives > 0;
  }
  // every hurdle on the course is real and solid from the start (real hurdles 0, 1, 2), bars in their cups
  placeHurdles() {
    let k = 0;
    for (const s of this.castle.stations) if (s.kind === "hurdles") s.slots = s.bars.map(([x, y, h]) => { this.bolt.placeReal(k, this.world([x, y]), this.yaw0, h); return k++; });
  }
  // after a fall: Bolt starts that station again, standing 1.5 m before it (the judge puts knocked bars back).
  // Only when it could not get up by itself (or the bundle has no get-up brain).
  respawn() {
    this.restarts += 1; this.rec = null;
    const s = this.station, before = [s.start[0] - 1.5, s.start[1]];
    this.bolt.teleport(this.start, this.world(before), this.yaw0);
    this.placeHurdles();
    this.shout = null; this.driver.speed = 0.2; this.phase = "ready"; this.readyFor = this.t + 2;
    this.cmd = { command: "stand", turn: 0, speed: 0.2 };
  }
  // course <-> world coordinates
  world([x, y]) {
    const c = Math.cos(this.yaw0), s = Math.sin(this.yaw0);
    return [this.origin[0] + c * x - s * y, this.origin[1] + s * x + c * y];
  }
  local() {
    const w = this.bolt.where(), dx = w.x - this.origin[0], dy = w.y - this.origin[1];
    const c = Math.cos(this.yaw0), s = Math.sin(this.yaw0);
    return [c * dx + s * dy, -s * dx + c * dy];
  }
  get station() { return this.castle.stations[this.index]; }
  say(text, kind = "info") { this.events.push({ t: this.t, text, kind }); }
  shoutNow(cmd) {   // the guard (or the viewer) shouts a command: Bolt must obey within 5 s, hold it 2 s
    if (this.over || this.shout) return false;
    this.shout = { cmd, since: this.t, held: 0, braking: 0, settle: 0 };
    this.say(SHOUTS[cmd], "shout");
    return true;
  }

  step() {
    const dt = this.bolt.dt;
    if (this.over) return;
    this.t += dt;
    if (this.phase === "getup") return this.recover(dt);
    if (this.phase === "ready") {          // stand for 2 s at the start line (or after a restart)
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if (this.t >= this.readyFor) { this.phase = "run"; this.begin(); }
    } else {
      // shouts and shots wait from braking for a hurdle until 1.5 s after landing (a "sit!" the moment
      // Bolt landed, still twisted from the jump, toppled it 3 times in 8 level-1 games)
      const midJump = (this.task && this.task.jumper && this.task.jumper.busy) || this.t < (this.calmUntil || 0);
      while (!midJump && this.pendingShouts.length && this.t >= this.pendingShouts[0].t + 2) this.shoutNow(this.pendingShouts.shift().cmd);
      while (!midJump && this.pendingShots.length && this.t >= this.pendingShots[0].t + 2) {   // the guard's cannon
        const shot = this.pendingShots.shift();
        this.bolt.shove(shot.side * this.L.cannon, 0, 0.2);
        this.say(`CANNON! ${this.L.cannon} N from the ${shot.side > 0 ? "right" : "left"}`, "shot");
      }
      if (this.shout) this.obey(dt); else this.play(dt);
    }
    const b = this.bolt;
    b.run(this.cmd);
    if (b.fell()) {
      this.falls += 1;
      this.results.push({ kind: "fall", title: `Fell at ${this.shout ? "a shout" : this.station.title}`, passed: false, note: "" });
      if (!this.loseLife("Bolt fell!")) return;
      if (!b.canGetUp()) return this.respawn();
      // get up where it fell; a shout in progress is dropped (as on a restart), the shouts and the cannon wait
      this.rec = { getup: new GetUp(b), from: this.phase, since: this.t };
      this.phase = "getup"; this.shout = null;
      this.cmd = { command: "rest", turn: 0, speed: 0.2 };
    }
  }

  // After a fall: Bolt gets up where it fell (GetUp), then the station goes on from there. Station clocks
  // pause meanwhile (as for a shout). Not up after 2 tries or 10 s: the old restart.
  recover(dt) {
    const r = this.rec, b = this.bolt;
    this.cmd = r.getup.step(dt);
    if (r.getup.failed) { this.say("Bolt could not get up: restart at the station.", "bad"); return this.respawn(); }
    b.run(this.cmd);
    if (!r.getup.done) return;
    this.getUps += 1;
    this.say(`Bolt got up by itself (${r.getup.secs.toFixed(1)} s)`, "good");
    const lost = this.t - r.since, k = this.task;
    this.rec = null; this.phase = r.from; this.driver.speed = 0.2; this.calmUntil = this.t + 1.5;
    if (r.from === "ready") { this.readyFor = Math.max(this.readyFor, this.t + 1); return; }
    if (!k) return;
    if (k.timer !== null && k.timer !== undefined) k.timer += lost;
    if (k.posedAt !== undefined) k.posedAt += lost;
    // a posture station starts again from the approach (it may have slid off the spot); a done one finishes
    if (["brake", "settle", "pose"].includes(k.stage)) { k.stage = "approach"; k.held = 0; k.settle = 0; delete k.posedAt; }
    if (k.stage === "rise") k.settle = 0;
    if (this.station.kind === "bridge") {   // back to before the plank (toppling off it, judged "stepped off"
      k.retry = true;                         // a moment before the fall, is part of the fall)
      if (k.stage === "done" && !k.passed) k.stage = "approach";
    }
    if (k.jumper) this.afterJumpFall(k);
  }
  // fell inside a hurdle jump: the judge puts that bar back, and Bolt jumps it again from a proper run-up;
  // if it ended up past the bar, that hurdle counts as jumped
  afterJumpFall(k) {
    const j = k.jumper, b = this.bolt;
    if (b.barDropped(j.k)) b.resetBar(j.k);
    b.aim(j.k);
    const ahead = b.hurdleAhead();
    k.jumper = null;
    if (ahead < -0.3) {
      this.say(`Hurdle ${k.bar + 1}: over, with a fall`, "info");
      k.bar += 1;
      if (k.bar >= this.station.bars.length) { k.stage = "done"; k.passed = k.dropped === 0; k.note = k.dropped ? `Knocked down ${k.dropped} of 3 bars.` : "All 3 bars stayed up."; }
    } else if (ahead < 0.45) k.backOff = true;   // too close to jump from (trained from 0.45-0.65 m): back off first
  }

  begin() {
    const s = this.station;
    this.task = { stage: "approach", since: this.t, held: 0, gate: 0, bad: 0, timer: null, settle: 0 };
    if (s.kind === "ice") this.bolt.setIce(this.world(s.rink), this.yaw0, s.mu);   // the rink: real low friction
    this.say(`${s.title}: ${s.how}`, "station");
  }
  finish(passed, note) {
    const s = this.station;
    if (s.kind === "ice") this.bolt.setIce([1e3, 0], 0, 1.0, false);
    this.results.push({ kind: s.kind, title: s.title, passed, note });
    if (passed) this.score += (s.kind === "gate" ? 50 : 100) * this.L.n;
    this.say(passed ? `${s.title}: passed!${note ? " " + note : ""}` : `${s.title}: failed. ${note || ""}`, passed ? "good" : "bad");
    if (!passed && !this.loseLife("A life lost.")) return;
    this.index += 1;
    if (this.index >= this.castle.stations.length) {
      this.over = true; this.phase = "finished";
      this.say(`Level ${this.L.n} (${this.L.name}) cleared! ${this.score} points`, "good");
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
    } else this.begin();
  }
  // stand still after a halt, then do the posture; then stand up again before walking on
  holdPosture(target, dt) {
    const k = this.task;
    if (k.stage === "brake") {             // trot slowly for 0.5 s before stopping (stopping dead tips it over)
      this.cmd = this.driver.toward(this.world(this.station.end), 0.2, true, dt); this.cmd.speed = 0.2;
      if ((k.settle += dt) > 0.5) { k.stage = "settle"; k.settle = 0; }
      return null;
    }
    if (k.stage === "settle") {            // stand still for a moment
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if ((k.settle += dt) > 0.6) { k.stage = "pose"; k.posedAt = this.t; }
      return null;
    }
    if (k.stage === "pose") {
      this.cmd = { command: target, turn: 0, speed: 0.2 };
      k.held = posture(this.bolt) === target ? k.held + dt : 0;
      if (k.held >= 2) { k.stage = "rise"; k.settle = 0; return "done"; }
      if (this.t - k.posedAt > 6) { k.stage = "rise"; k.settle = 0; return "timeout"; }
      return null;
    }
    return null;
  }
  // after a posture: stand up for 1 s before walking on (the walk takes over from a standing dog)
  rising(dt) {
    const k = this.task;
    if (k.stage !== "rise") return false;
    k.settle += dt;
    this.cmd = { command: rise(this.station.kind, k.settle), turn: 0, speed: 0.2 };
    if (k.settle > riseTime(this.station.kind)) k.stage = "done";
    return true;
  }

  play(dt) {
    const s = this.station, k = this.task, here = this.local();
    if (k.stage === "done") return this.finish(k.passed, k.note);
    if (this.rising(dt)) return;
    if (s.kind === "sit" || s.kind === "lie" || s.kind === "ice") {
      const target = s.kind === "ice" ? "sit" : s.kind;
      if (k.stage === "approach") {   // on the ice, it comes in fast: that is the challenge
        this.cmd = this.driver.toward(this.world(s.end), s.kind === "ice" ? 1.0 : 0.9, true, dt);
        if (this.cmd.dist < 0.25 || (this.cmd.dist < s.radius && this.driver.speed <= 0.25)) { k.stage = "brake"; k.settle = 0; }
        return;
      }
      const r = this.holdPosture(target, dt);
      if (r) {
        const d = Math.hypot(here[0] - s.end[0], here[1] - s.end[1]);
        k.passed = r === "done" && d < s.radius + 0.15;
        k.note = r === "timeout" ? "Too slow." : k.passed ? "" : s.kind === "ice" ? "Slid off the spot." : "Not in the box.";
      }
      return;
    }
    if (s.kind === "sprint") {
      const along = here[0];
      if (k.timer === null && along >= s.start[0]) k.timer = this.t;
      const p = along < s.start[0] ? [s.start[0] + 0.5, s.start[1]] : [s.end[0] + 2, s.end[1]];
      this.cmd = this.driver.toward(this.world(p), 1.2, false, dt);
      if (k.timer !== null && along >= s.end[0]) {
        const time = this.t - k.timer;
        k.stage = "done"; k.passed = time <= s.limit; k.note = `${time.toFixed(1)} s`;
      }
      return;
    }
    if (s.kind === "slalom") {
      const g = s.gates[k.gate];
      this.cmd = this.driver.toward(this.world(g), 0.6, false, dt);
      const n = s.gates.length, done = () => { k.stage = "done"; k.passed = k.bad === 0; k.note = k.bad ? `Missed ${k.bad} of ${n} gates.` : `All ${n} gates.`; };
      if (this.cmd.dist < s.radius) {
        k.gate += 1;
        if (k.gate >= n) done();
      } else if (here[0] > g[0] + 0.6) {     // went past this pair of poles without passing between them
        k.gate += 1; k.bad += 1;
        if (k.gate >= n) done();
      }
      return;
    }
    if (s.kind === "bridge") {
      // after a fall (in the water, or on the way, which can leave Bolt 2 m off the line: coming at the plank
      // from there it stepped off): walk to 2 m before the plank, on its line, and cross again, judged afresh
      if (k.retry) {
        if (here[0] > s.end[0] + 0.3 && k.retry !== "back") { k.retry = false; k.bad = 0; }
        else {
          k.retry = "back";
          this.cmd = this.driver.toward(this.world([s.start[0] - 2.0, s.start[1]]), 0.5, true, dt);
          if (this.cmd.dist < 0.35) { k.retry = false; k.bad = 0; }
          return;
        }
      }
      const on = here[0] >= s.start[0] && here[0] <= s.end[0];
      const p = here[0] < s.start[0] - 0.8 ? [s.start[0] - 0.4, s.start[1]] : [s.end[0] + 1.5, s.end[1]];
      this.cmd = this.driver.toward(this.world(p), here[0] < s.start[0] - 0.8 ? 0.6 : 0.3, false, dt);
      if (on) {
        const v = this.bolt.data.qvel, speed = Math.hypot(v[0], v[1]);
        if (Math.abs(here[1] - s.start[1]) > s.halfWidth) { k.stage = "done"; k.passed = false; k.note = "Stepped off the bridge."; }
        k.bad = speed > s.maxSpeed ? k.bad + dt : 0;
        if (k.bad > 1.0) { k.stage = "done"; k.passed = false; k.note = "Too fast on the bridge."; }
      }
      if (k.stage !== "done" && here[0] > s.end[0]) { k.stage = "done"; k.passed = true; k.note = ""; }
      return;
    }
    if (s.kind === "hurdles") {
      // three real hurdles; the jumper takes Bolt over each. A knocked-down bar fails the station; a
      // brushed bar that stays up does not (the agility rule)
      if (k.bar === undefined) { k.bar = 0; k.dropped = 0; }
      if (k.backOff) {   // got up too close to the bar: walk back to 1.6 m before it, then line up again
        const H = this.bolt.real[s.slots[k.bar]], p = [H.at[0] - 1.6 * Math.cos(H.yaw), H.at[1] - 1.6 * Math.sin(H.yaw)];
        this.cmd = this.driver.toward(p, 0.4, true, dt);
        if (this.cmd.dist < 0.35) k.backOff = false;
        return;
      }
      if (!k.jumper) k.jumper = new Jumper(this.bolt, this.driver, s.slots[k.bar]);
      this.cmd = k.jumper.step(dt);
      if (k.jumper.done) {
        const j = k.jumper;
        if (j.dropped) { k.dropped += 1; this.say(`Hurdle ${k.bar + 1}: bar knocked down!`, "bad"); }
        else if (j.touched) this.say(`Hurdle ${k.bar + 1}: brushed it, but the bar stayed up`, "info");
        else this.say(`Hurdle ${k.bar + 1}: clean jump!`, "good");
        k.bar += 1; k.jumper = null; this.calmUntil = this.t + 1.5;
        if (k.bar >= s.bars.length) {
          k.stage = "done"; k.passed = k.dropped === 0; k.note = k.dropped ? `Knocked down ${k.dropped} of 3 bars.` : "All 3 bars stayed up.";
        }
      }
      return;
    }
    if (s.kind === "gate") {
      this.cmd = this.driver.toward(this.world(s.end), 1.0, true, dt);
      if (this.cmd.dist < 0.5) { k.stage = "done"; k.passed = true; k.note = ""; }
    }
  }

  // a shout interrupts whatever Bolt is doing: brake, obey, hold for 2 s, then carry on
  obey(dt) {
    const sh = this.shout;
    if (sh.stage === undefined) sh.stage = this.cmd.command === "walk" ? "brake" : "settle";
    if (sh.stage === "brake") {
      const sp = this.cmd.speed;
      this.cmd = { ...this.cmd, command: "walk", speed: Math.max(0.2, sp - 1.0 * dt), turn: 0 };
      if (this.cmd.speed <= 0.2 && (sh.braking += dt) > 0.5) sh.stage = "settle";
      return;
    }
    if (sh.stage === "settle") {
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if ((sh.settle += dt) > 0.6 || sh.cmd === "still") sh.stage = "pose";
      return;
    }
    if (sh.stage === "pose") {
      this.cmd = { command: sh.cmd === "still" ? "stand" : sh.cmd, turn: 0, speed: 0.2 };
      sh.held = posture(this.bolt) === sh.cmd ? sh.held + dt : 0;
      const late = this.t - sh.since > this.L.obey + 2;   // obey within the level's time, then hold 2 s
      if (sh.held >= 2 || late) {
        const ok = sh.held >= 2;
        this.results.push({ kind: "shout", title: `Guard: ${SHOUTS[sh.cmd]}`, passed: ok, note: ok ? "" : "Too slow." });
        if (ok) this.score += 50;
        this.say(ok ? `${SHOUTS[sh.cmd]} obeyed!` : `${SHOUTS[sh.cmd]} too slow.`, ok ? "good" : "bad");
        sh.stage = "rise"; sh.rise = 0;
      }
      return;
    }
    if (sh.stage === "rise") {   // stand up again before walking on
      sh.rise += dt;
      this.cmd = { command: rise(sh.cmd, sh.rise), turn: 0, speed: 0.2 };
      if (sh.rise > (sh.cmd === "still" ? 0.2 : riseTime(sh.cmd))) {
        // station clocks pause for a shout: the sprint timer, and the time allowed to sit or lie down
        if (this.task && this.task.timer !== null && this.task.timer !== undefined) this.task.timer += this.t - sh.since;
        if (this.task && this.task.posedAt !== undefined) { this.task.posedAt += this.t - sh.since; this.task.held = 0; }
        this.shout = null;
        this.driver.speed = 0.2;
        if (this.task && ["brake", "settle"].includes(this.task.stage)) this.task.settle = 0;
      }
    }
  }
}
