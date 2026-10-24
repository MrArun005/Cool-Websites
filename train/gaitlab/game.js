// Bolt's Castle: a Takeshi's Castle-style course for Bolt, run on the live simulator (live.js).
//
// Each game builds a new course from shuffled stations: a sprint, a slalom between poles, a narrow
// bridge to cross slowly, a box to sit in, a mat to lie on, and the castle gate at the end. The castle
// guard also shouts surprise commands at any moment ("SIT!", "LIE DOWN!", "FREEZE!"), and so can the
// viewer. A fall eliminates Bolt. Bolt plays on its own: a course runner steers its trained brains
// from station to station, the same way the follow layer does. game_test.mjs plays many games headless.

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// What each posture looks like in the simulator (measured with the posture brain: stand 43 cm level,
// sit 20.7 cm nose up 30°, lie 12.5 cm level; each settles in under a second).
export function posture(bolt) {
  const R = bolt.rotation(), q = bolt.data.qpos, v = bolt.data.qvel;
  const h = q[2], pitch = Math.asin(clamp(R[6], -1, 1)), speed = Math.hypot(v[0], v[1]);
  if (h > 0.18 && h < 0.24 && pitch > 0.35 && pitch < 0.7 && speed < 0.15) return "sit";
  if (h < 0.16 && Math.abs(pitch) < 0.2 && speed < 0.15) return "lie";
  if (h > 0.38 && Math.abs(pitch) < 0.2 && speed < 0.1) return "still";
  return "moving";
}

export const STATIONS = {
  sprint: { title: "Sprint", say: "RUN!", how: "Cover 6 m in under 7 s" },
  slalom: { title: "Slalom", say: "WEAVE!", how: "Pass between every pair of poles" },
  bridge: { title: "Narrow bridge", say: "SLOWLY…", how: "Cross the 4 m plank without stepping off or rushing" },
  sit: { title: "Sit box", say: "SIT!", how: "Stop in the box and sit for 2 s" },
  lie: { title: "Lie-down mat", say: "LIE DOWN!", how: "Stop on the mat and lie down for 2 s" },
  hurdles: { title: "Hurdles", say: "OVER!", how: "Get over 3 bars without touching them" },
  gate: { title: "Castle gate", say: "HOME!", how: "Reach the castle" },
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
export function buildCastle(seed) {
  const r = rng(seed), kinds = ["sprint", "slalom", "bridge", "sit", "lie", "hurdles"];
  for (let i = kinds.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [kinds[i], kinds[j]] = [kinds[j], kinds[i]]; }
  kinds.push("gate");
  let x = 2.5, y = 0;
  const stations = kinds.map((kind) => {
    y = clamp(y + (r() * 2 - 1) * 1.2, -2, 2);   // each station a little to one side: Bolt must steer
    let s;
    if (kind === "sprint") s = { kind, start: [x, y], end: [x + 6, y], limit: 7 };
    else if (kind === "bridge") s = { kind, start: [x, y], end: [x + 4, y], halfWidth: 0.3, maxSpeed: 0.6 };
    else if (kind === "hurdles") {   // three bars 3 m apart, 15-25 cm high (the trot steps over them)
      s = { kind, bars: [0, 1, 2].map((k) => [x + 1.5 + 3 * k, y, Math.round((0.15 + 0.1 * r()) * 100) / 100]) };
      s.start = [x, y]; s.end = [x + 8.5, y];
    }
    else if (kind === "slalom") {
      const side = r() < 0.5 ? 1 : -1;
      s = { kind, gates: [0, 1, 2, 3].map((k) => [x + 1 + 2.5 * k, y + side * (k % 2 ? -0.9 : 0.9)]) };
      s.start = s.gates[0]; s.end = s.gates[3];
    } else s = { kind, start: [x, y], end: [x, y], radius: 0.5 };
    x = s.end[0] + (kind === "slalom" ? 3.0 : 2.5);
    return { ...s, ...STATIONS[kind] };
  });
  // two surprise shouts from the guard, at random moments of the run
  const shouts = [8 + 18 * r(), 30 + 25 * r()].map((t) => ({ t, cmd: ["sit", "lie", "still"][Math.floor(r() * 3)] }));
  return { seed, stations, shouts };
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

// One game. Call step() 50 times a second; read state for the screen.
export class Game {
  constructor(bolt, seed, start) {
    this.bolt = bolt; this.castle = buildCastle(seed);
    bolt.reset(start);
    const w = bolt.where();
    this.origin = [w.x, w.y]; this.yaw0 = w.yaw;       // course coordinates start where Bolt stands
    this.driver = new Driver(bolt);
    this.t = 0; this.index = 0; this.score = 0; this.results = [];
    this.phase = "ready"; this.task = null; this.shout = null; this.events = []; this.over = false;
    this.cmd = { command: "stand", turn: 0, speed: 0.2 };
    this.pendingShouts = this.castle.shouts.slice();
  }
  raise([x, y, h]) { this.bolt.setHurdle(this.world([x, y]), this.yaw0, h); }   // a bar at a course spot, across the course
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
    if (this.phase === "ready") {          // stand for 2 s at the start line
      this.cmd = { command: "stand", turn: 0, speed: 0.2 };
      if (this.t >= 2) { this.phase = "run"; this.begin(); }
    } else {
      while (this.pendingShouts.length && this.t >= this.pendingShouts[0].t + 2) this.shoutNow(this.pendingShouts.shift().cmd);
      if (this.shout) this.obey(dt); else this.play(dt);
    }
    const c = this.cmd, b = this.bolt;
    b.setCommand(c.command);
    if (c.command === "walk") { b.setTurn(c.turn); b.setSpeed(c.speed); }
    b.tick();
    if (b.fell()) { this.over = true; this.phase = "eliminated"; this.say("ELIMINATED!", "bad"); }
  }

  begin() {
    const s = this.station;
    this.task = { stage: "approach", since: this.t, held: 0, gate: 0, bad: 0, timer: null, settle: 0 };
    this.say(`${s.title}: ${s.how}`, "station");
  }
  finish(passed, note) {
    const s = this.station;
    this.results.push({ kind: s.kind, title: s.title, passed, note });
    if (passed) this.score += s.kind === "gate" ? 50 : 100;
    this.say(passed ? `${s.title}: passed!${note ? " " + note : ""}` : `${s.title}: failed. ${note || ""}`, passed ? "good" : "bad");
    this.index += 1;
    if (this.index >= this.castle.stations.length) {
      this.over = true; this.phase = "finished";
      this.say(`Castle reached! ${this.score} points`, "good");
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
    if (s.kind === "sit" || s.kind === "lie") {
      if (k.stage === "approach") {
        this.cmd = this.driver.toward(this.world(s.end), 0.9, true, dt);
        if (this.cmd.dist < 0.25 || (this.cmd.dist < s.radius && this.driver.speed <= 0.25)) { k.stage = "brake"; k.settle = 0; }
        return;
      }
      const r = this.holdPosture(s.kind, dt);
      if (r) {
        const d = Math.hypot(here[0] - s.end[0], here[1] - s.end[1]);
        k.passed = r === "done" && d < s.radius + 0.15; k.note = r === "timeout" ? "Too slow." : k.passed ? "" : "Not in the box.";
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
      if (this.cmd.dist < 0.45) {
        k.gate += 1;
        if (k.gate >= s.gates.length) { k.stage = "done"; k.passed = true; k.note = "All 4 gates."; }
      } else if (here[0] > g[0] + 0.6) {     // went past this pair of poles without passing between them
        k.gate += 1; k.bad += 1;
        if (k.gate >= s.gates.length) { k.stage = "done"; k.passed = false; k.note = `Missed ${k.bad} of 4 gates.`; }
      }
      return;
    }
    if (s.kind === "bridge") {
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
      // the simulator has one bar: it stands wherever the next bar is, and moves on once Bolt is past
      if (k.bar === undefined) { k.bar = 0; k.touched = 0; k.hit = false; this.raise(s.bars[0]); }
      this.cmd = this.driver.toward(this.world([s.end[0] + 1.5, s.end[1]]), 0.7, false, dt);
      if (!k.hit && this.bolt.hurdleHits() > 0) { k.hit = true; k.touched += 1; this.say(`Bar ${k.bar + 1} touched!`, "bad"); }
      if (here[0] > s.bars[k.bar][0] + 0.8) {
        k.bar += 1; k.hit = false;
        if (k.bar < s.bars.length) this.raise(s.bars[k.bar]);
        else { this.bolt.setHurdle([1e3, 0], 0, 0.1); k.stage = "done"; k.passed = k.touched === 0; k.note = k.touched ? `Touched ${k.touched} of 3 bars.` : "Clean over all 3 bars."; }
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
      const late = this.t - sh.since > 7;
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
