// Bolt's Day Out: five minutes of Bolt being a dog in a front yard, live. Nothing is pre-recorded: the physics
// runs here (live.js, MuJoCo), the trained brains pick every motor command at 50 Hz, and a small "mind" picks
// what to do next the way a dog would, from drives (energy, curiosity, play) plus chance, so no two runs match.
//
// Three layers, no DOM (doglife_test.mjs runs it headless, doglife_template.html draws it):
//   Yard   where things are: the lawn, the sidewalk and the kerb, the house and porch, the kennel, the bins,
//          trees, two backyard wing jumps. Solid things are in the physics too (live.js setProps).
//   Body   which trained brain moves the body, and the hand-overs between them (the rules measured in the
//          Agility Run, agilityrun.js, plus the ones the GPU brain learned: gait -> sit -> lie -> get up):
//            walk trot pace turns sit lie getup   GPU dog-motion brain (models/bolt-dog-imitate-gpu-360M)
//            canter, and canter -> trot           canter-switching brain (run 18g, 35.5M: canter->trot 93%)
//            sharp turns at a walk                run 12 walk brain (turns up to ~1.4 rad/s)
//            standing still, sniffing             posture brain
//            jumping a hurdle                     run-up jump brain (run 16i, 35.5M), handed over 1.1 m out
//            after a fall                         self-righting brain (run 17e) + posture brain (game.js GetUp)
//   Mind   what a dog does next: wander and sniff, trot about, zoomies, sit, nap, jump, fetch, come when
//          called, and never step off the kerb.
import { GetUp, rng } from "./game.js";
import { Imitate } from "./imitate.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const hyp = (x, y) => Math.hypot(x, y);
export { rng };

// ---------------------------------------------------------------- the yard (world metres: x east, y north)
export const YARD = {
  lawn: { x0: -15, x1: 15, y0: -13, y1: 9 },
  sidewalk: { y0: -15, y1: -13 },          // the kerb is the sidewalk's south edge
  kerb: -15,
  road: { y0: -22, y1: -15 },
  house: { x0: -9, x1: 3, y0: 9, y1: 17, door: -3 },
  porch: { x0: -5, x1: -1, y0: 7.4, y1: 9, h: 0.36 },
  garage: { x0: 4.5, x1: 10.5, y0: 9, y1: 16 },
  driveway: { x0: 5.5, x1: 9.5 },
  kennel: { at: [-12.4, 6.2], half: [0.55, 0.5], h: 0.85 },
  bin: { at: [12.3, 7.6], half: [0.3, 0.33], h: 1.0 },
  trash: [{ kind: "can", at: [11.5, 6.8] }, { kind: "bag", at: [12.0, 6.55] }, { kind: "paper", at: [12.9, 6.6] }, { kind: "bottle", at: [11.2, 7.4] }],
  trees: [{ at: [-11, -4], r: 0.2, h: 5.2 }, { at: [11.2, -7], r: 0.22, h: 6 }, { at: [-6.5, 5.6], r: 0.14, h: 3.6 }],
  bench: { at: [-13.4, -8.5], yaw: Math.PI / 2 },
  mailbox: [4.7, -12.6],
  lamp: [-13.2, -14.7],
  car: { at: [-5.5, -17.6], yaw: 0 },
  puddle: { at: [-2.2, -9.6], r: [0.9, 0.6] },
  ball: [3, -8],
  // the backyard wing jumps: real agility hurdles (live.js placeReal), k = which physics hurdle
  jumps: [{ k: 0, at: [-4, -5], yaw: 0, h: 0.40 }, { k: 1, at: [1.5, 2.6], yaw: Math.PI, h: 0.45 }],
  extra: 2,                                  // the hurdle the viewer sets up
  start: { at: [-1, -7.5], yaw: Math.PI / 2 },
  // where the mind may send Bolt (its centre), and the hard line it never crosses (the kerb, with a margin)
  roam: { x0: -13.6, x1: 13.6, y0: -12.4, y1: 6.6 },
  walkway: { y: -14.05, x0: -13.6, x1: 13.6 },   // the middle of the sidewalk
  stopAt: -14.45,
};

// Solid things for the physics (live.js setProps: static boxes and cylinders) — what Bolt could bump into.
export function yardProps(Y = YARD) {
  const out = [], box = (x0, x1, y0, y1, h, yaw = 0) => out.push({ kind: "box", size: [(x1 - x0) / 2, (y1 - y0) / 2, h / 2], pos: [(x0 + x1) / 2, (y0 + y1) / 2, h / 2], yaw });
  const H = Y.house, G = Y.garage, P = Y.porch;
  box(H.x0, H.x1, H.y0, H.y1, 3.0);
  box(G.x0, G.x1, G.y0, G.y1, 2.6);
  box(P.x0, P.x1, P.y0, P.y1, P.h);
  // fences: the two sides of the front yard and the back gaps between house, garage and the sides
  const L = Y.lawn, T = 0.05, fh = 1.0;
  box(L.x0 - T, L.x0 + T, L.y0, L.y1, fh); box(L.x1 - T, L.x1 + T, L.y0, L.y1, fh);
  box(L.x0, H.x0, L.y1 - T, L.y1 + T, fh); box(H.x1, G.x0, L.y1 - T, L.y1 + T, fh); box(G.x1, L.x1, L.y1 - T, L.y1 + T, fh);
  const K = Y.kennel; box(K.at[0] - K.half[0], K.at[0] + K.half[0], K.at[1] - K.half[1], K.at[1] + K.half[1], K.h);
  out.push({ kind: "cyl", size: [Y.bin.half[0], Y.bin.h / 2], pos: [...Y.bin.at, Y.bin.h / 2], yaw: 0 });
  for (const t of Y.trees) out.push({ kind: "cyl", size: [t.r, 1.5], pos: [...t.at, 1.5], yaw: 0 });
  out.push({ kind: "box", size: [0.25, 0.75, 0.25], pos: [...Y.bench.at, 0.25], yaw: 0 });
  out.push({ kind: "cyl", size: [0.05, 0.55], pos: [...Y.mailbox, 0.55], yaw: 0 });
  out.push({ kind: "cyl", size: [0.07, 2.0], pos: [...Y.lamp, 2.0], yaw: 0 });
  out.push({ kind: "box", size: [2.2, 0.9, 0.7], pos: [...Y.car.at, 0.7], yaw: Y.car.yaw });
  return out;
}

// What the mind steers round (circles, Bolt's centre keeps r away)
export function yardObstacles(Y = YARD, hurdles = []) {
  const o = [{ at: Y.kennel.at, r: 1.15, name: "kennel" }, { at: Y.bin.at, r: 0.95, name: "bin" }, { at: Y.bench.at, r: 1.1, name: "bench" },
    { at: Y.mailbox, r: 0.5, name: "mailbox" }, { at: [-3, 7.6], r: 2.3, name: "porch" }];
  for (const t of Y.trees) o.push({ at: t.at, r: t.r + 0.5, name: "tree" });
  for (const h of hurdles) for (const s of [-0.75, 0, 0.75]) o.push({ at: [h.at[0] - Math.sin(h.yaw) * s, h.at[1] + Math.cos(h.yaw) * s], r: 0.55, name: "hurdle", k: h.k });
  return o;
}

export function setupYard(bolt, bundle, Y = YARD) {
  bolt.reset(bundle.start);
  bolt.setProps(yardProps(Y));
  bolt.teleport(bundle.start, Y.start.at, Y.start.yaw);
  for (let k = 0; k < bolt.real.length; k++) bolt.placeReal(k, null, 0, 0);
  for (const j of Y.jumps) bolt.placeReal(j.k, j.at, j.yaw, j.h);
}

// ---------------------------------------------------------------- the body: which brain is in charge
export const TAKEOVER = 1.1;            // m before the bar the run-up jump brain takes over (teachers v3: 1.1 m from a trot)
const POSTURE = ["stand", "rise", "brake", "sniff", "settle"];
export let LANDING = "bridge";   // after a jump: "bridge" (a moment of canter, then trot: teachers v3) or "land" (the jump brain stops Bolt)
export function setLanding(l) { LANDING = l; }
export const BRAIN_NAMES = {
  gpu: "GPU dog-motion brain (360M steps)", g18: "canter-switching brain (run 18g)", posture: "posture brain",
  walk12: "run 12 walk brain (sharp turns)", jump: "run-up jump brain (run 16i)", getup: "self-righting brain (run 17e) + posture brain",
};

export class DogBody {
  // gpu, g18: Imitate instances on the same bolt; jumpBrain: the 16i brain (live.js Brain)
  constructor(bolt, gpu, g18) {
    this.bolt = bolt; this.gpu = gpu; this.g18 = g18; this.dog = gpu;
    this.want = { gait: "stand", steer: null };
    this.mode = "stand"; this.mt = 0; this.t = 0; this.speed = 0; this.events = [];
    this.armed = null; this.jumpInfo = null; this.falls = 0; this.recoveries = 0;
  }
  say(text, kind = "info") { this.events.push({ text, kind }); }
  get busy() { return ["jump", "land", "settle", "bridge", "fallen"].includes(this.mode); }
  get gait() { const m = this.mode; return m === "turnclip" || m === "sharp" || m === "brake" ? "walk" : m === "ctrot" ? "trot" : m === "sniff" ? "stand" : m; }
  brain() {
    const m = this.mode;
    if (m === "fallen") return "getup";
    if (m === "jump" || m === "land") return "jump";
    if (m === "settle") return "posture";
    if (m === "sharp" || m === "brake") return "walk12";
    if (POSTURE.includes(m)) return "posture";
    return this.dog === this.g18 ? "g18" : "gpu";
  }
  // hand the motion over between the two dog-motion brains: same clip, same clock, same last action
  use(to) {
    const from = this.dog;
    if (to === from) return;
    if (from.clip) { to.clip = to.clips[from.clip.name]; to.tau0 = from.tau0; to.t = from.t; to.lastAction = Float64Array.from(from.lastAction); }
    else to.clip = null;
    from.clip = null; this.dog = to;
  }
  // start a clip; a looping gait joins at the moment closest to Bolt's pose and joint speeds (agilityrun.js Body.clip)
  clip(name) {
    const d = this.dog;
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
    const b = this.bolt, was = this.mode;
    const clipMode = { walk: "walk", trot: "trot", ctrot: "trot", pace: "pace", canter: "canter", sit: "sit", lie: "lie", getup: "getup", bridge: "canter" }[mode];
    if (clipMode) {
      this.use(mode === "canter" || mode === "ctrot" || mode === "bridge" ? this.g18 : this.gpu);
      this.clip(clipMode);
    } else if (POSTURE.includes(mode) && !POSTURE.includes(was)) {
      this.dog.stop(); b.afterGetUp(false);
      // the run 12 brain takes the walk at the speed Bolt is going (the GPU walk is ~0.9 m/s; asked 0.4 it fell)
      if (mode === "brake") { b.lastAction = new Float64Array(12); b.command = "stand"; this.vs = clamp(this.speed, 0.3, 1.0); }
    } else if (mode === "sharp" && was !== "sharp" && was !== "brake") {
      this.dog.stop(); b.lastAction = new Float64Array(12); b.command = "stand"; this.vs = clamp(this.speed, 0.3, 1.0); b.speedCmd = this.vs;
    }
    this.prevMode = this.mode; this.mode = mode; this.mt = 0; Object.assign(this, extra);
  }
  fallen() { return this.bolt.up() < 0.5; }
  // arm a jump over hurdle k (aimed along its yaw)
  arm(k) { this.armed = k; this.bolt.aim(k); }

  step(dt) {
    const b = this.bolt, w = this.want, v = b.data.qvel;
    this.t += dt; this.mt += dt;
    this.speed += (Math.hypot(v[0], v[1]) - this.speed) * 0.1;
    if (this.mode !== "fallen" && this.fallen()) {
      this.say(`Bolt fell (${this.mode === "land" || this.mode === "bridge" ? "landing a jump" : `while ${this.gait}`})`, "bad");
      this.falls++; this.fellIn = this.mode;
      { const w = b.where(); this.fallInfo = { mode: this.mode, mt: +this.mt.toFixed(2), speed: +this.speed.toFixed(2), steer: this.want.steer === null ? null : +this.want.steer.toFixed(2), vs: this.vs, x: +w.x.toFixed(2), y: +w.y.toFixed(2), prev: this.prevMode }; }
      this.dog.stop(); this.armed = null; this.getup = new GetUp(b); this.go("fallen");
    }
    const m = this.mode;
    if (m === "fallen") {
      const cmd = this.getup.step(dt);
      if (this.getup.done) { this.recoveries++; this.say(`Bolt got itself up (${this.getup.secs.toFixed(1)} s)`, "good"); this.go("stand"); return b.run({ command: "stand", turn: 0, speed: 0.2 }); }
      if (this.getup.failed) { this.getup = new GetUp(b); this.say("Bolt tries again to get up", "info"); }   // a dog keeps trying
      return b.run(cmd);
    }
    if (m === "jump") {
      b.jumpTick();
      const J = this.jumpInfo;
      if (b.hurdleHits()) J.touched = true;
      if (!J.cleared && Math.min(...this.pawsPast()) > 0.08) { J.cleared = true; J.at = b.jt; }
      if ((J.cleared && b.jt - J.at >= 25 && b.up() > 0.85 && b.data.qpos[2] > 0.3) || b.jt >= this.jumpLimit) { J.landed = J.cleared; this.go(LANDING === "bridge" ? "bridge" : "land"); }
      return;
    }
    if (m === "land") {   // the jump brain slows Bolt down itself, then a moment sitting and standing (agilityrun.js)
      b.jumpTick();
      if ((Math.hypot(v[0], v[1]) < 1.0 && b.up() > 0.95 && Math.hypot(v[3], v[4]) < 0.5) || b.jt >= this.jumpLimit + 60) this.go("settle");
      return;
    }
    if (m === "settle") {
      b.run({ command: this.mt < 1.0 ? "sit" : "stand", turn: 0, speed: 0.2 });
      if (this.mt > 2.0) this.go("stand");
      return;
    }
    if (m === "bridge") {   // Bolt lands at a gallop: a moment of canter (18g), then its trot (canter -> trot practised)
      if (this.mt > 0.6) { this.go("ctrot"); this.dog.tick(); return; }
      this.dog.tick(); return;
    }
    // an armed jump: the jump brain takes over 1.1 m before the bar
    if (this.armed !== null && ["walk", "trot", "ctrot", "pace", "turnclip", "sharp"].includes(m)) {
      const ahead = b.hurdleAhead();
      // the run-up jump brain needs the speed of a trot: from anything slower Bolt runs out instead
      if (ahead <= TAKEOVER && ahead > 0.3 && m !== "trot" && m !== "ctrot") { this.armed = null; this.say("Bolt was not going fast enough and runs out of the jump"); }
      else if (ahead <= TAKEOVER && ahead > 0.3) {
        this.dog.stop(); b.startJump();
        this.jumpInfo = { k: this.armed, cleared: false, touched: false, landed: false, from: ahead, speed: this.speed };
        this.jumped = this.armed; this.armed = null;
        this.go("jump"); b.jumpTick(); return;
      }
    }
    const moving = ["walk", "trot", "canter", "pace"].includes(w.gait), e = w.steer;
    const sharp = e !== null && Math.abs(e) > 0.4;
    switch (m) {
      case "stand": case "sniff":
        if (w.gait === "sit" || w.gait === "lie") { if (this.mt > 0.3) this.go(w.gait); }   // the lie clip starts standing
        else if (w.gait === "sniff" && m !== "sniff") this.go("sniff");
        else if (moving && this.mt > 0.4) this.go(sharp ? "sharp" : "walk");
        break;
      case "walk": case "turnclip":
        if (w.gait === "sit" || (w.gait === "lie" && LIE_FROM_WALK)) this.go(w.gait);
        else if (w.gait === "stand" || w.gait === "sniff" || w.gait === "lie") this.go("brake");
        else if (sharp && this.armed === null) this.go("sharp");
        else if ((w.gait === "trot" || w.gait === "canter") && this.mt > 0.8 && (e === null || Math.abs(e) < 0.2)) this.go("trot");
        else if (w.gait === "pace" && this.mt > 0.8 && (e === null || Math.abs(e) < 0.2)) this.go("pace", { paceFor: null });
        else if (m === "walk" && e !== null && Math.abs(e) > 0.2 && this.mt > 0.4) { this.go("turnclip"); this.clip(e > 0 ? "turn_left" : "turn_right"); this.turnSign = Math.sign(e); }
        else if (m === "turnclip" && this.mt > 0.4 && (e === null || Math.abs(e) < 0.08 || Math.sign(e) !== this.turnSign)) this.go("walk");
        break;
      case "sharp":
        if (w.gait === "sit" || w.gait === "lie" || w.gait === "stand" || w.gait === "sniff") this.go("brake");
        else if (this.mt > 0.5 && (e === null || Math.abs(e) < 0.18)) this.go("walk");
        break;
      case "brake":
        if (moving && this.mt < 0.6) this.go("walk");
        else if (this.mt > 1.3 && this.vs <= 0.25) this.go(w.gait === "sniff" ? "sniff" : w.gait === "lie" ? "lie" : "stand");
        break;
      case "trot": case "ctrot":
        if (m === "ctrot" && this.mt > 1.5) { this.use(this.gpu); this.mode = "trot"; }   // back to the GPU brain's trot
        if (this.mt < 0.8) break;
        if (w.gait === "canter" && this.mt > 1.0 && this.armed === null && (e === null || Math.abs(e) < 0.2)) this.go("canter");
        else if (this.armed !== null) break;
        else if (w.gait === "pace") this.go("pace", { paceFor: null });
        else if (w.gait !== "trot" && w.gait !== "canter") this.go("pace", { next: "walk", paceFor: 0.8 });
        else if (e !== null && Math.abs(e) > 0.35) this.go("pace", { next: "walk", paceFor: 0.8 });   // slow down to turn
        break;
      case "canter":
        if (w.gait !== "canter" || (e !== null && Math.abs(e) > 0.3)) this.go("ctrot");   // 18g: canter -> trot 93%
        break;
      case "pace":
        if (this.paceFor !== null) { if (this.mt > this.paceFor) this.go(this.next || "walk"); }
        else if (this.mt > 0.8 && (w.gait !== "pace" || (e !== null && Math.abs(e) > 0.35))) this.go(w.gait === "trot" ? "trot" : "walk");
        break;
      case "sit":   // the GPU brain: the sit clip starts standing (it stops first), then holds the sit
        if (this.dog.finished() > 0.5 && w.gait === "lie") this.go("lie");
        else if (this.dog.finished() > 0.5 && w.gait !== "sit" && w.gait !== "lie") this.go(SIT_EXIT);
        break;
      case "lie":
        if (this.dog.finished() > 0.5 && w.gait !== "lie") this.go("getup");   // front end first, as a dog gets up
        break;
      case "getup":
        if (this.dog.finished() > 0.3) this.go(moving ? "walk" : GETUP_EXIT);
        break;
      case "rise":
        if (this.mt > 1.2) this.go("stand");
        break;
    }
    this.drive();
  }
  drive() {
    const b = this.bolt, m = this.mode, w = this.want;
    if (m === "stand" || m === "rise" || m === "sniff") return b.run({ command: "stand", turn: 0, speed: 0.2 });
    if (m === "sharp") {
      const steer = w.steer || 0, top = Math.min(1.4, b.maxTurn(this.vs || 0.5));
      const target = Math.abs(steer) < 0.3 ? 0.65 : clamp(0.25 + 0.6 * Math.cos(steer), TURN.vmin, 0.6);
      this.vs = target > this.vs ? Math.min(target, this.vs + 0.6 * b.dt) : Math.max(target, this.vs - 0.4 * b.dt);
      const cap = Math.abs(steer) > 1.2 ? TURN.cap : top;   // turning right round: gently (1.4 rad/s tipped it over)
      // the asked turn changes smoothly and keeps its side while the goal is behind (a goal straight behind
      // flips the heading error between +pi and -pi; a turn flipped at once tipped Bolt over)
      let want = clamp(2.5 * steer, -cap, cap);
      if (Math.abs(steer) > 1.8 && this.turnCmd && Math.sign(want) !== Math.sign(this.turnCmd)) want = -want;
      const prev = this.mt < 0.03 ? 0 : this.turnCmd || 0;
      this.turnCmd = prev + clamp(want - prev, -1.5 * b.dt, 1.5 * b.dt);
      return b.run({ command: "walk", turn: this.turnCmd, speed: this.vs });
    }
    if (m === "brake") { this.vs = Math.max(0.2, this.vs - 0.6 * b.dt); return b.run({ command: "walk", turn: 0, speed: this.vs }); }
    this.dog.tick();
  }
  pawsPast() {
    const d = this.bolt.data, [ux, uy] = this.bolt.hurdleDir, [hx, hy] = this.bolt.hurdleAt;
    return this.gpu.feet.map((g) => (d.geom_xpos[3 * g] - hx) * ux + (d.geom_xpos[3 * g + 1] - hy) * uy);
  }
}
// how Bolt gets up from a sit, and what follows the get-up clip (measured in doglife_test.mjs)
export const TURN = { cap: 0.7, vmin: 0.45 };
export function setTurnLimits(cap, vmin) { TURN.cap = cap; TURN.vmin = vmin; }
export let SIT_EXIT = "rise", GETUP_EXIT = "walk", LIE_FROM_WALK = false;
export function setLieFromWalk(v) { LIE_FROM_WALK = v; }
export function setExits(sit, getup) { SIT_EXIT = sit; GETUP_EXIT = getup; }

// ---------------------------------------------------------------- the mind
const LABEL = {
  wander: "exploring", sniffBin: "sniffing the bin", sniffSpot: "sniffing around", trot: "trotting about", zoomies: "zoomies!",
  sit: "sitting", nap: "napping", jump: "going for the jump", pace: "pacing along", sidewalk: "walking the sidewalk",
  ball: "chasing the ball", come: "coming to you", lookout: "watching the street",
};
export const PLACES = {
  porch: { at: [-3, 5.6], face: -Math.PI / 2, name: "by the porch steps" },
  kennel: { at: [-12.3, 4.85], face: -Math.PI / 2, name: "at the kennel door" },
  tree: { at: [-10.0, -5.2], face: 0, name: "in the shade of the oak" },
};
const SNIFF_SPOTS = [
  { at: [11.6, 6.0], name: "the bin", bin: true }, { at: [-10.2, -3.6], name: "the oak tree" }, { at: [10.4, -6.4], name: "the maple" },
  { at: [-7.5, 6.3], name: "the flower bed" }, { at: [4.7, -11.7], name: "the mailbox" }, { at: [-13.0, -7.2], name: "the bench" },
  { at: [-2.2, -9.0], name: "the puddle" }, { at: [-11.2, 5.2], name: "the kennel" }, { at: [13.6, 0], name: "the fence" }, { at: [-13.6, 1], name: "the fence" },
];

export class Mind {
  constructor(bolt, body, { seed = 1, Y = YARD } = {}) {
    this.bolt = bolt; this.body = body; this.Y = Y; this.rand = rng(seed); this.seed = seed;
    const r = this.rand;
    this.energy = 0.75 + 0.25 * r(); this.curiosity = 0.3 + 0.5 * r(); this.play = 0.3 + 0.5 * r();
    this.t = 0; this.task = null; this.log = []; this.counts = {};
    this.hurdles = Y.jumps.map((j) => ({ ...j }));
    this.extra = null;             // the viewer's jump {k, at, yaw, h}
    this.ball = { pos: [...Y.ball, 0.09], vel: [0, 0, 0], held: false, r: 0.09, flying: false };
    this.person = null;            // where the viewer stands (set by the page from the camera)
    this.stats = { jumps: 0, clean: 0, knocked: 0, distance: 0, fetches: 0, naps: 0, zoomies: 0 };
    this.last = null; this.lastName = null; this.kerbStops = 0;
    const w = bolt.where(); this.prev = [w.x, w.y];
  }
  say(text, kind = "info") { this.log.push({ t: this.t, text, kind }); }
  get label() { return this.task ? this.task.label : "standing"; }
  where() { const w = this.bolt.where(); return { x: w.x, y: w.y, yaw: w.yaw }; }
  obstacles(skip = null) { return yardObstacles(this.Y, [...this.hurdles, ...(this.extra ? [this.extra] : [])].filter((h) => h.k !== skip)); }

  // heading error to a target, steering round obstacles (a waypoint beside the first one in the way) and in
  // from the yard's edges
  steerTo(tx, ty, skipHurdle = null) {
    const p = this.where();
    let gx = tx, gy = ty;
    const d = hyp(tx - p.x, ty - p.y);
    if (d > 0.3) {
      const ux = (tx - p.x) / d, uy = (ty - p.y) / d;
      let first = null;
      for (const o of this.obstacles(skipHurdle)) {
        const ox = o.at[0] - p.x, oy = o.at[1] - p.y, along = ox * ux + oy * uy, side = -ox * uy + oy * ux, R = o.r + 0.35;
        if (along > 0 && along < d + R && Math.abs(side) < R && hyp(ox, oy) > 0.2 && (!first || along < first.along)) first = { o, along, side, R };
      }
      if (first) {   // pass on the side Bolt is already on
        const s = first.side > 0 ? -1 : 1, o = first.o;
        gx = o.at[0] + s * -uy * (first.R + 0.5); gy = o.at[1] + s * ux * (first.R + 0.5);
      }
    }
    return wrap(Math.atan2(gy - p.y, gx - p.x) - p.yaw);
  }
  // keep a point where Bolt may go
  inYard([x, y], margin = 0) { const R = this.Y.roam; return [clamp(x, R.x0 + margin, R.x1 - margin), clamp(y, R.y0 + margin, R.y1 - margin)]; }
  clear([x, y], r = 0.6) { return this.obstacles().every((o) => hyp(o.at[0] - x, o.at[1] - y) > o.r + r); }
  lineClear(a, b, r = 0.7) {
    const n = Math.ceil(hyp(b[0] - a[0], b[1] - a[1]) / 0.5);
    for (let i = 0; i <= n; i++) if (!this.clear([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n], r)) return false;
    return true;
  }
  randomSpot(margin = 1.0) {
    const R = this.Y.roam;
    for (let i = 0; i < 40; i++) {
      const p = [R.x0 + margin + (R.x1 - R.x0 - 2 * margin) * this.rand(), R.y0 + margin + (R.y1 - R.y0 - 2 * margin) * this.rand()];
      if (this.clear(p, 0.8)) return p;
    }
    return [0, -6];
  }

  // ---- the viewer
  call() { this.urgent = { kind: "come" }; this.say("You called Bolt", "cmd"); }
  throwBall(from, to) {
    const B = this.ball, T = 1.0;
    B.held = false; B.flying = true;
    B.pos = [from[0], from[1], 1.2];
    B.vel = [(to[0] - from[0]) / T, (to[1] - from[1]) / T, (0.09 - 1.2 + 0.5 * 9.81 * T * T) / T];
    this.urgent = { kind: "ball" }; this.say("You threw the ball", "cmd");
  }
  push(side) { this.bolt.shove(side, 0, 0.2); this.say(`You gave Bolt a shove (${Math.abs(side).toFixed(0)} N)`, "cmd"); }
  setJump() {
    const p = this.where(), k = this.Y.extra;
    let best = null;
    for (const da of [0, 0.4, -0.4, 0.8, -0.8, 1.6, -1.6, Math.PI]) {   // ahead of Bolt, where there is room
      const yaw = p.yaw + da, at = [p.x + 8.5 * Math.cos(yaw), p.y + 8.5 * Math.sin(yaw)];
      const run = [at[0] - 7 * Math.cos(yaw), at[1] - 7 * Math.sin(yaw)], land = [at[0] + 6 * Math.cos(yaw), at[1] + 6 * Math.sin(yaw)];
      const R = this.Y.roam, inside = (q) => q[0] > R.x0 && q[0] < R.x1 && q[1] > R.y0 && q[1] < R.y1;
      const saved = this.extra; this.extra = null;
      const ok = inside(run) && inside(land) && this.lineClear(run, land, 0.8);
      this.extra = saved;
      if (ok) { best = { at, yaw }; break; }
    }
    if (!best) { this.say("No room for a jump here: try again when Bolt is in the open", "info"); return false; }
    this.extra = { k, at: best.at, yaw: best.yaw, h: 0.40 };
    this.bolt.placeReal(k, best.at, best.yaw, 0.40);
    this.urgent = { kind: "jump", k }; this.say("You set up a 40 cm jump", "cmd");
    return true;
  }

  // ---- drives and choosing
  drift(dt) {
    const g = this.body.gait, m = this.body.mode;
    const cost = { walk: 0.0012, trot: 0.0035, pace: 0.003, canter: 0.012, jump: 0.03, land: 0.01, bridge: 0.01, fallen: 0.01 }[g] || 0;
    this.energy = clamp(this.energy - cost * dt + (m === "lie" ? 0.012 : m === "stand" || m === "sit" || m === "sniff" ? 0.0015 : 0) * dt, 0, 1);
    this.curiosity = clamp(this.curiosity + 0.004 * dt - (m === "sniff" ? 0.07 * dt : 0), 0, 1);
    this.play = clamp(this.play + 0.003 * dt * (0.5 + this.energy) - (g === "canter" || g === "jump" ? 0.05 * dt : 0), 0, 1);
  }
  choose() {
    const E = this.energy, C = this.curiosity, P = this.play, r = this.rand;
    const opts = {
      wander: 0.25 + 0.4 * C,
      sniffSpot: 0.15 + 0.7 * C,
      sniffBin: 0.1 + 0.6 * C,
      trot: E > 0.35 ? 0.2 + 0.4 * P * E : 0,
      zoomies: E > 0.45 ? 0.15 + 1.3 * P * E * E : 0,
      jump: E > 0.4 ? 0.15 + 0.6 * P * E : 0,
      pace: 0.12 + 0.15 * E,
      sit: 0.18 + 0.3 * (1 - E),
      nap: E < 0.55 ? 0.3 + 1.6 * (1 - E) - 0.4 : 0.02,
      sidewalk: 0.12 + 0.3 * C,
      lookout: 0.1 + 0.15 * C,
      ball: hyp(this.ball.pos[0] - this.where().x, this.ball.pos[1] - this.where().y) > 2 && !this.ball.held ? 0.1 + 0.5 * P * E : 0,
    };
    if (this.lastName) opts[this.lastName] *= 0.25;   // not the same thing twice in a row, usually
    let best = null, bv = -1;
    for (const [k, v] of Object.entries(opts)) { const s = v * (0.55 + 0.9 * r()); if (s > bv) { bv = s; best = k; } }
    return best;
  }
  start(name, extra = {}) {
    this.task = { name, label: LABEL[name], phase: "go", pt: 0, t0: this.t, ...extra };
    this.lastName = name;
    this.counts[name] = (this.counts[name] || 0) + 1;
    const T = this.task, r = this.rand, p = this.where();
    switch (name) {
      case "wander": { T.to = this.randomSpot(1.5); T.gait = r() < 0.25 && E_OK(this) ? "trot" : "walk"; this.say(T.gait === "trot" ? "Bolt trots off to explore" : "Bolt wanders off to explore"); break; }
      case "sniffSpot": case "sniffBin": {
        const spots = SNIFF_SPOTS.filter((s) => (name === "sniffBin") === !!s.bin && hyp(s.at[0] - p.x, s.at[1] - p.y) > 1.5);
        const s = spots[Math.floor(r() * spots.length)] || SNIFF_SPOTS[0];
        T.to = s.at; T.what = s.name; T.gait = "walk"; T.hold = 3 + 5 * r(); T.label = `sniffing ${s.name}`;
        this.say(`Bolt heads for ${s.name}, nose first`); break;
      }
      case "trot": { T.to = this.farSpot(9); T.gait = "trot"; this.say("Bolt trots across the lawn"); break; }
      case "pace": { T.to = this.farSpot(8); T.gait = "pace"; this.say("Bolt ambles along at a pace"); break; }
      case "zoomies": {
        const line = this.zoomLine();
        if (!line) { this.task = null; return this.start("trot"); }
        T.from = line[0]; T.to = line[1]; T.dir = line[2]; T.phase = "lineup"; T.canterFor = 2 + 2 * r(); T.label = "getting ready for zoomies";
        this.say("Bolt gets the zoomies!", "good"); this.stats.zoomies++; break;
      }
      case "sit": { T.hold = 4 + 8 * r(); T.phase = "sit"; this.say("Bolt sits down for a moment"); break; }
      case "nap": {
        const keys = Object.keys(PLACES), place = PLACES[keys[Math.floor(r() * keys.length)]];
        T.to = place.at; T.place = place; T.gait = "walk"; T.hold = 18 + 22 * r(); T.label = `off for a nap ${place.name}`;
        this.say(`Bolt is tired and heads off to nap ${place.name}`); break;
      }
      case "jump": {
        const k = extra.k !== undefined ? extra.k : null;
        const list = [...this.hurdles, ...(this.extra ? [this.extra] : [])];
        const h = k !== null ? list.find((q) => q.k === k) : list[Math.floor(r() * list.length)];
        // from whichever side Bolt is on: the hurdle is turned to face it (the bar sits the same either way)
        const fx = Math.cos(h.yaw), fy = Math.sin(h.yaw), side = (p.x - h.at[0]) * fx + (p.y - h.at[1]) * fy;
        let yaw = side > 0 ? wrap(h.yaw + Math.PI) : h.yaw;
        const ok = (y) => { const R = this.Y.roam, q = [h.at[0] - 7.5 * Math.cos(y), h.at[1] - 7.5 * Math.sin(y)], l = [h.at[0] + 6 * Math.cos(y), h.at[1] + 6 * Math.sin(y)]; return q[0] > R.x0 && q[0] < R.x1 && q[1] > R.y0 && q[1] < R.y1 && l[0] > R.x0 - 1 && l[0] < R.x1 + 1 && l[1] > R.y0 - 1 && l[1] < R.y1 + 1; };
        if (!ok(yaw)) yaw = wrap(yaw + Math.PI);
        T.h = h; T.yaw = yaw; T.phase = "go"; T.to = [h.at[0] - 7.5 * Math.cos(yaw), h.at[1] - 7.5 * Math.sin(yaw)]; T.gait = "walk";
        T.label = `lining up for the ${Math.round(h.h * 100)} cm jump`;
        this.say(`Bolt eyes the ${Math.round(h.h * 100)} cm jump`); break;
      }
      case "sidewalk": {
        const W = this.Y.walkway, x0 = clamp(p.x + (r() - 0.5) * 6, W.x0 + 2, W.x1 - 2);
        const dir = x0 > 0 ? -1 : 1, len = 6 + 8 * r();
        T.to = [x0, W.y + 1.2]; T.along = [clamp(x0 + dir * len, W.x0 + 1, W.x1 - 1), W.y]; T.gait = "walk";
        this.say("Bolt goes to walk along the sidewalk"); break;
      }
      case "lookout": {
        T.to = [clamp(p.x, -12, 12), -12.6]; T.gait = "walk"; T.hold = 5 + 6 * r(); T.label = "heading to watch the street";
        this.say("Bolt goes to watch the street"); break;
      }
      case "ball": { T.gait = "trot"; T.phase = "chase"; this.say(this.ball.flying ? "Bolt watches the ball fly and gives chase" : "Bolt spots the ball and goes for it", "good"); break; }
      case "come": { T.gait = "trot"; T.phase = "go"; this.say("Bolt comes running"); break; }
    }
  }
  farSpot(min) {
    const p = this.where();
    for (let i = 0; i < 30; i++) { const q = this.randomSpot(2); if (hyp(q[0] - p.x, q[1] - p.y) > min && this.lineClear([p.x, p.y], q)) return q; }
    return this.randomSpot(2);
  }
  // a straight line for a canter: lined up at a walk, 1.5 s trot, 2-4 s canter, and ~6 m to slow down, all on clear lawn
  zoomLine() {
    const R = this.Y.roam, p = this.where();
    for (let i = 0; i < 80; i++) {
      // first: bolt off from right here, roughly the way Bolt faces; then from a spot nearby
      const here = i < 30, a = here ? [p.x, p.y] : this.randomSpot(1.5);
      const ang = here ? p.yaw + (this.rand() - 0.5) * (i < 12 ? 1.2 : 3.0) : this.rand() * 2 * Math.PI, d = [Math.cos(ang), Math.sin(ang)];
      const b = [a[0] + 17 * d[0], a[1] + 17 * d[1]];
      if (b[0] < R.x0 + 0.5 || b[0] > R.x1 - 0.5 || b[1] < R.y0 + 0.5 || b[1] > R.y1 - 0.5) continue;
      if (!this.lineClear(a, b, 0.7)) continue;
      if (hyp(a[0] - p.x, a[1] - p.y) > 8) continue;
      return [a, b, ang];
    }
    return null;
  }

  // ---- one control step: what Bolt wants now
  step(dt) {
    this.t += dt;
    this.drift(dt);
    const body = this.body, w = body.want, p = this.where();
    if (body.mode !== "fallen") { this.stats.distance += hyp(p.x - this.prev[0], p.y - this.prev[1]); }
    this.prev = [p.x, p.y];
    this.ballStep(dt);
    // the kerb: a good dog never steps into the road
    if (p.y < this.Y.stopAt && !body.busy) {
      if (!this.atKerb) { this.atKerb = true; this.kerbStops++; this.say("Bolt stops at the kerb", "good"); this.task = { name: "kerb", label: "stopping at the kerb", phase: "stop", pt: 0, t0: this.t }; }
    } else if (p.y > this.Y.stopAt + 0.6) this.atKerb = false;
    if (body.busy) { if (body.mode === "fallen") this.task = this.task && this.task.name === "jump" ? null : this.task; return; }
    if (this.urgent && (!this.task || !["come", "ball", "jump"].includes(this.task.name) || this.urgent.kind !== this.task.name)) {
      const u = this.urgent; this.urgent = null;
      if (this.task && (this.task.name === "nap" || this.task.name === "sit") && ["sit", "lie", "getup"].includes(body.mode)) this.say("Bolt hears you and gets up");
      this.task = null; this.start(u.kind, u.k !== undefined ? { k: u.k } : {});
    }
    if (!this.task) this.start(this.choose());
    const T = this.task;
    T.pt += dt;
    const done = (text, kind) => { if (text) this.say(text, kind); this.task = null; w.gait = body.mode === "sit" || body.mode === "lie" ? w.gait : "walk"; };
    const go = (to, gait, near = 0.6, skip = null) => {   // walk/trot to a spot; true on arrival
      const d = hyp(to[0] - p.x, to[1] - p.y), e = this.steerTo(to[0], to[1], skip);
      // circling a spot closer than the turn allows: count it as reached
      if (body.mode === "sharp" && body.mt > 5 && d < 2.0) return true;
      w.steer = e;
      w.gait = d < near ? "stand" : gait === "walk" || d < 3.0 ? "walk" : gait;
      return d < near;
    };
    if (T.pt > 70 && !["nap", "kerb"].includes(T.name)) return done(null);   // stuck: think again
    switch (T.name) {
      case "kerb": {
        w.steer = null; w.gait = "stand";
        if (T.pt > 2.5) { this.task = null; this.start("wander"); this.task.to = this.inYard([p.x, -6]); this.say("Bolt turns back to the lawn"); }
        break;
      }
      case "wander": case "trot": case "pace":
        if (go(T.to, T.gait, 1.0)) done(T.name === "wander" && this.rand() < 0.5 ? null : null);
        break;
      case "sniffSpot": case "sniffBin":
        if (T.phase === "go") { if (go(T.to, "walk", 0.7)) { T.phase = "sniff"; T.pt = 0; this.say(`Bolt sniffs ${T.what}`); } }
        else { w.steer = null; w.gait = "sniff"; if (T.pt > T.hold) { this.curiosity = Math.max(0, this.curiosity - 0.25); done(T.name === "sniffBin" && this.rand() < 0.5 ? "Nothing to eat in there. Bolt moves on" : null); } }
        break;
      case "lookout":
        if (T.phase === "go") { if (go(T.to, "walk", 0.8)) { T.phase = "sit"; T.pt = 0; this.say("Bolt sits at the edge of the lawn and watches the street"); } }
        else { w.steer = null; w.gait = T.pt < T.hold ? "sit" : "walk"; if (T.pt > T.hold && body.mode !== "sit") done(null); }
        break;
      case "sit":
        w.steer = null;
        if (T.pt < T.hold) w.gait = "sit";
        else { w.gait = "walk"; if (!["sit", "rise", "lie", "getup"].includes(body.mode)) done("Bolt is up again"); }
        break;
      case "nap": {
        const P = T.place;
        if (T.phase === "go") { if (go(T.to, "walk", 0.8)) { T.phase = "turn"; T.pt = 0; } }
        else if (T.phase === "turn") {   // circle round to face out, as dogs do before lying down
          const e = wrap(P.face - p.yaw); w.steer = e; w.gait = "walk";
          if (Math.abs(e) < 0.35 || T.pt > 6) { T.phase = "down"; T.pt = 0; this.say(`Bolt turns round, sits and lies down ${P.name}`); this.stats.naps++; }
        } else if (T.phase === "down") {
          w.steer = null; w.gait = "lie"; T.label = "napping";
          if (body.mode === "lie" && this.energy > 0.92 && T.pt > 10) T.phase = "wake";
          if (T.pt > T.hold + 10 && body.mode === "lie") T.phase = "wake";
        } else if (T.phase === "wake") {
          w.gait = "walk"; w.steer = null; T.label = "waking up";
          if (!T.woke) { T.woke = true; this.say("Bolt wakes up and gets up, front legs first"); }
          if (!["lie", "getup", "sit", "rise"].includes(body.mode)) done(null);
        }
        break;
      }
      case "zoomies": {
        const [a, b] = [T.from, T.to];
        if (T.phase === "lineup") {
          if (go(a, "walk", 1.0)) { T.phase = "face"; T.pt = 0; }
        } else if (T.phase === "face") {
          const e = wrap(T.dir - p.yaw); w.steer = e; w.gait = "walk";
          if (Math.abs(e) < 0.12 || T.pt > 8) { T.phase = "run"; T.pt = 0; T.label = "zoomies!"; }
        } else if (T.phase === "run") {
          // a canter cannot steer (the clip runs straight; asked to turn it drops to a trot), so Bolt runs on
          // where it points and slows down early if the lawn ahead runs out or something is in the way
          const fx = Math.cos(p.yaw), fy = Math.sin(p.yaw), R = this.Y.roam, look = body.mode === "canter" ? 8 : 6;
          const ahead = [p.x + look * fx, p.y + look * fy];
          const room = ahead[0] > R.x0 && ahead[0] < R.x1 && ahead[1] > R.y0 && ahead[1] < R.y1 && this.lineClear([p.x + fx, p.y + fy], [p.x + 5 * fx, p.y + 5 * fy], 0.4);
          w.steer = ["canter", "ctrot", "trot", "pace"].includes(body.mode) ? null : this.steerTo(b[0], b[1]);
          if (!room) T.slow = true;
          if (body.mode === "canter") T.cantered = true;
          w.gait = T.slow ? "walk" : body.mode === "canter" && body.mt > T.canterFor ? (T.slow = true, "walk") : "canter";
          if ((T.cantered && ["walk", "turnclip", "sharp", "stand"].includes(body.mode)) || T.pt > 20) { this.play = Math.max(0, this.play - 0.5); done(T.cantered ? "Zoomies over: Bolt slows to a walk" : null); }
        }
        break;
      }
      case "jump": {
        const h = T.h, fx = Math.cos(T.yaw), fy = Math.sin(T.yaw);
        const along = (p.x - h.at[0]) * fx + (p.y - h.at[1]) * fy, side = -(p.x - h.at[0]) * fy + (p.y - h.at[1]) * fx;   // along < 0: before the bar
        if (T.phase === "go") {
          if (go(T.to, "walk", 0.8, h.k)) { T.phase = "line"; T.pt = 0; this.bolt.placeReal(h.k, h.at, T.yaw, h.h); }
        } else if (T.phase === "line") {   // walk along the line to the bar, square to it, then trot in
          const aim = [h.at[0] + (along + 2.0) * fx, h.at[1] + (along + 2.0) * fy];
          w.steer = wrap(Math.atan2(aim[1] - p.y, aim[0] - p.x) - p.yaw); w.gait = "walk";
          const askew = Math.abs(wrap(p.yaw - T.yaw));
          if (-along < 4.8 && -along > 2.6 && Math.abs(side) < 0.25 && askew < 0.14) {
            T.phase = "commit"; T.pt = 0; body.arm(h.k); w.gait = "trot"; T.label = `jumping the ${Math.round(h.h * 100)} cm hurdle`;
          } else if (-along < 2.4 || T.pt > 15) { T.phase = "go"; T.pt = 0; T.tries = (T.tries || 0) + 1; this.say("Bolt was not lined up: it circles round for another go"); if (T.tries > 2) { this.say("Bolt gives up on the jump for now"); done(null); } }
        } else if (T.phase === "commit") {
          w.gait = "trot"; w.steer = 0;
          if (body.jumped === h.k && !body.busy) {
            const J = body.jumpInfo, dropped = this.bolt.barDropped(h.k);
            this.stats.jumps++;
            if (J.cleared && !dropped) { this.stats.clean++; this.say(J.touched ? "Over! It brushed the bar, the bar stayed up" : `Clear jump over ${Math.round(h.h * 100)} cm!`, "good"); }
            else if (dropped) { this.stats.knocked++; this.say("Over, but the bar came down", "bad"); }
            else this.say("Bolt ran out of the jump", "bad");
            body.jumped = null; T.phase = "after"; T.pt = 0; this.play = Math.max(0, this.play - 0.35);
            if (h === this.extra) this.extra.done = true;
          } else if (body.armed === null && !body.busy && body.jumped !== h.k && T.pt > 1.0) { T.phase = "go"; T.pt = 0; }
          if (T.pt > 10) done(null);
        } else if (T.phase === "after") {
          w.gait = "walk"; w.steer = null;
          if (T.pt > 1.5) { for (const k of [h.k]) if (this.bolt.barDropped(k)) this.bolt.placeReal(k, h.at, T.yaw, h.h); done(null); }
        }
        break;
      }
      case "sidewalk":
        if (T.phase === "go") { if (go(T.to, "walk", 0.8)) { T.phase = "along"; T.pt = 0; T.label = "walking the sidewalk"; this.say("Bolt walks along the sidewalk"); } }
        else {
          const tgt = T.along, d = hyp(tgt[0] - p.x, tgt[1] - p.y);
          w.steer = wrap(Math.atan2(tgt[1] - p.y, tgt[0] - p.x) - p.yaw); w.gait = d < 1 ? "stand" : "walk";
          if (d < 1 || T.pt > 30) { T.phase = "back"; done(null); this.start("wander"); this.task.to = this.inYard([p.x, -9]); this.task.label = "back onto the lawn"; }
        }
        break;
      case "come": {
        const me = this.person || [p.x, p.y], to = this.inYard([me[0], me[1]], 0.5);
        if (T.phase === "go") { if (go(to, "trot", 1.2)) { T.phase = "sit"; T.pt = 0; this.say("Bolt is here, and sits for you", "good"); } }
        else { w.steer = null; w.gait = "sit"; if (T.pt > 6) { w.gait = "walk"; if (body.mode !== "sit") done(null); } }
        break;
      }
      case "ball": {
        const B = this.ball;
        if (T.phase === "chase") {
          if (B.flying && T.pt < 0.8) { w.gait = "stand"; w.steer = null; break; }   // watch it go
          const snoutD = hyp(B.pos[0] - (p.x + 0.38 * Math.cos(p.yaw)), B.pos[1] - (p.y + 0.38 * Math.sin(p.yaw)));
          const e = this.steerTo(B.pos[0], B.pos[1]), d = hyp(B.pos[0] - p.x, B.pos[1] - p.y);
          w.steer = e; w.gait = d < 2.5 ? "walk" : "trot";
          if (snoutD < 0.35 && !B.flying) { B.held = true; T.phase = "bring"; T.pt = 0; T.label = "bringing the ball back"; this.say("Got it! Bolt picks up the ball", "good"); }
          else if (d < 0.6 && !B.flying) { w.gait = "walk"; }
        } else if (T.phase === "bring") {
          const me = this.person || [0, -6], to = this.inYard(me, 0.5);
          if (go(to, "trot", 1.2)) { B.held = false; B.pos = [p.x + 0.45 * Math.cos(p.yaw), p.y + 0.45 * Math.sin(p.yaw), 0.09]; B.vel = [0.3 * Math.cos(p.yaw), 0.3 * Math.sin(p.yaw), 0]; this.stats.fetches++; T.phase = "sit"; T.pt = 0; this.say("Bolt drops the ball at your feet", "good"); this.play = Math.max(0, this.play - 0.3); }
        } else { w.gait = "sit"; w.steer = null; if (T.pt > 4) { w.gait = "walk"; if (body.mode !== "sit") done(null); } }
        if (T.pt > 40) done(null);
        break;
      }
    }
  }
  // the ball: a light ball thrown through the air, bouncing and rolling to a stop on the grass (simple physics
  // here; carried in front of Bolt's snout while it holds it)
  ballStep(dt) {
    const B = this.ball;
    if (B.held) {
      const d = this.bolt.data, snout = this.body.snout;
      if (snout !== undefined) B.pos = [d.geom_xpos[3 * snout], d.geom_xpos[3 * snout + 1], d.geom_xpos[3 * snout + 2] - 0.02];
      else { const w = this.bolt.where(); B.pos = [w.x + 0.36 * Math.cos(w.yaw), w.y + 0.36 * Math.sin(w.yaw), 0.3]; }
      B.vel = [0, 0, 0]; return;
    }
    const v = B.vel, q = B.pos, L = this.Y.lawn;
    v[2] -= 9.81 * dt;
    for (let i = 0; i < 3; i++) q[i] += v[i] * dt;
    if (q[2] < B.r) {
      q[2] = B.r;
      if (v[2] < -0.6) { v[2] = -0.45 * v[2]; v[0] *= 0.7; v[1] *= 0.7; }
      else { v[2] = 0; B.flying = false; const s = hyp(v[0], v[1]), f = Math.max(0, s - 1.6 * dt); if (s > 0) { v[0] *= f / s; v[1] *= f / s; } }
    }
    // the fences and the house stop it; it may roll onto the sidewalk but the kerb stops it
    if (q[0] < L.x0 + 0.15 || q[0] > L.x1 - 0.15) { q[0] = clamp(q[0], L.x0 + 0.15, L.x1 - 0.15); v[0] *= -0.4; }
    if (q[1] > 6.8) { q[1] = 6.8; v[1] *= -0.4; }
    if (q[1] < -14.6) { q[1] = -14.6; v[1] *= -0.3; }
  }
}
function E_OK(m) { return m.energy > 0.45; }

// A new day in the same world (the page's "another five minutes"): the yard reset, a fresh body and mind.
export function newDay(L, bundle, seed) {
  setupYard(L.bolt, bundle);
  L.gpu.stop(); L.g18.stop();
  const body = new DogBody(L.bolt, L.gpu, L.g18);
  body.jumpLimit = L.body.jumpLimit; body.snout = L.body.snout;
  L.body = body; L.mind = new Mind(L.bolt, body, { seed });
  return L;
}

// Everything for a page or a test. brains: { gpu, g18, jump16i } brain specs; clips: dog_clips.json
export function makeLife(live, mj, bundle, brains, clips, seed) {
  const bolt = new live.Bolt(mj, bundle);
  if (brains.jump16i) { bolt.brains.jump = new live.Brain(brains.jump16i); bolt.c = { ...bolt.c, jump_steps: brains.jump16i.jump_steps || 200 }; }
  setupYard(bolt, bundle);
  const gpu = new Imitate(bolt, brains.gpu, clips), g18 = new Imitate(bolt, brains.g18, clips);
  const body = new DogBody(bolt, gpu, g18);
  body.jumpLimit = brains.jump16i ? brains.jump16i.jump_steps || 200 : 150;
  const sn = (bundle.geoms || []).find((g) => g.part === "snout");
  body.snout = sn ? sn.id : undefined;
  const mind = new Mind(bolt, body, { seed });
  return { bolt, body, mind, gpu, g18 };
}
