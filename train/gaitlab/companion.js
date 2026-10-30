// Bolt Companion (simulation): talk to Bolt, and Bolt answers with a personality and acts it out with its
// MuJoCo body. Three layers, no DOM here (companion_template.html draws and wires it, companion_test.mjs
// tests it headless):
//
//   Pet      the physics body (live.js Bolt, doglife.js DogBody, the same trained brains as Bolt's Day Out and
//            Bolt Dash) and a fixed set of ACTIONS it can really do. Learned moves go through the trained
//            brains; a few expressive moves are assisted by a guide force in the physics (like Bolt Dash's
//            hop), and a few are drawn only (head tilt, tail, ears, eyes). ACTIONS says which is which.
//   Mind     what Bolt says: companion_mind.js (in-house, no network; an optional local model on the person's
//            own machine can stand in for it). Its replies name moves from ACTIONS.
//   Dance    a built-in tune with known beats, beat finding for the person's own music (beatsFromSamples, offline
//            on the decoded file) or the microphone (BeatTracker, live), and a choreography on the beats.
//
// Nothing here touches the network.
import { DogBody } from "./doglife.js";
import { Imitate } from "./imitate.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// ------------------------------------------------------------------------------------------- the actions
// kind: learned = the trained brains move every motor; assisted = the brains hold the legs, a guide force or
// torque on the body makes the move (disclosed on the page); drawn = only the drawing changes (no physics).
// secs: about how long the move takes (the model is told, so it does not ask for an hour of tricks).
export const ACTIONS = {
  stand:        { kind: "learned",  secs: 1,   label: "stand",          how: "posture brain" },
  sit:          { kind: "learned",  secs: 4,   label: "sit",            how: "dog-motion brain, sit clip" },
  lie_down:     { kind: "learned",  secs: 6,   label: "lie down",       how: "dog-motion brain, lie clip" },
  get_up:       { kind: "learned",  secs: 7,   label: "get up",         how: "dog-motion brain, get-up clip" },
  walk_forward: { kind: "learned",  secs: 3,   label: "walk",           how: "dog-motion brain, walk" },
  come_closer:  { kind: "learned",  secs: 4,   label: "come closer",    how: "dog-motion brain, walk and turns" },
  turn_left:    { kind: "learned",  secs: 2.5, label: "turn left",      how: "dog-motion brain, turn clip" },
  turn_right:   { kind: "learned",  secs: 2.5, label: "turn right",     how: "dog-motion brain, turn clip" },
  spin:         { kind: "learned",  secs: 8,   label: "spin round",     how: "sharp-turn walk brain, a full circle" },
  zoomies:      { kind: "learned",  secs: 12,   label: "zoomies",        how: "dog-motion brain, trot out and back" },
  hop:          { kind: "assisted", secs: 1.2, label: "hop",            how: "posture brain holds the legs, an upward kick lifts the body" },
  happy_bounce: { kind: "assisted", secs: 1.8, label: "happy bounce",   how: "three small assisted hops" },
  wiggle:       { kind: "assisted", secs: 1.6, label: "wiggle",         how: "a side-to-side twist torque on the body" },
  play_bow:     { kind: "assisted", secs: 2.2, label: "play bow",       how: "posture brain stands, a nose-down torque tips the front down" },
  twirl:        { kind: "assisted", secs: 1.2, label: "twirl",          how: "an assisted hop with a small twist" },
  head_tilt:    { kind: "drawn",    secs: 0,   label: "head tilt",      how: "drawn only" },
  nod:          { kind: "drawn",    secs: 0,   label: "nod",            how: "drawn only" },
  tail_wag:     { kind: "drawn",    secs: 0,   label: "tail wag",       how: "drawn only" },
  ears_down:    { kind: "drawn",    secs: 0,   label: "ears down",      how: "drawn only" },
  dance:        { kind: "assisted", secs: 8,   label: "dance",          how: "a short routine on Bolt's own tune (Dance mode)" },
};
export const EMOTIONS = ["happy", "excited", "playful", "silly", "proud", "curious", "calm", "caring", "sad", "sleepy", "worried"];
// what other names for the moves mean (models and people are inventive): synonym -> action id
const ACTION_ALIASES = {
  sit_down: "sit", sitting: "sit", lie: "lie_down", lay_down: "lie_down", laydown: "lie_down", lie_down_sad: "lie_down", sad_lie_down: "lie_down",
  down: "lie_down", rest: "lie_down", nap: "lie_down", sleep: "lie_down", getup: "get_up", stand_up: "get_up", rise: "get_up", up: "stand",
  walk: "walk_forward", forward: "walk_forward", come: "come_closer", come_here: "come_closer", approach: "come_closer", snuggle: "come_closer",
  turn: "turn_left", turn_around: "spin", excited_spin: "spin", chase_tail: "spin", spin_around: "spin", run: "zoomies", trot: "zoomies", run_around: "zoomies",
  jump: "hop", leap: "hop", bounce: "happy_bounce", bounce_happy: "happy_bounce", hop_happy: "happy_bounce", happy_hop: "happy_bounce",
  wag: "tail_wag", wag_tail: "tail_wag", tailwag: "tail_wag", butt_wiggle: "wiggle", shimmy: "wiggle", wiggle_body: "wiggle", shake_body: "wiggle",
  bow: "play_bow", playbow: "play_bow", tilt_head: "head_tilt", headtilt: "head_tilt", tilt: "head_tilt", nod_head: "nod", ears: "ears_down", droop_ears: "ears_down",
  twist: "twirl", pirouette: "twirl", dance_moves: "dance", boogie: "dance",
};
export function actionId(a) {
  if (typeof a !== "string") return null;
  const k = a.trim().toLowerCase().replace(/[\s-]+/g, "_").replace(/[^a-z_]/g, "");
  if (ACTIONS[k]) return k;
  return ACTION_ALIASES[k] || null;
}

// the guide's settings (assisted moves only; see Pet.guide)
export const ASSIST = {
  hop: 1.7, bounce: 1.0, float: 0.3,            // kick (m/s straight up on the root joint), lift while in the air (share of g)
  wiggle: 24, wiggleHz: 2.6,                     // twist torque (N m) and how fast
  bow: 35, bowMax: 0.42,                        // nose-down torque (N m), released past this pitch (rad)
  twirl: 0.9, level: 60, levelD: 4,             // twirl: yaw rate kick (rad/s) in a hop; level torque while assisted (N m per unit tilt, per rad/s)
};
const POSE_MODES = ["stand", "rise", "brake", "sniff", "settle"];

// ------------------------------------------------------------------------------------------- the body
export class Pet {
  // live: the live.js module; mj: MuJoCo; bundle: bolt_run.json; brains: {gpu, g18} imitation brains; clips: dog_clips.json
  constructor(live, mj, bundle, brains, clips) {
    const bolt = new live.Bolt(mj, bundle);
    bolt.baseXml = bundle.xml.replace(/<body name="(hurdle\d*|ice|bar\d+)"[^]*?<\/body>/g, "");   // only Bolt and the floor
    bolt.build("");
    const m = bolt.model;
    this.dogBodies = []; this.dogMass = [];
    for (let i = 1; i < m.nbody; i++) if (m.body_rootid[i] === m.body_rootid[bolt.torso]) { this.dogBodies.push(i); this.dogMass.push(m.body_mass[i]); }
    this.mass = this.dogMass.reduce((a, b) => a + b, 0);
    const self = this, M = bolt.c.motor;
    bolt.applyPush = () => {};
    bolt.physics = function (n) {   // live.js Bolt.physics (the motor limits used in training) plus the guide
      const { mj: J, model: mm, data: d } = this, fr = mm.actuator_forcerange, v = d.qvel, h = mm.opt.timestep;
      for (let s = 0; s < n; s++) {
        if (M) for (let j = 0; j < 12; j++) {
          const w = Math.abs(v[6 + j]);
          const drive = Math.min(M.torque, M.power / Math.max(w, 1e-6)) * clamp((M.cutoff - w) / (M.cutoff - M.speed), 0, 1);
          fr[2 * j] = v[6 + j] < 0 ? -drive : -M.torque; fr[2 * j + 1] = v[6 + j] >= 0 ? drive : M.torque;
        }
        self.guide(h); J.mj_step(mm, d);
      }
    };
    const first = Math.min(...bundle.geoms.map((g) => g.id));
    this.geoms = bundle.geoms.map((g) => ({ ...g, id: g.id - first + m.body_geomadr[bolt.torso] }));
    this.bolt = bolt; this.bundle = bundle;
    this.gpu = new Imitate(bolt, brains.gpu, clips); this.g18 = new Imitate(bolt, brains.g18, clips);
    this.home = [0, 0]; this.viewer = [1.6, 0];   // where Bolt lives, and where the person is (the camera side)
    this.reset();
  }
  reset() {
    const b = this.bolt;
    this.gpu.stop(); this.g18.stop();
    b.teleport(this.bundle.start, this.home, 0);
    this.body = new DogBody(b, this.gpu, this.g18);
    this.t = 0; this.queue = []; this.cur = null; this.assist = null; this.events = []; this.log = [];
    this.look = { tilt: 0, nod: 0, ears: 0, wag: 0.4, emotion: "happy" };
    this.drawn = [];   // drawn-only moves running alongside: { id, t0, secs }
    this.stats = { falls: 0, done: 0, skipped: 0 };
  }
  emit(type, extra = {}) { this.events.push({ type, t: this.t, ...extra }); }
  where() { return this.bolt.where(); }
  pitch() { return Math.asin(clamp(this.bolt.rotation()[6], -1, 1)); }   // nose up positive
  posture() {   // what the body is doing now, in words the Mind can use
    const m = this.body.mode;
    if (m === "fallen") return "fallen over (getting up)";
    if (m === "sit") return "sitting";
    if (m === "lie") return "lying down";
    if (m === "getup") return "getting up";
    if (["walk", "turnclip", "sharp", "brake", "pace"].includes(m)) return "walking";
    if (["trot", "ctrot", "canter"].includes(m)) return "trotting";
    return "standing";
  }

  // ---- what to do: a list of action ids (unknown ones are dropped), played in order; replaces what is queued
  perform(list, { keep = false } = {}) {
    const ids = (Array.isArray(list) ? list : [list]).map(actionId).filter(Boolean);
    if (!keep) { this.queue = []; if (this.cur && !this.cur.essential) this.finish(true); }
    for (const id of ids) {
      if (ACTIONS[id].kind === "drawn") this.queue.push({ id, drawn: true });
      else this.queue.push({ id });
    }
    return ids;
  }
  stop() { this.queue = []; if (this.cur) this.finish(true); this.body.want.gait = this.body.mode === "sit" || this.body.mode === "lie" ? this.body.want.gait : "stand"; this.body.want.steer = null; }
  get idle() { return !this.cur && !this.queue.length; }
  finish(skipped = false) {
    const c = this.cur;
    if (!c) return;
    this.assist = null;
    if (skipped) this.stats.skipped++; else this.stats.done++;
    this.log.push({ id: c.id, t0: +c.t0.toFixed(2), secs: +(this.t - c.t0).toFixed(2), skipped, posture: this.posture() });
    this.emit("done", { id: c.id, skipped });
    this.cur = null;
  }
  startNext() {
    while (this.queue.length) {
      const a = this.queue.shift();
      if (a.drawn) { this.drawn.push({ id: a.id, t0: this.t, secs: a.id === "ears_down" ? 3 : a.id === "tail_wag" ? 2.5 : 1.4 }); this.emit("start", { id: a.id }); this.log.push({ id: a.id, t0: +this.t.toFixed(2), secs: 0, drawn: true }); continue; }
      this.cur = { id: a.id, t0: this.t, st: 0, stage: 0 }; this.emit("start", { id: a.id });
      return;
    }
  }
  // ---- the assisted moves need Bolt standing on the posture brain; true once it is
  standReady() {
    const B = this.body;
    B.want.gait = "stand"; B.want.steer = null;
    return this.settled(0.6);
  }
  // standing still on the posture brain (the assisted moves start only from here)
  settled(minT = 0.4, landGap = 0.25, vmax = 0.25) {
    const B = this.body, v = this.bolt.data.qvel;
    return (B.mode === "stand" || B.mode === "sniff") && B.mt > minT && this.bolt.up() > 0.92 && Math.hypot(v[0], v[1]) < vmax && Math.hypot(v[3], v[4]) < 0.6 && !(this.assist && this.assist.kind !== "land") && this.t - (this.landedAt ?? -9) > landGap;
  }
  // go to a spot: a learned sharp turn on the spot to face it (the run 12 brain, from standing), then walk with
  // gentle steering (sharp turns while already walking tipped Bolt over in testing); true once within `stop` m
  goTo(c, [gx, gy], stop) {
    const B = this.body, w = B.want, p = this.where(), dx = gx - p.x, dy = gy - p.y, err = wrap(Math.atan2(dy, dx) - p.yaw), dist = Math.hypot(dx, dy);
    if (dist < stop) return true;
    c.go = c.go || "face";
    if (c.go === "face") {
      if (Math.abs(err) < 0.3 || dist < stop + 0.3) { c.go = "walk"; }
      else if (["walk", "turnclip", "trot", "ctrot", "pace", "canter"].includes(B.mode)) { w.gait = "stand"; w.steer = null; }   // stop first
      else if (B.mode === "sharp" || ((B.mode === "stand" || B.mode === "sniff") && B.mt > 0.3)) { w.gait = "walk"; w.steer = Math.sign(err) * 1.15; }
      else { w.gait = "stand"; w.steer = null; }
      return false;
    }
    if (B.mode === "sharp" && Math.abs(err) < 0.3) w.steer = 0;
    w.gait = "walk"; w.steer = B.mode === "sharp" ? Math.sign(err) * Math.min(1.2, Math.abs(err)) : clamp(err, -0.35, 0.35);
    if (Math.abs(err) > 1.4 && B.mode !== "sharp") c.go = "face";
    return false;
  }
  // a learned turn on the spot by `angle` rad (run 12 sharp-turn walk brain); true when turned
  turnBy(c, angle) {
    const B = this.body, w = B.want, yaw = this.where().yaw;
    if (c.turned === undefined) {   // start only from a real stand (a sharp turn asked mid-walk tipped Bolt over)
      if (!((B.mode === "stand" || B.mode === "sniff") && B.mt > 0.3)) { w.gait = "stand"; w.steer = null; return false; }
      c.turned = 0; c.prev = yaw;
    }
    c.turned += wrap(yaw - c.prev); c.prev = yaw;
    const left = angle - c.turned;
    if (Math.abs(left) < 0.25 || Math.sign(left) !== Math.sign(angle)) return true;
    w.gait = "walk"; w.steer = Math.sign(angle) * 1.15;
    return false;
  }
  // one control step (1/50 s)
  step() {
    const b = this.bolt, B = this.body, dt = b.dt;
    this.t += dt;
    if (!this.cur) this.startNext();
    const c = this.cur;
    if (c) { c.st += dt; if (this.run(c, dt)) this.finish(); else if (c.st > (c.limit || 14)) this.finish(true); }
    const falls0 = B.falls;
    B.step(dt);
    for (const e of B.events.splice(0)) if (/got itself up/.test(e.text)) this.emit("up");
    if (B.falls > falls0) { this.stats.falls++; this.emit("fall"); if (this.cur) this.finish(true); this.queue = []; this.assist = null; }
    // drawn moves and the look
    this.drawn = this.drawn.filter((d) => this.t - d.t0 < d.secs);
    const L = this.look, u = (id) => { const d = this.drawn.find((x) => x.id === id); return d ? (this.t - d.t0) / d.secs : -1; };
    const ti = u("head_tilt"), no = u("nod"), ea = u("ears_down"), wa = u("tail_wag");
    L.tilt = ti >= 0 ? 0.42 * Math.sin(Math.PI * Math.min(1, ti * 1.6)) * (ti < 0.62 ? 1 : 1 - (ti - 0.62) / 0.38) : L.tilt * 0.9;
    L.nod = no >= 0 ? 0.28 * Math.sin(ti >= 0 ? 0 : no * Math.PI * 4) : 0;
    const sadness = ["sad", "worried", "sleepy"].includes(L.emotion) ? 1 : 0;
    L.ears += ((ea >= 0 ? 1 : sadness * 0.7) - L.ears) * 0.08;
    const wagWant = wa >= 0 ? 1 : { excited: 0.9, happy: 0.7, playful: 0.8, silly: 0.8, proud: 0.6, curious: 0.4, calm: 0.25, caring: 0.35, sad: 0.05, sleepy: 0.05, worried: 0.12 }[L.emotion] ?? 0.4;
    L.wag += (wagWant - L.wag) * 0.05;
  }
  // the moves: true when done
  run(c, dt) {
    const B = this.body, b = this.bolt, w = B.want, d = b.data;
    const posture = () => POSE_MODES.includes(B.mode);
    switch (c.id) {
      case "stand": w.gait = "stand"; w.steer = null; return posture() && B.mt > 0.6;
      case "get_up": w.gait = "stand"; w.steer = null; c.limit = 12; return posture() && B.mode !== "brake" && B.mt > 0.6;
      case "sit": c.limit = 10; w.gait = "sit"; w.steer = null; return B.mode === "sit" && this.gpu.finished() > 0.6;
      case "lie_down": {   // the lie clip starts standing: from a sit, stand up first
        c.limit = 16; w.steer = null;
        if (c.stage === 0 && (B.mode === "sit" || B.mode === "rise" || B.mode === "brake")) { w.gait = "stand"; return false; }
        c.stage = 1; w.gait = "lie"; return B.mode === "lie" && this.gpu.finished() > 0.4;
      }
      case "walk_forward": {
        c.limit = 8;
        if (c.stage === 0) { w.gait = "walk"; w.steer = this.homeSteer(0); if (["walk", "turnclip", "sharp"].includes(B.mode)) c.moved = (c.moved || 0) + dt; if (c.moved > 2.2 || c.st > 5) c.stage = 1; return false; }
        w.gait = "stand"; w.steer = null; return posture() && B.mode !== "brake";
      }
      case "come_closer": {   // to 0.8 m in front of the person, then stand
        c.limit = 14;
        if (c.stage === 0) { if (this.goTo(c, this.viewer, 0.85) || c.st > 11) c.stage = 1; return false; }
        w.gait = "stand"; w.steer = null; return posture() && B.mode !== "brake";
      }
      case "turn_left": case "turn_right": {   // a quarter turn on the spot
        c.limit = 8;
        if (c.stage === 0) { if (this.turnBy(c, c.id === "turn_left" ? Math.PI / 2 : -Math.PI / 2) || c.st > 7) c.stage = 1; return false; }
        w.gait = "stand"; w.steer = null; return posture() && B.mode !== "brake";
      }
      case "spin": {   // right round on the spot
        c.limit = 14;
        if (c.stage === 0) { if (this.turnBy(c, 2 * Math.PI - 0.25) || c.st > 11) c.stage = 1; return false; }
        w.gait = "stand"; w.steer = null; return posture() && B.mode !== "brake";
      }
      case "zoomies": {   // a trot out, then back home (facing it first)
        c.limit = 20;
        if (c.stage === 0) { w.gait = "trot"; w.steer = this.homeSteer(0); if ((["trot", "ctrot"].includes(B.mode) && B.mt > 1.2) || c.st > 4.5) c.stage = 1; return false; }
        if (c.stage === 1) { if (this.goTo(c, this.home, 0.6) || c.st > 17) c.stage = 2; return false; }
        w.gait = "stand"; w.steer = null; return posture() && B.mode !== "brake";
      }
      case "hop": case "twirl": {
        if (c.stage === 0) { if (!this.standReady() || !this.kick(c.id === "hop" ? ASSIST.hop : 1.8, c.id === "twirl" ? ASSIST.twirl : 0)) return false; c.stage = 1; c.ts = c.st; return false; }
        if (c.stage === 1) { if (!this.assist || this.assist.kind === "land") { c.stage = 2; c.ts = c.st; } else if (c.st - c.ts > 1.0) { this.assist = null; c.stage = 2; c.ts = c.st; } return false; }
        return c.st - c.ts > 0.45 && !this.assist;
      }
      case "happy_bounce": {
        if (c.stage === 0) { if (!this.standReady()) return false; c.stage = 1; c.n = 0; c.ts = c.st - 1; }
        if (c.stage === 1) {
          if (!this.assist && c.st - c.ts > 0.12) { if (c.n >= 3) { c.stage = 2; c.ts = c.st; return false; } if (this.kick(ASSIST.bounce, 0)) c.n++; else if (c.st > 4) c.n = 3; c.ts = c.st; }
          if (this.assist && this.assist.kind === "air" && c.st - c.ts > 0.7) { this.assist = null; c.ts = c.st; }
          return false;
        }
        return c.st - c.ts > 0.3;
      }
      case "wiggle": {
        if (c.stage === 0) { if (!this.standReady()) return false; c.stage = 1; c.ts = c.st; this.assist = { kind: "wiggle", t0: this.t }; }
        if (c.st - c.ts > 1.5) { this.assist = null; return true; }
        return false;
      }
      case "play_bow": {
        if (c.stage === 0) { if (!this.standReady()) return false; c.stage = 1; c.ts = c.st; this.assist = { kind: "bow", t0: this.t }; }
        if (c.st - c.ts > 1.9) { this.assist = null; return c.st - c.ts > 2.4; }
        return false;
      }
      case "dance": {   // a short routine without music: the moves of one phrase of Dance on a 112 bpm count
        if (c.stage === 0) { if (!this.standReady()) return false; c.stage = 1; c.ts = c.st; c.plan = choreograph(beatGrid(112, 0.3, 8.5), { seed: 3 }); c.k = 0; }
        while (c.k < c.plan.length && c.plan[c.k].t - (MOVE_LEAD[c.plan[c.k].move] || 0) <= c.st - c.ts) { this.beatMove(c.plan[c.k]); c.k++; }
        return c.k >= c.plan.length && c.st - c.ts > 8.8 && !this.assist;
      }
    }
    return true;
  }
  // straight on, unless Bolt is far from home: then curve back toward it
  homeSteer(bias) {
    const p = this.where(), dx = this.home[0] - p.x, dy = this.home[1] - p.y, far = Math.hypot(dx, dy);
    if (far < 1.8) return bias;
    return clamp(wrap(Math.atan2(dy, dx) - p.yaw), -0.35, 0.35);
  }
  // an assisted kick: straight up on the root joint (all of Bolt rises together, the legs keep the brain's motion)
  kick(up, spin = 0) {
    const d = this.bolt.data;
    if (!this.settled(0.3, up <= 1.3 ? 0.08 : 0.25, up <= 1.3 ? 0.25 : 0.15)) return false;
    this.assist = { kind: "air", t0: this.t, z0: d.qpos[2], spin: !!spin };
    d.qvel[2] += up; if (spin) d.qvel[5] += spin * (Math.random() < 0.5 ? -1 : 1);
    this.emit("kick", { up });
    return true;
  }
  // a beat of the dance: m = { t, move, beat } from choreograph(); learned moves change the body's wish, assisted ones kick
  beatMove(m) {
    const B = this.body;
    this.emit("beat", { move: m.move, beat: m.beat });
    if (m.move === "bounce") return this.kick(ASSIST.bounce * (m.accent ? 1.25 : 1), 0);
    if (m.move === "twirl") return this.kick(1.8, ASSIST.twirl);
    const upright = () => POSE_MODES.includes(B.mode) && B.mode !== "brake" && this.bolt.up() > 0.9 && (!this.assist || this.assist.kind === "land");
    if (m.move === "wiggle") { if (!upright()) return false; this.assist = { kind: "wiggle", t0: this.t, until: this.t + (m.len || 0.9) }; return true; }
    if (m.move === "bow") { if (!upright()) return false; this.assist = { kind: "bow", t0: this.t, until: this.t + (m.len || 1.2) }; return true; }
    if (m.move === "tilt") { this.drawn.push({ id: "head_tilt", t0: this.t, secs: 0.9 }); return true; }
    return false;
  }
  // ---- the guide (assisted moves only): forces and torques on Bolt's bodies for one physics step of h s
  guide(h) {
    const f = this.bolt.data.xfrc_applied;
    for (const i of this.dogBodies) f.fill(0, 6 * i, 6 * i + 6);
    const a = this.assist;
    if (!a) return;
    if (a.until && this.t > a.until) { this.assist = null; return; }
    const b = this.bolt, d = b.data, v = d.qvel, R = d.xmat, t = 9 * b.torso, j = 6 * b.torso;
    if (a.kind === "air") {   // a light lift while in the air, and the body kept level
      for (let k = 0; k < this.dogBodies.length; k++) f[6 * this.dogBodies[k] + 2] = this.dogMass[k] * 9.81 * ASSIST.float;
      f[j + 3] = ASSIST.level * R[t + 5] - ASSIST.levelD * v[3];
      f[j + 4] = -ASSIST.level * R[t + 2] - ASSIST.levelD * v[4];
      if (d.qvel[2] < 0 && d.qpos[2] < a.z0 + 0.02 && this.t - a.t0 > 0.1) { this.landedAt = this.t; this.assist = { kind: "land", t0: this.t, until: this.t + (a.spin ? 0.45 : 0.2) }; }
      return;
    }
    if (a.kind === "land") {   // the touchdown: keep level and stop the twist
      f[j + 3] = ASSIST.level * R[t + 5] - ASSIST.levelD * v[3];
      f[j + 4] = -ASSIST.level * R[t + 2] - ASSIST.levelD * v[4];
      f[j + 5] = -3 * v[5];
      return;
    }
    if (a.kind === "wiggle") {   // the twist, with the body held level so it does not tip
      f[j + 3] = ASSIST.level * R[t + 5] - ASSIST.levelD * v[3];
      f[j + 4] = -ASSIST.level * R[t + 2] - ASSIST.levelD * v[4];
      f[j + 5] = ASSIST.wiggle * Math.sin(2 * Math.PI * ASSIST.wiggleHz * (this.t - a.t0)) - 1.0 * v[5];
      return;
    }
    if (a.kind === "bow") {   // nose down (about the torso's own side axis), eased in, let go past bowMax
      const ramp = clamp((this.t - a.t0) / 0.4, 0, 1), p = this.pitch(), tq = ASSIST.bow * ramp * clamp((ASSIST.bowMax + p) / 0.1, 0, 1);
      // the torso's y axis in the world: column 1 of xmat; a positive torque about it pitches the nose down
      f[j + 3] = tq * R[t + 1]; f[j + 4] = tq * R[t + 4]; f[j + 5] = tq * R[t + 7];
    }
  }
}

// ------------------------------------------------------------------------------------------- the dance
// how early each dance move starts so that it shows on the beat (s): a kick shows within a control step; the
// twist and the bow build up (measured in companion_test.mjs body: the twist reaches 0.05 rad ~90 ms after it
// starts, the bow dips 0.05 rad ~370 ms after)
export const MOVE_LEAD = { bounce: 0.02, twirl: 0.02, wiggle: 0.09, bow: 0.36, tilt: 0 };
// beats on a fixed tempo: bpm, the first beat at t0 (s), until t1 (s)
export function beatGrid(bpm, t0, t1) { const out = [], p = 60 / bpm; for (let t = t0, k = 0; t <= t1 + 1e-9; t += p, k++) out.push({ t: +t.toFixed(4), beat: k }); return out; }
// the choreography: one move per beat in 8-beat phrases. Phrases alternate: bounces with a wiggle at the end,
// a play bow held two beats, a twirl on the last beat of every second phrase (and a beat of rest after it, to land). Every move is assisted except the
// head tilt (drawn): learned moves (sit, turns) take seconds, too slow for a single beat.
export function choreograph(beats, { seed = 1 } = {}) {
  let r = seed * 9301 + 49297; const R = () => ((r = (r * 1103515245 + 12345) >>> 0) / 4294967296);
  const P = [
    ["bounce", "rest", "bounce", "rest", "bounce", "bounce", "wiggle", "rest"],
    ["tilt", "rest", "bounce", "rest", "bow", "rest", "rest", "twirl"],
    ["bounce", "bounce", "wiggle", "rest", "bounce", "rest", "bounce", "bounce"],
    ["wiggle", "rest", "tilt", "rest", "bounce", "rest", "rest", "twirl"],
  ];
  const out = [];
  let phrase = P[0];
  for (let i = 0; i < beats.length; i++) {
    const k = i % 8;
    if (k === 0) phrase = P[Math.floor(i / 8) % P.length === 0 ? 0 : Math.floor(R() * P.length)];
    const move = phrase[k];
    if (move === "rest") continue;
    if (out.length && out[out.length - 1].move === "twirl" && out[out.length - 1].beat === (beats[i - 1] && (beats[i - 1].beat ?? i - 1))) continue;   // a beat to land after a twirl
    const p = i + 1 < beats.length ? beats[i + 1].t - beats[i].t : 0.5;
    out.push({ t: beats[i].t, beat: beats[i].beat ?? i, move, accent: k === 0 || k === 4, len: move === "bow" ? 2 * p : move === "wiggle" ? 1.6 * p : p });
  }
  return out;
}

// Beats in decoded audio (one channel, sampleRate): an onset envelope (energy rise in 23 ms hops, low end
// weighted by a one-pole low-pass), the tempo from the envelope's autocorrelation (70-180 bpm), then the beat
// phase that lines up with the most onset energy. Returns { bpm, beats: [s...], confidence }.
const OFFSET = +(globalThis.__BEAT_OFFSET ?? -0.4);   // hops between a sound's start and its onset peak (measured in companion_test.mjs beats)
export function beatsFromSamples(x, sampleRate, { minBpm = 70, maxBpm = 180 } = {}) {
  const hop = Math.round(sampleRate * 0.0232), n = Math.floor(x.length / hop);
  if (n < 40) return { bpm: 0, beats: [], confidence: 0 };
  const env = new Float32Array(n), lowA = Math.exp(-2 * Math.PI * 180 / sampleRate);
  let lp = 0, prev = 0;
  for (let i = 0; i < n; i++) {
    let e = 0, el = 0;
    for (let k = i * hop, end = k + hop; k < end; k++) { const s = x[k]; lp = lowA * lp + (1 - lowA) * s; e += s * s; el += lp * lp; }
    const v = Math.log1p(1000 * (e + 4 * el) / hop);
    env[i] = Math.max(0, v - prev); prev = v;
  }
  // remove the slow average so loud passages do not dominate
  const fr = 1 / (hop / sampleRate), win = Math.round(fr * 0.5), on = new Float32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) { acc += env[i]; if (i >= win) acc -= env[i - win]; on[i] = Math.max(0, env[i] - acc / Math.min(i + 1, win)); }
  const lagMin = Math.floor(fr * 60 / maxBpm), lagMax = Math.ceil(fr * 60 / minBpm);
  let mu = 0; for (let i = 0; i < n; i++) mu += on[i]; mu /= n;   // the onsets are never negative: compare against their mean
  let best = 0, bestLag = lagMin, mean = 0;
  const ac = [];
  for (let L = lagMin; L <= lagMax; L++) {
    let s = 0; for (let i = L; i < n; i++) s += on[i] * on[i - L];
    s = s / (n - L) - mu * mu;
    // a gentle preference for tempos near 120 bpm (resolves half/double tempo ties)
    const bpm = 60 * fr / L, wgt = Math.exp(-0.5 * (Math.log2(bpm / 120) / 0.9) ** 2);
    ac.push(s); mean += s;
    if (s * wgt > best) { best = s * wgt; bestLag = L; }
  }
  mean /= ac.length;
  let ac0 = 0; for (let i = 0; i < n; i++) ac0 += on[i] * on[i]; ac0 = ac0 / n - mu * mu;
  // half the lag (double the tempo) when that beat is nearly as strong: a kick on every beat beats one on every other
  const at = (L) => ac[L - lagMin] ?? 0;
  if (bestLag / 2 >= lagMin) { const h = Math.round(bestLag / 2); if (Math.max(at(h), at(h - 1), at(h + 1)) > 0.5 * at(bestLag)) bestLag = [h - 1, h, h + 1].reduce((p, q) => (at(q) > at(p) ? q : p)); }
  // refine the period with a parabola through the peak
  const k = bestLag - lagMin, y0 = ac[k - 1] ?? ac[k], y1 = ac[k], y2 = ac[k + 1] ?? ac[k];
  const den = y0 - 2 * y1 + y2;
  let period = bestLag + (den < 0 ? 0.5 * (y0 - y2) / den : 0);
  // phase: the offset whose comb catches the most onset energy
  let bestPh = 0, bestS = -1;
  for (let ph = 0; ph < period; ph += 0.25) {
    let s = 0; for (let t = ph; t < n; t += period) { const i = Math.round(t); s += on[i] || 0; }
    if (s > bestS) { bestS = s; bestPh = ph; }
  }
  // then lock period and phase to the onset peaks themselves: for each beat on the grid, the strongest onset
  // within a quarter beat (its centre of mass), and a least-squares line through them; twice, so drift goes away
  for (let pass = 0; pass < 2; pass++) {
    const pts = [];
    for (let j = 0, t = bestPh; t < n; t += period, j++) {
      const lo = Math.max(0, Math.round(t - period / 4)), hi = Math.min(n - 1, Math.round(t + period / 4));
      let bi = -1, bv = 0; for (let i = lo; i <= hi; i++) if (on[i] > bv) { bv = on[i]; bi = i; }
      if (bi < 0 || bv < 0.2 * Math.sqrt(Math.max(ac0, 1e-9))) continue;
      const a0 = on[bi - 1] || 0, a2 = on[bi + 1] || 0;
      pts.push([j, bi + (a0 + a2 > 0 ? 0.5 * (a2 - a0) / (2 * bv - a0 - a2 || 1) : 0)]);
    }
    if (pts.length < 6) break;
    let sx = 0, sy = 0, sxx = 0, sxy = 0; for (const [x, y] of pts) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
    const m = pts.length, slope = (m * sxy - sx * sy) / (m * sxx - sx * sx || 1);
    if (!(slope > 0) || Math.abs(slope - period) > 0.1 * period) break;
    period = slope; bestPh = (sy - slope * sx) / m;
    while (bestPh - period >= 0) bestPh -= period;
    if (bestPh < 0) bestPh += period;
  }
  // the onset envelope rises a fraction of a hop after the sound starts: centre the beat on the hop
  // no beats before the music starts: begin at the first clear onset
  let first = 0, lastOn = n - 1; const thr = 0.3 * on.reduce((a, b) => Math.max(a, b), 0); while (first < n && on[first] < thr) first++; while (lastOn > 0 && on[lastOn] < thr) lastOn--;
  const beats = []; for (let t = bestPh; t < n; t += period) if (t > first - period / 4 && t < lastOn + period / 4) beats.push(+(((t - OFFSET) * hop) / sampleRate).toFixed(4));
  const bpm = 60 * fr / period;
  return { bpm: +bpm.toFixed(1), beats, confidence: +clamp((at(bestLag) || ac[k]) / (ac0 || 1), 0, 1).toFixed(2) };
}

// Live beats (the microphone): feed it energy readings as they come (time s, energy >= 0); it keeps an onset
// history and predicts the next beat. next(now) returns the next predicted beat time after now, or null.
export class BeatTracker {
  constructor() { this.onsets = []; this.prev = 0; this.avg = 0; this.lastOnset = -1; this.period = 0; this.phase = 0; }
  push(t, energy) {
    const v = Math.log1p(1000 * energy), rise = Math.max(0, v - this.prev);
    this.prev = v; this.avg += (rise - this.avg) * 0.05;
    if (rise > Math.max(0.25, 2.2 * this.avg) && t - this.lastOnset > 0.28) {
      this.lastOnset = t; this.onsets.push(t);
      if (this.onsets.length > 24) this.onsets.shift();
      this.fit();
      return true;
    }
    return false;
  }
  // tempo: the most common gap between onsets (folded into 0.33-0.86 s); phase: the latest onset on the grid
  fit() {
    const o = this.onsets; if (o.length < 4) return;
    const votes = new Map();
    for (let i = 1; i < o.length; i++) for (let j = Math.max(0, i - 4); j < i; j++) {
      let g = o[i] - o[j]; while (g > 0.86) g /= 2; while (g < 0.33) g *= 2;
      const key = Math.round(g * 50); votes.set(key, (votes.get(key) || 0) + 1);
    }
    let bk = 0, bv = 0; for (const [k, v] of votes) if (v > bv) { bv = v; bk = k; }
    const near = []; for (let i = 1; i < o.length; i++) { let g = o[i] - o[i - 1]; while (g > 0.86) g /= 2; while (g < 0.33) g *= 2; if (Math.abs(g - bk / 50) < 0.03) near.push(g); }
    this.period = near.length ? near.reduce((a, b) => a + b, 0) / near.length : bk / 50;
    this.phase = o[o.length - 1];
  }
  get bpm() { return this.period ? 60 / this.period : 0; }
  next(now) { if (!this.period) return null; const k = Math.ceil((now - this.phase) / this.period + 1e-6); return this.phase + k * this.period; }
}

// The built-in tune: 4 bars of 4/4 at 112 bpm, looped; drums, bass and a little lead. Notes as data, so the page
// plays them with WebAudio and the beats are known exactly (no detection needed).
export const TUNE = (() => {
  const bpm = 112, bars = 8, beat = 60 / bpm, notes = [];
  const lead = [72, 0, 76, 79, 77, 0, 76, 74, 72, 0, 76, 79, 81, 79, 77, 76, 74, 0, 77, 81, 79, 0, 77, 76, 74, 76, 77, 79, 76, 0, 72, 0];
  const bassRoots = [48, 48, 53, 53, 50, 50, 55, 55];
  for (let b = 0; b < bars * 4; b++) {
    const t = b * beat, bar = Math.floor(b / 4);
    notes.push({ t, kind: "kick" });
    if (b % 2 === 1) notes.push({ t, kind: "snare" });
    notes.push({ t: t + beat / 2, kind: "hat" });
    const root = bassRoots[bar % bassRoots.length];
    notes.push({ t, kind: "bass", midi: root, dur: beat * 0.45 }, { t: t + beat / 2, kind: "bass", midi: root + (b % 4 === 3 ? 7 : 12), dur: beat * 0.35 });
    const m = lead[b % lead.length]; if (m && bar >= 2) notes.push({ t, kind: "lead", midi: m, dur: beat * 0.8 });
  }
  return { bpm, beat, length: bars * 4 * beat, notes, beats: Array.from({ length: bars * 4 }, (_, k) => k * beat) };
})();
