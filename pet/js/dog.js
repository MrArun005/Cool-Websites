/* dog.js: the robot dog's body and its tricks.
 *
 * DogSim is a physics simulation, stepped at a fixed rate with update(dt):
 *   stance  feet planted, body held up by spring-damper "muscles"; feet take
 *           trot steps when the body moves away from them
 *   push    legs extend and accelerate the body for takeoff
 *   flight  ballistic: gravity pulls, angular momentum is conserved, and
 *           tucking the legs changes the inertia (so the spin speeds up)
 *   body    the body itself touches the ground (rolling over, playing dead)
 *
 * Tricks are async functions that set targets and await sim time, so they run
 * the same in a browser and in a Node test.
 */
(function (root) {
  'use strict';
  const P = typeof module === 'object' && module.exports ? require('./physics.js') : root.PetPhysics;
  const { G, TAU, DOG, LEGS, TROT_PAIRS } = P;

  class Cancelled extends Error {}

  const STAND = { height: DOG.standHeight, pitch: 0, roll: 0, shift: 0 };
  const STAGE_RADIUS = 2.6; // keep the dog on the floor we draw

  class DogSim {
    constructor() {
      this.time = 0;
      this.pos = [0, DOG.standHeight, 0];
      this.vel = [0, 0, 0];
      this.yaw = 0; this.pitch = 0; this.roll = 0;
      this.yawRate = 0; this.pitchRate = 0; this.rollRate = 0;
      this.mode = 'stance';
      this.posture = 'stand';
      this.target = { height: DOG.standHeight, pitch: 0, roll: 0, yaw: 0, x: 0, z: 0, freq: 2.2, yawFreq: 1.2 };
      this.autoStep = true;
      this.walking = null;
      this.push = null;
      this.flight = null;
      this.tuck = 0;
      this.feet = LEGS.map((_, i) => ({ pos: this.nominalFoot(i), mode: 'planted', swing: null, local: null, localTarget: null }));
      this.legs = [];
      this.waiters = [];
      this.trace = [];
      this.lastJump = null;
      this.spinTravelled = 0;     // total pitch rotation, for tests and the HUD
      this.viewer = [1.5, 1.5];   // where the human is (the camera), for "come here"
      this.expression = 'normal'; // normal | happy | sleep | dead | dizzy | sad
      this.excitement = 0.4;      // drives the tail wag
      this.speaking = false;
      this.head = { pitch: 0, tilt: 0, pitchV: 0, tiltV: 0, pitchT: 0, tiltT: 0 };
      this.updateLegs();
    }

    // ---------- geometry helpers ----------
    heading(extraYaw = 0) { return P.rotY([1, 0, 0], this.yaw + extraYaw); }
    right(extraYaw = 0) { return P.rotY([0, 0, 1], this.yaw + extraYaw); }
    toWorld(local) { return P.bodyToWorld(local, this.pos, this.yaw, this.pitch, this.roll); }
    toBody(world) { return P.worldToBody(world, this.pos, this.yaw, this.pitch, this.roll); }

    // Where a foot "wants" to be: under its hip, a little ahead if moving.
    nominalFoot(i, lead = 0) {
      const h = LEGS[i].hip, r = P.rotY([h[0], 0, h[2]], this.yaw);
      return [this.pos[0] + r[0] + this.vel[0] * lead, 0, this.pos[2] + r[2] + this.vel[2] * lead];
    }
    feetCentre() {
      const c = [0, 0, 0];
      this.feet.forEach(f => { c[0] += f.pos[0] / 4; c[2] += f.pos[2] / 4; });
      return c;
    }

    // ---------- time & waiting ----------
    until(pred) { return new Promise((resolve, reject) => this.waiters.push({ pred, resolve, reject })); }
    wait(seconds) { const end = this.time + seconds; return this.until(() => this.time >= end); }
    cancel() {
      const ws = this.waiters; this.waiters = [];
      ws.forEach(w => w.reject(new Cancelled()));
      this.walking = null;
    }
    checkWaiters() {
      if (!this.waiters.length) return;
      const ready = this.waiters.filter(w => w.pred());
      if (!ready.length) return;
      this.waiters = this.waiters.filter(w => !ready.includes(w));
      ready.forEach(w => w.resolve());
    }

    // ---------- simulation step ----------
    update(dt) {
      this.time += dt;
      if (this.mode === 'stance') this.stepStance(dt);
      else if (this.mode === 'push') this.stepPush(dt);
      else if (this.mode === 'flight') this.stepFlight(dt);
      else if (this.mode === 'body') this.stepBody(dt);
      this.stepFeet(dt);
      this.stepHead(dt);
      this.updateLegs();
      this.checkWaiters();
    }

    stepStance(dt) {
      const t = this.target;
      if (this.walking) {
        const w = this.walking, dx = w.goal[0] - t.x, dz = w.goal[2] - t.z, d = Math.hypot(dx, dz);
        const stepLen = Math.min(d, w.speed * dt);
        if (d > 1e-4) { t.x += dx / d * stepLen; t.z += dz / d * stepLen; }
        if (d > 0.15) t.yaw = this.yaw + P.wrapAngle(Math.atan2(-dz, dx) - this.yaw);
        if (d < 1e-3) this.walking = null;
      }
      [this.pos[1], this.vel[1]] = P.spring(this.pos[1], this.vel[1], t.height, t.freq, 0.7, dt);
      [this.pos[0], this.vel[0]] = P.spring(this.pos[0], this.vel[0], t.x, 1.6, 1.0, dt);
      [this.pos[2], this.vel[2]] = P.spring(this.pos[2], this.vel[2], t.z, 1.6, 1.0, dt);
      [this.pitch, this.pitchRate] = P.spring(this.pitch, this.pitchRate, t.pitch, 2.0, 0.8, dt);
      [this.roll, this.rollRate] = P.spring(this.roll, this.rollRate, t.roll, 2.0, 0.8, dt);
      [this.yaw, this.yawRate] = P.spring(this.yaw, this.yawRate, t.yaw, t.yawFreq, 1.0, dt);
      if (this.pos[1] < 0.1) { this.pos[1] = 0.1; this.vel[1] = Math.max(0, this.vel[1]); }
      if (this.autoStep) this.maybeStep();
    }

    // Trot: when the body has drifted from the feet, step a diagonal pair.
    maybeStep() {
      if (this.feet.some(f => f.mode !== 'planted')) return;
      let best = null, bestErr = 0.05;
      for (const pair of TROT_PAIRS) {
        const err = Math.max(...pair.map(i => {
          const n = this.nominalFoot(i, 0.25), f = this.feet[i].pos;
          return Math.hypot(n[0] - f[0], n[2] - f[2]);
        }));
        if (err > bestErr) { best = pair; bestErr = err; }
      }
      if (best) best.forEach(i => this.swingFoot(i, this.nominalFoot(i, 0.25), 0.2));
    }

    swingFoot(i, to, duration, height = 0.06) {
      const f = this.feet[i];
      f.mode = 'swing';
      f.swing = { from: f.pos.slice(), to: [to[0], 0, to[2]], t: 0, T: duration, h: height };
      f.localTarget = null;
    }

    attachFoot(i, localTarget) {
      const f = this.feet[i];
      if (f.mode !== 'body') { f.local = this.toBody(f.pos); f.mode = 'body'; f.swing = null; }
      f.localTarget = localTarget;
    }

    stepPush(dt) {
      const p = this.push;
      p.t += dt;
      const ramp = Math.min(1, p.t / p.T);
      for (let k = 0; k < 3; k++) this.vel[k] += p.accel[k] * dt;
      this.pos = P.add(this.pos, P.scale(this.vel, dt));
      this.pitchRate = p.spin0 * ramp;
      this.pitch += this.pitchRate * dt;
      this.spinTravelled += this.pitchRate * dt;
      if (p.t >= p.T || this.pos[1] >= DOG.takeoffHeight) this.takeoff();
    }

    takeoff() {
      const plan = this.push.plan;
      // Re-plan with the real state at the moment the feet leave the ground.
      const T = P.flightTime(this.vel[1], this.pos[1], DOG.takeoffHeight);
      const remaining = plan.wantAngle - this.pitch;  // aim to land level
      const L = P.clamp(remaining / P.spinIntegral(T, plan.tuck), -DOG.maxSpinMomentum, DOG.maxSpinMomentum);
      const yawLeft = this.push.turn - (this.yaw - this.push.startYaw);
      this.flight = { t: 0, T, L, tuck: plan.tuck, yawRate: yawLeft / T, startLocal: this.feet.map(f => this.toBody(f.pos)) };
      this.feet.forEach((f, i) => { f.mode = 'body'; f.swing = null; f.localTarget = null; f.local = this.flight.startLocal[i]; });
      this.trace = [this.pos.slice()];
      this.push = null;
      this.mode = 'flight';
    }

    stepFlight(dt) {
      const f = this.flight;
      f.t += dt;
      this.tuck = f.tuck * P.tuckProfile(Math.min(1, f.t / f.T));
      this.pitchRate = f.L / P.pitchInertia(this.tuck);  // L is conserved; I changes
      this.pitch += this.pitchRate * dt;
      this.spinTravelled += this.pitchRate * dt;
      this.yawRate = f.yawRate;
      this.yaw += f.yawRate * dt;
      this.vel[1] -= G * dt;
      this.pos = P.add(this.pos, P.scale(this.vel, dt));
      const blend = Math.min(1, f.t / 0.08);
      this.feet.forEach((foot, i) => { foot.local = P.lerp3(f.startLocal[i], P.legPose(i, this.tuck), blend); });
      this.trace.push(this.pos.slice());
      if (this.vel[1] < 0 && this.pos[1] <= DOG.takeoffHeight) this.land();
    }

    land() {
      const residual = P.wrapAngle(this.pitch);
      const crashed = Math.abs(residual) > 0.7;
      this.pitch = residual;
      this.roll = P.wrapAngle(this.roll);
      this.pitchRate = crashed ? this.pitchRate * 0.2 : 0;
      this.yawRate = 0;
      this.mode = 'stance';
      this.tuck = 0;
      const t = this.target;
      t.x = this.pos[0] + this.vel[0] * 0.12;
      t.z = this.pos[2] + this.vel[2] * 0.12;
      t.yaw = this.yaw;
      t.pitch = 0; t.roll = 0; t.height = DOG.standHeight; t.freq = 3.0; // stiff legs to absorb the landing
      this.vel[0] *= 0.5; this.vel[2] *= 0.5;
      this.feet.forEach((f, i) => { f.mode = 'planted'; f.pos = this.nominalFoot(i); f.local = null; f.localTarget = null; });
      if (this.lastJump) Object.assign(this.lastJump, { landed: !crashed, landingErrorDeg: residual * 180 / Math.PI, impactSpeed: -this.vel[1] });
      if (crashed) this.expression = 'dizzy';
      this.flight = null;
    }

    stepBody(dt) {
      const before = this.roll;
      [this.roll, this.rollRate] = P.spring(this.roll, this.rollRate, this.target.roll, 0.8, 0.95, dt);
      [this.pitch, this.pitchRate] = P.spring(this.pitch, this.pitchRate, 0, 2, 1, dt);
      const support = P.rollSupport(this.roll);
      // Rolls without slipping: the body travels sideways by radius × angle.
      const r = this.right(), travel = support * (this.roll - before);
      this.pos[0] += r[0] * travel; this.pos[2] += r[2] * travel;
      this.pos[1] += (support - this.pos[1]) * (1 - Math.exp(-dt * 10));
      this.vel = [0, 0, 0];
      this.target.x = this.pos[0]; this.target.z = this.pos[2];
    }

    stepFeet(dt) {
      const k = 1 - Math.exp(-dt * 12);
      for (const f of this.feet) {
        if (f.mode === 'swing') {
          const s = f.swing; s.t += dt;
          const u = Math.min(1, s.t / s.T), e = P.smooth(u);
          f.pos = P.lerp3(s.from, s.to, e);
          f.pos[1] += 4 * s.h * u * (1 - u);
          if (u >= 1) { f.mode = 'planted'; f.pos = s.to.slice(); f.swing = null; }
        } else if (f.mode === 'body') {
          if (f.localTarget) f.local = P.lerp3(f.local, f.localTarget, k);
          f.pos = this.toWorld(f.local);
          if (this.mode !== 'flight') f.pos[1] = Math.max(0.02, f.pos[1]);
        }
      }
    }

    stepHead(dt) {
      const h = this.head;
      const bob = this.speaking ? Math.sin(this.time * 14) * 0.08 : 0;
      [h.pitch, h.pitchV] = P.spring(h.pitch, h.pitchV, h.pitchT + bob, 2.5, 0.6, dt);
      [h.tilt, h.tiltV] = P.spring(h.tilt, h.tiltV, h.tiltT, 2.5, 0.6, dt);
    }

    updateLegs() {
      const fwd = this.heading();
      this.legs = LEGS.map((leg, i) => {
        const hip = this.toWorld(leg.hip);
        // knees: front ones point backward like a dog's elbow, rear ones forward
        const hint = P.bodyToWorld([leg.front ? -1 : 1, 0.15, 0], [0, 0, 0], this.yaw, this.pitch, this.roll);
        const s = P.solveLeg(hip, this.feet[i].pos, P.len(hint) ? hint : fwd);
        return { hip, knee: s.knee, foot: s.foot };
      });
    }

    // ---------- building blocks for tricks ----------
    setPose({ height = STAND.height, pitch = 0, roll = 0, shift = 0, freq = 2.2 } = {}) {
      const c = this.feetCentre(), h = this.heading();
      Object.assign(this.target, { height, pitch, roll, freq, x: c[0] + h[0] * shift, z: c[2] + h[2] * shift });
    }

    async settle(seconds = 0.5) { await this.wait(seconds); }

    async ensureStanding() {
      if (this.mode === 'push' || this.mode === 'flight') await this.until(() => this.mode === 'stance');
      if (this.mode === 'body') await this.getUp();
      this.releaseFeet();
      if (this.posture !== 'stand') {
        this.posture = 'stand';
        this.autoStep = true;
        this.setPose(STAND);
        await this.wait(0.6);
      }
      this.autoStep = true;
      if (this.expression === 'sleep' || this.expression === 'dead') this.expression = 'normal';
    }

    // Put any lifted paws back on the ground.
    releaseFeet() {
      this.feet.forEach((f, i) => {
        if (f.mode === 'body') this.swingFoot(i, f.home || this.nominalFoot(i), 0.25);
        f.home = null;
      });
    }

    async getUp() {
      if (this.mode === 'body') {
        this.target.roll = TAU * Math.round(this.roll / TAU);
        await this.until(() => Math.abs(this.roll - this.target.roll) < 0.03 && Math.abs(this.rollRate) < 0.2);
        this.roll = 0; this.target.roll = 0; this.rollRate = 0;
        this.mode = 'stance';
        this.target.height = STAND.height;
        this.feet.forEach((f, i) => this.swingFoot(i, this.nominalFoot(i), 0.3, 0.03));
        await this.wait(0.5);
      }
      this.posture = 'stand';
      this.setPose(STAND);
    }

    async jump({ rise = 0.15, flips = 0, forward = 0, side = 0, turn = 0, tuck = 0.35, name = 'jump' } = {}) {
      await this.ensureStanding();
      this.autoStep = false;
      // 1. Crouch: bend the legs to have room to push.
      this.setPose({ height: DOG.crouchHeight, pitch: flips > 0 ? -0.12 : flips < 0 ? 0.08 : 0 });
      this.excitement = 1;
      await this.wait(0.4);
      // 2. Plan the flight and the push that produces it.
      const plan = P.planJump({ rise, rotations: flips, tuck, forward: Math.hypot(forward, side) });
      const dir = P.add(P.scale(this.heading(turn / 2), forward), P.scale(this.right(turn / 2), side));
      const v0 = [dir[0] / plan.T, plan.vy, dir[2] / plan.T];
      const pushDist = Math.max(0.08, DOG.takeoffHeight - this.pos[1]);
      const tp = 2 * pushDist / plan.vy;
      this.push = {
        t: 0, T: tp, plan, turn, startYaw: this.yaw,
        accel: v0.map((v, k) => (v - this.vel[k]) / tp),
        spin0: plan.L / P.pitchInertia(0),
      };
      this.lastJump = { name, ...plan, pushTime: tp, bodyWeights: plan.legForce / (DOG.mass * G) };
      this.mode = 'push';
      await this.until(() => this.mode === 'stance');
      await this.wait(0.4);
      this.target.freq = 2.2;
      this.posture = 'stand';
      this.autoStep = true;
      this.setPose(STAND);
      const result = this.lastJump;
      if (!result.landed) {
        await this.wait(0.8);
        this.expression = 'normal';
      }
      this.excitement = 0.5;
      return result;
    }

    async walkTo(goal, speed = 0.55) {
      await this.ensureStanding();
      const d = Math.hypot(goal[0], goal[2]);
      if (d > STAGE_RADIUS) goal = [goal[0] * STAGE_RADIUS / d, 0, goal[2] * STAGE_RADIUS / d];
      this.walking = { goal, speed };
      this.target.yawFreq = 1.2;
      await this.until(() => !this.walking && Math.hypot(this.pos[0] - goal[0], this.pos[2] - goal[2]) < 0.06);
    }

    async turn(angle) {
      await this.ensureStanding();
      this.target.yaw = this.yaw + angle;
      this.target.yawFreq = 0.9;
      await this.until(() => Math.abs(this.yaw - this.target.yaw) < 0.04 && Math.abs(this.yawRate) < 0.3);
      this.target.yawFreq = 1.2;
    }

    async lieDown() {
      if (this.posture === 'lie' && this.mode === 'stance') return;
      await this.ensureStanding();
      this.autoStep = false;
      this.posture = 'lie';
      const h = this.heading();
      [0, 1].forEach(i => { const n = this.nominalFoot(i); this.swingFoot(i, [n[0] + h[0] * 0.15, 0, n[2] + h[2] * 0.15], 0.3); });
      await this.wait(0.3);
      this.setPose({ height: 0.13, shift: 0.03, freq: 1.6 });
      await this.wait(0.9);
    }

    async rollOnGround(targetRoll, legs = 'tuck') {
      if (this.posture !== 'lie' || this.mode !== 'stance') await this.lieDown();
      this.mode = 'body';
      this.feet.forEach((f, i) => {
        const hip = LEGS[i].hip;
        this.attachFoot(i, legs === 'stiff'
          ? [hip[0] * 1.15, hip[1] - 0.3, hip[2] * 2.2]
          : P.legPose(i, 0.9));
      });
      this.target.roll = targetRoll;
      await this.until(() => Math.abs(this.roll - targetRoll) < 0.04 && Math.abs(this.rollRate) < 0.3);
    }

    async liftPaw(i, local, seconds) {
      const f = this.feet[i];
      f.home = f.pos.slice();
      this.attachFoot(i, local);
      await this.wait(seconds);
    }
  }

  // ---------- the trick library ----------
  const hipOf = i => LEGS[i].hip;

  const Tricks = {
    async stand(d) { await d.ensureStanding(); d.expression = 'normal'; },

    async sit(d) {
      await d.ensureStanding();
      d.autoStep = false;
      d.posture = 'sit';
      d.setPose({ height: 0.24, pitch: 0.55, shift: -0.08, freq: 1.8 });
      await d.wait(0.9);
    },

    async lie(d) { await d.lieDown(); },

    async beg(d) {
      if (d.posture !== 'sit') await Tricks.sit(d);
      d.posture = 'beg';
      d.setPose({ height: 0.27, pitch: 0.95, shift: -0.1, freq: 1.6 });
      d.excitement = 1;
      await d.wait(0.4);
      await Promise.all([0, 1].map(i => d.liftPaw(i, [hipOf(i)[0] + 0.06, hipOf(i)[1] - 0.13, hipOf(i)[2] * 0.7], 1.6)));
      d.releaseFeet();
      d.setPose({ height: 0.24, pitch: 0.55, shift: -0.08 });
      d.posture = 'sit';
      await d.wait(0.5);
    },

    async shake(d) {
      if (d.posture !== 'sit') await Tricks.sit(d);
      const h = hipOf(1);
      d.feet[1].home = d.feet[1].pos.slice();
      for (let k = 0; k < 3; k++) {
        d.attachFoot(1, [h[0] + 0.2, h[1] - 0.1, h[2]]);
        await d.wait(0.25);
        d.attachFoot(1, [h[0] + 0.2, h[1] - 0.17, h[2]]);
        await d.wait(0.25);
      }
      d.releaseFeet();
      await d.wait(0.35);
    },

    async bow(d) {
      await d.ensureStanding();
      d.autoStep = false;
      d.posture = 'bow';
      d.excitement = 1;
      d.setPose({ height: 0.26, pitch: -0.42, shift: 0.05 });
      await d.wait(1.4);
      await d.ensureStanding();
    },

    async jump(d) { return d.jump({ rise: 0.22, tuck: 0.4, name: 'jump' }); },

    async backflip(d, n = 1) {
      d.head.pitchT = 0.3;
      const r = await d.jump({ rise: n > 1 ? 0.7 : 0.5, flips: n, tuck: 1, name: n > 1 ? `backflip ×${n}` : 'backflip' });
      d.head.pitchT = 0;
      return r;
    },

    async frontflip(d, n = 1) {
      const r = await d.jump({ rise: n > 1 ? 0.7 : 0.5, flips: -n, forward: 0.5, tuck: 1, name: n > 1 ? `somersault ×${n}` : 'somersault' });
      return r;
    },

    async jumpAround(d) {
      for (let k = 0; k < 5; k++) {
        // Hop in a random direction, curving back toward the middle of the stage.
        let turn = (Math.random() < 0.5 ? -1 : 1) * (0.5 + Math.random() * 1.2);
        const home = Math.atan2(d.pos[2], -d.pos[0]);
        if (Math.hypot(d.pos[0], d.pos[2]) > 1.2) turn = P.wrapAngle(home - d.yaw);
        await d.jump({ rise: 0.12 + Math.random() * 0.08, forward: 0.3, turn, tuck: 0.3, name: 'hop' });
      }
    },

    async rollOver(d) {
      await d.rollOnGround(d.roll + TAU);
      d.roll -= TAU; d.target.roll -= TAU;
      await d.getUp();
    },

    async playDead(d) {
      d.expression = 'normal';
      d.head.tiltT = 0;
      await d.rollOnGround(-Math.PI / 2, 'stiff');  // flop onto its left side, belly to the camera
      d.expression = 'dead';
      d.excitement = 0;
      await d.wait(2.4);
      d.expression = 'happy';
      d.excitement = 1;
      await d.getUp();
    },

    async spin(d) {
      d.excitement = 1;
      await d.turn(TAU * (Math.random() < 0.5 ? 1 : -1));
      d.yaw = P.wrapAngle(d.yaw); d.target.yaw = d.yaw;
    },

    async dance(d) {
      await d.ensureStanding();
      d.excitement = 1;
      d.expression = 'happy';
      for (let k = 0; k < 4; k++) {
        const s = k % 2 ? 1 : -1;
        d.setPose({ height: 0.3, roll: 0.22 * s, pitch: 0.12 * s });
        d.target.yaw = d.yaw + 0.25 * s;
        await d.wait(0.35);
      }
      await d.jump({ rise: 0.14, turn: Math.PI, tuck: 0.3, name: 'dance hop' });
      await d.jump({ rise: 0.14, turn: Math.PI, tuck: 0.3, name: 'dance hop' });
    },

    async wag(d) {
      await d.ensureStanding();
      d.excitement = 1.4;
      d.expression = 'happy';
      for (let k = 0; k < 4; k++) {
        d.setPose({ roll: 0.07 * (k % 2 ? 1 : -1) });
        await d.wait(0.22);
      }
      d.setPose(STAND);
      await d.wait(0.3);
    },

    async come(d) {
      const v = d.viewer, dx = v[0] - d.pos[0], dz = v[1] - d.pos[2], dist = Math.hypot(dx, dz);
      const stop = Math.max(0, dist - 1.0);
      await d.walkTo([d.pos[0] + dx / dist * stop, 0, d.pos[2] + dz / dist * stop]);
      await d.turn(P.wrapAngle(Math.atan2(-dz, dx) - d.yaw));
    },

    async walk(d, dir = 'forward') {
      const turnBy = { forward: 0, left: Math.PI / 2, right: -Math.PI / 2, back: Math.PI }[dir] || 0;
      if (turnBy) await d.turn(turnBy);
      const h = d.heading();
      await d.walkTo([d.pos[0] + h[0] * 0.8, 0, d.pos[2] + h[2] * 0.8]);
    },

    async turnAround(d) { await d.turn(Math.PI); },

    async nod(d) {
      for (let k = 0; k < 2; k++) { d.head.pitchT = -0.3; await d.wait(0.18); d.head.pitchT = 0.15; await d.wait(0.18); }
      d.head.pitchT = 0;
    },

    async confused(d) {
      d.head.tiltT = 0.4; d.excitement = 0.1;
      await d.wait(1.2);
      d.head.tiltT = 0; d.excitement = 0.4;
    },

    async sad(d) {
      d.expression = 'sad'; d.excitement = 0; d.head.pitchT = -0.25;
      await d.wait(1.5);
      d.head.pitchT = 0; d.expression = 'normal'; d.excitement = 0.3;
    },

    async happy(d) { d.expression = 'happy'; await Tricks.wag(d); },

    async rest(d, seconds = 3) {
      await d.lieDown();
      d.expression = 'sleep'; d.excitement = 0; d.head.pitchT = -0.2;
      await d.wait(seconds);
      d.expression = 'normal'; d.excitement = 0.5; d.head.pitchT = 0;
    },

    async talk(d, seconds = 2) {
      d.speaking = true;
      try { await d.wait(seconds); } finally { d.speaking = false; }
    },

    async recover(d) {
      d.cancel();
      d.speaking = false;
      d.head.tiltT = 0; d.head.pitchT = 0;
      d.excitement = 0.4;
      await d.ensureStanding();
      d.setPose(STAND);
      await d.wait(0.8);
    },
  };

  const api = { DogSim, Tricks, Cancelled };
  root.PetDog = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
