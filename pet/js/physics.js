/* physics.js: the physics the digital robot dog has to obey.
 *
 * No graphics in here, so the same code runs in the browser and in the Node
 * tests. Units: metres, seconds, kilograms, radians.
 *
 * Body frame: +x forward (nose), +y up, +z to the dog's right.
 * Orientation is yaw (about world Y), then pitch (about the body's side axis,
 * positive = nose up), then roll (about the body's long axis, positive = top
 * tips to the right). That is the matrix Ry * Rz * Rx, which is three.js Euler
 * order 'YZX' with x = roll, y = yaw, z = pitch.
 */
(function (root) {
  'use strict';

  const G = 9.81;
  const TAU = Math.PI * 2;

  // A medium-sized robot dog, roughly the size and weight of a beagle.
  const DOG = {
    bodyLength: 0.56, bodyWidth: 0.24, bodyHeight: 0.15,
    hipX: 0.22, hipY: -0.03, hipZ: 0.12,
    upper: 0.2, lower: 0.2,          // thigh and shin length
    standHeight: 0.33,               // body-centre height when standing
    crouchHeight: 0.22,              // ... when crouched, ready to jump
    takeoffHeight: 0.40,             // ... when the legs are fully extended
    bodyMass: 9.0, legMass: 0.75,    // per leg
    maxLaunchSpeed: 3.6,             // m/s: the most the leg motors can deliver
    maxSpinMomentum: 5.0,            // kg·m²/s: the most spin a push can create
  };
  DOG.mass = DOG.bodyMass + 4 * DOG.legMass;

  // 0 front-left, 1 front-right, 2 rear-left, 3 rear-right.
  const LEGS = [
    { name: 'FL', front: true,  hip: [ DOG.hipX, DOG.hipY, -DOG.hipZ] },
    { name: 'FR', front: true,  hip: [ DOG.hipX, DOG.hipY,  DOG.hipZ] },
    { name: 'RL', front: false, hip: [-DOG.hipX, DOG.hipY, -DOG.hipZ] },
    { name: 'RR', front: false, hip: [-DOG.hipX, DOG.hipY,  DOG.hipZ] },
  ];
  // Diagonal pairs step together (a trot): FL+RR, then FR+RL.
  const TROT_PAIRS = [[0, 3], [1, 2]];

  // ---------- small vector helpers (plain [x, y, z] arrays) ----------
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const len = a => Math.sqrt(dot(a, a));
  const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const smooth = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
  const wrapAngle = a => a - TAU * Math.round(a / TAU);

  function rotX(v, a) { const c = Math.cos(a), s = Math.sin(a); return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]; }
  function rotY(v, a) { const c = Math.cos(a), s = Math.sin(a); return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]]; }
  function rotZ(v, a) { const c = Math.cos(a), s = Math.sin(a); return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]; }

  function bodyToWorld(local, pos, yaw, pitch, roll) {
    return add(pos, rotY(rotZ(rotX(local, roll), pitch), yaw));
  }
  function worldToBody(world, pos, yaw, pitch, roll) {
    return rotX(rotZ(rotY(sub(world, pos), -yaw), -pitch), -roll);
  }

  // Damped spring: how every joint controller in a real robot behaves (a PD
  // controller). freq in Hz, zeta = damping ratio (1 = no overshoot).
  function spring(x, v, target, freq, zeta, dt) {
    const w = TAU * freq;
    v += (w * w * (target - x) - 2 * zeta * w * v) * dt;
    return [x + v * dt, v];
  }

  // Two-segment leg inverse kinematics: where must the knee be so the foot
  // lands on `foot`? `bendHint` says which way the knee points.
  function solveLeg(hip, foot, bendHint, L1 = DOG.upper, L2 = DOG.lower) {
    const d = sub(foot, hip);
    const dist = len(d);
    const dir = dist > 1e-6 ? scale(d, 1 / dist) : [0, -1, 0];
    const reach = clamp(dist, 0.06, L1 + L2 - 1e-3);
    const footC = add(hip, scale(dir, reach));
    const a = (L1 * L1 - L2 * L2 + reach * reach) / (2 * reach);
    const h = Math.sqrt(Math.max(0, L1 * L1 - a * a));
    let perp = sub(bendHint, scale(dir, dot(bendHint, dir)));
    if (len(perp) < 1e-4) perp = [0, 1, 0];
    perp = scale(perp, 1 / len(perp));
    return { knee: add(add(hip, scale(dir, a)), scale(perp, h)), foot: footC, stretched: dist > reach + 1e-4 };
  }

  // Leg pose in the air, in the body frame. tuck 0 = legs straight down,
  // tuck 1 = feet pulled in tight under the belly.
  function legPose(i, tuck) {
    const hip = LEGS[i].hip;
    const straight = [hip[0], hip[1] - 0.37, hip[2]];
    const tucked = [hip[0] * 0.4, hip[1] - 0.12, hip[2]];
    return lerp3(straight, tucked, tuck);
  }

  // Moment of inertia about the pitch axis, computed from where the legs are.
  // Tucking pulls mass toward the centre, so the same angular momentum spins
  // the dog faster: the figure-skater effect.
  function computePitchInertia(tuck) {
    let I = DOG.bodyMass * (DOG.bodyLength ** 2 + DOG.bodyHeight ** 2) / 12;
    LEGS.forEach((leg, i) => {
      const foot = legPose(i, tuck);
      const knee = solveLeg(leg.hip, foot, [leg.front ? -1 : 1, 0, 0]).knee;
      for (const p of [lerp3(leg.hip, knee, 0.5), lerp3(knee, foot, 0.5)]) {
        I += (DOG.legMass / 2) * (p[0] * p[0] + p[1] * p[1]);
      }
    });
    return I;
  }
  const INERTIA_TABLE = Array.from({ length: 41 }, (_, k) => computePitchInertia(k / 40));
  function pitchInertia(tuck) {
    const x = clamp(tuck, 0, 1) * 40, k = Math.min(39, Math.floor(x));
    return INERTIA_TABLE[k] + (INERTIA_TABLE[k + 1] - INERTIA_TABLE[k]) * (x - k);
  }

  // How tucked the legs are at fraction s (0..1) of a flight.
  function tuckProfile(s) {
    return smooth((s - 0.05) / 0.25) * (1 - smooth((s - 0.62) / 0.25));
  }

  // Time in the air when leaving at height y0 with upward speed vy and landing at y1.
  function flightTime(vy, y0, y1) {
    return (vy + Math.sqrt(Math.max(0, vy * vy + 2 * G * (y0 - y1)))) / G;
  }

  // With angular momentum L fixed in the air, the spin angle is L × ∫ dt / I(t).
  function spinIntegral(T, tuckAmp, steps = 120) {
    let sum = 0;
    for (let k = 0; k <= steps; k++) {
      const w = k === 0 || k === steps ? 1 : (k % 2 ? 4 : 2);  // Simpson's rule
      sum += w / pitchInertia(tuckAmp * tuckProfile(k / steps));
    }
    return sum * (T / steps) / 3;
  }

  // Plan a ballistic jump. Gravity decides how long we're in the air; the
  // angular momentum from the push decides how far we spin in that time.
  function planJump({ rise = 0.15, rotations = 0, tuck = 0.35, forward = 0 } = {}) {
    let vy = Math.sqrt(2 * G * rise);
    const speedLimited = vy > DOG.maxLaunchSpeed;
    vy = Math.min(vy, DOG.maxLaunchSpeed);
    const T = flightTime(vy, DOG.takeoffHeight, DOG.takeoffHeight);
    const want = rotations * TAU;
    const integral = spinIntegral(T, tuck);
    let L = want / integral;
    const spinLimited = Math.abs(L) > DOG.maxSpinMomentum;
    L = clamp(L, -DOG.maxSpinMomentum, DOG.maxSpinMomentum);
    const vx = forward / T;
    const launchSpeed = Math.hypot(vx, vy);
    const pushDistance = DOG.takeoffHeight - DOG.crouchHeight;
    const accel = launchSpeed * launchSpeed / (2 * pushDistance);
    return {
      vy, vx, T, L, tuck,
      wantAngle: want,
      achievedAngle: L * integral,
      feasible: !spinLimited,
      speedLimited,
      apexRise: vy * vy / (2 * G),
      launchSpeed,
      legForce: DOG.mass * (accel + G),                    // newtons, all legs together
      energy: 0.5 * DOG.mass * launchSpeed ** 2 + 0.5 * L * L / pitchInertia(0), // joules
    };
  }

  // Lying on its side, the body is roughly an ellipse in cross-section: how high
  // is the body centre when rolled by `roll`?
  function rollSupport(roll) {
    const w = DOG.bodyWidth / 2, h = DOG.bodyHeight / 2 + 0.01;
    return Math.sqrt((w * Math.sin(roll)) ** 2 + (h * Math.cos(roll)) ** 2);
  }

  const api = {
    G, TAU, DOG, LEGS, TROT_PAIRS,
    add, sub, scale, dot, len, lerp3, clamp, smooth, wrapAngle,
    rotX, rotY, rotZ, bodyToWorld, worldToBody,
    spring, solveLeg, legPose, pitchInertia, tuckProfile,
    flightTime, spinIntegral, planJump, rollSupport,
  };
  root.PetPhysics = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
