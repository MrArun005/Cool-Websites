const test = require('node:test');
const assert = require('node:assert/strict');
const P = require('../js/physics.js');

test('flight time follows from gravity: T = 2·vy/g for equal takeoff and landing heights', () => {
  const plan = P.planJump({ rise: 0.5 });
  assert.ok(Math.abs(plan.T - 2 * plan.vy / P.G) < 1e-9);
  assert.ok(Math.abs(plan.apexRise - 0.5) < 1e-9);
});

test('tucking the legs lowers the inertia, so the same spin momentum turns faster', () => {
  assert.ok(P.pitchInertia(1) < P.pitchInertia(0) * 0.7);
});

test('a single backflip is feasible and plans exactly one full turn', () => {
  const plan = P.planJump({ rise: 0.5, rotations: 1, tuck: 1 });
  assert.equal(plan.feasible, true);
  assert.ok(Math.abs(plan.achievedAngle - P.TAU) < 1e-9);
});

test('a double backflip exceeds what the leg motors can produce', () => {
  const plan = P.planJump({ rise: 0.7, rotations: 2, tuck: 1 });
  assert.equal(plan.feasible, false);
  assert.ok(plan.achievedAngle < 2 * P.TAU - 0.5);
});

test('leg inverse kinematics keeps both segments at their real length', () => {
  const hip = [0, 0.3, 0], foot = [0.05, 0, 0.02];
  const { knee, foot: f } = P.solveLeg(hip, foot, [-1, 0, 0]);
  assert.ok(Math.abs(P.len(P.sub(knee, hip)) - P.DOG.upper) < 1e-9);
  assert.ok(Math.abs(P.len(P.sub(f, knee)) - P.DOG.lower) < 1e-9);
  assert.ok(knee[0] < 0, 'front knee points backward');
});

test('body/world transforms are inverses of each other', () => {
  const pos = [0.3, 0.4, -0.2], local = [0.1, -0.2, 0.05];
  const back = P.worldToBody(P.bodyToWorld(local, pos, 0.7, -0.4, 1.1), pos, 0.7, -0.4, 1.1);
  back.forEach((v, i) => assert.ok(Math.abs(v - local[i]) < 1e-12));
});
