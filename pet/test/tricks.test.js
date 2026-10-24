const test = require('node:test');
const assert = require('node:assert/strict');
const { run, DogSim } = require('./run.js');
const { TAU } = require('../js/physics.js');

const upright = d => Math.abs(d.pitch) < 0.05 && Math.abs(d.roll) < 0.05 && Math.abs(d.pos[1] - 0.33) < 0.02;

test('backflip: one full backward turn, lands feet first, back on the same spot', async () => {
  const d = new DogSim();
  const r = await run(d, 'backflip', 1);
  assert.equal(r.landed, true);
  assert.ok(Math.abs(r.landingErrorDeg) < 3, `landing error ${r.landingErrorDeg}°`);
  assert.ok(Math.abs(d.spinTravelled - TAU) < 0.15);
  assert.ok(Math.hypot(d.pos[0], d.pos[2]) < 0.1);
  assert.ok(upright(d));
});

test('somersault: one forward turn and travels forward', async () => {
  const d = new DogSim();
  const r = await run(d, 'frontflip', 1);
  assert.equal(r.landed, true);
  assert.ok(Math.abs(d.spinTravelled + TAU) < 0.15);
  assert.ok(d.pos[0] > 0.4, `moved ${d.pos[0]} m`);
  assert.ok(upright(d));
});

test('double backflip under-rotates and crashes, because physics', async () => {
  const d = new DogSim();
  const r = await run(d, 'backflip', 2);
  assert.equal(r.landed, false);
  assert.ok(upright(d), 'gets back up afterwards');
});

test('roll over: rolls a full turn on the ground and stands back up', async () => {
  const d = new DogSim();
  await run(d, 'rollOver');
  assert.ok(Math.abs(d.pos[2]) > 0.2, 'rolled sideways');
  assert.ok(upright(d));
});

test('sit tilts the body nose-up and lowers the hips', async () => {
  const d = new DogSim();
  await run(d, 'sit');
  assert.ok(d.pitch > 0.45 && d.pos[1] < 0.27);
  assert.equal(d.posture, 'sit');
});

test('every trick finishes without numerical blow-ups', async () => {
  const d = new DogSim();
  for (const t of ['sit', 'shake', 'beg', 'bow', 'jump', 'jumpAround', 'playDead', 'spin', 'dance', 'come', 'lie', 'wag', 'stand']) {
    await run(d, t);
    for (const x of [...d.pos, d.pitch, d.roll, d.yaw]) assert.ok(Number.isFinite(x), `${t} produced ${x}`);
    assert.ok(Math.hypot(d.pos[0], d.pos[2]) < 3, `${t} wandered off the stage`);
  }
});

test('stop cancels a trick mid-air and the dog still lands safely', async () => {
  const d = new DogSim();
  const { Tricks } = require('../js/dog.js');
  const flip = Tricks.backflip(d, 1).catch(e => e);
  for (let k = 0; k < 240 * 0.6; k++) d.update(1 / 240), await null;
  assert.equal(d.mode, 'flight');
  d.cancel();
  assert.ok((await flip).constructor.name === 'Cancelled');
  await run(d, 'recover');
  assert.ok(upright(d));
});
