const test = require('node:test');
const assert = require('node:assert/strict');
const { Brain } = require('../js/brain.js');

const memStore = () => { const m = new Map(); return { getItem: k => m.get(k) ?? null, setItem: (k, v) => m.set(k, v) }; };
const tricks = steps => steps.map(s => s.trick);

test('understands chained commands in order', () => {
  const b = new Brain();
  assert.deepEqual(tricks(b.understand('sit, then do a backflip and roll over')), ['sit', 'backflip', 'rollOver']);
});

test('hears "somersault" however speech-to-text spells it', () => {
  const b = new Brain();
  for (const s of ['somersault', 'summersault', 'summer salt', 'summer slam', 'front flip']) {
    b.battery = 100;
    assert.deepEqual(tricks(b.understand(s)), ['frontflip'], s);
  }
});

test('counts flips', () => {
  const b = new Brain();
  assert.equal(b.understand('double backflip')[0].arg, 2);
  assert.equal(b.understand('backflip')[0].arg, 1);
});

test('remembers its name in long-term storage across restarts', () => {
  const store = memStore();
  new Brain(store).understand('your name is Rex');
  const reborn = new Brain(store);
  assert.equal(reborn.name, 'Rex');
  assert.match(reborn.understand('what is your name')[0].say, /Rex/);
});

test('praise is saved as feedback on the trick it was about (training labels)', () => {
  const store = memStore();
  const b = new Brain(store);
  b.understand('backflip');
  b.understand('good boy');
  const entry = new Brain(store).log.find(e => e.said === 'backflip');
  assert.equal(entry.feedback, 1);
});

test('refuses big tricks on a flat battery, and recharges after sleep', () => {
  const b = new Brain();
  b.battery = 10;
  assert.deepEqual(tricks(b.understand('backflip')), ['sad']);
  const [nap] = b.understand('go to sleep');
  assert.equal(nap.trick, 'rest');
  nap.onDone();
  assert.equal(b.battery, 100);
});

test('talks and tells jokes', () => {
  const b = new Brain();
  assert.equal(b.understand('talk to me')[0].trick, 'talk');
  assert.ok(b.understand('tell me a joke')[0].joke);
  assert.equal(b.understand('blorp')[0].trick, 'confused');
});
