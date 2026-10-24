// Drive the simulation with a fixed time step until a trick finishes.
const { DogSim, Tricks } = require('../js/dog.js');
async function run(dog, trick, arg, maxSeconds = 20) {
  let done = false, result, error;
  Tricks[trick](dog, arg).then(r => { done = true; result = r; }, e => { done = true; error = e; });
  const dt = 1 / 240;
  for (let k = 0; k < maxSeconds / dt && !done; k++) {
    dog.update(dt);
    if (k % 4 === 0) await new Promise(r => setImmediate(r));
  }
  for (let k = 0; k < 5 && !done; k++) await new Promise(r => setImmediate(r));
  if (error) throw error;
  if (!done) throw new Error(`${trick} did not finish within ${maxSeconds}s of sim time`);
  return result;
}
module.exports = { run, DogSim, Tricks };
