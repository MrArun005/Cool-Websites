// A track = a source of tasks + how to show them + how to judge them.
//   load()                       -> { tasks: Map<id, task>, train: id[], holdout: id[] }
//   nextTrainTask(ctx)           -> id | null (ctx: { mem, runIndex, doneThisRun:Set, solved:Set })
//   onTrainResult(id, passed, ctx) -> optional event (e.g. curriculum level change), persisted by the track
//   extractCode(reply, task), judge(code, task) -> { type, output, durationMs }
//   templates: { solveUser, reflectUser }, storeCode: boolean, oracle(id) for the mock
// Every task has: id, entryPoint (or entryPoints), vars (template variables), keywordText (for retrieval).
import { createMultiplETrack } from './multipl-e.js';

export async function createTrack(cfg, deps = {}) {
  const name = cfg.track ?? cfg.defaultTrack;
  switch (name) {
    case 'multipl-e': return createMultiplETrack(cfg, deps);
    case 'mutation': return (await import('./mutation.js')).createMutationTrack(cfg, deps);
    default: throw new Error(`unknown track "${name}"`);
  }
}
