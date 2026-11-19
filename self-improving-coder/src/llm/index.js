import { createOpenAICompatible } from './openai-compatible.js';
import { createMock } from './mock.js';

/**
 * Provider interface: { name, chat(messages, { temperature, meta }) -> Promise<string> }
 * `meta` ({ kind, problemId }) is only informational; real providers ignore it.
 */
export function createLLM(cfg, deps = {}) {
  switch (cfg.llm.provider) {
    case 'openai-compatible': return createOpenAICompatible(cfg, deps);
    case 'mock': return createMock(cfg, deps);
    default: throw new Error(`Unknown llm.provider ${cfg.llm.provider}`);
  }
}
