// OpenAI-style /chat/completions client with retry + exponential backoff.
// Default target is a local Ollama server on the Actions runner (see DECISIONS.md:
// GitHub Models was retired on 2026-07-30). Any OpenAI-compatible endpoint works.
import { sleep, MS_PER_SECOND } from '../util.js';

export class LLMError extends Error {
  constructor(message, { status, retryable } = {}) {
    super(message);
    this.status = status;
    this.retryable = retryable;
  }
}

export function backoffDelay(attempt, { baseDelayMs, maxDelayMs }, retryAfterSec) {
  if (Number.isFinite(retryAfterSec) && retryAfterSec > 0) return Math.min(retryAfterSec * MS_PER_SECOND, maxDelayMs);
  return Math.min(baseDelayMs * 2 ** (attempt - 1), maxDelayMs);
}

export function createOpenAICompatible(cfg, deps = {}) {
  const fetchFn = deps.fetch ?? globalThis.fetch;
  const sleepFn = deps.sleep ?? sleep;
  const { endpoint, model, maxTokens, seed, retry, requestTimeoutMs, apiKeyEnv } = cfg.llm;
  const apiKey = apiKeyEnv ? (deps.env ?? process.env)[apiKeyEnv] : undefined;

  async function once(messages, temperature, seedOverride) {
    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    const body = { model, messages, temperature, max_tokens: maxTokens, stream: false };
    const useSeed = Number.isFinite(seedOverride) ? seedOverride : seed;
    if (Number.isFinite(useSeed)) body.seed = useSeed;
    let res;
    try {
      res = await fetchFn(endpoint, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(requestTimeoutMs) });
    } catch (e) {
      throw new LLMError(`network error: ${e.message}`, { retryable: true });
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      const retryable = retry.retryOnStatus.includes(res.status);
      const err = new LLMError(`HTTP ${res.status}: ${text.slice(0, cfg.llm.errorSnippetChars)}`, { status: res.status, retryable });
      err.retryAfter = Number(res.headers?.get?.('retry-after'));
      throw err;
    }
    const json = await res.json();
    const content = json?.choices?.[0]?.message?.content;
    if (typeof content !== 'string') throw new LLMError('malformed response: no choices[0].message.content', { retryable: true });
    const u = json.usage ?? {};
    return { content, usage: { tokensIn: u.prompt_tokens ?? 0, tokensOut: u.completion_tokens ?? 0 } };
  }

  const self = {
    name: 'openai-compatible',
    lastUsage: { tokensIn: 0, tokensOut: 0 },
    async chat(messages, { temperature = cfg.llm.temperature, seed: seedOverride } = {}) {
      let lastErr;
      for (let attempt = 1; attempt <= retry.maxAttempts; attempt++) {
        try {
          const { content, usage } = await once(messages, temperature, seedOverride);
          self.lastUsage = usage;
          return content;
        } catch (e) {
          lastErr = e;
          if (!e.retryable || attempt === retry.maxAttempts) break;
          await sleepFn(backoffDelay(attempt, retry, e.retryAfter));
        }
      }
      throw lastErr;
    },
  };
  return self;
}
