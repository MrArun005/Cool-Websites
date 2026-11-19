// Prints the runtime values the workflow needs as GITHUB_OUTPUT lines, so the YAML never
// hard-codes a model name, version or URL. Usage: node src/ci/config-out.js >> "$GITHUB_OUTPUT"
import { loadConfig } from '../config.js';

const cfg = loadConfig();
const rt = cfg.runtime;
const out = {
  model: cfg.llm.model,
  provider: cfg.llm.provider,
  ollama_version: rt.ollamaVersion,
  ollama_url: rt.ollamaDownloadUrl.replace('{version}', rt.ollamaVersion),
  ollama_host: rt.ollamaHost,
  ollama_ready_timeout: rt.ollamaReadyTimeoutSec,
  artifact_name: rt.artifactName,
  artifact_retention_days: rt.artifactRetentionDays,
  state_dir: cfg.paths.stateDir,
};
for (const [k, v] of Object.entries(out)) {
  if (/[\r\n]/.test(String(v))) throw new Error(`config value for ${k} contains a newline`);
  console.log(`${k}=${v}`);
}
