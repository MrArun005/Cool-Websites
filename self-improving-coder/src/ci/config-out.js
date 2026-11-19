// Prints the runtime values the workflow needs as GITHUB_OUTPUT lines, so the YAML never hard-codes a
// model name, version, URL or variant list.
//   node src/ci/config-out.js                    -> global values + matrix (JSON list of variant names)
//   node src/ci/config-out.js --variant=<name>   -> also that variant's track and state dir
import { loadConfig } from '../config.js';
import { variantConfig } from '../variants.js';

const cfg = loadConfig();
const rt = cfg.runtime;
const variant = process.argv.find((a) => a.startsWith('--variant='))?.slice('--variant='.length);
const out = {
  model: cfg.llm.model,
  provider: cfg.llm.provider,
  ollama_version: rt.ollamaVersion,
  ollama_url: rt.ollamaDownloadUrl.replace('{version}', rt.ollamaVersion),
  ollama_host: rt.ollamaHost,
  ollama_ready_timeout: rt.ollamaReadyTimeoutSec,
  ollama_context_length: rt.ollamaContextLength,
  artifact_prefix: rt.artifactName,
  publish_artifact: rt.publishArtifactName,
  artifact_retention_days: rt.artifactRetentionDays,
  state_dir: cfg.paths.stateDir,
  variants_dir: `${cfg.paths.stateDir}/${cfg.paths.variantsDir}`,
  site_dir: cfg.site.outFile.split('/').slice(0, -1).join('/') || '.',
  curriculum_cache: cfg.curriculum.cacheDir,
  dataset_cache: cfg.dataset.cacheDir,
  matrix: JSON.stringify(cfg.variants.map((v) => v.name)),
};
if (variant) {
  const vc = variantConfig(cfg, variant);
  out.track = vc.track;
  out.variant_state_dir = vc.paths.stateDir;
}
for (const [k, v] of Object.entries(out)) {
  if (/[\r\n]/.test(String(v))) throw new Error(`config value for ${k} contains a newline`);
  console.log(`${k}=${v}`);
}
