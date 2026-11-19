// Used by the workflow's aggregate job: copy downloaded per-variant artifacts into state/variants/<name>/.
// Only directories named after a configured variant are accepted; symlinks are refused. Data only:
// nothing in the artifacts is executed.
//   node src/ci/collect.js <downloadDir>
import fs from 'node:fs';
import path from 'node:path';
import { loadConfig, rootPath } from '../config.js';

export function collect(cfg, downloadDir) {
  const prefix = `${cfg.runtime.artifactName}-`;
  const known = new Set(cfg.variants.map((v) => v.name));
  const copied = [];
  if (!fs.existsSync(downloadDir)) return copied;
  for (const entry of fs.readdirSync(downloadDir).sort()) {
    const src = path.join(downloadDir, entry);
    if (!entry.startsWith(prefix) || !fs.lstatSync(src).isDirectory()) throw new Error(`unexpected artifact entry ${entry}`);
    const name = entry.slice(prefix.length);
    if (!known.has(name)) throw new Error(`artifact for unknown variant "${name}"`);
    for (const f of fs.readdirSync(src, { recursive: true })) {
      if (fs.lstatSync(path.join(src, f)).isSymbolicLink()) throw new Error(`artifact ${entry} contains a symlink: ${f}`);
    }
    const dest = rootPath(cfg, cfg.paths.stateDir, cfg.paths.variantsDir, name);
    fs.rmSync(dest, { recursive: true, force: true });
    fs.cpSync(src, dest, { recursive: true });
    copied.push(name);
  }
  return copied;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const dir = process.argv[2];
  if (!dir) throw new Error('usage: node src/ci/collect.js <downloadDir>');
  console.log(`collected: ${collect(loadConfig(), path.resolve(dir)).join(', ') || 'nothing'}`);
}
