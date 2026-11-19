// The aggregate job merges artifacts written by jobs that ran untrusted code: only configured variant
// names are accepted, symlinks are refused, and nothing is executed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { collect } from '../src/ci/collect.js';
import { loadConfig } from '../src/config.js';
import { tmpDir } from './helpers.js';

const setup = () => {
  const stateDir = tmpDir();
  const cfg = loadConfig({ env: {}, overrides: { paths: { stateDir } } });
  const dl = tmpDir();
  return { cfg, dl, prefix: `${cfg.runtime.artifactName}-` };
};

test('copies known variants into state/variants/<name>/', () => {
  const { cfg, dl, prefix } = setup();
  fs.mkdirSync(path.join(dl, `${prefix}lessons`));
  fs.writeFileSync(path.join(dl, `${prefix}lessons`, 'champion.json'), '{}');
  assert.deepEqual(collect(cfg, dl), ['lessons']);
  assert.ok(fs.existsSync(path.join(cfg.paths.stateDir, cfg.paths.variantsDir, 'lessons', 'champion.json')));
});

test('refuses unknown variant names, foreign entries and symlinks', () => {
  let { cfg, dl, prefix } = setup();
  fs.mkdirSync(path.join(dl, `${prefix}../../evil`), { recursive: true });
  assert.throws(() => collect(cfg, dl), /unexpected artifact entry|unknown variant/);
  ({ cfg, dl, prefix } = setup());
  fs.mkdirSync(path.join(dl, `${prefix}not-a-variant`));
  assert.throws(() => collect(cfg, dl), /unknown variant/);
  ({ cfg, dl, prefix } = setup());
  fs.mkdirSync(path.join(dl, `${prefix}baseline`));
  fs.symlinkSync('/etc/passwd', path.join(dl, `${prefix}baseline`, 'champion.json'));
  assert.throws(() => collect(cfg, dl), /symlink/);
});
