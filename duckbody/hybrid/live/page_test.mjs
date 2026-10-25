// The live page in headless Chromium: node page_test.mjs URL OUT_DIR [wall_seconds=180] [speed_clicks=0] [size=5v5]
// CDN scripts are served from CDN_DIR (env, local copies of the exact versions the page loads) so the test needs no
// internet. Samples window.__live every 10 s, screenshots now and then, writes OUT_DIR/page_test.json.
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire('/opt/node-tools/node_modules/x.js');
const { chromium } = require('playwright');
const [url, out, wallArg = '180', speedClicks = '0', size = '5v5'] = process.argv.slice(2);
const cdn = process.env.CDN_DIR;
fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
const logs = [];
page.on('console', m => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', e => logs.push(`pageerror: ${e.message}`));
const map = { 'three.min.js': 'three.min.js', 'OrbitControls.js': 'OrbitControls.js', 'mujoco.js': 'mujoco.js' };
await page.route(/cdnjs\.cloudflare\.com|cdn\.jsdelivr\.net/, route => {
  const f = path.basename(new URL(route.request().url()).pathname);
  if (map[f]) return route.fulfill({ path: path.join(cdn, map[f]), contentType: 'text/javascript', headers: { 'access-control-allow-origin': '*' } });
  return route.abort();
});
await page.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
const t0 = Date.now();
await page.goto(url);
await page.waitForFunction(() => window.__live && window.__live.summary, null, { timeout: 120000 });
console.log(`page ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (size === '3v3') { await page.click('#size'); await page.waitForTimeout(3000); }
for (let i = 0; i < +speedClicks; i++) await page.click('#speed');
const samples = [];
const wall = +wallArg;
for (let s = 0; s <= wall; s += 10) {
  const L = await page.evaluate(() => ({ fps: window.__live.fps, rate: window.__live.rate, load: window.__live.load, s: window.__live.summary }));
  const sm = L.s;
  samples.push({ wall: s, fps: +L.fps.toFixed(1), rate: +L.rate.toFixed(2), load: +L.load.toFixed(0), sim_s: sm.sim_s, half: sm.half, clock: sm.clock, phase: sm.phase, score: sm.score, kicks: sm.kicks, connected: sm.connected, falls: sm.falls });
  console.log(JSON.stringify(samples[samples.length - 1]));
  if (s % 60 === 0) await page.screenshot({ path: path.join(out, `live_${String(s).padStart(3, '0')}.png`) });
  if (s < wall) await page.waitForTimeout(10000);
}
const final = await page.evaluate(() => {
  const g = window.__liveControl.game;
  return { summary: g.summary(), events: g.events.map(e => `${e.half}H ${e.clock.toFixed(0)} ${e.kind} ${e.text}`), kicks: g.kicks,
    moving: g.players.map(p => [p.name, p.stats.samples ? +(p.stats.moving / p.stats.samples).toFixed(2) : null]),
    stepMs: g.world.stepMs, brainMs: g.brainMs, bamMs: g.world.bamMs };
});
await page.screenshot({ path: path.join(out, 'live_end.png') });
fs.writeFileSync(path.join(out, 'page_test.json'), JSON.stringify({ samples, final, logs: logs.slice(0, 50) }, null, 1));
console.log('events:\n' + final.events.join('\n'));
console.log('moving share:', JSON.stringify(final.moving));
console.log(`per sim second: mj_step ${(final.stepMs / final.summary.sim_s).toFixed(0)} ms, BAM ${(final.bamMs / final.summary.sim_s).toFixed(0)} ms, brains ${(final.brainMs / final.summary.sim_s).toFixed(0)} ms`);
console.log('console:', logs.slice(0, 10).join(' | '));
await browser.close();
