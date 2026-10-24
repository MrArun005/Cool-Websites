/* app.js: wires the brain, the body simulation, the 3D view, the chat and
 * the voice together. */
(() => {
  'use strict';
  const { DogSim, Tricks, Cancelled } = PetDog;
  const { Brain } = PetBrain;
  const $ = id => document.getElementById(id);

  let storage = null;
  try { storage = window.localStorage; storage.getItem('pet-probe'); } catch { storage = null; }

  const brain = new Brain(storage);
  const sim = new DogSim();
  let view = null;
  try {
    if (!window.THREE) throw new Error('three.js did not load');
    view = new PetView.DogView($('scene'));
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => view.applyTheme());
  } catch (e) {
    console.warn(e);
    $('nogl').style.display = 'grid';
  }

  const settings = { slowmo: false, trace: true, voice: true };

  // ---------- chat ----------
  const chat = $('chat');
  function addMsg(who, text) {
    const el = document.createElement('div');
    el.className = 'msg ' + who;
    el.textContent = text;
    chat.append(el);
    chat.scrollTop = chat.scrollHeight;
  }

  // ---------- voice out ----------
  const synth = window.speechSynthesis;
  let voice = null;
  function pickVoice() {
    const voices = synth ? synth.getVoices() : [];
    voice = voices.find(v => /en[-_](US|GB)/i.test(v.lang) && /google|samantha|daniel|natural/i.test(v.name))
      || voices.find(v => /^en/i.test(v.lang)) || null;
  }
  if (synth) { pickVoice(); synth.addEventListener?.('voiceschanged', pickVoice); }

  let bubbleTimer = 0;
  function showBubble(text, seconds) {
    const b = $('bubble');
    b.textContent = text;
    b.classList.add('show');
    clearTimeout(bubbleTimer);
    bubbleTimer = setTimeout(() => b.classList.remove('show'), seconds * 1000);
  }

  // Speak a line; resolves when done. "*sits*" stage directions are shown, not spoken.
  function speak(text) {
    if (!text) return Promise.resolve();
    const spoken = text.replace(/\*[^*]*\*/g, ' ').replace(/\s+/g, ' ').trim();
    const estimate = Math.max(1.2, text.split(/\s+/).length * 0.36);
    showBubble(text, estimate + 1);
    sim.speaking = true;
    const done = () => { sim.speaking = false; };
    if (!settings.voice || !synth || !spoken) {
      return new Promise(r => setTimeout(() => { done(); r(); }, (spoken ? estimate : 0.8) * 1000));
    }
    return new Promise(resolve => {
      const u = new SpeechSynthesisUtterance(spoken);
      if (voice) u.voice = voice;
      u.pitch = 1.45; u.rate = 1.06;
      let finished = false;
      const finish = () => { if (!finished) { finished = true; done(); resolve(); } };
      u.onend = finish; u.onerror = finish;
      setTimeout(finish, (estimate + 4) * 1000); // some browsers never fire onend
      synth.speak(u);
    });
  }

  // ---------- doing things ----------
  const queue = [];
  let busy = false;

  function handle(text) {
    text = text.trim();
    if (!text) return;
    addMsg('you', text);
    const steps = brain.understand(text);
    if (steps.some(s => s.trick === 'stop')) {
      queue.length = 0;
      synth && synth.cancel();
      sim.cancel();
    }
    queue.push(...steps);
    updateMemStat();
    pump();
  }

  async function pump() {
    if (busy) return;
    busy = true;
    while (queue.length) {
      const step = queue.shift();
      if (step.say) addMsg('pet', step.say);
      try {
        if (step.trick === 'stop') { speak(step.say); await Tricks.recover(sim); continue; }
        if (step.trick === 'talk') { await speak(step.say); continue; }
        const speech = speak(step.say);
        const result = await Tricks[step.trick](sim, step.arg);
        if (result && result.T) showPhysics(result);
        await speech;
        if (step.onDone) { const line = step.onDone(); addMsg('pet', line); await speak(line); }
      } catch (e) {
        if (!(e instanceof Cancelled)) console.error(e);
      }
    }
    busy = false;
  }

  // ---------- physics readout ----------
  const fmt = (x, d = 2) => x.toFixed(d);
  function showPhysics(j) {
    const rows = [
      ['Trick', j.name],
      ['Launch speed', `${fmt(j.launchSpeed)} m/s`],
      ['Push-off time', `${Math.round(j.pushTime * 1000)} ms`],
      ['Height gained', `${Math.round(j.apexRise * 100)} cm`],
      ['Time in the air', `${fmt(j.T)} s`],
      ['Peak leg force', `${Math.round(j.legForce)} N (${fmt(j.bodyWeights, 1)}× body weight)`],
      ['Energy', `${Math.round(j.energy)} J`],
    ];
    if (j.wantAngle) {
      rows.push(['Spin momentum', `${fmt(Math.abs(j.L))} kg·m²/s${j.feasible ? '' : ' (motor limit!)'}`]);
      rows.push(['Tucking', `spins ${fmt(PetPhysics.pitchInertia(0) / PetPhysics.pitchInertia(1), 1)}× faster`]);
    }
    const landed = j.landed
      ? `<span class="ok">✓ Landed, ${fmt(Math.abs(j.landingErrorDeg), 1)}° off level</span>`
      : `<span class="bad">✗ Crashed: ${Math.round(Math.abs(j.landingErrorDeg))}° short of upright</span>`;
    rows.push(['Landing', landed]);
    $('physics').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    brain.recordOutcome({ trick: j.name, launchSpeed: +fmt(j.launchSpeed), airTime: +fmt(j.T), landed: j.landed, landingErrorDeg: +fmt(j.landingErrorDeg, 1) });
    if (!j.landed) {
      const line = 'Ow. I under-rotated. My legs can’t make enough spin for that. Physics: 1, me: 0.';
      addMsg('pet', line);
      speak(line);
    }
  }

  const MODE_NAMES = { stance: 'Standing', push: 'Pushing off', flight: 'In the air', body: 'Rolling' };
  function updateLive() {
    const spin = Math.abs(sim.pitchRate) > 0.5 ? Math.abs(sim.pitchRate) : Math.abs(sim.rollRate);
    $('live').innerHTML = `<b>${MODE_NAMES[sim.mode]}</b> · height ${Math.round(sim.pos[1] * 100)} cm · ` +
      `vertical ${fmt(sim.vel[1], 1)} m/s · spin ${Math.round(spin * 57.3)}°/s · battery ${Math.round(brain.battery)}%`;
  }

  function updateMemStat() {
    $('petName').textContent = brain.name;
    const rated = brain.log.filter(e => e.feedback !== undefined).length;
    $('memStat').textContent = `Long-term memory: ${brain.log.length} interactions, ${rated} with your feedback ("good boy" / "no"). ` +
      (storage ? 'Saved in this browser.' : 'Storage is blocked, so this resets on reload.');
  }

  // ---------- main loop: fixed 240 Hz physics, render every frame ----------
  const DT = 1 / 240;
  let last = performance.now(), acc = 0;
  function frame(now) {
    const real = Math.min(0.1, (now - last) / 1000);
    last = now;
    acc += real * (settings.slowmo ? 0.25 : 1);
    while (acc >= DT) { sim.update(DT); acc -= DT; }
    if (!busy) brain.battery = Math.min(100, brain.battery + real * 0.3); // trickle charge while idle
    if (view) {
      sim.viewer = view.cameraXZ();
      view.render(sim, { battery: brain.battery, showTrace: settings.trace }, real);
      const head = sim.toWorld([0.4, 0.28, 0]);
      const p = view.project(head), b = $('bubble');
      b.style.left = Math.max(80, Math.min(p.x, $('scene').clientWidth - 80)) + 'px';
      b.style.top = Math.max(60, p.y) + 'px';
    }
    updateLive();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  // ---------- controls ----------
  $('sayForm').addEventListener('submit', e => {
    e.preventDefault();
    handle($('sayInput').value);
    $('sayInput').value = '';
  });

  const CHIPS = [['Sit', 'sit'], ['Jump', 'jump'], ['Backflip', 'backflip'], ['Somersault', 'somersault'], ['Roll over', 'roll over'],
    ['Play dead', 'play dead'], ['Shake', 'shake'], ['Beg', 'beg'], ['Spin', 'spin'], ['Dance', 'dance'], ['Jump around', 'jump around'],
    ['Come here', 'come here'], ['Lie down', 'lie down'], ['Talk', 'talk'], ['Tell a joke', 'tell me a joke'], ['Good boy!', 'good boy'],
    ['Double backflip', 'double backflip'], ['Sleep', 'sleep'], ['Stop', 'stop']];
  CHIPS.forEach(([label, text]) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.addEventListener('click', () => handle(text));
    $('chips').append(b);
  });

  function toggle(id, key) {
    $(id).addEventListener('click', () => {
      settings[key] = !settings[key];
      $(id).setAttribute('aria-pressed', String(settings[key]));
      if (key === 'voice' && !settings.voice && synth) synth.cancel();
    });
  }
  toggle('slowmo', 'slowmo');
  toggle('traceBtn', 'trace');
  toggle('voiceBtn', 'voice');
  if (!synth) { settings.voice = false; $('voiceBtn').disabled = true; $('voiceBtn').setAttribute('aria-pressed', 'false'); }

  $('exportBtn').addEventListener('click', () => {
    const data = { exportedAt: new Date().toISOString(), pet: brain.name, interactions: brain.log };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    a.download = `pet-interactions-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('forgetBtn').addEventListener('click', () => {
    if (!confirm('Erase the dog’s name and every saved interaction?')) return;
    brain.clearMemory();
    addMsg('sys', 'Long-term memory wiped.');
    updateMemStat();
  });

  // ---------- voice in ----------
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const mic = $('mic');
  if (!Recognition) {
    mic.disabled = true;
    mic.title = 'Voice input is not supported in this browser (try Chrome or Edge)';
  } else {
    const rec = new Recognition();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    let listening = false;
    rec.onresult = e => handle(e.results[0][0].transcript);
    rec.onend = () => { listening = false; mic.classList.remove('listening'); };
    rec.onerror = e => { if (e.error !== 'no-speech' && e.error !== 'aborted') addMsg('sys', `Microphone: ${e.error}`); };
    mic.addEventListener('click', () => {
      if (listening) { rec.stop(); return; }
      try { rec.start(); listening = true; mic.classList.add('listening'); } catch { /* already started */ }
    });
  }

  // ---------- hello ----------
  updateMemStat();
  const seen = brain.log.length;
  addMsg('pet', seen
    ? `Welcome back! It’s me, ${brain.name}. I remember our ${seen} chats. What trick should I do?`
    : `Hi! I’m ${brain.name}, a robot dog made of physics. Try "sit", "backflip", "roll over", "play dead" or "tell me a joke". You can chain them: "sit then backflip".`);

  window.pet = { sim, brain, handle }; // handy in the browser console
})();
