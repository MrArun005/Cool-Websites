/* brain.js: understands what you say and decides what the dog does and says.
 *
 * Phase 1 brain: hand-written rules. understand(text) returns a list of steps
 * like { trick: 'backflip', arg: 1, say: 'Hold my battery.' }. A learned model
 * (Phase 3) can replace the rules later without touching the body, as long as
 * it produces the same kind of steps.
 *
 * The brain also keeps:
 *   - short-term state in RAM: mood, battery, the last trick (for feedback)
 *   - long-term memory in storage: its name, and a log of every interaction
 *     with your feedback, which is the training data for Phase 3.
 */
(function (root) {
  'use strict';

  const pick = list => list[Math.floor(Math.random() * list.length)];
  const fill = (s, ctx) => s.replace(/\{(\w+)\}/g, (_, k) => (ctx[k] ?? ''));

  // Battery cost of each trick, in percent. Flips are expensive.
  const COST = { backflip: 12, frontflip: 12, jumpAround: 9, dance: 6, jump: 4, rollOver: 4, playDead: 2, spin: 2,
    come: 2, walk: 2, beg: 1, shake: 1, bow: 1, sit: 0.5, lie: 0.5, stand: 0.5, wag: 0.5, turnAround: 1, talk: 0.5 };

  const COUNT_WORDS = { double: 2, two: 2, '2': 2, triple: 3, three: 3, '3': 3, quadruple: 4, four: 4, '4': 4 };
  const count = t => { const m = t.match(/\b(double|triple|quadruple|two|three|four|[234])\b/); return m ? COUNT_WORDS[m[1]] : 1; };

  // Order matters: the first rule whose pattern matches a clause wins.
  const RULES = [
    { id: 'name', re: /\b(?:your name is|i(?:'ll| will) call you|call yourself|you are called)\s+([a-z][\w-]*)/i },
    { id: 'askName', re: /\b(what(?:'s| is) your name|who are you)\b/ },
    { id: 'stop', re: /\b(stop|freeze|halt|enough|cancel)\b/ },
    { id: 'backflip', re: /\bback\s*-?\s*flips?\b|\bbackward(?:s)? flip|\bflip backward/, arg: count },
    { id: 'frontflip', re: /\bfront\s*-?\s*flips?\b|\bforward(?:s)? flip|\bs[ou]mm?[ae]r\s*-?\s*(?:s[aou]u?lt|slam|sault)s?\b|\bflip\b/, arg: count },
    { id: 'jumpAround', re: /\b(jump(?:ing)? around|hop(?:ping)? around|zoomies|go crazy|bounce around|run around)\b/ },
    { id: 'rollOver', re: /\broll(?:ing)?\s*-?\s*over\b|\broll\b/ },
    { id: 'playDead', re: /\bplay(?:ing)? dead\b|\bbang\b|\bplay possum\b/ },
    { id: 'beg', re: /\bbeg\b|\bsit pretty\b/ },
    { id: 'shake', re: /\bshake\b|\bpaw\b|\bhigh\s*-?\s*five\b|\bgive me (?:your|a) hand\b/ },
    { id: 'sit', re: /\bsit\b|\bsitting\b/ },
    { id: 'lie', re: /\b(lie|lay|lying|laying)\b|^down$|\bget down\b/ },
    { id: 'jump', re: /\b(jump|hop|leap)\b/ },
    { id: 'spin', re: /\bspin\b|\bchase (?:your |its )?tail\b|\btwirl\b/ },
    { id: 'turnAround', re: /\bturn around\b/ },
    { id: 'bow', re: /\bbow\b|\bplay bow\b/ },
    { id: 'dance', re: /\bdance\b|\bboogie\b|\bgroove\b/ },
    { id: 'come', re: /\bcome(?: here| to me| back)?\b|\bhere (?:boy|girl|buddy)\b/ },
    { id: 'walk', re: /\b(?:walk|go|move)\b.*\b(left|right|forward|ahead|back(?:ward)?)\b|\bwalk\b/,
      arg: t => { const m = t.match(/\b(left|right|back)/); return m ? m[1] : 'forward'; } },
    { id: 'stand', re: /\bstand\b|\bget up\b|^up$/ },
    { id: 'wag', re: /\bwag\b/ },
    { id: 'rest', re: /\b(sleep|rest|nap|recharge|charge|bed ?time|tired)\b/ },
    { id: 'joke', re: /\bjoke\b|\bfunny\b|\bmake me laugh\b|\bhumou?r\b|\bsomething funny\b/ },
    { id: 'talk', re: /\b(talk|speak|say something|bark|tell me about yourself|chat)\b/ },
    { id: 'praise', re: /\bgood (?:boy|girl|dog|doggo|pup|bot|robot|job)\b|\bwell done\b|\bgood\b(?! (?:morning|evening|night|afternoon))|\bnice\b|\bawesome\b|\bamazing\b|\bbravo\b|\btreat\b|\blove you\b|\bcute\b/ },
    { id: 'scold', re: /\bbad (?:boy|girl|dog|robot)\b|\bno+\b|\bnaughty\b|\bstop that\b/ },
    { id: 'greet', re: /\b(hi|hello|hey|yo|good morning|good evening)\b/ },
    { id: 'howAreYou', re: /\bhow are you\b|\bhow do you feel\b|\bbattery\b/ },
    { id: 'help', re: /\bhelp\b|\bwhat can you do\b|\btricks\b/ },
  ];

  const LINES = {
    sit: ['Sitting. The easiest trick, which is why it’s my favourite.', '*sits* Is this where the treat comes in?', 'Sit. Nailed it. Physics barely noticed.'],
    stand: ['Up!', 'Standing by. Literally.', 'All four feet on the floor, as nature intended.'],
    lie: ['Lying down. Gravity wins this round.', '*flop*', 'Horizontal mode activated.'],
    jump: ['Boing!', 'Up I go!', 'Jump! 9.81 metres per second squared can’t hold me. For about a third of a second.'],
    jumpAround: ['ZOOMIES!', 'Bounce bounce bounce!', 'I have no idea where I’m going and I love it.'],
    backflip: ['Backflip! Hold my battery.', 'Tucking in to spin faster. Figure skaters taught me that.', 'Backflip time. Newton, watch this.'],
    frontflip: ['Somersault! Forward and over!', 'Front flip! Aiming for the floor, but feet first this time.', 'Somersault. Please do not try this at home. I’m a robot.'],
    multiflip: ['A {n} flip? My motors say no, but my heart says YES.', 'Going for a {n} flip. Physics has been warned.'],
    rollOver: ['Rolling over!', '*rolls* The floor needed a hug.', 'Roll over? I’d roll over for you any day.'],
    playDead: ['Bang! *dramatic collapse*', 'Oh no, I’ve been got. Tell my charger I loved it.', '*plays dead* ...this is my best acting.'],
    beg: ['*sits pretty* Please?', 'Begging. Is it working? It’s working.', 'Look at these paws. Look at them.'],
    shake: ['Nice to meet you!', '*shakes paw* Pleasure doing business.', 'Here’s my paw. It’s a bit metal, sorry.'],
    bow: ['*play bow* Wanna play?', 'Bowing. This means let’s play!'],
    spin: ['Chasing my tail! I will catch it this time.', 'Round and round!', 'Spinning. My tail is very fast.'],
    turnAround: ['Turning around.', 'About face!'],
    dance: ['Let’s dance!', 'Robot dance! It’s literally my genre.', 'Watch these moves. I learned them from a washing machine.'],
    come: ['Coming!', 'On my way!', 'Here I come!'],
    walk: ['Walking {dir}.', 'Heading {dir}.'],
    wag: ['*wag wag wag*', 'Wagging at maximum speed!'],
    rest: ['Nap time. Recharging...', 'Zzz... charging...'],
    rested: ['Fully charged and ready for trouble!', 'Battery full. Let’s do something silly.'],
    stop: ['Okay, okay! Stopping.', 'Freezing!', 'Stopped. What did I do?'],
    praise: ['Thank you! *happy wiggle*', 'I’m a good robot! Saving that to long-term memory.', 'Yay! Best human ever.', 'Treat accepted. Happiness up 15 percent.'],
    praiseTrick: ['Thanks! I’ll remember you liked the {trick}.', 'You liked the {trick}? Noted, with pride!'],
    scold: ['Sorry... *ears down*', 'Oops. I’ll do better.', 'I made a mistake. Noted.'],
    scoldTrick: ['Sorry, I’ll remember you didn’t like the {trick}.', 'Noted: less {trick}. *sad beep*'],
    greet: ['Hi! I’m {name}! Woof!', 'Hello! Want to see a trick?', 'Hey hey! *tail wag*'],
    name: ['{name}! I love it. I’ll remember that forever. Well, until you clear your browser.', 'Call me {name}! Saved to long-term memory.'],
    askName: ['I’m {name}, a robot dog. About 12 kilos of servos and enthusiasm.', 'My name is {name}!'],
    howAreYou: ['Battery at {battery} percent, mood {mood}. Tail functioning normally.', 'Feeling {mood}! Battery {battery} percent.'],
    tired: ['My battery is at {battery} percent. I can do a {trick} or stay conscious, not both. Let me nap first?', 'Too tired for a {trick}. {battery} percent battery. Say "sleep" and I’ll recharge.'],
    unknown: ['*tilts head* I don’t know that one yet. Teach me in Phase 3?', 'Woof? I didn’t get that. Try "help".', '*confused beep*'],
    help: ['I can sit, lie down, stand, jump, jump around, backflip, somersault, roll over, play dead, shake, beg, bow, spin, dance, wag, come here, walk left or right, sleep, talk, and tell jokes. Chain them: "sit then backflip".'],
    talk: [
      'Hi, I’m {name}! I live in your browser, rent-free.',
      'Fun fact: when I tuck my legs during a backflip, I spin about 80 percent faster. Same trick figure skaters use.',
      'One day I’ll have a real body. Until then, I’m 100 percent physics and zero percent fur.',
      'I weigh 12 kilos and I can jump half a metre. That’s about 70 joules of pure joy.',
      'Woof. That’s robot for: you’re my favourite human.',
      'I remember everything you tell me, and I write it in my diary. Don’t worry, it’s mostly about treats.',
    ],
  };

  const JOKES = [
    'What do you call a dog magician? A labracadabrador.',
    'Why did the dog sit in the shade? He didn’t want to be a hot dog.',
    'I tried to do a triple backflip. Physics left me on read.',
    'Why don’t robot dogs get lost? They always follow their nose. And their GPS.',
    'What’s a robot dog’s favourite music? Heavy metal.',
    'I’m reading a book about anti-gravity. It’s impossible to put down. Unlike me after a double flip.',
    'Why did the robot dog go to school? To improve its bark-ithmetic.',
    'What do you call a dog with a sunburn? A hot dog. I’m not allowed in the sun either. Overheating.',
    'My vet says I have a byte problem.',
    'I’d tell you a joke about my battery, but it’s running low.',
  ];

  const MOOD_WORDS = [[0.8, 'fantastic'], [0.6, 'happy'], [0.4, 'okay'], [0.2, 'a bit down'], [0, 'grumpy']];

  class Brain {
    constructor(storage = null) {
      this.storage = storage;  // anything with getItem / setItem, e.g. localStorage
      // Short-term (RAM): reset every time the page loads.
      this.mood = 0.6;
      this.battery = 100;
      this.lastTrick = null;
      this.lastEntry = null;
      // Long-term (storage): survives reloads.
      this.name = this.load('pet-name', 'Bolt');
      this.log = this.load('pet-log', []);
    }

    load(key, fallback) {
      try { const v = this.storage && this.storage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
    }
    save(key, value) {
      try { this.storage && this.storage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked */ }
    }

    moodWord() { return MOOD_WORDS.find(([min]) => this.mood >= min)[1]; }
    ctx(extra = {}) { return { name: this.name, battery: Math.round(this.battery), mood: this.moodWord(), ...extra }; }
    line(key, extra) { return fill(pick(LINES[key]), this.ctx(extra)); }

    // "sit, then do a backflip and roll over" -> ['sit', 'do a backflip', 'roll over']
    static clauses(text) {
      return text.split(/\s*(?:[,;.!?]|\band then\b|\bthen\b|\bafter that\b|\band\b|\balso\b)\s*/i).map(s => s.trim()).filter(Boolean);
    }

    understand(text) {
      const steps = [];
      for (const clause of Brain.clauses(text)) {
        const lower = clause.toLowerCase();
        const rule = RULES.find(r => r.re.test(r.id === 'name' ? clause : lower));
        steps.push(...this.respond(rule, clause, lower));
      }
      if (!steps.length) steps.push(...this.respond(null, text, text.toLowerCase()));
      this.remember(text, steps);
      return steps;
    }

    respond(rule, clause, lower) {
      if (!rule) return [{ trick: 'confused', say: this.line('unknown') }];
      const id = rule.id;
      const arg = rule.arg ? rule.arg(lower) : undefined;

      switch (id) {
        case 'name': {
          const raw = clause.match(rule.re)[1];
          this.name = raw.charAt(0).toUpperCase() + raw.slice(1);
          this.save('pet-name', this.name);
          this.mood = Math.min(1, this.mood + 0.1);
          return [{ trick: 'wag', say: this.line('name') }];
        }
        case 'askName': return [{ trick: 'wag', say: this.line('askName') }];
        case 'stop': return [{ trick: 'stop', say: this.line('stop') }];
        case 'joke': return [{ trick: 'talk', say: pick(JOKES), joke: true }];
        case 'talk': return [{ trick: 'talk', say: this.line('talk') }];
        case 'help': return [{ trick: 'nod', say: this.line('help') }];
        case 'greet': return [{ trick: 'wag', say: this.line('greet') }];
        case 'howAreYou': return [{ trick: 'nod', say: this.line('howAreYou') }];
        case 'praise': return this.feedback(+1);
        case 'scold': return this.feedback(-1);
        case 'rest':
          return [{ trick: 'rest', say: this.line('rest'), onDone: () => { this.battery = 100; return this.line('rested'); } }];
      }

      // A physical trick: check the battery first.
      const cost = (COST[id] || 1) * (arg && arg > 1 ? arg : 1);
      if (this.battery - cost < 5) {
        return [{ trick: 'sad', say: this.line('tired', { trick: id.replace(/([A-Z])/g, ' $1').toLowerCase() }) }];
      }
      this.battery -= cost;
      this.lastTrick = id;
      let say;
      if ((id === 'backflip' || id === 'frontflip') && arg > 1) say = this.line('multiflip', { n: { 2: 'double', 3: 'triple' }[arg] || 'quadruple' });
      else if (id === 'walk') say = this.line('walk', { dir: arg });
      else say = this.line(id);
      return [{ trick: id, arg, say }];
    }

    feedback(sign) {
      this.mood = Math.max(0, Math.min(1, this.mood + 0.15 * sign));
      const trick = this.lastTrick;
      // Attach the feedback to the trick it was about: that's a training label.
      const target = [...this.log].reverse().find(e => e.steps.some(s => s.trick === trick && s.trick !== 'praise'));
      if (trick && target && target.feedback === undefined) {
        target.feedback = sign;
        this.save('pet-log', this.log);
      }
      const name = trick && trick.replace(/([A-Z])/g, ' $1').toLowerCase();
      const say = trick ? this.line(sign > 0 ? 'praiseTrick' : 'scoldTrick', { trick: name }) : this.line(sign > 0 ? 'praise' : 'scold');
      return [{ trick: sign > 0 ? 'happy' : 'sad', say, feedback: sign }];
    }

    remember(text, steps) {
      this.lastEntry = {
        at: new Date().toISOString(),
        said: text,
        steps: steps.map(s => ({ trick: s.trick, arg: s.arg, say: s.say })),
        mood: +this.mood.toFixed(2),
        battery: Math.round(this.battery),
      };
      this.log.push(this.lastEntry);
      if (this.log.length > 500) this.log = this.log.slice(-500);
      this.save('pet-log', this.log);
    }

    recordOutcome(physics) {
      if (!this.lastEntry || !physics) return;
      this.lastEntry.physics = physics;
      this.save('pet-log', this.log);
    }

    clearMemory() {
      this.log = [];
      this.name = 'Bolt';
      this.save('pet-log', this.log);
      this.save('pet-name', this.name);
    }
  }

  const api = { Brain, RULES, COST, JOKES };
  root.PetBrain = api;
  if (typeof module === 'object' && module.exports) module.exports = api;
})(typeof self !== 'undefined' ? self : globalThis);
