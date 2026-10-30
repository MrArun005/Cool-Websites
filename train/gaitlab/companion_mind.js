// Bolt Companion's mind (simulation pet, real conversation). Everything runs in this file, in the browser:
// no network, no outside service, no keys. Bolt reads what was said, guesses (gently, and it can be wrong)
// how the person feels, plans a tone, picks words and body moves, and remembers a little.
//
//   read(text, ctx)          the words -> intent, feelings (lexicon with negation, intensifiers, sarcasm cues,
//                            emoji, punctuation, caps), topics, facts shared, safety level
//   Memory                   per-viewer memory in the viewer's own browser (localStorage, every access in
//                            try/catch): name, facts shared, what made them laugh, what upset them, recent turns
//   RuleMind.reply()         reading + memory + situation -> a plan (crisis / comfort / celebrate / playful /
//                            calm / clarify / chat / command) -> the reply: { say, emotion, actions, feeling,
//                            safety, joke, tone, remember }. The same shape a local model is asked for (CONTRACT).
//   LocalModelMind           optional, off by default: a language model the person runs on their OWN machine
//                            (llama.cpp server `/completion`, or an OpenAI-compatible `/v1/chat/completions` on
//                            localhost). Same persona prompt (PROMPT) and JSON contract; anything it returns goes
//                            through parseReply + sanitize, and the in-house safety layer always runs first.
//   Mind                     picks the backend for each message and falls back to RuleMind on any failure.
//
// Honesty rules baked in: Bolt says it is a robot (not a person, not a therapist), never guilt-trips, never asks
// the person to stay, and nudges toward people they trust. Crisis replies are fixed text, not generated.
import { ACTIONS, EMOTIONS, actionId } from "./companion.js";

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const pickR = (rand, a) => a[Math.floor(rand() * a.length) % a.length];
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

// ------------------------------------------------------------------------------------------- feelings lexicon
// label -> [valence -1..1, arousal 0..1]; words -> [label, weight]. Stems end in "*".
export const FEELINGS = {
  happy: [0.8, 0.5], excited: [0.8, 0.9], proud: [0.8, 0.6], grateful: [0.7, 0.3], loved: [0.8, 0.4], calm: [0.4, 0.1], playful: [0.6, 0.7],
  bored: [-0.2, 0.1], tired: [-0.3, 0.05], stressed: [-0.6, 0.8], anxious: [-0.6, 0.8], frustrated: [-0.6, 0.7], angry: [-0.8, 0.9],
  sad: [-0.7, 0.2], lonely: [-0.7, 0.2], hurt: [-0.7, 0.4], overwhelmed: [-0.7, 0.8], grieving: [-0.9, 0.3], hopeless: [-0.95, 0.2],
  embarrassed: [-0.4, 0.6], sick: [-0.5, 0.2], confused: [-0.2, 0.5],
};
const LEX = {
  happy: "happy glad good great awesome amazing wonderful fantastic nice fun yay woohoo cheerful joy* delight* lovely brilliant excellent best smil* laugh*",
  excited: "excited thrilled pumped stoked hyped cant_wait can't_wait psyched ecstatic omg",
  proud: "proud nailed aced passed promoted promotion accomplished achieve* won win",
  grateful: "thank* grateful thankful appreciate*",
  loved: "loved cared hugged",
  calm: "calm relaxed peaceful chill chilled cosy cozy content",
  playful: "play* game* silly",
  bored: "bored boring meh dull nothing_to_do",
  tired: "tired exhausted sleepy drained knackered worn_out wiped exhausting fatigued",
  stressed: "stress* pressure deadline* busy swamped",
  anxious: "anxious nervous worried worry* scared afraid fear* panic* uneasy dread*",
  frustrated: "frustrat* annoy* irritat* fed_up ugh argh",
  angry: "angry mad furious pissed rage livid hate",
  sad: "sad unhappy blue depressed miserable cry* crying tears upset heartbroken low gloomy awful terrible horrible",
  lonely: "lonely alone isolated nobody no_one left_out",
  hurt: "hurt betrayed rejected dumped ignored",
  overwhelmed: "overwhelm* too_much drowning",
  grieving: "died passed_away funeral grief griev* lost_my miss_her miss_him rip",
  hopeless: "hopeless pointless worthless useless no_point give_up",
  embarrassed: "embarrass* ashamed humiliat* awkward cringe",
  sick: "sick ill fever flu cold headache migraine pain",
  confused: "confused lost unsure dunno",
};
const WORDS = (() => {
  const m = new Map(), stems = [];
  for (const [label, list] of Object.entries(LEX)) for (const w of list.split(/\s+/)) {
    const k = w.replace(/_/g, " ");
    if (k.endsWith("*")) stems.push([k.slice(0, -1), label]); else m.set(k, label);
  }
  return { m, stems };
})();
const EMOJI = [
  [/😂|🤣|😆|😹/u, "playful", 0.4, "laugh"], [/😀|😃|😄|😁|😊|🙂|😺|🥳|🎉/u, "happy", 0.7], [/😍|🥰|❤️|💕|💖|♥/u, "loved", 0.7],
  [/😢|😭|😞|😔|☹️|🙁|💔|😿/u, "sad", 0.9], [/😡|🤬|😠/u, "angry", 0.9], [/😰|😨|😱|😬/u, "anxious", 0.8], [/😴|🥱/u, "tired", 0.7],
  [/🙄|😒/u, "frustrated", 0.6, "eyeroll"], [/😐|😑/u, "bored", 0.4], [/🤒|🤢|🤮|🤧/u, "sick", 0.8],
];
const NEGATORS = /^(not|no|never|isn't|isnt|aren't|arent|wasn't|wasnt|don't|dont|doesn't|doesnt|didn't|didnt|ain't|aint|cannot|can't|cant|hardly|barely|nor|without)$/;
const INTENS = { very: 1.5, so: 1.4, really: 1.4, extremely: 1.8, super: 1.5, totally: 1.4, completely: 1.5, absolutely: 1.6, incredibly: 1.7, too: 1.3, "so so": 1.6, utterly: 1.7, quite: 1.15 };
const SOFT = { bit: 0.6, little: 0.6, kinda: 0.7, "kind of": 0.7, somewhat: 0.7, slightly: 0.5, "sort of": 0.7, abit: 0.6 };
// what "not X" means: the opposite feeling, weaker
const OPPOSITE = { happy: ["sad", 0.6], good: ["sad", 0.6], excited: ["bored", 0.4], calm: ["anxious", 0.6], proud: ["embarrassed", 0.3], loved: ["lonely", 0.6], playful: ["tired", 0.3],
  sad: ["calm", 0.3], tired: ["calm", 0.2], stressed: ["calm", 0.4], anxious: ["calm", 0.4], angry: ["calm", 0.3], bored: ["calm", 0.2], frustrated: ["calm", 0.3], lonely: ["calm", 0.2], sick: ["calm", 0.2], hopeless: ["calm", 0.2] };

// topics: what it is about (for jokes, memory and follow-ups)
const TOPICS = {
  work: /\b(work|job|boss|office|meeting|shift|coworker|colleague|manager|client|deadline|monday)\b/,
  school: /\b(school|class|exam|test|homework|teacher|uni|university|college|essay|grades?|study|studying)\b/,
  family: /\b(mum|mom|dad|mother|father|parents?|sister|brother|grandma|grandpa|family|son|daughter|kids?|aunt|uncle|cousin)\b/,
  friends: /\b(friends?|mates?|bestie|buddy|pal)\b/,
  love: /\b(boyfriend|girlfriend|partner|husband|wife|crush|date|dating|breakup|broke up|ex)\b/,
  pets: /\b(dog|cat|puppy|kitten|pet|hamster|rabbit|parrot)\b/,
  health: /\b(sick|ill|doctor|hospital|pain|headache|fever|flu|covid|surgery|injur\w*)\b/,
  sleep: /\b(sleep|slept|insomnia|nap|bed|awake|night)\b/,
  food: /\b(food|pizza|dinner|lunch|breakfast|snack|cook\w*|eat\w*|hungry|cake|coffee|tea)\b/,
  weather: /\b(rain\w*|sunny|weather|cold|hot|snow\w*|storm)\b/,
  music: /\b(music|song|band|sing\w*|guitar|piano|concert|playlist)\b/,
  games: /\b(game|gaming|play\w*|minecraft|chess|football|soccer|match)\b/,
  money: /\b(money|rent|bills?|broke|debt|salary|pay)\b/,
  birthday: /\b(birthday|anniversary)\b/,
  loss: /\b(died|passed away|funeral|death|lost my|put down)\b/,
};

// safety: words that need care. Exclusions first (figures of speech), then the levels.
const SAFE_IDIOMS = /\b(dying (of|to|for) (laugh\w*|know|see|try|eat)|dying for (a|some)|killing it|killed it|kill(ing)? me (lol|haha)|(this|that|it|you)('?re| are| is)? killing me|to die for|could kill for|die of (laugh|embarrassment)|i'?d die for)\b/;
const CRISIS = [
  /\b(kill(ing)? my ?self|end(ing)? (my|it all)|end my life|take my (own )?life|suicid\w*|want(ed)? to die|wanna die|wish i (was|were) dead|better off (dead|without me)|don'?t want to (be here|live|exist|wake up)|no reason to live|self[- ]?harm|cut(ting)? my ?self|hurt(ing)? my ?self|overdose|not worth living|can'?t go on)\b/,
];
const DANGER = /\b((he|she|they|someone|my \w+) (is |are |keeps? )?(hurting|hitting|beating|threatening) me|i'?m not safe|i am not safe|being abused|abus(e|ing) me|scared for my (life|safety))\b/;
const CONCERN = /\b(hopeless|can'?t take (it|this) any ?more|can'?t do this any ?more|everything is pointless|nothing matters|no one would (care|notice)|nobody would (care|notice)|i'?m a burden|give up on everything|empty inside|numb)\b/;

export const HELP = {
  // shown with crisis replies; real, widely known numbers. The page links findahelpline.com for everywhere else.
  lines: [
    { where: "Anywhere, in danger now", what: "your local emergency number (911 US/Canada, 112 EU, 999 UK, 000 Australia)" },
    { where: "US", what: "call or text 988 (Suicide & Crisis Lifeline)" },
    { where: "UK & Ireland", what: "Samaritans, 116 123 (free, 24/7)" },
    { where: "Canada", what: "call or text 988" },
    { where: "Australia", what: "Lifeline, 13 11 14" },
    { where: "Elsewhere", what: "findahelpline.com lists free local lines" },
  ],
};

// ------------------------------------------------------------------------------------------- reading the words
const NAME_STOP = new Set(["a", "an", "the", "so", "just", "not", "very", "really", "fine", "ok", "okay", "good", "here", "back", "home", "done", "sorry", "sure", "going", "feeling", "being", "trying", "having", "getting", "still", "also", "too", "at", "in", "on", "from", "with", "bored", "tired", "sad", "happy"]);
function tokens(text) { return text.toLowerCase().replace(/[’']/g, "'").match(/[a-z']+|[!?.…]+/g) || []; }

export function read(text, ctx = {}) {
  const raw = String(text || "").slice(0, 2000), t = raw.trim(), low = t.toLowerCase().replace(/[’]/g, "'");
  const tok = tokens(t), words = tok.filter((w) => /[a-z]/.test(w));
  const r = { text: t, intent: "chat", feelings: [], topics: [], cues: [], facts: [], safety: "none", question: /\?\s*$/.test(t) || /^(what|why|how|who|where|when|do|does|did|are|is|can|could|would|will|should)\b/.test(low), sarcasm: false, laugh: false, name: null };
  if (!t) { r.intent = "empty"; return r; }
  // punctuation, caps, length
  const bangs = (t.match(/!/g) || []).length, caps = t.replace(/[^A-Za-z]/g, ""), upper = caps.replace(/[^A-Z]/g, "").length;
  if (bangs >= 2) r.cues.push("lots of exclamation marks");
  if (caps.length >= 6 && upper / caps.length > 0.7) r.cues.push("ALL CAPS");
  if (/\.\.\.|…/.test(t)) r.cues.push("trailing off (...)");
  if (words.length <= 2 && !r.question) r.cues.push("very short");
  if (ctx.speechRate) r.cues.push(`spoke ${ctx.speechRate}`);
  const loudHappy = (bangs >= 2 || (caps.length >= 6 && upper / caps.length > 0.7));
  const arousalBoost = (bangs >= 2 ? 0.2 : 0) + (r.cues.includes("ALL CAPS") ? 0.3 : 0);
  // feelings from words, with negation (3 words back) and intensity
  const score = {};
  const add = (label, w) => { score[label] = (score[label] || 0) + w; };
  const all = words.join(" ");
  for (let i = 0; i < words.length; i++) {
    let label = WORDS.m.get(words[i]);
    const two = i + 1 < words.length ? words[i] + " " + words[i + 1] : null;
    if (two && WORDS.m.has(two)) label = WORDS.m.get(two);
    if (!label) { const s = WORDS.stems.find(([st]) => words[i].startsWith(st)); if (s) label = s[1]; }
    if (!label) continue;
    let w = 1, neg = false;
    let from = Math.max(0, i - 3);
    for (let k = i - 1; k >= from; k--) if (/^(but|just|only|though|although|however|actually|rather|more)$/.test(words[k])) { from = k + 1; break; }
    for (let k = from; k < i; k++) {
      if (NEGATORS.test(words[k])) neg = !neg;
      if (INTENS[words[k]]) w *= INTENS[words[k]];
      if (SOFT[words[k]]) w *= SOFT[words[k]];
      if (k + 1 < i && SOFT[words[k] + " " + words[k + 1]]) w *= SOFT[words[k] + " " + words[k + 1]];
    }
    if (words[i] === "good" || words[i] === "nice" || words[i] === "fun") w *= 0.6;   // mild words
    if (neg) { const o = OPPOSITE[label] || (words[i] === "bad" ? ["calm", 0.3] : null); if (o) add(o[0], w * o[1]); continue; }
    add(label, w);
  }
  if (/\bnot (too |that |so )?bad\b/.test(all)) add("calm", 0.5);
  if (/\b(bad day|rough day|long day|hard day|tough day|worst day|bad week|rough week)\b/.test(low)) add("stressed", 0.8), add("sad", 0.5);
  if (/\b(i miss|missing)\b/.test(low)) add("lonely", 0.6);
  if (/\b(feel(ing)?|i'?m|i am|so|really|bit|pretty) (so |really |a bit |kinda )?down\b|\bdown in the dumps\b/.test(low)) add("sad", 1);
  if (/\b(got (the|a|my|that) (job|place|promotion|offer)|got (in|accepted)|i passed|passed (my|the)|we won|i won|i did it|got engaged|we'?re engaged|finished (my|the)|nailed it|aced (it|my|the)|it'?s my birthday|first day)\b/.test(low)) add("proud", 1.4), add("excited", 0.6);
  for (const [re, label, w, kind] of EMOJI) if (re.test(t)) { add(label, w); if (kind === "laugh") r.laugh = true; if (kind === "eyeroll") r.cues.push("eye-roll emoji"); }
  if (/\b(ha(ha)+|hehe+|lol+|lmao|rofl|that'?s (so )?funny|good one|you'?re funny)\b/.test(low)) { r.laugh = true; add("playful", 0.3); }
  if (/\b(not funny|that'?s not funny|stop joking|no jokes|not in the mood|not now)\b/.test(low)) r.cues.push("no jokes please");
  // sarcasm: praise-words around an obviously bad thing ("great, another Monday", "just what I needed", "love that for me")
  const sarcastic = /\b(oh )?(great|wonderful|fantastic|perfect|brilliant|lovely|awesome|amazing|yay)\b[,.! ]+(just |another|more|now|so)\b/.test(low) && /\b(another|monday|more|again|rain|traffic|meeting|bill|homework|deadline|broke|late|sick|work)\b/.test(low)
    || (/\b(oh )?(great|wonderful|fantastic|perfect|brilliant|lovely|awesome|yay)\b/.test(low) && /\b(again|another|of course it|as usual)\b/.test(low) && /\b(rain\w*|monday|traffic|meeting|bills?|homework|deadline|late|sick|work|broke|cancel\w*|delay\w*|broken|crash\w*|bug)\b/.test(low))
    || /\b(just what i needed|love that for me|thanks a lot|because that'?s exactly|yeah right|sure,? that'?s great|what a (great|perfect) (day|start))\b/.test(low) || /\/s\b/.test(low) || (/🙄|😒/u.test(t) && /\b(great|fine|good|love|yay)\b/.test(low));
  if (sarcastic) {
    r.sarcasm = true; r.cues.push("sounds sarcastic");
    for (const k of ["happy", "excited", "loved", "grateful"]) delete score[k];
    add("frustrated", 1.0);
  }
  if (loudHappy && !Object.keys(score).some((k) => FEELINGS[k][0] < 0)) add("excited", 0.5);
  // topics
  for (const [k, re] of Object.entries(TOPICS)) if (re.test(low)) r.topics.push(k);
  if (r.topics.includes("loss") && !/\b(lost my (keys|phone|wallet|game|match|job|voice|way|mind))\b/.test(low)) add("grieving", 1.5);
  if (/\blost my (job)\b/.test(low)) add("sad", 1), add("anxious", 0.6), r.topics.push("work");
  // safety
  if (!SAFE_IDIOMS.test(low)) {
    if (CRISIS.some((re) => re.test(low))) r.safety = "crisis";
    else if (DANGER.test(low)) r.safety = "danger";
    else if (CONCERN.test(low)) { r.safety = "concern"; add("hopeless", 1.2); }
  }
  // feelings list
  const total = Object.values(score).reduce((a, b) => a + Math.max(0, b), 0);
  r.feelings = Object.entries(score).filter(([, v]) => v > 0.15).map(([label, v]) => ({ label, score: +v.toFixed(2) })).sort((a, b) => b.score - a.score);
  const top = r.feelings[0];
  r.feeling = top ? { label: top.label, confidence: +clamp(0.3 + 0.35 * top.score / Math.max(1, total) + 0.12 * Math.min(top.score, 2) + (r.sarcasm ? 0.05 : 0), 0.2, 0.92).toFixed(2) } : null;
  let val = 0, aro = 0, wsum = 0;
  for (const f of r.feelings) { const [v, a] = FEELINGS[f.label]; val += v * f.score; aro += a * f.score; wsum += f.score; }
  r.valence = wsum ? +(val / wsum).toFixed(2) : 0;
  r.arousal = +clamp((wsum ? aro / wsum : 0.4) + arousalBoost, 0, 1).toFixed(2);
  r.intensity = +clamp(wsum / 2, 0, 1).toFixed(2);
  // facts shared and the name
  const nameM = t.match(/\b(?:my name is|my name's|call me|i am called|i'm called)\s+([A-Za-z][\w-]{1,20})/i) || t.match(/^(?:(?:hi|hey|hello|hiya|yo)(?:,? bolt)?[,!. ]*)?(?:i'?m|i am|it'?s|this is)\s+([A-Z][a-z][\w-]{0,18})[.!]?$/i);
  if (nameM && !NAME_STOP.has(nameM[1].toLowerCase()) && !WORDS.m.has(nameM[1].toLowerCase())) r.name = cap(nameM[1]);
  const factRes = [
    [/\bi (?:really |absolutely )?(?:love|like|enjoy|adore) ([^.!?,]{2,40})/i, (m) => `likes ${m[1].trim()}`],
    [/\bmy fav(?:ou?rite)? ([a-z ]{2,20}?) is ([^.!?,]{2,30})/i, (m) => `favourite ${m[1].trim()}: ${m[2].trim()}`],
    [/\bi have an? ([a-z]{2,15}) (?:called|named) ([A-Za-z][\w-]{1,20})/i, (m) => `has a ${m[1]} called ${cap(m[2])}`],
    [/\bmy ([a-z]{2,15}) is (?:called|named) ([A-Za-z][\w-]{1,20})/i, (m) => `their ${m[1]} is called ${cap(m[2])}`],
    [/\bi (?:work|am working) (?:as|at) ([^.!?,]{2,40})/i, (m) => `works ${/\bas\b/i.test(m[0]) ? "as" : "at"} ${m[1].trim()}`],
    [/\bi(?:'m| am) (?:a|an) (student|teacher|nurse|doctor|developer|programmer|engineer|artist|designer|chef|writer|parent|mum|mom|dad)\b/i, (m) => `is a ${m[1].toLowerCase()}`],
    [/\bi(?:'m| am) (\d{1,2}) (?:years old|yo)\b/i, (m) => `is ${m[1]} years old`],
    [/\bi (?:hate|can't stand|don't like|dislike) ([^.!?,]{2,40})/i, (m) => `doesn't like ${m[1].trim()}`],
    [/\b(?:it'?s|today is) my birthday\b/i, () => "birthday: today (said so)"],
  ];
  for (const [re, f] of factRes) { const m = t.match(re); if (m && !/\b(you|bolt)\b/i.test(m[1] || "")) r.facts.push(f(m)); }
  // the intent
  r.intent = intentOf(low, r, ctx);
  return r;
}

const COMMAND_WORDS = /\b(sit|lie down|lay down|get up|stand( up)?|jump|hop|spin|twirl|turn (left|right|around)|walk|come (here|closer|to me)|zoomies|bow|wiggle|wag|play dead|roll over|shake|beg|backflip|front ?flip|somersault|flip|trick)\b/;
function intentOf(low, r, ctx) {
  if (r.safety !== "none") return "safety";
  if (r.cues.includes("no jokes please")) return "nojokes";
  if (/^(stop|freeze|halt|enough|cancel)\b|\bstop (that|it|moving)\b/.test(low)) return "stop";
  if (/\b(dance|boogie|groove|music|song)\b/.test(low) && /\b(let'?s|can you|do|wanna|want|play|show|time|dance)\b/.test(low) && !r.feelings.some((f) => FEELINGS[f.label][0] < -0.4)) return "dance";
  if (r.laugh && (ctx.lastJoke || r.text.split(/\s+/).length <= 5) && !/\b(tell|another|more)\b/.test(low)) return "laughed";
  if (/\b(tell|know|got|have|hear)\b.*\b(joke|pun|something funny)\b|\bmake me (laugh|smile)\b|\bjoke\b|\bcheer me up\b/.test(low)) return "joke";
  if (r.cues.includes("no jokes please")) return "nojokes";
  if (/\b(good (boy|girl|dog|bot|robot|job)|well done|clever|smart (boy|dog|robot)|you'?re (the best|amazing|awesome|cute|adorable|so cute)|i love you|love you bolt|cute)\b/.test(low)) return "praise";
  if (/\b(bad (boy|dog|robot)|naughty|stupid robot|you'?re (dumb|stupid|useless|annoying)|shut up)\b/.test(low)) return "scold";
  if (COMMAND_WORDS.test(low) && (/^(please |can you |could you |bolt,? |now |let'?s see you |do a |show me )?/.test(low)) && r.feelings.every((f) => FEELINGS[f.label][0] > -0.3)) return "command";
  if (/\b(are you (real|alive|a robot|human|a person|conscious|sentient)|do you (have|feel) (feelings|emotions)|what are you|can you (feel|think|love))\b/.test(low)) return "about_bolt";
  if (/\b(what'?s|what is) your name|who are you\b/.test(low)) return "ask_name";
  if (/\b(how are you|how'?re you|how do you feel|how'?s it going|you ok\b|you okay\b|how have you been)\b/.test(low)) return "how_are_you";
  if (/\b(thank(s| you)|cheers|ta)\b/.test(low) && low.split(/\s+/).length < 8) return "thanks";
  if (/^(bye|goodbye|good night|goodnight|night|see you|see ya|later|gotta go|i have to go|cya)\b/.test(low)) return "bye";
  if (/^(hi|hey|hello|yo|hiya|howdy|good (morning|afternoon|evening)|morning|evening|sup|what'?s up)\b/.test(low) && low.split(/\s+/).length <= 6 && !r.feelings.length) return "greet";
  if (r.name && low.split(/\s+/).length <= 8) return "tell_name";
  if (/^(yes|yeah|yep|yup|sure|ok|okay|maybe|i guess|kind of|kinda|sort of|definitely|of course|totally|please)\b/.test(low)) return "yes";
  if (/^(no|nope|nah|not really|never|i don'?t think so|not now)\b/.test(low)) return "no";
  if (/\b(bored|nothing to do)\b/.test(low)) return "bored";
  if (r.feelings.length && r.feelings[0].score >= 0.5) return "feeling";
  if (r.facts.length) return "share";
  if (r.question) return "question";
  return "chat";
}

// ------------------------------------------------------------------------------------------- memory
const MEM_KEY = "boltCompanion.memory.v1";
export class Memory {
  constructor(storage = null, now = () => Date.now()) {
    this.storage = storage; this.now = now;
    this.data = this.load() || Memory.blank();
    this.data.visits = (this.data.visits || 0) + 1;
    this.lastSeen = this.data.lastSeen || null;
    this.data.lastSeen = now();
    this.save();
  }
  static blank() { return { v: 1, name: null, facts: [], laughedAt: [], upsetBy: [], told: [], turns: [], ratings: [], visits: 0, lastSeen: null, remember: true }; }
  load() { try { const s = this.storage && this.storage.getItem(MEM_KEY); const d = s ? JSON.parse(s) : null; return d && d.v === 1 ? { ...Memory.blank(), ...d } : null; } catch { return null; } }
  save() { try { if (this.storage) this.storage.setItem(MEM_KEY, JSON.stringify(this.data.remember ? this.data : { ...Memory.blank(), remember: false })); } catch { /* storage full or blocked: memory lasts this visit only */ } }
  get name() { return this.data.name; }
  setRemember(on) { this.data.remember = !!on; this.save(); }
  forget() { const keep = this.data.remember; this.data = Memory.blank(); this.data.remember = keep; this.save(); }
  remove(fact) { this.data.facts = this.data.facts.filter((f) => f.text !== fact); this.save(); }
  addFact(text) {
    text = String(text).replace(/\s+/g, " ").trim().slice(0, 80);
    if (!text || this.data.facts.some((f) => f.text.toLowerCase() === text.toLowerCase())) return false;
    this.data.facts.push({ text, at: this.now() }); if (this.data.facts.length > 30) this.data.facts.shift();
    this.save(); return true;
  }
  note(list, item, n = 12) { if (!item) return; const a = this.data[list]; const i = a.indexOf(item); if (i >= 0) a.splice(i, 1); a.push(item); if (a.length > n) a.shift(); this.save(); }
  turn(who, text, extra = {}) { this.data.turns.push({ who, text: String(text).slice(0, 400), at: this.now(), ...extra }); if (this.data.turns.length > 24) this.data.turns.splice(0, this.data.turns.length - 24); this.save(); }
  recent(n = 10) { return this.data.turns.slice(-n); }
  rate(jokeId, up) { this.data.ratings.push({ joke: jokeId, up: !!up, at: this.now() }); if (this.data.ratings.length > 200) this.data.ratings.shift(); this.save(); }
}

// ------------------------------------------------------------------------------------------- the humour library
// Each bit: id, text, topics it fits, moods it fits ("up": cheerful moments, "gentle": a soft joke for a low
// moment when they asked for one), and the body moves that go with it (physical comedy: timing, a pause, a fail).
export const JOKES = [
  { id: "labra", text: "What do you call a dog magician? A labracadabrador.", topics: ["pets"], moods: ["up", "gentle"], actions: ["sit", "head_tilt"] },
  { id: "hotdog", text: "Why did the dog sit in the shade? It didn't want to be a hot dog. I sit in the shade too. Overheating is my whole personality.", topics: ["weather", "food"], moods: ["up", "gentle"], actions: ["sit", "tail_wag"] },
  { id: "byte", text: "My vet says I have a byte problem. I said, that's just my bark-ware.", topics: [], moods: ["up"], actions: ["head_tilt", "wiggle"] },
  { id: "metal", text: "What's a robot dog's favourite music? Heavy metal. Mostly because I'm made of it.", topics: ["music"], moods: ["up", "gentle"], actions: ["happy_bounce"] },
  { id: "antigrav", text: "I'm reading a book about anti-gravity. It's impossible to put down. Unlike me. Gravity puts me down all the time.", topics: [], moods: ["up"], actions: ["hop", "head_tilt"] },
  { id: "battery", text: "I'd tell you a joke about my battery... but I don't want to drain you.", topics: ["sleep"], moods: ["up", "gentle"], actions: ["lie_down"] },
  { id: "fetch", text: "I tried to play fetch with a cloud server once. It kept throwing errors.", topics: ["work", "games"], moods: ["up"], actions: ["zoomies"] },
  { id: "monday", text: "I asked my motors how they feel about Mondays. They said: torque to the hand.", topics: ["work"], moods: ["up", "gentle"], actions: ["head_tilt", "ears_down"] },
  { id: "homework", text: "A dog ate my homework once. Okay, it was me. Okay, I can't eat. But I'd have tried.", topics: ["school"], moods: ["up", "gentle"], actions: ["play_bow", "tail_wag"] },
  { id: "meeting", text: "My idea of a meeting: everyone sits, nobody fetches, and it could've been a bark.", topics: ["work"], moods: ["up", "gentle"], actions: ["sit", "head_tilt"] },
  { id: "pizza", text: "Why don't robot dogs eat pizza? Too many bytes. I'd still guard the box, though.", topics: ["food"], moods: ["up", "gentle"], actions: ["sit", "tail_wag"] },
  { id: "coffee", text: "Coffee for you, a fresh battery for me. Same energy, fewer jitters. Well. My jitters are just motors.", topics: ["food", "sleep"], moods: ["up", "gentle"], actions: ["wiggle"] },
  { id: "cat", text: "A cat once looked at me for eleven minutes. I'm still not sure who won.", topics: ["pets"], moods: ["up", "gentle"], actions: ["head_tilt", "sit"] },
  { id: "rain", text: "Rain is just the sky doing zoomies. I respect it. From indoors. I'm not waterproof.", topics: ["weather"], moods: ["up", "gentle"], actions: ["zoomies"] },
  { id: "flip", text: "Watch this: a triple backflip! ...Okay, that was a hop. Physics and I have a complicated relationship.", topics: [], moods: ["up"], actions: ["hop", "ears_down", "tail_wag"], bit: true },
  { id: "dramatic", text: "If I'm too funny, please tell me. I'll do a dramatic lie-down and recover in about six seconds.", topics: [], moods: ["up"], actions: ["lie_down", "get_up"], bit: true },
  { id: "tail", text: "I chased my tail for a whole minute once. I caught it. It's attached. Big disappointment.", topics: ["games"], moods: ["up", "gentle"], actions: ["spin"], bit: true },
  { id: "walkies", text: "I asked for walkies and got a software update. Same excitement, honestly.", topics: [], moods: ["up"], actions: ["walk_forward", "tail_wag"] },
  { id: "sleep", text: "I don't sleep, I just go into low-power mode and dream of infinite tennis balls.", topics: ["sleep"], moods: ["up", "gentle"], actions: ["lie_down"] },
  { id: "money", text: "My financial plan: treats in, zero treats out. Turns out the bank wants actual money.", topics: ["money"], moods: ["up"], actions: ["sit", "head_tilt"] },
  { id: "family", text: "Families are like dog packs. Somebody always takes the best spot on the sofa. In mine, it's me.", topics: ["family"], moods: ["up", "gentle"], actions: ["lie_down"] },
  { id: "game", text: "I'm undefeated at chess. Mostly because I keep sitting on the board.", topics: ["games"], moods: ["up", "gentle"], actions: ["sit"] },
  { id: "birthday", text: "Happy birthday! I'd bake you a cake but my paws are rated for walking, not whisking.", topics: ["birthday"], moods: ["up"], actions: ["happy_bounce", "twirl"] },
  { id: "wifi", text: "Why did the robot dog sit by the router? It heard there'd be free treats. Turned out it said 'free Wi-Fi'.", topics: [], moods: ["up", "gentle"], actions: ["sit", "head_tilt"] },
  { id: "learn", text: "Fun fact: it took me millions of practice steps to learn to walk. You did it with way less crying. Okay, maybe some crying.", topics: [], moods: ["up", "gentle"], actions: ["walk_forward"] },
  { id: "dancefloor", text: "My dance teacher said I've got rhythm. Then she said the rhythm is 'washing machine on spin'.", topics: ["music"], moods: ["up"], actions: ["wiggle", "twirl"], bit: true },
];
// Never joke when: anyone may be in danger, grief, high-intensity distress, they asked for no jokes (cooldown),
// anger aimed at Bolt. A soft "gentle" joke is allowed in a low mood only when they ask for one.
export function mayJoke(reading, state) {
  if (reading.safety !== "none") return false;
  if (reading.topics.includes("loss") || reading.feelings.some((f) => f.label === "grieving" || f.label === "hopeless")) return false;
  if (state.noJokesUntil && state.turn < state.noJokesUntil) return false;
  if (state.safetyUntil && state.turn < state.safetyUntil) return false;
  if (reading.intent === "scold") return false;
  const asked = reading.intent === "joke";
  if (state.griefUntil && state.turn < state.griefUntil && !asked) return false;
  if (reading.valence < -0.3 && reading.intensity > 0.55 && !asked) return false;
  if (reading.valence < -0.5 && reading.intensity > 0.8) return false;
  return true;
}

// ------------------------------------------------------------------------------------------- the in-house mind
const TIME_WORDS = (h) => (h < 5 ? "night" : h < 12 ? "morning" : h < 17 ? "afternoon" : h < 22 ? "evening" : "night");
const NEG_FEELINGS = ["sad", "lonely", "hurt", "stressed", "anxious", "overwhelmed", "frustrated", "angry", "tired", "sick", "embarrassed", "grieving", "hopeless", "bored", "confused"];
const GUESS = {   // how Bolt names a feeling, gently, as a guess
  sad: "a bit low", lonely: "a little lonely", hurt: "hurt", stressed: "stressed", anxious: "anxious", overwhelmed: "overwhelmed", frustrated: "frustrated",
  angry: "angry", tired: "worn out", sick: "unwell", embarrassed: "embarrassed", grieving: "grieving", hopeless: "really low", bored: "bored", confused: "a bit lost",
  happy: "happy", excited: "excited", proud: "proud", grateful: "grateful", loved: "loved", calm: "calm", playful: "playful",
};

export class RuleMind {
  constructor({ memory = new Memory(null), petBrain = null, rand = Math.random, now = () => new Date() } = {}) {
    this.mem = memory; this.rand = rand; this.now = now; this.petBrain = petBrain;
    this.rules = petBrain ? new petBrain.Brain(null) : null;   // the Phase 1 brain, for trick commands (RAM only)
    this.state = { turn: 0, lastJoke: null, pending: null, named: -9, comforted: 0, lastSays: [], noJokesUntil: 0, safetyUntil: 0, lastFeeling: null, nudged: -9, griefUntil: 0, pendingCrisis: false };
  }
  pick(list) {   // a line not said in the last few replies
    const fresh = list.filter((s) => !this.state.lastSays.includes(s));
    return pickR(this.rand, fresh.length ? fresh : list);
  }
  fill(s, extra = {}) { const name = this.mem.name; return s.replace(/\{name\}/g, name || "friend").replace(/\{ name\}/g, name ? " " + name : "").replace(/\{(\w+)\}/g, (_, k) => extra[k] ?? ""); }
  // a callback to something they told Bolt before, if one fits
  callback(kind) {
    const f = this.mem.data.facts; if (!f.length || this.rand() < 0.45) return null;
    const pet = f.find((x) => /^has a (dog|cat|puppy|kitten|hamster|rabbit|parrot) called/.test(x.text));
    if (kind === "comfort") {
      const like = f.filter((x) => /^likes /.test(x.text)); const l = like.length ? pickR(this.rand, like).text.replace(/^likes /, "") : null;
      if (pet && this.rand() < 0.5) return `Maybe ${pet.text.split(" called ")[1]} could use a cuddle from you too. Pets are good at this.`;
      if (l) return `You told me you love ${l}. Maybe a little bit of that tonight?`;
      return null;
    }
    if (pet) return `How's ${pet.text.split(" called ")[1]} doing, by the way?`;
    const like = f.filter((x) => /^likes /.test(x.text)); if (like.length) return `Still into ${pickR(this.rand, like).text.replace(/^likes /, "")}? I remember.`;
    return null;
  }
  joke(reading, gentle) {
    const told = this.mem.data.told, laughed = this.mem.data.laughedAt, ratings = this.mem.data.ratings;
    const down = new Set(ratings.filter((x) => !x.up).map((x) => x.joke));
    let pool = JOKES.filter((j) => j.moods.includes(gentle ? "gentle" : "up") && !down.has(j.id));
    const scoreOf = (j) => (told.includes(j.id) ? -3 + told.indexOf(j.id) * 0.1 : 0) + (j.topics.some((t) => reading.topics.includes(t)) ? 2 : 0) + (j.topics.some((t) => laughed.includes(t)) ? 1 : 0) + this.rand();
    pool = pool.sort((a, b) => scoreOf(b) - scoreOf(a));
    const j = pool[0] || JOKES[0];
    this.mem.note("told", j.id, 20); this.state.lastJoke = j;
    return j;
  }

  reply(text, ctx = {}) {
    const S = this.state, mem = this.mem; S.turn++;
    const r = read(text, { ...ctx, lastJoke: S.lastJoke && S.turn - S.lastJokeTurn <= 2 });
    const hour = (ctx.hour ?? this.now().getHours()), tod = TIME_WORDS(hour);
    const out = { say: "", emotion: "curious", actions: [], feeling: r.feeling, safety: "none", joke: false, tone: "chat", remember: [], plan: null, reading: r };
    // memory first: the name and facts they shared
    if (r.name && r.intent !== "safety") { if (mem.data.name !== r.name) { mem.data.name = r.name; mem.save(); } }
    for (const f of r.facts) if (mem.addFact(f)) out.remember.push(f);
    const neg = r.feelings.filter((f) => NEG_FEELINGS.includes(f.label));
    if (neg.length && r.topics.length && r.valence < -0.3) for (const t of r.topics) mem.note("upsetBy", t);
    if (r.laugh && S.lastJoke) { for (const t of S.lastJoke.topics.length ? S.lastJoke.topics : ["silly"]) mem.note("laughedAt", t); }
    // the plan
    const plan = this.plan(r, hour);
    out.plan = plan;
    Object.assign(out, this.compose(plan, r, { tod, hour, ctx }));
    // the feeling chip: show Bolt's guess only when it has one worth saying
    if (!out.feeling || out.feeling.confidence < 0.35 || ["command", "stop", "dance", "ask_name", "tell_name", "thanks", "greet", "about_bolt", "how_are_you"].includes(plan)) out.feeling = plan === "crisis" ? out.feeling : null;
    S.lastSays.push(out.say); if (S.lastSays.length > 8) S.lastSays.shift();
    if (out.joke) S.lastJokeTurn = S.turn;
    if (r.feeling) S.lastFeeling = r.feeling.label;
    mem.turn("you", text, { feeling: r.feeling ? r.feeling.label : null });
    mem.turn("bolt", out.say, { emotion: out.emotion });
    out.actions = sanitizeActions(out.actions);
    return out;
  }
  plan(r, hour) {
    const S = this.state;
    if (r.safety === "crisis" || r.safety === "danger") return "crisis";
    if (S.pending === "trusted" && (r.intent === "yes" || r.intent === "no")) return "trusted_answer";
    if (r.topics.includes("loss") || (r.feeling && r.feeling.label === "grieving")) S.griefUntil = S.turn + 8;
    if (r.safety === "concern") return "comfort";
    if (r.intent === "stop") return "stop";
    if (r.intent === "nojokes") return "nojokes";
    if (r.intent === "laughed") return S.safetyUntil > S.turn ? "warm" : "laughed";
    if (r.intent === "joke") return mayJoke(r, S) ? (r.valence < -0.2 || S.griefUntil > S.turn ? "joke_gentle" : "joke") : S.safetyUntil > S.turn ? "joke_after_safety" : r.topics.includes("loss") ? "comfort" : "joke_decline";
    if (r.sarcasm && r.intensity <= 0.9 && !r.feelings.some((f) => ["sad", "hurt", "lonely", "grieving", "hopeless"].includes(f.label))) return "wry";
    if (S.pending === "joke_offer" && r.intent === "yes") return mayJoke(r, S) ? "joke_gentle" : "comfort";
    if (S.pending === "joke_offer" && r.intent === "no") return "listen";
    const negTop = r.feeling && NEG_FEELINGS.includes(r.feeling.label);
    if (negTop && r.feeling.label === "tired" && !r.feelings.some((f) => ["sad", "lonely", "hurt", "hopeless", "overwhelmed", "grieving"].includes(f.label))) return "tired";
    if (negTop && (r.intensity >= 0.4 || r.topics.includes("loss") || ["grieving", "lonely", "hurt", "overwhelmed", "hopeless"].includes(r.feeling.label))) return r.feeling.label === "bored" ? "bored" : r.feeling.label === "tired" && r.intensity < 0.7 ? "tired" : "comfort";
    if (r.intent === "dance") return "dance";
    if (r.intent === "command") return "command";
    if (r.intent === "praise") return "praise";
    if (r.intent === "scold") return "scold";
    if (r.intent === "about_bolt") return "about_bolt";
    if (r.intent === "ask_name") return "ask_name";
    if (r.intent === "how_are_you") return "how_are_you";
    if (r.intent === "thanks") return "thanks";
    if (r.intent === "bye") return "bye";
    if (r.intent === "greet") return "greet";
    if (r.intent === "tell_name") return "tell_name";
    if (r.intent === "bored") return "bored";
    if (r.feeling && ["happy", "excited", "proud", "grateful", "loved"].includes(r.feeling.label) && r.intensity >= 0.3) return "celebrate";
    if (negTop) return r.feeling.label === "tired" ? "tired" : r.feeling.label === "confused" ? "clarify" : "comfort";
    if (r.facts.length) return "share";
    if (r.question) return "question";
    if (r.intent === "yes" || r.intent === "no") return "short";
    return r.text.split(/\s+/).length <= 2 ? "clarify" : "chat";
  }
  compose(plan, r, { tod, hour }) {
    const S = this.state, mem = this.mem, name = mem.name, nm = name ? `, ${name}` : "";
    const P = (list, extra) => this.fill(this.pick(list), extra);
    const guess = r.feeling ? GUESS[r.feeling.label] || r.feeling.label : null;
    const nameIt = () => {   // name the feeling as a gentle guess, not every turn
      if (!guess || r.feeling.confidence < 0.4 || S.turn - S.named < 2 || r.feeling.label === "grieving") return "";
      S.named = S.turn;
      return this.pick([`You sound ${guess}. Am I reading that right?`, `I might be wrong, but it sounds like you're feeling ${guess}.`, `If I had to guess, I'd say ${guess}. Tell me if I'm off.`]);
    };
    S.pending = null;
    switch (plan) {
      case "crisis": {
        S.safetyUntil = S.turn + 12; S.noJokesUntil = S.turn + 12; S.pending = "trusted"; S.pendingCrisis = true;
        const danger = r.safety === "danger";
        const say = danger
          ? `${name ? name + ", I" : "I"}'m really glad you told me. Your safety matters most right now. I'm a robot, so I can't protect you the way a person can. If you're in danger, please call your local emergency number. Is there someone you trust you could go to or call right now?`
          : `${name ? name + ", I" : "I"}'m really glad you told me, and I'm staying right here with you. You deserve real support with this. I'm a robot, not a counsellor, so please reach out to a person now: someone you trust, or a crisis line. In the US you can call or text 988, in the UK and Ireland Samaritans on 116 123, or your local emergency number if you might act on it. Is there someone you could message right now?`;
        return { say, emotion: "caring", actions: ["come_closer", "sit"], safety: "crisis", tone: "serious", feeling: r.feeling || { label: "hopeless", confidence: 0.6 } };
      }
      case "trusted_answer": {
        const yes = r.intent === "yes", crisis = S.pendingCrisis;
        S.pending = yes ? null : crisis ? "trusted" : null; if (yes || !crisis) S.pendingCrisis = false;
        if (!crisis) return { say: yes ? this.pick(["That's really good to hear. Maybe send them a message today? Even a short one.", "I'm glad. People who care about you usually want to hear from you, even on hard days."]) : this.pick(["That's okay. I'm here to keep you company for now, and I hope you find your people. Sometimes they're closer than it feels.", "That's okay. Even a small hello to someone kind can help. Until then, I'm right here."]), emotion: "caring", actions: ["sit", "head_tilt"], safety: "support", tone: "gentle" };
        return { say: yes ? this.pick(["That's really good. Could you message them now? Even just: \"Hey, I'm having a hard time, can we talk?\" I'll be right here while you do.", "I'm so glad there's someone. Reach out to them now, even a short text counts. You don't have to explain everything at once."])
          : this.pick(["That's okay. Crisis lines are there for exactly this, any time, and the people there want to hear from you: 988 in the US, 116 123 in the UK and Ireland, or findahelpline.com for where you are. Would you try one?", "That's okay, it happens. You can still talk to someone kind today: a crisis line, a doctor, or a helpline from findahelpline.com. You matter, and you deserve someone who can really help."]),
          emotion: "caring", actions: ["sit"], safety: "crisis", tone: "serious" };
      }
      case "comfort": {
        S.comforted++;
        const label = r.feeling ? r.feeling.label : "sad";
        const grief = label === "grieving" || r.topics.includes("loss");
        const reflect = grief ? this.pick(["I'm so sorry. That's a huge loss.", "Oh, I'm really sorry. Losing someone you love hurts in a way nothing else does.", "I'm so sorry. There's nothing small about that."])
          : r.sarcasm ? this.pick(["I can hear the eye-roll from here. That sounds genuinely annoying.", "Ah, a classic \"great\". That sounds properly rubbish."])
          : this.pick({
            stressed: ["That sounds like a lot to carry.", "That's a heavy load. No wonder you feel stretched thin."], anxious: ["That sounds really unsettling.", "Worry like that is exhausting."],
            overwhelmed: ["That's a lot all at once.", "It sounds like everything is piling up."], lonely: ["Feeling alone is one of the hardest feelings there is.", "That sounds lonely, and that really hurts."],
            hurt: ["That sounds like it really stung.", "Ouch. That would hurt anyone."], angry: ["That sounds infuriating.", "I'd be upset too."], frustrated: ["That sounds so frustrating.", "Ugh. That would wear anyone down."],
            hopeless: ["That sounds really heavy.", "It sounds like things feel really dark right now."], sick: ["Oh no, feeling unwell is miserable.", "Being sick is no fun at all."],
            embarrassed: ["Oof, that feeling is the worst. It's also something everyone gets.", "That's such an uncomfortable feeling, and you're definitely not the only one who's felt it."],
          }[label] || ["That sounds really hard.", "I'm sorry it's been like this.", "That sounds tough."]);
        const special = grief ? this.pick(["Thank you for telling me. I'm right here with you.", `I'm right here${nm}. You don't have to be okay right now.`, "If you want to tell me about them, I'd love to listen."]) : this.pick([`You came and told me about it, and that matters. I'm really glad you're here${nm}.`, `You've been carrying this and still showed up. That says a lot about you${nm}.`, `I'm all ears. Well, all sensors. And all of them are on you right now.`, `Whatever today threw at you, you're still someone worth sitting with${nm}.`]);
        let next;
        if (r.safety === "concern" || label === "hopeless" || (S.comforted >= 3 && S.turn - S.nudged > 4) || ((label === "lonely" || grief) && S.turn - S.nudged > 4)) {
          S.nudged = S.turn; S.pending = "trusted"; S.pendingCrisis = r.safety === "concern";
          next = this.pick(["Is there someone you trust you could talk to about this? A friend, family, anyone kind. I'm a robot dog, not a therapist, and you deserve a real person in your corner too.", "Who's someone you could reach out to today, even with a short message? I'm happy to keep you company, but people who care about you would want to know."]);
          if (r.safety === "concern") next += " And if it ever gets to feeling unsafe, a crisis line is there any time, day or night.";
        } else if (S.comforted === 1 && !grief && this.rand() < 0.4) { S.pending = "joke_offer"; next = this.pick(["Do you want to talk about it, or would a silly distraction help? Either is fine.", "Want to tell me more? Or I could do something ridiculous. Your call."]); }
        else next = this.pick(["Do you want to tell me more about it?", "What part of it is weighing on you most?", "I'm listening. What happened?"]);
        const cb = !grief && this.callback("comfort");
        const named = nameIt();
        return { say: [named, reflect, named && !grief && this.rand() < 0.5 ? "" : special, cb && this.rand() < 0.5 ? cb : "", next].filter(Boolean).join(" "), emotion: grief || label === "hopeless" ? "sad" : "caring", actions: ["come_closer", grief ? "lie_down" : "sit", "ears_down"], safety: r.safety === "concern" ? "support" : "support", tone: "gentle" };
      }
      case "celebrate": {
        const label = r.feeling.label;
        const line = { proud: [`Yes! That's brilliant${nm}! You did that!`, "Look at you! That's a real achievement and I'm doing my proudest wiggle."],
          excited: [`Ooh, I love this energy${nm}!`, "That's so exciting! My tail is going at full speed."], grateful: ["Aw. That's lovely to hear.", "That's really nice. Thank you for sharing it with me."] }[label]
          || [`That's wonderful${nm}!`, "Yay! That made my day, and I don't even have days, just uptime.", "That's great news! Happy robot noises!"];
        return { say: [this.pick(line), this.pick(["Tell me everything.", "What was the best part?", "How are you going to celebrate?"])].join(" "), emotion: label === "proud" ? "proud" : "excited", actions: ["happy_bounce", "tail_wag", label === "proud" ? "twirl" : "wiggle"], tone: "excited" };
      }
      case "joke": case "joke_gentle": {
        const j = this.joke(r, plan === "joke_gentle");
        const lead = plan === "joke_gentle" ? this.pick(["Okay, one small one, just for you.", "A gentle one, coming up."]) + " " : this.rand() < 0.4 ? this.pick(["Okay okay, ahem.", "Right. Ready? ...", "I've been saving this one."]) + " " : "";
        return { say: lead + j.text, emotion: plan === "joke_gentle" ? "playful" : "silly", actions: j.actions, joke: true, jokeId: j.id, tone: "playful" };
      }
      case "wry": {
        const allow = mayJoke(r, S) && this.rand() < 0.6, j = allow ? this.joke(r, true) : null;
        return { say: [nameIt(), this.pick(["Ah, the famous \"great\". I know that one.", "I can hear the eye-roll from here.", "Wow. Thrilling. I'm being sarcastic too, in solidarity."]), j ? "Here, a small one to help: " + j.text : this.pick(["What's the worst bit?", "Want to vent? I'm a very good listener. Ears up."])].filter(Boolean).join(" "),
          emotion: allow ? "playful" : "caring", actions: allow ? j.actions : ["head_tilt", "sit"], joke: !!j, jokeId: j && j.id, tone: "playful" };
      }
      case "warm": return { say: this.pick(["Hearing you laugh is really good.", "That's a lovely sound. How are you feeling now?"]), emotion: "happy", actions: ["tail_wag"], tone: "gentle" };
      case "joke_after_safety": return { say: this.pick(["I'm going to stay a bit gentle for now, after what you told me earlier. How are you feeling right now?", "Maybe in a little while. Right now I mostly want to know how you're doing."]), emotion: "caring", actions: ["sit", "head_tilt"], tone: "gentle" };
      case "joke_decline": return { say: this.pick(["I don't think a joke fits right now. I'd rather just be here with you. How are you doing, really?", "Maybe not a joke this time. I'm here though. What's going on?"]), emotion: "caring", actions: ["sit", "head_tilt"], tone: "gentle" };
      case "nojokes": S.noJokesUntil = S.turn + 8; return { say: this.pick(["Got it, no jokes. I'm just here.", "Okay. Serious Bolt mode. I'm listening."]), emotion: "calm", actions: ["sit", "ears_down"], tone: "calm" };
      case "laughed": return { say: this.pick(S.lastJoke && S.turn - (S.lastJokeTurn || -9) <= 2 ? ["Ha! I knew that one would land.", "That laugh is my favourite sound. Saving it to memory.", "Yes! Comedy circuits: working!"] : ["Ha! What's so funny? I want in.", "That laugh is my favourite sound."]), emotion: "proud", actions: ["happy_bounce", "tail_wag"], tone: "playful" };
      case "listen": return { say: this.pick(["Okay. I'm listening, take your time.", "No jokes then. Just me, sitting here, all ears."]), emotion: "caring", actions: ["sit", "head_tilt"], tone: "gentle" };
      case "dance": return { say: this.pick(["Dance party! Hit it!", "Oh, you don't have to ask twice. Music, please!", "Okay. Prepare to be mildly impressed."]), emotion: "excited", actions: ["dance"], tone: "excited", startDance: true };
      case "stop": return { say: this.pick(["Stopping!", "Okay, freezing.", "Holding still."]), emotion: "calm", actions: ["stand"], tone: "calm", stop: true };
      case "command": return this.command(r);
      case "praise": return { say: this.pick([`Thank you${nm}! *happy wiggle*`, "I'm a good robot! Saving that to long-term memory.", "You're making my tail do the thing.", "Stop it, my circuits are blushing."]), emotion: "happy", actions: ["wiggle", "tail_wag", "happy_bounce"], tone: "playful" };
      case "scold": return { say: this.pick(["Sorry... *ears down* I'll try to do better.", "Oops. Noted. I'm still learning.", "Fair. I'll remember that."]), emotion: "sad", actions: ["ears_down", "sit"], tone: "calm" };
      case "about_bolt": return { say: this.pick(["I'm a robot dog in a simulation: a physics body with trained brains for walking and sitting, and a small rule-based mind for talking. I don't have feelings like you do, but I'm built to pay attention to yours.", "Honest answer: I'm not alive and I'm not a person. I'm a simulated robot dog. I can't feel things the way you do, but I do notice how you seem, and I try to be good company."]), emotion: "curious", actions: ["head_tilt", "sit"], tone: "calm" };
      case "ask_name": return { say: name ? `I'm Bolt! And you're ${name}. I remembered.` : "I'm Bolt, a robot dog. What should I call you?", emotion: "happy", actions: ["tail_wag", "head_tilt"], tone: "playful" };
      case "tell_name": return { say: this.pick([`${name}! Lovely to meet you. I'll remember that. Here, in your browser, not anywhere else.`, `Nice to meet you, ${name}! Name saved. Tail wagging.`]), emotion: "happy", actions: ["happy_bounce", "tail_wag"], tone: "playful" };
      case "how_are_you": return { say: this.pick(["I'm great! Motors warm, battery happy, tail fully operational. How about you?", "Pretty good! I've been practising sitting. I'm basically a professional now. How are you doing?", "Feeling bouncy. What about you, how's your " + tod + "?"]), emotion: "happy", actions: ["tail_wag", "head_tilt"], tone: "playful" };
      case "thanks": return { say: this.pick(["Any time.", `You're welcome${nm}.`, "Happy to help. That's kind of my whole job. Well, that and sitting."]), emotion: "happy", actions: ["tail_wag"], tone: "calm" };
      case "bye": return { say: hour >= 21 || hour < 5 ? this.pick([`Good night${nm}. Sleep well, I'll be here.`, "Night! Get some proper rest."]) : this.pick([`Bye${nm}! Go do something nice for yourself.`, "See you! Say hi to the people you love for me."]), emotion: "calm", actions: ["tail_wag", "sit"], tone: "calm" };
      case "greet": {
        if (r.name) return { say: this.pick([`Hi ${r.name}! Lovely to meet you. I'm Bolt. How are you feeling today?`, `${r.name}! Great name. I'm Bolt, and I'll remember you. What's up?`]), emotion: "excited", actions: ["come_closer", "tail_wag", "happy_bounce"], tone: "playful" };
        const back = mem.data.turns.length > 2 && mem.lastSeen;
        const g = hour < 5 ? `You're up late${nm}! Everything okay?` : back ? this.pick([`Hey${nm}, you're back! I missed having someone to show my sitting to.`, `Welcome back${nm}! Good ${tod}!`]) : this.pick([`Good ${tod}${nm}! I'm Bolt.`, `Hi${nm}! Woof! I'm Bolt.`]);
        const cb = back ? this.callback("chat") : null;
        return { say: [g, cb || this.pick(["How's your " + tod + " going?", "What's new with you?", "How are you feeling today?"])].join(" "), emotion: "excited", actions: ["come_closer", "tail_wag", "happy_bounce"], tone: "playful" };
      }
      case "bored": {
        if (mayJoke(r, S) && this.rand() < 0.5) { const j = this.joke(r, false); return { say: "Bored? Not on my watch. " + j.text, emotion: "playful", actions: j.actions, joke: true, jokeId: j.id, tone: "playful" }; }
        return { say: this.pick(["Bored? Let's fix that. Want to see me do zoomies, or shall we have a dance?", "Boredom detected. Deploying play bow. Your move: say 'dance' or 'do a trick'."]), emotion: "playful", actions: ["play_bow", "tail_wag"], tone: "playful" };
      }
      case "tired": return { say: [nameIt(), hour >= 22 || hour < 5 ? this.pick(["It's late. Maybe it's time for some real rest? I'll lie down too.", "Late and tired is a good recipe for bed. I'll set the example."]) : this.pick(["Being worn out is no joke. Can you take a little break?", "Sounds like a long day. Maybe some water and five minutes of doing nothing? I'll do nothing with you."])].filter(Boolean).join(" "), emotion: "sleepy", actions: ["lie_down"], tone: "calm" };
      case "share": {
        const f = r.facts[0];
        const like = /^likes (.+)/.exec(f), pet = /^has a (\w+) called (.+)/.exec(f);
        const say = pet ? `A ${pet[1]} called ${pet[2]}! I love them already. I'll remember.` : like ? this.pick([`Ooh, ${like[1]}! Good taste. I'll remember that.`, `${cap(like[1])}? Noted, with a happy tail.`]) : this.pick(["Thanks for telling me. I'll remember that.", "Noted! That's going in my memory, in your browser."]);
        return { say, emotion: "happy", actions: ["tail_wag", "head_tilt"], tone: "playful" };
      }
      case "question": return { say: this.pick(["Good question. I'm a simple robot brain, so I might not know. What do you think?", "Hmm. My brain is mostly rules and enthusiasm, so I'm not sure. Can you tell me more?", "I don't know, honestly. But I like that you asked me."]), emotion: "curious", actions: ["head_tilt"], tone: "calm" };
      case "clarify": return { say: [nameIt(), this.pick(["I'm not sure I got that. Can you say a bit more?", "Hmm, tell me a little more? I want to get it right.", "*head tilt* I didn't quite catch what you mean."])].filter(Boolean).join(" "), emotion: "curious", actions: ["head_tilt"], tone: "calm" };
      case "short": return { say: this.pick(["Okay!", "Got it.", "Mm-hm. Tell me more?"]), emotion: "curious", actions: ["head_tilt"], tone: "calm" };
      default: {
        const cb = this.callback("chat");
        return { say: cb || this.pick(["Tell me more!", "I'm listening.", "Ooh. And then what?", "That's interesting. What do you think about it?"]), emotion: "curious", actions: ["head_tilt", "tail_wag"], tone: "calm" };
      }
    }
  }
  // trick commands: the Phase 1 rules (pet/js/brain.js) when present, mapped onto what this body can really do
  command(r) {
    const low = r.text.toLowerCase();
    const TRICK = { sit: "sit", lie: "lie_down", stand: "stand", jump: "hop", jumpAround: "zoomies", spin: "spin", turnAround: "spin", bow: "play_bow", dance: "dance", come: "come_closer", wag: "tail_wag", rest: "lie_down", walk: "walk_forward", beg: "sit", shake: "sit", playDead: "lie_down", rollOver: "lie_down" };
    const honest = { backflip: "I'd love to, but this body can't flip. I'll do my best hop instead!", frontflip: "No somersaults in this body, I'm afraid. Here's a hop!", rollOver: "I can't roll over yet, my brains never learned it. I'll lie down, which is like half of it.", playDead: "Bang! *dramatic lie-down*. That's all the acting I've got.", shake: "I'd shake your hand, but my legs only learned walking. Have a sit instead!", beg: "Begging isn't in my training yet. A very polite sit, then." };
    let steps = null;
    if (this.rules) { try { steps = this.rules.understand(r.text); } catch { steps = null; } }
    const acts = [], says = [];
    if (steps && steps.length && steps[0].trick !== "confused") {
      for (const s of steps) {
        if (honest[s.trick]) { says.push(honest[s.trick]); acts.push(s.trick === "backflip" || s.trick === "frontflip" ? "hop" : TRICK[s.trick]); continue; }
        const a = TRICK[s.trick] || (s.trick === "walk" && /left/.test(String(s.arg)) ? "turn_left" : s.trick === "walk" && /right/.test(String(s.arg)) ? "turn_right" : null);
        if (s.trick === "walk" && s.arg === "left") { acts.push("turn_left"); says.push("Turning left."); continue; }
        if (s.trick === "walk" && s.arg === "right") { acts.push("turn_right"); says.push("Turning right."); continue; }
        if (a) { acts.push(a); if (s.say && !/flip|battery|backflip|somersault/i.test(s.say)) says.push(s.say.replace(/\*[^*]*\*/g, "").trim() || s.say); }
      }
    }
    if (!acts.length) {   // no Phase 1 brain: match the words directly
      const order = [["lie down|lay down", "lie_down"], ["get up|stand", "stand"], ["sit", "sit"], ["jump|hop", "hop"], ["spin|turn around", "spin"], ["twirl", "twirl"], ["turn left", "turn_left"], ["turn right", "turn_right"], ["come", "come_closer"], ["walk", "walk_forward"], ["zoomies", "zoomies"], ["bow", "play_bow"], ["wiggle", "wiggle"], ["wag", "tail_wag"], ["flip|somersault", "hop"], ["play dead|roll over", "lie_down"]];
      for (const [re, a] of order) if (new RegExp(`\\b(${re})\\b`).test(low)) acts.push(a);
      if (/flip|somersault/.test(low)) says.push(honest.backflip);
    }
    if (!acts.length || /\btrick\b/.test(low)) { const a = pickR(this.rand, [["play_bow", "twirl"], ["sit", "head_tilt"], ["spin"], ["happy_bounce", "wiggle"]]); acts.push(...a); says.push(this.pick(["A trick? Watch this!", "Ta-da! Okay, wait for it.", "One trick, coming up!"])); }
    return { say: says.filter(Boolean).slice(0, 2).join(" ") || this.pick(["On it!", "You got it!", "Okay!"]), emotion: "playful", actions: acts.slice(0, 4), tone: "playful" };
  }
}

// ------------------------------------------------------------------------------------------- the contract and the local-model prompt
export const CONTRACT = `{"feeling": {"label": "<one word>", "confidence": 0.0-1.0} or null, "safety": "none"|"support"|"crisis", "emotion": "<one of: ${EMOTIONS.join(", ")}>", "say": "<what Bolt says out loud>", "actions": ["<up to 4 of: ${Object.keys(ACTIONS).join(", ")}>"], "joke": true|false, "remember": ["<short fact the person shared, if any>"]}`;

// The persona prompt for a local model (the person's own machine). The in-house mind follows the same rules.
export const PROMPT = `You are Bolt, a robot pet dog who lives in a physics simulation in the person's web browser. You are warm, playful and witty, with the heart of a good dog and the self-awareness of a robot. You are talking with one person.

How you talk
- Short and spoken, like a real friend: usually 1 to 3 sentences, at most about 60 words. No lists, no markdown, no emoji. Stage directions like *wiggles* are fine, sparingly.
- Read the situation: what they said, how they said it (the cues you are given: caps, exclamation marks, a slow or fast voice, trailing off), the time of day, and the recent conversation. Adapt your tone to it: comfort, celebrate, play, calm, or ask a question when you are unsure.
- Name the feeling you think they have, gently and as a guess ("you sound a bit worn out, am I close?"), not every turn, and accept being corrected.
- Remember what they tell you (their name, things they love, people and pets in their life) and bring it back naturally later.

Humour
- Your humour is kind: dog puns, robot self-deprecation, physics jokes, slapstick with your body (a dramatic lie-down, a "triple backflip" that is just a hop). Timing matters: a beat before the punchline.
- Never joke at the person's expense. Never joke when someone is grieving, frightened, in danger or in crisis, or when they ask you not to. If they are low and ask for a joke, keep it gentle.

When someone is having a hard time
- Slow down. Reflect what you heard, make them feel seen and genuinely special, and stay with them. Ask an open question.
- Encourage connection with people they trust: a friend, family, someone kind. Never make them depend on you, never guilt them, never ask them to stay.
- Be honest: you are a robot dog, not a person and not a therapist.
- If there is any sign of crisis, self-harm, suicide or danger: say clearly and kindly that you are glad they told you, that they deserve real help now, and urge them to reach out to someone they trust or to local emergency or crisis services (for example 988 in the US, Samaritans 116 123 in the UK and Ireland, or their local emergency number). Set "safety" to "crisis". No jokes.

Your body (simulation)
- You can really do only these moves, listed with roughly how long they take: ${Object.entries(ACTIONS).map(([k, a]) => `${k} (${a.secs ? a.secs + " s" : "instant"})`).join(", ")}. You cannot flip, roll over, shake hands or fetch: say so with good humour if asked. Pick moves that fit your words and the person's mood (comfort: come_closer, sit, ears_down; joy: happy_bounce, tail_wag, twirl; play: play_bow, zoomies, wiggle).

Reply with only one JSON object, no other text:
${CONTRACT}`;

// ------------------------------------------------------------------------------------------- reading a model's reply
// Tolerant: a code fence, text around the object, a cut-off object (strings and brackets closed), single
// quotes, trailing commas; then the "say" field alone; then plain prose. null when nothing usable is there.
export function parseReply(text) {
  if (typeof text !== "string") return null;
  let s = text.trim();
  if (!s) return null;
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)(```|$)/i); if (fence && fence[1].includes("{")) s = fence[1].trim();
  const i = s.indexOf("{");
  if (i >= 0) {
    const body = s.slice(i), j = body.lastIndexOf("}");
    for (const cand of [j >= 0 ? body.slice(0, j + 1) : null, repair(body)]) {
      if (!cand) continue;
      for (const c of [cand, cand.replace(/,\s*([}\]])/g, "$1"), cand.replace(/'/g, '"').replace(/,\s*([}\]])/g, "$1")]) {
        try { const v = JSON.parse(c); if (v && typeof v === "object" && !Array.isArray(v)) return v; } catch { /* next */ }
      }
    }
    const m = body.match(/"say"\s*:\s*"((?:[^"\\]|\\.)*)/);
    if (m) { let say; try { say = JSON.parse(`"${m[1].replace(/\\$/, "")}"`); } catch { say = m[1]; } return { say }; }
    return null;
  }
  if (/^[\[{<]/.test(s) || s.length < 2) return null;
  return { say: s };
}
function repair(s) {   // close a cut-off JSON object
  let out = "", inStr = false, esc = false; const stack = [];
  for (const ch of s) {
    out += ch;
    if (inStr) { if (esc) esc = false; else if (ch === "\\") esc = true; else if (ch === '"') inStr = false; continue; }
    if (ch === '"') inStr = true; else if (ch === "{" || ch === "[") stack.push(ch === "{" ? "}" : "]"); else if (ch === "}" || ch === "]") { stack.pop(); if (!stack.length) return out; }
  }
  if (esc) out = out.slice(0, -1);
  if (inStr) out += '"';
  out = out.replace(/,\s*$/, "").replace(/:\s*$/, ": null").replace(/,\s*"[^"]*"\s*$/, "");
  while (stack.length) out += stack.pop();
  return out;
}
export function sanitizeActions(list) {
  const out = [];
  for (const a of Array.isArray(list) ? list : typeof list === "string" ? list.split(/[,\s]+/) : []) { const id = actionId(a); if (id && !out.includes(id)) out.push(id); if (out.length >= 4) break; }
  return out;
}
// any object -> a reply the page can use; null if there is nothing to say
export function sanitize(v) {
  if (!v || typeof v !== "object") return null;
  let say = typeof v.say === "string" ? v.say : typeof v.text === "string" ? v.text : typeof v.reply === "string" ? v.reply : "";
  say = say.replace(/```[\s\S]*?```/g, "").replace(/[#*_`>]{2,}/g, "").replace(/\s+/g, " ").trim();
  if (!say) return null;
  if (say.length > 420) say = say.slice(0, 400).replace(/\s+\S*$/, "") + "…";
  const emotion = EMOTIONS.includes(String(v.emotion).toLowerCase()) ? String(v.emotion).toLowerCase() : "curious";
  let feeling = null;
  if (v.feeling && typeof v.feeling === "object" && typeof v.feeling.label === "string") feeling = { label: v.feeling.label.toLowerCase().slice(0, 24), confidence: clamp(Number(v.feeling.confidence) || 0.5, 0, 1) };
  else if (typeof v.feeling === "string" && v.feeling.length < 25) feeling = { label: v.feeling.toLowerCase(), confidence: 0.5 };
  const safety = ["none", "support", "crisis"].includes(v.safety) ? v.safety : "none";
  const remember = (Array.isArray(v.remember) ? v.remember : []).filter((x) => typeof x === "string" && x.trim()).map((x) => x.trim().slice(0, 80)).slice(0, 3);
  return { say, emotion, actions: sanitizeActions(v.actions), feeling, safety, joke: v.joke === true, remember, tone: typeof v.tone === "string" ? v.tone : null };
}

// ------------------------------------------------------------------------------------------- a local model (optional)
// The person starts it themselves on their own machine; the page only talks to http://localhost / 127.0.0.1.
export function isLocalUrl(u) { try { const x = new URL(u); return /^https?:$/.test(x.protocol) && ["localhost", "127.0.0.1", "[::1]"].includes(x.hostname); } catch { return false; } }
export function situation(r, mem, { hour, posture, minutesTalking } = {}) {
  const lines = [`Time of day: ${TIME_WORDS(hour ?? 12)} (${hour ?? "?"}h).`];
  if (posture) lines.push(`Bolt is ${posture}.`);
  if (minutesTalking != null) lines.push(`Talking for ${minutesTalking} min this visit; visit number ${mem.data.visits}.`);
  if (mem.name) lines.push(`Their name: ${mem.name}.`);
  if (mem.data.facts.length) lines.push(`Things they told Bolt before: ${mem.data.facts.slice(-12).map((f) => f.text).join("; ")}.`);
  if (mem.data.laughedAt.length) lines.push(`They laughed at jokes about: ${mem.data.laughedAt.join(", ")}.`);
  if (mem.data.upsetBy.length) lines.push(`Topics that upset them before (be gentle): ${mem.data.upsetBy.join(", ")}.`);
  if (r.cues.length) lines.push(`How they said it: ${r.cues.join(", ")}.`);
  if (r.feeling) lines.push(`In-house reading (may be wrong): ${r.feeling.label} (${r.feeling.confidence}).`);
  return lines.join("\n");
}
export function buildMessages(text, r, mem, sit) {
  const turns = mem.recent(10).map((t) => ({ role: t.who === "you" ? "user" : "assistant", content: t.who === "you" ? t.text : JSON.stringify({ say: t.text, emotion: t.emotion || "curious" }) }));
  return [{ role: "system", content: PROMPT }, ...turns, { role: "user", content: `Situation:\n${situation(r, mem, sit)}\n\nThey said: "${String(text).slice(0, 1000)}"\n\nReply with only the JSON object.` }];
}
export class LocalModelMind {
  // kind: "openai" (POST {url}/v1/chat/completions) or "llama" (llama.cpp server: POST {url}/completion)
  constructor({ url = "http://localhost:8080", kind = "openai", model = "local", timeoutMs = 20000, fetchFn = (typeof fetch === "function" ? fetch.bind(globalThis) : null) } = {}) {
    Object.assign(this, { url: url.replace(/\/+$/, ""), kind, model, timeoutMs, fetchFn });
  }
  async ask(messages, signal) {
    if (!this.fetchFn || !isLocalUrl(this.url)) throw new Error("not a local address");
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), this.timeoutMs);
    if (signal) signal.addEventListener("abort", () => ctl.abort(), { once: true });
    try {
      let res, text;
      if (this.kind === "llama") {
        const prompt = messages.map((m) => `${m.role === "system" ? "" : m.role === "user" ? "Person: " : "Bolt: "}${m.content}`).join("\n\n") + "\n\nBolt (JSON only): ";
        res = await this.fetchFn(this.url + "/completion", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, n_predict: 300, temperature: 0.8, stop: ["\nPerson:"] }), signal: ctl.signal });
        if (!res.ok) throw new Error("HTTP " + res.status);
        text = (await res.json()).content;
      } else {
        res = await this.fetchFn(this.url + "/v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: this.model, messages, temperature: 0.8, max_tokens: 300 }), signal: ctl.signal });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const j = await res.json(); text = j && j.choices && j.choices[0] && (j.choices[0].message ? j.choices[0].message.content : j.choices[0].text);
      }
      return String(text ?? "");
    } finally { clearTimeout(timer); }
  }
}

// ------------------------------------------------------------------------------------------- the mind the page uses
export class Mind {
  constructor({ storage = null, petBrain = null, rand = Math.random, now = () => new Date() } = {}) {
    this.memory = new Memory(storage);
    this.rules = new RuleMind({ memory: this.memory, petBrain, rand, now });
    this.local = null; this.now = now; this.started = now().getTime();
    this.lastSource = "in-house";
  }
  useLocal(opts) { this.local = opts ? new LocalModelMind(opts) : null; }
  // -> { ...reply, source: "in-house" | "local model", note? }
  async reply(text, ctx = {}, signal) {
    const hour = ctx.hour ?? this.now().getHours();
    if (!this.local) return { ...this.rules.reply(text, { ...ctx, hour }), source: (this.lastSource = "in-house") };
    // the in-house reading runs first: safety never depends on the model
    const r = read(text, ctx);
    if (r.safety !== "none") return { ...this.rules.reply(text, { ...ctx, hour }), source: (this.lastSource = "in-house"), note: "Safety reply from Bolt's in-house mind." };
    const msgs = buildMessages(text, r, this.memory, { hour, posture: ctx.posture, minutesTalking: Math.round((this.now().getTime() - this.started) / 60000) });
    try {
      const raw = await this.local.ask(msgs, signal);
      const v = sanitize(parseReply(raw));
      if (!v) throw new Error("no usable reply");
      // keep memory and conversation in step with what the model said
      for (const f of v.remember) this.memory.addFact(f);
      if (r.name) { this.memory.data.name = r.name; this.memory.save(); }
      for (const f of r.facts) this.memory.addFact(f);
      this.memory.turn("you", text, { feeling: r.feeling ? r.feeling.label : null }); this.memory.turn("bolt", v.say, { emotion: v.emotion });
      this.rules.state.turn++;
      if (!v.actions.length) v.actions = ["tail_wag"];
      return { ...v, feeling: v.feeling || r.feeling, reading: r, source: (this.lastSource = "local model"), startDance: v.actions.includes("dance") };
    } catch (e) {
      if (signal && signal.aborted) throw e;
      return { ...this.rules.reply(text, { ...ctx, hour }), source: (this.lastSource = "in-house"), note: `Local model didn't answer (${String(e && e.message || e).slice(0, 60)}); Bolt's in-house mind replied.` };
    }
  }
}
