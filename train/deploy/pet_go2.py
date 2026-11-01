"""Bolt's pet brain on the Go2 (run_go2.py --pet): what you say -> what the dog says and does. Built in-house, no
external AI service: the rules, lines, jokes, mood and memory of the browser pet (pet/js/brain.js), ported to Python,
with every trick mapped onto skills the Go2 brain was actually trained on (models/bolt-go2-blind-695M clips: walk,
trot, pace, canter, turn_left, turn_right, sit, lie, getup; steering on the gaits).

    pet = PetBrain(memory="~/.bolt/pet.json")
    for step in pet.understand("hi Bolt, sit then dance"):
        step["say"]        # what it says (printed; spoken if a local TTS, espeak-ng, is installed)
        step["moves"]      # [(seconds, command), ...]: --interactive commands, sent in order, each held that long

Feelings: mood 0..1 goes up with praise and play, down with scolding; it changes the lines and the moves (a grumpy
dog walks instead of trotting and skips the zoomies; a fantastic one adds a happy wiggle). Battery: the robot's real
charge when run_go2 passes it (LowState bms soc), else the browser pet's per-trick costs. Memory (name, every
interaction, your feedback on the trick it was about) is a JSON file: the training data for later.
Tricks the Go2 body has not learned (flips, roll over, shake, beg, jump) are said honestly, with a safe substitute.
"""
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

COST = {"jumpAround": 9, "dance": 6, "spin": 2, "come": 2, "walk": 2, "sit": 0.5, "lie": 0.5, "stand": 0.5,
        "turnAround": 1, "playDead": 2, "bow": 1, "wag": 0.5, "talk": 0.5}
NOT_YET = {"backflip", "frontflip", "rollOver", "shake", "beg", "jump"}   # browser tricks the Go2 has not learned

RULES = [   # first match wins (pet/js/brain.js RULES, same order)
    ("name", r"\b(?:your name is|i(?:'ll| will) call you|call yourself|you are called)\s+([a-z][\w-]*)"),
    ("askName", r"\b(what(?:'s| is) your name|who are you)\b"),
    ("stop", r"\b(stop|freeze|halt|enough|cancel)\b"),
    ("backflip", r"\bback\s*-?\s*flips?\b|\bbackward(?:s)? flip|\bflip backward"),
    ("frontflip", r"\bfront\s*-?\s*flips?\b|\bforward(?:s)? flip|\bs[ou]mm?[ae]r\s*-?\s*(?:s[aou]u?lt|slam|sault)s?\b|\bflip\b"),
    ("jumpAround", r"\b(jump(?:ing)? around|hop(?:ping)? around|zoomies|go crazy|bounce around|run around)\b"),
    ("rollOver", r"\broll(?:ing)?\s*-?\s*over\b|\broll\b"),
    ("playDead", r"\bplay(?:ing)? dead\b|\bbang\b|\bplay possum\b"),
    ("beg", r"\bbeg\b|\bsit pretty\b"),
    ("shake", r"\bshake\b|\bpaw\b|\bhigh\s*-?\s*five\b|\bgive me (?:your|a) hand\b"),
    ("sit", r"\bsit\b|\bsitting\b"),
    ("lie", r"\b(lie|lay|lying|laying)\b|^down$|\bget down\b"),
    ("jump", r"\b(jump|hop|leap)\b"),
    ("spin", r"\bspin\b|\bchase (?:your |its )?tail\b|\btwirl\b"),
    ("turnAround", r"\bturn around\b"),
    ("bow", r"\bbow\b|\bplay bow\b"),
    ("dance", r"\bdance\b|\bboogie\b|\bgroove\b"),
    ("come", r"\bcome(?: here| to me| back)?\b|\bhere (?:boy|girl|buddy)\b"),
    ("walk", r"\b(?:walk|go|move)\b.*\b(left|right|forward|ahead|back(?:ward)?)\b|\bwalk\b"),
    ("stand", r"\bstand\b|\bget up\b|^up$"),
    ("wag", r"\bwag\b"),
    ("rest", r"\b(sleep|rest|nap|recharge|charge|bed ?time|tired)\b"),
    ("joke", r"\bjoke\b|\bfunny\b|\bmake me laugh\b|\bhumou?r\b|\bsomething funny\b"),
    ("talk", r"\b(talk|speak|say something|bark|tell me about yourself|chat)\b"),
    ("praise", r"\bgood (?:boy|girl|dog|doggo|pup|bot|robot|job)\b|\bwell done\b|\bgood\b(?! (?:morning|evening|night|afternoon))"
               r"|\bnice\b|\bawesome\b|\bamazing\b|\bbravo\b|\btreat\b|\blove you\b|\bcute\b"),
    ("scold", r"\bbad (?:boy|girl|dog|robot)\b|\bno+\b|\bnaughty\b|\bstop that\b"),
    ("greet", r"\b(hi|hello|hey|yo|good morning|good evening)\b"),
    ("howAreYou", r"\bhow are you\b|\bhow do you feel\b|\bbattery\b"),
    ("help", r"\bhelp\b|\bwhat can you do\b|\btricks\b"),
]
RULES = [(k, re.compile(p, re.I)) for k, p in RULES]

LINES = {
    "sit": ["Sitting. The easiest trick, which is why it's my favourite.", "*sits* Is this where the treat comes in?"],
    "stand": ["Up!", "All four feet on the floor, as nature intended."],
    "lie": ["Lying down. Gravity wins this round.", "*flop*", "Horizontal mode activated."],
    "jumpAround": ["ZOOMIES!", "I have no idea where I'm going and I love it."],
    "playDead": ["Bang! *dramatic collapse*", "Oh no, I've been got. Tell my charger I loved it."],
    "bow": ["*bows low* Wanna play?"],
    "spin": ["Chasing my tail! I will catch it this time.", "Round and round!"],
    "turnAround": ["Turning around.", "About face!"],
    "dance": ["Let's dance!", "Robot dance! It's literally my genre.", "Watch these moves. I learned them from a washing machine."],
    "come": ["Coming!", "On my way!"],
    "walk": ["Walking {dir}.", "Heading {dir}."],
    "wag": ["*wiggles the whole back end, no tail fitted*", "Wagging! Well, swaying. Tail sold separately."],
    "rest": ["Nap time. Lying down to save power.", "Zzz..."],
    "stop": ["Okay, okay! Stopping.", "Freezing!", "Stopped. What did I do?"],
    "praise": ["Thank you! *happy wiggle*", "I'm a good robot! Saving that to long-term memory.", "Yay! Best human ever."],
    "praiseTrick": ["Thanks! I'll remember you liked the {trick}.", "You liked the {trick}? Noted, with pride!"],
    "scold": ["Sorry... *ears down*", "Oops. I'll do better."],
    "scoldTrick": ["Sorry, I'll remember you didn't like the {trick}.", "Noted: less {trick}. *sad beep*"],
    "greet": ["Hi! I'm {name}! Woof!", "Hello! Want to see a trick?"],
    "name": ["{name}! I love it. I'll remember that forever.", "Call me {name}! Saved to long-term memory."],
    "askName": ["I'm {name}, a robot dog. Fifteen kilos of motors and enthusiasm.", "My name is {name}!"],
    "howAreYou": ["Battery at {battery} percent, mood {mood}.", "Feeling {mood}! Battery {battery} percent."],
    "tired": ["My battery is at {battery} percent. Let me rest before a {trick}?"],
    "grumpy": ["Fine. A {trick}. A slow one.", "*sighs in servo* {trick}. If I must."],
    "notYet": ["I haven't learned to {trick} in this body yet. Here's a {instead} instead!",
               "A {trick}? Not with these legs yet. Training is on it. Have a {instead}."],
    "unknown": ["*tilts head* I don't know that one yet.", "Woof? I didn't get that. Try \"help\"."],
    "help": ["I can sit, lie down, stand up, walk (left or right), come here, spin, turn around, dance, do zoomies, "
             "bow, play dead, rest, tell jokes and chat. Chain them: \"sit then dance\". Flips and roll-overs: still training."],
    "talk": ["Hi, I'm {name}! I run entirely on this robot. No cloud, no fur.",
             "Fun fact: I learned to walk from recordings of a real dog, then practised billions of steps in simulation.",
             "If I fall over, I roll back onto my belly and get up. Practised it a lot. Please don't test it.",
             "Woof. That's robot for: you're my favourite human."],
}
JOKES = [
    "What do you call a dog magician? A labracadabrador.",
    "Why did the dog sit in the shade? He didn't want to be a hot dog.",
    "What's a robot dog's favourite music? Heavy metal.",
    "Why did the robot dog go to school? To improve its bark-ithmetic.",
    "My vet says I have a byte problem.",
    "I'd tell you a joke about my battery, but it's running low.",
    "I tried a backflip in simulation. Physics left me on read.",
]
MOOD_WORDS = [(0.8, "fantastic"), (0.6, "happy"), (0.4, "okay"), (0.2, "a bit down"), (0.0, "grumpy")]
SUBSTITUTE = {"backflip": "spin", "frontflip": "spin", "rollOver": "playDead", "shake": "sit", "beg": "sit", "jump": "jumpAround"}


def moves_for(trick, arg, mood):
    """A trick -> [(seconds, command), ...] in skills the Go2 brain has. The dog ends sitting after moving tricks, so a
    try never ends mid-gait; "stand" is the getup clip (it ends standing still)."""
    happy, grumpy = mood >= 0.8, mood < 0.2
    gait = "walk" if grumpy else "trot"
    m = {
        "sit": [(4, "sit")], "lie": [(5, "lie")], "rest": [(5, "lie")], "playDead": [(5, "lie")],
        "stand": [(5, "getup")], "stop": [(3, "sit")],
        "bow": [(3, "lie"), (5, "getup")],
        "come": [(3.0, "steer 0"), (3.0, "walk"), (3, "sit")],
        "walk": [(0.0, "steer " + {"left": "0.5", "right": "-0.5"}.get(arg, "0")), (4.0, "walk"), (0, "steer 0"), (3, "sit")],
        "turnAround": [(0.0, "steer 0.8"), (4.0, "walk"), (0, "steer 0"), (3, "sit")],
        "spin": [(0.0, "steer 0.8"), (5.0, gait), (0, "steer 0"), (3, "sit")],
        "jumpAround": [(0, "steer 0.6"), (2.0, "trot"), (0, "steer -0.6"), (2.0, "trot"), (0, "steer 0.6"), (2.0, "trot"),
                       (0, "steer 0"), (3, "sit")] if not grumpy else [(4, "sit")],
        "dance": [(0, "steer 0.6"), (1.5, "walk"), (0, "steer -0.6"), (1.5, "walk"), (0, "steer 0.8"), (2.5, gait),
                  (0, "steer 0"), (2.0, "pace"), (3, "sit")],
        "wag": [(0, "steer 0.5"), (1.0, "walk"), (0, "steer -0.5"), (1.0, "walk"), (0, "steer 0"), (3, "sit")],
    }.get(trick, [])
    if happy and m and trick in ("sit", "come", "walk", "dance"):   # a happy wiggle before settling
        m = m[:-1] + [(0, "steer 0.5"), (1.0, "walk"), (0, "steer -0.5"), (1.0, "walk"), (0, "steer 0")] + m[-1:]
    return m


class PetBrain:
    def __init__(self, memory=None, seed=None, speak=False):
        self.rng = random.Random(seed)
        self.mood, self.battery, self.real_battery = 0.6, 100.0, False
        self.last_trick, self.memory = None, Path(memory).expanduser() if memory else None
        saved = {}
        if self.memory and self.memory.exists():
            try:
                saved = json.loads(self.memory.read_text())
            except (OSError, ValueError):
                saved = {}
        self.name, self.log = saved.get("name", "Bolt"), saved.get("log", [])
        self.tts = shutil.which("espeak-ng") or shutil.which("espeak") if speak else None

    def _save(self):
        if self.memory:
            self.memory.parent.mkdir(parents=True, exist_ok=True)
            self.memory.write_text(json.dumps({"name": self.name, "log": self.log[-500:]}, indent=1))

    def mood_word(self):
        return next(w for lo, w in MOOD_WORDS if self.mood >= lo)

    def line(self, key, **kw):
        ctx = dict(name=self.name, battery=round(self.battery), mood=self.mood_word(), **kw)
        return re.sub(r"\{(\w+)\}", lambda m: str(ctx.get(m.group(1), "")), self.rng.choice(LINES[key]))

    def set_battery(self, percent):
        """The robot's real charge (LowState bms soc), replacing the per-trick estimate."""
        self.battery, self.real_battery = float(percent), True

    @staticmethod
    def clauses(text):
        return [c.strip() for c in re.split(r"\s*(?:[,;.!?]|\band then\b|\bthen\b|\bafter that\b|\band\b|\balso\b)\s*",
                                            text, flags=re.I) if c.strip()]

    def understand(self, text):
        steps = []
        for clause in self.clauses(text):
            rule = next(((k, r) for k, r in RULES if r.search(clause)), None)
            steps += self.respond(rule, clause)
        if not steps:
            steps = self.respond(None, text)
        self.log.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "said": text, "mood": round(self.mood, 2),
                         "battery": round(self.battery), "steps": [{"trick": s["trick"], "say": s["say"]} for s in steps]})
        self._save()
        return steps

    def respond(self, rule, clause):
        step = lambda trick, say, arg=None: {"trick": trick, "say": say,   # noqa: E731
                                             "moves": moves_for(trick, arg, self.mood)}
        if rule is None:
            return [step("confused", self.line("unknown"))]
        key, r = rule
        if key == "name":
            self.name = r.search(clause).group(1).capitalize()
            self.mood = min(1.0, self.mood + 0.1)
            return [step("wag", self.line("name"))]
        if key in ("askName", "greet", "talk", "help", "howAreYou"):
            return [step("wag" if key in ("askName", "greet") else "talk", self.line(key))]
        if key == "joke":
            self.mood = min(1.0, self.mood + 0.05)   # it likes making you laugh
            return [step("talk", self.rng.choice(JOKES))]
        if key in ("praise", "scold"):
            return self.feedback(1 if key == "praise" else -1)
        if key == "stop":
            return [step("stop", self.line("stop"))]
        arg = None
        if key == "walk":
            m = re.search(r"\b(left|right|back)", clause, re.I)
            arg = m.group(1).lower() if m else "forward"
        if key in NOT_YET:
            instead = SUBSTITUTE[key]
            say = self.line("notYet", trick=re.sub(r"([A-Z])", r" \1", key).lower(),
                            instead=re.sub(r"([A-Z])", r" \1", instead).lower())
            self.last_trick = instead
            return [step(instead, say)]
        cost = COST.get(key, 1)
        if key != "rest" and self.battery - cost < 5:
            return [step("rest", self.line("tired", trick=key))]
        if key == "rest" and not self.real_battery:
            self.battery = 100.0   # the browser pet's nap; on the robot, the charger decides
        elif not self.real_battery:
            self.battery -= cost
        self.last_trick = key
        if self.mood < 0.2 and key in ("dance", "jumpAround", "spin"):
            say = self.line("grumpy", trick=key)
        elif key == "walk":
            say = self.line("walk", dir=arg)
        else:
            say = self.line(key)
        if key in ("dance", "jumpAround"):
            self.mood = min(1.0, self.mood + 0.05)   # play lifts its mood a little
        return [step(key, say, arg)]

    def feedback(self, sign):
        self.mood = max(0.0, min(1.0, self.mood + 0.15 * sign))
        t = self.last_trick
        target = next((e for e in reversed(self.log) if any(s["trick"] == t for s in e["steps"])), None)
        if t and target is not None and "feedback" not in target:
            target["feedback"] = sign   # the training label: what it did, what you thought
        name = t and re.sub(r"([A-Z])", r" \1", t).lower()
        say = self.line(("praiseTrick" if sign > 0 else "scoldTrick"), trick=name) if t else \
            self.line("praise" if sign > 0 else "scold")
        return [{"trick": "happy" if sign > 0 else "sad", "say": say,
                 "moves": moves_for("wag", None, self.mood) if sign > 0 else [(3, "sit")]}]

    def speak(self, text):
        print(f"{self.name}: {text}", flush=True)
        if self.tts:   # a local voice, on the robot's computer
            subprocess.Popen([self.tts, "-s", "165", text.replace("*", "")], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)


def selftest():
    p = PetBrain(seed=1)
    s = p.understand("hi, sit then dance")
    assert [x["trick"] for x in s] == ["wag", "sit", "dance"], s
    assert s[1]["moves"][-1] == (4, "sit") and s[2]["moves"][-1][1] == "sit"
    assert p.understand("do a backflip")[0]["trick"] == "spin"            # honest substitute
    assert "haven't learned" in p.understand("roll over")[0]["say"] or "Not with" in p.understand("roll over")[0]["say"]
    m0 = p.mood
    p.understand("good boy")
    assert p.mood > m0 and p.log[-2]["steps"][0]["trick"] == "playDead" and p.log[-2].get("feedback") == 1
    for _ in range(6):
        p.understand("bad dog")
    assert p.mood_word() == "grumpy"
    s = p.understand("dance")[0]
    assert "If I must" in s["say"] or "slow one" in s["say"]
    assert all(c != "trot" for _, c in s["moves"]), s["moves"]                # a grumpy dog walks, no trot
    assert p.understand("walk left")[0]["moves"][0] == (0.0, "steer 0.5")
    assert p.understand("your name is Rex")[0]["say"].startswith("Rex") or "Rex" in p.understand("what's your name")[0]["say"]
    from controller import ALL_CLIPS
    for k, _ in RULES:   # every move is a real command of the robot program
        for trick in (k, SUBSTITUTE.get(k)):
            for _, c in moves_for(trick, "left", 0.9) + moves_for(trick, None, 0.1):
                assert c.split()[0] in ALL_CLIPS or c.startswith("steer "), (trick, c)
    print(f"pet: {len(RULES)} rules; every move maps to a trained Go2 skill; mood, feedback labels, substitutes OK")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    selftest()
