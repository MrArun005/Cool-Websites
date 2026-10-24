"""The robot's brain. Runs on the Raspberry Pi (or any computer, in simulation).

    python3 brain.py                      # simulated body, try it anywhere
    python3 brain.py --port /dev/ttyACM0  # real Arduino plugged into USB

Type "help" to see what it understands.
"""

import argparse
import re
import time

from body import HOME, Body, SimBody
from memory import Memory

HELP = """I understand:
  my name is <name>          remember <thing> is <value>
  what is my name            what is <thing>
  forget <thing>             what do you know
  what did i just say        diary
  wave | nod | dance | look left | look right | home
  forward | back | turn left | turn right   (add seconds, e.g. "forward 2")
  distance                   quit"""


class Robot:
    def __init__(self, body, memory):
        self.body = body
        self.memory = memory

    # --- movements ----------------------------------------------------------
    def pose(self, pause=0.4, **angles):
        for joint, angle in angles.items():
            self.body.move(joint, angle)
        time.sleep(pause)  # give the servos time to get there

    def home(self):
        self.pose(**HOME)

    def wave(self):
        self.pose(r_shoulder=90, r_elbow=60)
        for _ in range(3):
            self.pose(pause=0.3, r_elbow=120)
            self.pose(pause=0.3, r_elbow=60)
        self.home()

    def nod(self):
        for _ in range(2):
            self.pose(pause=0.25, head_tilt=70)
            self.pose(pause=0.25, head_tilt=110)
        self.pose(head_tilt=90)

    def dance(self):
        for _ in range(2):
            self.pose(pause=0.3, l_shoulder=150, r_shoulder=30, head_pan=60)
            self.pose(pause=0.3, l_shoulder=30, r_shoulder=150, head_pan=120)
        self.body.drive(180, -180, 0.6)
        self.home()

    # --- conversation ---------------------------------------------------------
    def respond(self, text):
        said = text.strip()
        t = said.lower().rstrip("?!. ")
        self.memory.hear(said)

        if t in ("help", "?"):
            return HELP
        if re.fullmatch(r"my name is (.+)", t):
            name = said[len("my name is "):].strip()
            self.memory.remember("name", name)
            self.memory.log(f"met {name}")
            return f"Nice to meet you, {name}! I'll remember that, even after I'm switched off."
        # Match on the original text so values keep their capitals ("my dog is Rex").
        if m := re.fullmatch(r"remember (?:that )?(.+?) (?:is|are) (.+?)[.!]?", said, re.IGNORECASE):
            self.memory.remember(m.group(1), m.group(2))
            return f"Saved to long-term memory: {m.group(1)} = {m.group(2)}"
        if t in ("what is my name", "what's my name", "who am i"):
            name = self.memory.recall("name")
            return f"You are {name}." if name else "I don't know yet. Tell me: 'my name is ...'"
        if m := re.fullmatch(r"(?:what is|what's|what are) (?:the )?(.+)", t):
            key = m.group(1)
            value = self.memory.recall(key)
            return f"{key} is {value}." if value else f"I don't remember anything about '{key}'."
        if m := re.fullmatch(r"forget (.+)", t):
            return "Forgotten." if self.memory.forget(m.group(1)) else "I didn't know that anyway."
        if t in ("what do you know", "list memories"):
            facts = self.memory.all_facts()
            return "\n".join(f"  {k}: {v}" for k, v in facts) or "My long-term memory is empty."
        if t in ("what did i just say", "what did i say"):
            recent = self.memory.recent()[:-1]  # skip this question itself
            return "Recently (short-term / RAM):\n" + "\n".join(f"  - {r}" for r in recent[-5:]) if recent else "Nothing yet."
        if t == "diary":
            entries = self.memory.diary()
            return "\n".join(f"  {at}  {txt}" for at, txt in entries) or "My diary is empty."

        if t in ("wave", "nod", "dance", "home"):
            getattr(self, t)()
            self.memory.log(t)
            return f"*{t}s*" if t != "home" else "Back to home pose."
        if t in ("look left", "look right"):
            self.pose(head_pan=140 if t.endswith("left") else 40)
            return "Looking."
        if m := re.fullmatch(r"(forward|back|turn left|turn right)(?: ([\d.]+))?", t):
            speed = 180
            left, right = {
                "forward": (speed, speed),
                "back": (-speed, -speed),
                "turn left": (-speed, speed),
                "turn right": (speed, -speed),
            }[m.group(1)]
            seconds = min(float(m.group(2) or 1), 5)  # cap it, for safety
            cm = self.body.distance_cm()
            if m.group(1) == "forward" and 0 < cm < 20:
                return f"Something is {cm} cm ahead. I won't drive into it."
            self.body.drive(left, right, seconds)
            self.memory.log(f"drove {m.group(1)} {seconds}s")
            return "Done."
        if t == "distance":
            cm = self.body.distance_cm()
            return f"The nearest thing ahead is {cm} cm away." if cm >= 0 else "Nothing in range."

        return "I don't understand that yet. Type 'help'."


def main():
    parser = argparse.ArgumentParser(description="Robot brain")
    parser.add_argument("--port", help="Arduino serial port, e.g. /dev/ttyACM0 (omit to simulate)")
    parser.add_argument("--memory", default="robot_memory.db", help="long-term memory file")
    args = parser.parse_args()

    body = Body(args.port) if args.port else SimBody()
    memory = Memory(args.memory)
    robot = Robot(body, memory)

    name = memory.recall("name")
    print("Robot online." + (f" Welcome back, {name}!" if name else " Type 'help' to start."))
    robot.home()
    try:
        while True:
            try:
                text = input("you> ")
            except EOFError:
                break
            if text.strip().lower() in ("quit", "exit", "bye"):
                break
            if text.strip():
                print("robot>", robot.respond(text))
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down. Short-term memory (RAM) is gone; long-term memory is saved.")
        body.close()
        memory.close()


if __name__ == "__main__":
    main()
