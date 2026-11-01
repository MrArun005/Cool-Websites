"""The Go2's own wireless remote as Bolt's controls (run_go2.py --remote). Reads LowState.wireless_remote (40 bytes,
the layout of unitree_sdk2py.utils.joystick, which Unitree's simulator also fills) and turns it into the same text
commands --interactive takes, so the remote and the keyboard share one tested path.

    Left stick forward     a little: walk; most of the way: trot; all the way with R1 held: canter.
    Left stick released    for 0.5 s while walking/trotting: sit (the dog's "stop"; it never stands frozen).
    Right stick sideways   steer (left = turn left), up to 0.6 rad/s, on the gaits.
    A sit   B lie   Y get up
    L2 + B, or SELECT      stop: damping mode (Unitree's own damping combo is L2 + B too).

Hysteresis on the stick bands, so a stick near a band edge does not flip the gait back and forth.
"""
import struct

BUTTONS = {   # byte, bit (MSB first, as unitree_sdk2py.utils.joystick.extract reads them)
    "L2": (2, 2), "R2": (2, 3), "SELECT": (2, 4), "START": (2, 5), "L1": (2, 6), "R1": (2, 7),
    "left": (3, 0), "down": (3, 1), "right": (3, 2), "up": (3, 3), "Y": (3, 4), "X": (3, 5), "B": (3, 6), "A": (3, 7)}
GAITS = ("walk", "trot", "canter")
MAX_STEER = 0.6


def decode(raw):
    """40 remote bytes -> ({button: pressed}, {"lx", "rx", "ry", "ly": -1..1})."""
    raw = bytes(bytearray(raw))
    buttons = {k: bool((raw[b] >> (7 - i)) & 1) for k, (b, i) in BUTTONS.items()}
    f = lambda a: struct.unpack("f", raw[a:a + 4])[0]   # noqa: E731
    return buttons, {"lx": f(4), "rx": f(8), "ry": f(12), "ly": f(20)}


def encode(buttons=(), lx=0.0, rx=0.0, ry=0.0, ly=0.0):
    """The inverse (tests and the simulator's scripted remote)."""
    raw = bytearray(40)
    for k in buttons:
        b, i = BUTTONS[k]
        raw[b] |= 1 << (7 - i)
    for a, v in ((4, lx), (8, rx), (12, ry), (20, ly)):
        raw[a:a + 4] = struct.pack("f", float(v))
    return raw


class RemoteCommander:
    def __init__(self, dt=0.02):
        self.dt, self.prev, self.band, self.released, self.steer, self.stopped = dt, {}, "none", 0.0, 0.0, False

    def _band(self, ly, r1):
        up = {"none": 0, "walk": 1, "trot": 2, "canter": 3}[self.band]   # hysteresis: leave a band 0.1 inside it
        if ly >= 0.95 and r1:
            return "canter"
        if ly >= (0.6 if up >= 2 else 0.7):
            return "trot"
        if ly >= (0.15 if up >= 1 else 0.25):
            return "walk"
        return "none"

    def update(self, raw):
        """One poll (every dt): the commands to send now, as --interactive text ("trot", "steer 0.3", "stop")."""
        if self.stopped:
            return []
        b, s = decode(raw)
        pressed = lambda k: b[k] and not self.prev.get(k, False)   # noqa: E731  (on the press, not while held)
        out = []
        if (b["L2"] and b["B"]) or b["SELECT"]:
            self.stopped = True
            return ["stop"]
        for key, clip in (("A", "sit"), ("Y", "getup")):
            if pressed(key):
                out.append(clip)
                self.band = "none"
        if pressed("B") and not b["L2"]:
            out.append("lie")
            self.band = "none"
        band = self._band(s["ly"], b["R1"])
        if band != "none":
            self.released = 0.0
            if band != self.band:
                out.append(band)
                self.band = band
        elif self.band in GAITS:
            self.released += self.dt
            if self.released >= 0.5:
                out.append("sit")
                self.band = "none"
        steer = 0.0 if abs(s["rx"]) < 0.1 else round(-MAX_STEER * max(-1.0, min(1.0, s["rx"])), 2)
        if abs(steer - self.steer) >= 0.05 or (steer == 0.0 and self.steer != 0.0):
            out.append(f"steer {steer}")
            self.steer = steer
        self.prev = b
        return out


def selftest():
    """Decode = the SDK's own reader on the same bytes, and the mapping on a short scripted session."""
    from unitree_sdk2py.utils.joystick import Joystick
    raw = encode(["L2", "B", "R1", "up", "A"], lx=0.1, rx=-0.5, ry=0.3, ly=0.8)
    j = Joystick()
    j.extract(list(raw))
    b, s = decode(raw)
    assert (b["L2"], b["B"], b["R1"], b["up"], b["A"], b["Y"]) == (j.LT.pressed, j.B.pressed, j.RB.pressed,
                                                                     j.up.pressed, j.A.pressed, False)
    k = j.ly.smooth   # the SDK's reader low-passes the sticks (one call from 0: smooth x value)
    assert abs(k * s["ly"] - j.ly.data) < 1e-6 and abs(k * s["rx"] - j.rx.data) < 1e-6, (s, j.ly.data, j.rx.data)
    rc, got = RemoteCommander(), []
    script = [encode(ly=0.5)] * 5 + [encode(ly=0.68)] * 5 + [encode(ly=0.8)] * 5 + [encode(ly=0.65, rx=-0.5)] * 5 \
        + [encode()] * 30 + [encode(["B"])] * 3 + [encode()] * 3 + [encode(["Y"])] * 2 + [encode(["L2", "B"])] * 2
    for raw in script:
        got += rc.update(raw)
    want = ["walk", "trot", "steer 0.3", "steer 0.0", "sit", "lie", "getup", "stop"]
    assert got == want, got
    print(f"remote: decode matches the SDK's reader; scripted session -> {got}")


if __name__ == "__main__":
    selftest()
