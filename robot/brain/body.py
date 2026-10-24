"""The brain's connection to the body.

`Body` talks to the Arduino (robot/firmware/limbs/limbs.ino) over USB serial.
`SimBody` has the same methods but just prints what it would do, so you can
try the brain on any laptop before you own a single motor.
"""

import time

# Same numbering as the firmware and the website.
JOINTS = {
    "head_pan": 0,
    "head_tilt": 1,
    "l_shoulder": 2,
    "l_elbow": 3,
    "r_shoulder": 4,
    "r_elbow": 5,
}
HOME = {"head_pan": 90, "head_tilt": 90, "l_shoulder": 10, "l_elbow": 20, "r_shoulder": 10, "r_elbow": 20}


class Body:
    def __init__(self, port, baud=115200):
        import serial  # pip install pyserial

        self.link = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # the Arduino resets when the port opens
        self.link.reset_input_buffer()

    def send(self, command):
        self.link.write((command + "\n").encode())
        return self.link.readline().decode().strip()

    def move(self, joint, angle):
        return self.send(f"S {JOINTS[joint]} {int(angle)}")

    def drive(self, left, right, seconds):
        # Re-send while driving so the firmware watchdog doesn't stop us.
        end = time.time() + seconds
        while time.time() < end:
            self.send(f"M {int(left)} {int(right)}")
            time.sleep(min(0.3, max(0, end - time.time())))
        self.stop()

    def stop(self):
        return self.send("X")

    def distance_cm(self):
        reply = self.send("D")  # "D 42"
        try:
            return int(reply.split()[1])
        except (IndexError, ValueError):
            return -1

    def close(self):
        self.stop()
        self.link.close()


class SimBody:
    """Pretend body: prints actions instead of moving motors."""

    def __init__(self):
        self.angles = dict(HOME)

    def move(self, joint, angle):
        self.angles[joint] = int(angle)
        print(f"   [body] {joint} -> {int(angle)} deg   (serial: S {JOINTS[joint]} {int(angle)})")
        return "OK"

    def drive(self, left, right, seconds):
        print(f"   [body] wheels L={left} R={right} for {seconds}s   (serial: M {left} {right})")
        return "OK"

    def stop(self):
        print("   [body] wheels stop   (serial: X)")
        return "OK"

    def distance_cm(self):
        return 57  # a pretend wall

    def close(self):
        pass
