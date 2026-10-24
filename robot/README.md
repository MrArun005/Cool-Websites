# Robot

A beginner robot: wheeled base, a head that pans and tilts, two arms with shoulder and elbow joints, and a brain that remembers things.

```
                ┌──────────────── Raspberry Pi (brain) ───────────────┐
 camera, mic ──▶│ brain.py  ─ memory.py (RAM: recent  /  SQLite: facts) │
                └───────────────┬─────────────────────────────────────┘
                                │ USB serial, text commands ("S 4 90")
                ┌───────────────▼──────── Arduino (spinal cord) ──────┐
                │ limbs.ino: smooth servo moves, wheels, distance,     │
                │            safety watchdog                           │
                └──┬──────────────┬──────────────────┬────────────────┘
               6 servos     TB6612 driver        HC-SR04 sensor
             (head, arms)   → 2 wheel motors       ("eyes")
```

## Try it now, no hardware needed

```bash
cd robot/brain
python3 brain.py
```

Type `help`. Tell it `my name is <you>`, quit, run it again, and it greets you by name. That's long-term memory (a `robot_memory.db` file). Ask `what did i just say` to see short-term memory, which lives in RAM and is gone after a restart.

## With the real robot

1. Open `firmware/limbs/limbs.ino` in the Arduino IDE, select your board, upload.
2. Test it in the Serial Monitor (115200 baud, newline): `S 0 45`, `M 150 150`, `X`, `D`.
3. Plug the Arduino into the Pi. On the Pi: `pip install pyserial`, then
   `python3 brain.py --port /dev/ttyACM0` (sometimes `/dev/ttyUSB0`).

## Wiring

| Arduino pin | Goes to |
|---|---|
| 2 | Servo 0: head pan (SG90) |
| 4 | Servo 1: head tilt (SG90) |
| 7 | Servo 2: left shoulder (MG996R) |
| 8 | Servo 3: left elbow (MG996R) |
| 12 | Servo 4: right shoulder (MG996R) |
| 13 | Servo 5: right elbow (MG996R) |
| 5 / A0 / A1 | TB6612 PWMA / AIN1 / AIN2 (left motor) |
| 6 / A2 / A3 | TB6612 PWMB / BIN1 / BIN2 (right motor) |
| 5V | TB6612 VCC and STBY |
| A4 / A5 | HC-SR04 TRIG / ECHO |
| GND | Every ground: battery, both regulators, driver, sensor, servos |

Power:

```
7.4 V battery ─ switch ─┬─ buck converter #1 (5.1 V, 5 A) ─▶ Raspberry Pi (USB-C or 5V pins)
                        ├─ buck converter #2 (5.1-6 V, 5 A) ─▶ servo red (+) wires
                        └─ TB6612 VM (motor power)
```

Servos and motors must **not** be powered from the Arduino or Pi 5 V pins. Set each buck converter's voltage with a multimeter *before* connecting anything to it.

## Before attaching arm parts

Upload the firmware with only the servos connected. They go to the home pose (head 90°, shoulders 10°, elbows 20°). Fit the servo horns and limbs in that pose so the joint limits in `limbs.ino` (`MIN_ANGLE` / `MAX_ANGLE`) match the real robot. If a joint hits its frame, tighten that limit.

## Ideas for next

- **Voice:** feed speech-to-text into `Robot.respond()` and speak the reply with text-to-speech.
- **Vision:** use the Pi camera with OpenCV to find faces and point the head at them (`head_pan`).
- **Smarter memory:** store more in `memory.py`, e.g. who it saw and when, or connect an AI model for conversation.
- **Legs:** once wheels feel easy, try a 4-legged walker (12 servos + a PCA9685 servo board).
