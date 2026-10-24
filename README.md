# Cool Websites

## 🐕 Robot dog sim

Open `pet/index.html`. Everything it needs is in the repo, so it works offline. It's a physics-simulated robot dog you can talk to, by typing or with your voice. It can sit, jump, jump around, backflip, somersault, roll over, play dead, shake, beg, dance and tell jokes, and it talks back. Every jump is real ballistics with conserved angular momentum, and the panel shows the leg force and energy each trick needs. It remembers its name and logs your feedback as training data.

The plan for what comes next (full physics with learned tricks, understanding people and humor, then a real body) is in [`pet/ROADMAP.md`](pet/ROADMAP.md).

```bash
node --test pet/test/*.test.js   # physics, trick and brain tests
```

### Training the real-physics dog

[`train/`](train/README.md) has Bolt as a MuJoCo physics model (12 motors, real contacts and friction) and a reinforcement-learning setup that teaches it to stand, sit, lie down and walk on command. It trains on a normal 4-core CPU in about an hour. The roadmap includes a [budget](pet/ROADMAP.md#budget) for every phase.

## 🤖 Build a Robot

Open `index.html` in a browser for an interactive guide to building your first robot. It covers:

- **Robot lab:** move the head and arm servos with sliders, and see the serial commands the brain sends
- **Anatomy:** what each part does (brain, spinal cord, muscles, senses, heart)
- **Memory & RAM:** a hands-on demo of RAM versus storage
- **Shopping list**, **8 build steps** and **safety rules**

The real robot code is in [`robot/`](robot/README.md): a Python brain with memory for the Raspberry Pi, and Arduino firmware for the limbs. You can try the brain right now without any hardware:

```bash
cd robot/brain && python3 brain.py
```
