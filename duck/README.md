# Bolt Duck

A small two-legged robot pet you build yourself: Microduck-style walking body, Bolt's personality.

**Body:** [Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini), an open-source (Apache-2.0)
biped by Antoine Pirrone, sponsored by Hugging Face and Pollen Robotics. It is the do-it-yourself
ancestor of Pollen's Microduck: same idea, same 14-joint layout, same "train in simulation, then
run on the robot" approach. Everything is 3D-printed and fully documented.

**Personality (this repo):** talking, memory, humor, reacting to praise, following you, from
`pet/js/brain.js` and later Phase 3 of `pet/ROADMAP.md`.

## Build kit

Everything for the physical build is in [`build/`](build/):
- [SHOPPING.md](build/SHOPPING.md): every part to buy, with quantities and prices (about €432 with the pet package)
- [PRINT.md](build/PRINT.md): all 51 printed pieces with sizes and materials (needs a 200 × 200 mm printer bed)
- [CHECKLIST.md](build/CHECKLIST.md): the build, step by step, with a test at the end of each stage
- [get_parts.sh](build/get_parts.sh): downloads the print files and official guides into `duck/parts/`

## What you build

| | |
|---|---|
| Height | about 42 cm with legs extended |
| Weight | about 2 kg (simulation model: 2.06 kg) |
| Joints | 14 motors: per leg hip yaw, hip roll, hip pitch, knee, ankle; neck pitch, head pitch, head yaw, head roll; plus 2 small antenna motors |
| Brain | Raspberry Pi Zero 2 W (in the head) |
| Senses | IMU (balance), camera, microphone, foot switches |
| Expression | speaker, 2 eye LEDs, 2 antennas, a little "flashlight" |
| Power | 2 × 18650 cells (7.4 V) with BMS, USB-C charging, power switch |

## Parts (approximate, Oct 2026)

The project's own BOM says the full build should come in under $400. Rough breakdown:

| Part | Qty | ≈ Cost |
|---|---|---|
| Feetech STS3215 serial bus servos (7.4 V) | 14 | ~$210 (about €14 each) |
| Raspberry Pi Zero 2 W | 1 | ~$28 |
| BNO055 IMU | 1 | ~$40 |
| Bus servo driver board (USB) | 1 | ~$15 |
| 18650 cells, BMS, USB-C charger, switch, UBEC (5 V) | 1 set | ~$30 |
| Camera, mic, speaker, LEDs, small antenna servos | 1 set | ~$35 |
| Bearings, M3 heat-set inserts, M3 screws, threadlocker | 1 set | ~$20 |
| PLA (most parts) + a little TPU (foot soles) | ~600 g | ~$25 |

Exact list: the [BOM spreadsheet](https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing).

**No 3D printer?** A library or makerspace printer, or an online print service. All parts print in
standard PLA at 15% infill except the foot soles (TPU, 40%).

**Tools:** screwdrivers, a soldering iron (for heat-set inserts and battery wiring), a multimeter.

## Build steps

1. **Print** the 37 parts listed in the [print guide](https://github.com/apirrone/Open_Duck_Mini/blob/main/docs/print_guide.md).
2. **Configure every motor** (ID and zero position) before assembly:
   [configure_motors.md](https://github.com/apirrone/Open_Duck_Mini/blob/main/docs/configure_motors.md).
3. **Assemble** trunk → feet → shins → thighs → hips → neck → head:
   [assembly guide](https://github.com/apirrone/Open_Duck_Mini/blob/main/docs/assembly_guide.md),
   or the illustrated [Tnkr guide](https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home).
4. **Wire** it: [wiring diagram](https://github.com/apirrone/Open_Duck_Mini/blob/main/docs/open_duck_mini_v2_wiring_diagram.png).
5. **Run the walking policy** with the onboard runtime:
   [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime).
6. **Add Bolt's personality** (next section).

## What this repo adds

The duck's makers built the body and walking. Bolt turns it into a pet:

- **Talk:** speech in through the head mic, the brain (`pet/js/brain.js`, later a language model)
  decides, speech out of the head speaker.
- **Obey:** commands map to the duck's skills: walk, turn, stop, look around, and later sit,
  stand up and the forward roll (Microduck's training tasks show all of these are learnable at this size).
- **Follow you:** the head camera finds a person; head yaw keeps them centred and the body turns
  and walks toward them.
- **Feel alive:** eye LEDs and antennas show mood; it remembers your name and what you praise.

## Simulation

The duck's MuJoCo model loads in this repo's training setup on a normal CPU, so we can test commands
and the brain before building. Training new skills on this body needs an NVIDIA GPU: the duck
projects train with MuJoCo Playground / mjlab, thousands of robots at once.

## Smaller later: a 25 cm Microduck-size version

Pollen Robotics published Microduck's robot model and part meshes
([microduck_rl](https://github.com/pollen-robotics/microduck_rl)). The code and model files are
Apache-2.0; the part meshes are CC BY-NC-SA 4.0, so they're fine for a personal, non-commercial
build but must not be sold. There's no official assembly guide or BOM, so it's a harder second
project once Bolt Duck walks.
