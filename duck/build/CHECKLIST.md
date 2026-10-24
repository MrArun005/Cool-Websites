# Bolt Duck: build checklist

Work through it in order. Each stage ends with a test, so a mistake is caught before it's buried.
Detailed photos and steps for stages 3–6 are in the official
[assembly guide](https://github.com/apirrone/Open_Duck_Mini/blob/main/docs/assembly_guide.md) and the
[Tnkr guide](https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home).

## 1. Get the parts
- [ ] Order everything in [SHOPPING.md](SHOPPING.md). Servos and the IMU take longest to arrive.
- [ ] Run `./get_parts.sh` to download the print files.

## 2. Print
- [ ] **Test print first:** one `leg_spacer.stl` and one `knee_to_ankle_left_sheet.stl`. Check a servo
      and its screws fit before printing the rest.
- [ ] Print all 51 pieces in [PRINT.md](PRINT.md) (PLA 15%; foot soles in TPU).
- [ ] Remove supports, clean holes, press in the M3 heat-set inserts with the soldering iron.

## 3. Set up the motors (before assembly)
- [ ] Flash Raspberry Pi OS Lite to the SD card; enable Wi-Fi and SSH.
- [ ] Install the runtime: [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime) (`v2` branch).
- [ ] For each servo: `python configure_motor.py --id <id>` sets its ID and moves it to zero.
      Label each servo with its ID. Fit the horn in the zero position.

| Joint | ID | Joint | ID |
|---|---|---|---|
| left_hip_yaw | 20 | right_hip_yaw | 10 |
| left_hip_roll | 21 | right_hip_roll | 11 |
| left_hip_pitch | 22 | right_hip_pitch | 12 |
| left_knee | 23 | right_knee | 13 |
| left_ankle | 24 | right_ankle | 14 |
| neck_pitch | 30 | head_pitch | 31 |
| head_yaw | 32 | head_roll | 33 |


## 4. Build the body
- [ ] Trunk: bearings into `trunk_bottom`, join with `trunk_top`, mount the middle motor.
- [ ] Feet: foot switches in, servo in with its driver side toward `foot_top`.
- [ ] Shins, then thighs (watch the `hip_pitch` motor direction: it sets the zero position).
- [ ] Hips: `roll_motor_top` onto the `hip_yaw` servo, then onto the trunk.
- [ ] Threadlocker on every metal-to-metal screw, never on plastic screws.
- [ ] **Test:** with the robot held in the air, move each leg joint from the runtime. No grinding, correct direction.

## 5. Power and electronics
- [ ] Battery pack: 2 × 18650 in the holder → BMS → switch → XT30. Check with the multimeter: ~7.4 V, correct polarity.
- [ ] UBEC set to 5 V → Pi Zero 2 W. Servo board straight from 7.4 V.
- [ ] IMU on the Pi's I2C pins; servo board on USB.
- [ ] **Test:** power from the battery, the Pi boots, all 14 servos answer.

## 6. Neck and head
- [ ] Neck motor, head mechanism (pitch → yaw → roll), head shell.
- [ ] Antenna servos, eye LEDs and diffusers, flashlight.
- [ ] **Test:** look left/right/up/down from the runtime.

## 7. Bolt's pet hardware (our additions)
These are not wired in the original design yet; the pin plan below is ours and untested on this robot.
- [ ] **Camera:** Pi camera on the Zero's camera connector (needs the narrow 22-pin Zero cable), lens through the head's front.
- [ ] **Speaker:** MAX98357A I2S amplifier: BCLK → GPIO18, LRCLK → GPIO19, DIN → GPIO21, power 5 V.
- [ ] **Microphone:** INMP441 I2S mic sharing BCLK/LRCLK: SD → GPIO20, L/R → GND, power 3.3 V.
- [ ] **Test:** `arecord` a few seconds and play it back through the speaker.

## 8. First steps
- [ ] Stand it up on the floor, hand near it, and run the standing policy.
- [ ] Run the walking policy (`BEST_WALK_ONNX_2.onnx` from the Open Duck Mini repo).
- [ ] Charge habit: stop when it slows down; don't run the cells flat.

## 9. Make it Bolt
- [ ] Install Bolt's brain on the Pi: voice in → command → duck skill → voice out.
- [ ] Give it a name. It should remember it after a reboot.
