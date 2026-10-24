// limbs.ino — the robot's "spinal cord".
//
// Runs on an Arduino Uno / Nano. It does not think; it just obeys short text
// commands from the brain (Raspberry Pi) over USB serial and moves the body:
//
//   S <joint> <angle>   move a joint (0-5) to an angle (degrees), smoothly
//   M <left> <right>    drive the wheels, each -255 (reverse) .. 255 (forward)
//   X                   stop the wheels
//   D                   measure distance; replies "D <cm>"
//   P                   print every joint angle; replies "P a0 a1 a2 a3 a4 a5"
//
// Every command gets "OK" or "ERR <reason>" back. Lines end with '\n'.
//
// Safety: if the brain goes quiet for WATCHDOG_MS while the wheels are
// turning, the wheels stop by themselves.
//
// Wiring (see robot/README.md for the full diagram):
//   Servos (signal wires)  : pins 2, 4, 7, 8, 12, 13
//   TB6612FNG motor driver : PWMA=5 AIN1=A0 AIN2=A1  PWMB=6 BIN1=A2 BIN2=A3  STBY->5V
//   HC-SR04 distance sensor: TRIG=A4 ECHO=A5
//   Servos and motors MUST be powered from the battery (via a 5-6V regulator),
//   never from the Arduino's 5V pin. Connect all grounds together.

#include <Servo.h>

const int JOINTS = 6;
// Joint numbers are shared with robot/brain/body.py and the website.
// 0 head pan, 1 head tilt, 2 left shoulder, 3 left elbow, 4 right shoulder, 5 right elbow
const char *JOINT_NAMES[JOINTS] = {"head_pan", "head_tilt", "l_shoulder", "l_elbow", "r_shoulder", "r_elbow"};
const int SERVO_PINS[JOINTS] = {2, 4, 7, 8, 12, 13};
// Mechanical limits: never drive a joint past these or it may jam and burn out.
const int MIN_ANGLE[JOINTS] = {30, 60, 0, 0, 0, 0};
const int MAX_ANGLE[JOINTS] = {150, 120, 180, 150, 180, 150};
const int HOME_ANGLE[JOINTS] = {90, 90, 10, 20, 10, 20};

const int PWMA = 5, AIN1 = A0, AIN2 = A1;  // left wheel
const int PWMB = 6, BIN1 = A2, BIN2 = A3;  // right wheel
const int TRIG = A4, ECHO = A5;

const unsigned long STEP_MS = 15;       // servo smoothing tick: 1 degree per tick
const unsigned long WATCHDOG_MS = 1000; // stop wheels if no command for this long

Servo servos[JOINTS];
int current[JOINTS];
int target[JOINTS];

bool wheelsMoving = false;
unsigned long lastCommandAt = 0;
unsigned long lastStepAt = 0;

char line[40];
byte lineLen = 0;

void setMotor(int pwmPin, int in1, int in2, int speed) {
  speed = constrain(speed, -255, 255);
  digitalWrite(in1, speed > 0);
  digitalWrite(in2, speed < 0);
  analogWrite(pwmPin, abs(speed));
}

void drive(int left, int right) {
  setMotor(PWMA, AIN1, AIN2, left);
  setMotor(PWMB, BIN1, BIN2, right);
  wheelsMoving = (left != 0 || right != 0);
}

long readDistanceCm() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);
  unsigned long us = pulseIn(ECHO, HIGH, 30000UL);  // ~5 m max
  if (us == 0) return -1;                           // nothing in range
  return us / 58;                                   // sound: ~58 us per cm round trip
}

void handleCommand(char *cmd) {
  char op = cmd[0];
  int a = 0, b = 0;

  if (op == 'S' && sscanf(cmd + 1, "%d %d", &a, &b) == 2) {
    if (a < 0 || a >= JOINTS) { Serial.println(F("ERR bad joint")); return; }
    target[a] = constrain(b, MIN_ANGLE[a], MAX_ANGLE[a]);
    Serial.println(F("OK"));
  } else if (op == 'M' && sscanf(cmd + 1, "%d %d", &a, &b) == 2) {
    drive(a, b);
    Serial.println(F("OK"));
  } else if (op == 'X') {
    drive(0, 0);
    Serial.println(F("OK"));
  } else if (op == 'D') {
    Serial.print(F("D "));
    Serial.println(readDistanceCm());
  } else if (op == 'P') {
    Serial.print('P');
    for (int i = 0; i < JOINTS; i++) { Serial.print(' '); Serial.print(current[i]); }
    Serial.println();
  } else {
    Serial.println(F("ERR unknown command"));
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(AIN1, OUTPUT); pinMode(AIN2, OUTPUT); pinMode(PWMA, OUTPUT);
  pinMode(BIN1, OUTPUT); pinMode(BIN2, OUTPUT); pinMode(PWMB, OUTPUT);
  pinMode(TRIG, OUTPUT); pinMode(ECHO, INPUT);
  drive(0, 0);

  for (int i = 0; i < JOINTS; i++) {
    current[i] = target[i] = HOME_ANGLE[i];
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(current[i]);
  }
  Serial.println(F("READY"));
}

void loop() {
  // 1. Read commands one character at a time so we never block the servos.
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[lineLen] = '\0';
      if (lineLen > 0) handleCommand(line);
      lineLen = 0;
      lastCommandAt = millis();
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = c;
    }
  }

  // 2. Ease each joint one degree toward its target, so limbs move smoothly
  //    instead of snapping (snapping draws huge current and shakes the robot).
  if (millis() - lastStepAt >= STEP_MS) {
    lastStepAt = millis();
    for (int i = 0; i < JOINTS; i++) {
      if (current[i] != target[i]) {
        current[i] += (target[i] > current[i]) ? 1 : -1;
        servos[i].write(current[i]);
      }
    }
  }

  // 3. Watchdog: if the brain crashed or the cable fell out, don't drive off the table.
  if (wheelsMoving && millis() - lastCommandAt > WATCHDOG_MS) {
    drive(0, 0);
  }
}
