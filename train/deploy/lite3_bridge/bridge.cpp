// Bolt <-> DEEP Robotics Lite3 bridge: runs on the Lite3's motion host (or a laptop on its network) and speaks, towards
// our Python program (deploy/run_go2.py --body lite3, deploy/lite3_link.py), exactly the UDP protocol of DEEP's own
// simulator (Lite3_rl_deploy mujoco_simulation.py); towards the robot, DEEP's Lite3_MotionSDK (Sender / Receiver).
//
//   state  -> <bolt host>:30010   float64 time, float32 roll pitch yaw (rad), acc (3), gyro (3), q (12), dq (12), tau (12)
//   cmd    <- UDP port 20001       float32 x 12 each: kp | position | kd | velocity | torque  (240 bytes)
//
// Joint values pass through in DEEP's own order and sign (lite3_link.py converts to ours). Every 1 ms: read the robot,
// send the state, send the newest command. No command for 50 ms (our program stopped or the network dropped): damping
// (kp 0, kd 1) instead, so the robot sags down slowly; DEEP's SDK itself gives control back after 50 ms of silence.
// Ctrl-C hands control back to DEEP's own controller (ControlGet(ROBOT)).
//
// Units: the IMU angles arrive in degrees and are sent in radians (as DEEP's Lite3_rl_deploy HardwareInterface does).
// The angular velocity is passed as DEEP's deploy passes it (no conversion); robot_types.h calls it degrees/second:
// --gyro-deg converts it. HARDWARE_DAY check: turn the robot 90 degrees by hand and read the logged yaw rate.
//
// Build (x86_64 laptop or the robot's aarch64 host; DEEP's SDK is MIT-licensed, github.com/DeepRoboticsLab/Lite3_MotionSDK):
//   g++ -O2 -std=c++17 bridge.cpp -I$SDK/include -I$SDK/include/common -L$SDK/lib \
//       -ldeeprobotics_legged_sdk_$(uname -m) -lpthread -o lite3_bridge
//   ./lite3_bridge [--robot 192.168.2.1] [--bolt 127.0.0.1] [--gyro-deg]
#include <arpa/inet.h>
#include <csignal>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <thread>

#include "receiver.h"
#include "sender.h"

static volatile bool g_run = true;
static void on_sigint(int) { g_run = false; }

int main(int argc, char** argv) {
  std::string robot_ip = "192.168.2.1", bolt_ip = "127.0.0.1";
  bool gyro_deg = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--robot" && i + 1 < argc) robot_ip = argv[++i];
    else if (a == "--bolt" && i + 1 < argc) bolt_ip = argv[++i];
    else if (a == "--gyro-deg") gyro_deg = true;
    else { std::cerr << "usage: lite3_bridge [--robot IP] [--bolt IP] [--gyro-deg]\n"; return 2; }
  }
  std::signal(SIGINT, on_sigint);

  int rx = socket(AF_INET, SOCK_DGRAM, 0), tx = socket(AF_INET, SOCK_DGRAM, 0);
  sockaddr_in me{}, to{};
  me.sin_family = AF_INET; me.sin_port = htons(20001); me.sin_addr.s_addr = INADDR_ANY;
  if (bind(rx, (sockaddr*)&me, sizeof me) < 0) { perror("bind 20001"); return 1; }
  fcntl(rx, F_SETFL, O_NONBLOCK);
  to.sin_family = AF_INET; to.sin_port = htons(30010); inet_pton(AF_INET, bolt_ip.c_str(), &to.sin_addr);

  Sender* sender = new Sender(robot_ip, 43893);
  Receiver* receiver = new Receiver();
  RobotData* rd = &receiver->GetState();
  receiver->StartWork();
  sender->RobotStateInit();   // DEEP's call: joints to zero and SDK control (their examples start here)

  float cmd[60] = {0};
  auto last_cmd = std::chrono::steady_clock::now() - std::chrono::seconds(1);
  const double d2r = M_PI / 180.0;
  auto next = std::chrono::steady_clock::now();
  long n_cmd = 0, n_damp = 0;
  while (g_run) {
    next += std::chrono::milliseconds(1);
    float buf[60];
    ssize_t k;
    while ((k = recv(rx, buf, sizeof buf, 0)) == (ssize_t)sizeof buf) {   // newest command wins
      std::memcpy(cmd, buf, sizeof cmd);
      last_cmd = std::chrono::steady_clock::now();
    }
    // state out
    unsigned char out[8 + 45 * 4];
    double t = rd->tick * 0.001;
    float f[45];
    f[0] = rd->imu.angle_roll * d2r; f[1] = rd->imu.angle_pitch * d2r; f[2] = rd->imu.angle_yaw * d2r;
    f[3] = rd->imu.acc_x; f[4] = rd->imu.acc_y; f[5] = rd->imu.acc_z;
    double g = gyro_deg ? d2r : 1.0;
    f[6] = rd->imu.angular_velocity_roll * g; f[7] = rd->imu.angular_velocity_pitch * g; f[8] = rd->imu.angular_velocity_yaw * g;
    for (int j = 0; j < 12; ++j) {
      f[9 + j] = rd->joint_data.joint_data[j].position;
      f[21 + j] = rd->joint_data.joint_data[j].velocity;
      f[33 + j] = rd->joint_data.joint_data[j].torque;
    }
    std::memcpy(out, &t, 8); std::memcpy(out + 8, f, sizeof f);
    sendto(tx, out, sizeof out, 0, (sockaddr*)&to, sizeof to);
    // command in
    RobotCmd rc;
    std::memset(&rc, 0, sizeof rc);
    bool fresh = std::chrono::steady_clock::now() - last_cmd < std::chrono::milliseconds(50);
    for (int j = 0; j < 12; ++j) {
      if (fresh) {
        rc.joint_cmd[j].kp = cmd[j]; rc.joint_cmd[j].position = cmd[12 + j]; rc.joint_cmd[j].kd = cmd[24 + j];
        rc.joint_cmd[j].velocity = cmd[36 + j]; rc.joint_cmd[j].torque = cmd[48 + j];
      } else {
        rc.joint_cmd[j].kd = 1.0f;   // damping only
      }
    }
    fresh ? ++n_cmd : ++n_damp;
    sender->SendCmd(rc);
    std::this_thread::sleep_until(next);
  }
  sender->ControlGet(ROBOT);   // control back to DEEP's own controller
  std::cout << "lite3_bridge: " << n_cmd << " ms under Bolt, " << n_damp << " ms damping; control returned to the robot\n";
  return 0;
}
