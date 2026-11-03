#!/usr/bin/env bash
# One-command setup of the Bolt Go2 robot software (HARDWARE_DAY.md step 0), on a Linux laptop or the Go2's Jetson.
#
#   bash train/deploy/setup_go2.sh            # robot software only
#   bash train/deploy/setup_go2.sh --sim      # + Unitree's simulator, for the rehearsal before the robot
#   bash train/deploy/setup_go2.sh --lite3    # + the DEEP Robotics Lite3: their MJCF, simulator and MotionSDK, our bridge
#                                             #   (flags combine: --sim --lite3)
#
# Installs into ./bolt-deps next to the repo checkout (nothing system-wide; delete the folder to undo):
#   CycloneDDS 0.10.x (C library, built from source; Unitree's SDK needs exactly this line), a Python 3.11 venv with
#   cyclonedds 0.10.2, unitree_sdk2_python, numpy, mujoco 3.17.0, gymnasium; with --sim also unitree_mujoco.
# Versions are pinned to the ones every rehearsal in PLAN.md ran with. Ends with check_ready.py.
# Needs: git, cmake, a C compiler, python3.11 (or uv), internet. Nothing here calls an external AI service.
set -euo pipefail

SIM=0; LITE3=0
for arg in "$@"; do
  case "$arg" in --sim) SIM=1;; --lite3) LITE3=1;; *) echo "unknown option: $arg"; exit 2;; esac
done
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS="${BOLT_DEPS:-$REPO/bolt-deps}"
CYCLONE_REF=5041f3560c088c99e5088b2b8520b69169621196      # eclipse-cyclonedds/cyclonedds releases/0.10.x
SDK_REF=814556d15970dd2ecf1c9984e845ca02ab07e206          # unitreerobotics/unitree_sdk2_python
SIM_REF=1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d          # unitreerobotics/unitree_mujoco
L3_MODEL_REF=52352f5bbf2a30b5d02ece91900415188e7abc74     # DeepRoboticsLab/URDF_model (Lite3/mjcf only, sparse)
L3_DEPLOY_REF=1c4a8f14c9c531903fa5a26783e3f9b99d1039d9    # DeepRoboticsLab/Lite3_rl_deploy (their simulator)
L3_SDK_REF=b30a3ec09619e1dd0f3cfa6c49a50eb59670f141       # DeepRoboticsLab/Lite3_MotionSDK

say() { printf '\n== %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing: $1 ($2)"; exit 1; }; }
need git "sudo apt install git"
need cmake "sudo apt install cmake"
need cc "sudo apt install build-essential"
mkdir -p "$DEPS"

fetch() {   # fetch <url> <dir> <commit>: a pinned checkout, reused when already there
  if [[ ! -d "$2/.git" ]]; then git clone -q "$1" "$2"; fi
  git -C "$2" fetch -q origin "$3" 2>/dev/null || git -C "$2" fetch -q --all
  git -C "$2" checkout -q "$3"
}

say "CycloneDDS C library (0.10.x, pinned)"
fetch https://github.com/eclipse-cyclonedds/cyclonedds "$DEPS/cyclonedds" "$CYCLONE_REF"
if [[ ! -f "$DEPS/cyclonedds/install/lib/libddsc.so" ]]; then
  cmake -S "$DEPS/cyclonedds" -B "$DEPS/cyclonedds/build" -DCMAKE_INSTALL_PREFIX="$DEPS/cyclonedds/install" \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SSL=OFF >/dev/null
  cmake --build "$DEPS/cyclonedds/build" --target install -j"$(nproc)" >/dev/null
fi
export CYCLONEDDS_HOME="$DEPS/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"

say "Python 3.11 environment"
if [[ ! -x "$DEPS/venv/bin/python" ]]; then
  if command -v uv >/dev/null; then uv venv -q -p 3.11 "$DEPS/venv"
  else need python3.11 "sudo apt install python3.11 python3.11-venv, or install uv"; python3.11 -m venv "$DEPS/venv"; fi
fi
PY="$DEPS/venv/bin/python"
if command -v uv >/dev/null; then PIP=(uv pip install -q -p "$PY"); else "$PY" -m pip install -q -U pip; PIP=("$PY" -m pip install -q); fi
"${PIP[@]}" "cyclonedds==0.10.2" "numpy>=1.26,<3" "mujoco==3.17.0" "gymnasium>=1.0"

say "Unitree SDK (pinned)"
fetch https://github.com/unitreerobotics/unitree_sdk2_python "$DEPS/unitree_sdk2_python" "$SDK_REF"
"${PIP[@]}" -e "$DEPS/unitree_sdk2_python"

if [[ $SIM == 1 ]]; then
  say "Unitree's simulator (rehearsal only, pinned)"
  fetch https://github.com/unitreerobotics/unitree_mujoco "$DEPS/unitree_mujoco" "$SIM_REF"
  "${PIP[@]}" pygame
fi

if [[ $LITE3 == 1 ]]; then
  say "DEEP Robotics Lite3: model, simulator, MotionSDK (pinned) and our bridge"
  if [[ ! -d "$DEPS/URDF_model/.git" ]]; then   # 0.5 GB of every DEEP robot: only the Lite3 MJCF is checked out
    git clone -q --filter=blob:none --no-checkout https://github.com/DeepRoboticsLab/URDF_model "$DEPS/URDF_model"
    git -C "$DEPS/URDF_model" sparse-checkout set Lite3/mjcf
  fi
  git -C "$DEPS/URDF_model" checkout -q "$L3_MODEL_REF"
  fetch https://github.com/DeepRoboticsLab/Lite3_rl_deploy "$DEPS/Lite3_rl_deploy" "$L3_DEPLOY_REF"
  fetch https://github.com/DeepRoboticsLab/Lite3_MotionSDK "$DEPS/Lite3_MotionSDK" "$L3_SDK_REF"
  "${PIP[@]}" "colorama==0.4.6"
  L3SDK="$DEPS/Lite3_MotionSDK"
  if command -v g++ >/dev/null && [[ -f "$L3SDK/lib/libdeeprobotics_legged_sdk_$(uname -m).so" ]]; then
    g++ -O2 -std=c++17 "$REPO/train/deploy/lite3_bridge/bridge.cpp" -I"$L3SDK/include" -I"$L3SDK/include/common" \
      -L"$L3SDK/lib" -ldeeprobotics_legged_sdk_"$(uname -m)" -lpthread -o "$DEPS/lite3_bridge"
    echo "built $DEPS/lite3_bridge (run it on the robot's network: ./lite3_bridge --robot <ip> --bolt <this machine>)"
  else
    echo "lite3_bridge not built (needs g++ and DEEP's SDK library for $(uname -m)); the simulator rehearsal works without it"
  fi
fi

cat > "$DEPS/env.sh" <<EOF
# source this before running the robot programs:  source $DEPS/env.sh
export CYCLONEDDS_HOME="$CYCLONEDDS_HOME"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:\${LD_LIBRARY_PATH:-}"
$( [[ $SIM == 1 ]] && echo "export UNITREE_MUJOCO=\"$DEPS/unitree_mujoco\"" )
$( [[ $LITE3 == 1 ]] && printf 'export LITE3_MJCF="%s"\nexport LITE3_RL_DEPLOY="%s"\nexport LD_LIBRARY_PATH="%s:${LD_LIBRARY_PATH:-}"' "$DEPS/URDF_model/Lite3/mjcf/Lite3.xml" "$DEPS/Lite3_rl_deploy" "$DEPS/Lite3_MotionSDK/lib" )
export PATH="$DEPS/venv/bin:\$PATH"
EOF

say "Readiness check"
# shellcheck disable=SC1091
source "$DEPS/env.sh"
cd "$REPO/train"
python deploy/check_ready.py
if [[ $LITE3 == 1 ]]; then python deploy/check_ready.py --body lite3; fi
echo
echo "Next time: source $DEPS/env.sh && cd $REPO/train   (then HARDWARE_DAY.md)"
