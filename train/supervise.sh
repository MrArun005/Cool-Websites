#!/usr/bin/env bash
# Keep a training run going until it reaches its step target, whatever happens in the chat.
#
#   setsid nohup ./supervise.sh <name> <steps> <watch-dir> "<label>" <train/watch args...> \
#       > supervise.log 2>&1 < /dev/null &
#
# Runs train.py and watch.py side by side. If either one dies (crash, killed, out of memory),
# it is restarted within 30 s: training resumes from its latest checkpoint, the watcher picks up
# where its data.json left off. The watcher runs with --patience 0, so it never asks training to
# stop early; it still copies the best checkpoint to runs/<name>/best/. Ends only when
# runs/<name>/policy.zip exists (the step target was reached) and the watcher has finished.
set -u
export MUJOCO_GL=${MUJOCO_GL:-osmesa}   # draw video clips without a screen
cd "$(dirname "$0")"
name=$1 steps=$2 watch=$3 label=$4
shift 4
extra=("$@")                    # e.g. --skills walk --gait-generator tall
run=runs/$name
mkdir -p "$run" "$watch"
log() { echo "$(date -u +%FT%TZ) supervisor: $*"; }

start_train() {
  local resume=()
  local latest
  latest=$(ls "$run"/checkpoints/policy_*_steps.zip 2>/dev/null | sort -V | tail -1)
  [ -n "$latest" ] && resume=(--resume "$latest")
  log "starting training ${resume[*]:-from scratch}"
  python -u train.py --name "$name" --steps "$steps" --envs 8 "${extra[@]}" "${resume[@]}" >> "$watch/train.log" 2>&1 &
  train_pid=$!
}
start_watch() {
  log "starting watcher"
  python -u watch.py "$run" --log "$watch/train.log" --out "$watch" --target "$steps" --patience 0 \
    --label "$label" "${extra[@]}" >> "$watch/watch.log" 2>&1 &
  watch_pid=$!
}

rm -f "$run/STOP"
start_train
start_watch
while true; do
  sleep 30
  if ! kill -0 "$train_pid" 2>/dev/null; then
    wait "$train_pid"; code=$?
    if [ -f "$run/policy.zip" ] && grep -q "^done:" "$watch/train.log" && [ "$code" = 0 ]; then
      log "training reached its target"
      train_pid=0
    else
      log "training stopped unexpectedly (exit $code); restarting from the latest checkpoint"
      sleep 5
      start_train
      continue
    fi
  fi
  if ! kill -0 "$watch_pid" 2>/dev/null; then
    if [ "$train_pid" = 0 ] && grep -q "^finished" "$watch/watch.log"; then
      log "all done"
      exit 0
    fi
    log "watcher stopped unexpectedly; restarting it"
    start_watch
  fi
done
