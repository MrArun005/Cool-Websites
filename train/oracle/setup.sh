#!/usr/bin/env bash
# Turn an Oracle Cloud server (Oracle Linux 9 or Ubuntu, Arm or x86) into a Bolt training machine.
#
#   export GITHUB_TOKEN=...   # fine-grained, only MrArun005/Cool-Websites, Contents: read and write
#   curl -fsSL https://raw.githubusercontent.com/MrArun005/Cool-Websites/claude/robot-build-memory-ju76nt/train/oracle/setup.sh | bash -s imitate18
#
# The argument picks the run (see RUNS below). The script installs Python 3.12 and the training stack, gets
# the run's newest checkpoint from GitHub (so a run moved here continues where it was), starts it under
# keepalive (restarts after crashes), and pushes progress to the branch claude/train-oracle every 10 minutes.
# Cron restarts the job and the pushes after a reboot. Safe to run again.
set -euo pipefail
RUN="${1:-imitate18}"
: "${GITHUB_TOKEN:?export GITHUB_TOKEN first (fine-grained, Cool-Websites only, Contents read/write)}"
CODE_BRANCH=claude/robot-build-memory-ju76nt
OUT_BRANCH=claude/train-oracle
HOME_DIR="$HOME/bolt"
REPO="$HOME_DIR/Cool-Websites"

case "$RUN" in   # job file | branch holding the newest checkpoint to continue from | its folder there
  imitate18) JOB=train/jobs/run18-imitate.json; FROM=claude/train-imitate18; FROM_DIR=train/remote/imitate18 ;;
  run20)     JOB=train/jobs/run20-run-fast.json; FROM=claude/train-run20;    FROM_DIR=train/remote/run20 ;;
  jump16f)   JOB=train/jobs/run16f-jump-a1motors.json; FROM=claude/train-jump16e; FROM_DIR=train/remote/jump16f ;;
  *) echo "unknown run $RUN"; exit 1 ;;
esac

echo "== packages"
if command -v dnf >/dev/null; then
  sudo dnf install -y -q git python3.12 python3.12-pip cronie || sudo dnf install -y -q git python3.11 python3.11-pip cronie
  sudo dnf install -y -q mesa-libOSMesa 2>/dev/null || true
  sudo systemctl enable --now crond
else
  sudo apt-get update -q && sudo apt-get install -y -q git python3-venv python3-pip cron libosmesa6 || true
fi
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)

echo "== code"
mkdir -p "$HOME_DIR"
umask 077; printf '%s\n' "$GITHUB_TOKEN" > "$HOME_DIR/.token"; umask 022
URL="https://x-access-token:$(cat "$HOME_DIR/.token")@github.com/MrArun005/Cool-Websites"
if [ ! -d "$REPO/.git" ]; then git clone -q -b "$CODE_BRANCH" "$URL" "$REPO"; else git -C "$REPO" pull -q origin "$CODE_BRANCH"; fi
git -C "$REPO" config user.email "oracle@bolt"; git -C "$REPO" config user.name "Bolt Oracle"

echo "== python ($PY)"
[ -d "$HOME_DIR/venv" ] || "$PY" -m venv "$HOME_DIR/venv"
"$HOME_DIR/venv/bin/pip" install -q --upgrade pip
"$HOME_DIR/venv/bin/pip" install -q -r "$REPO/train/requirements.txt" || {
  echo "pinned install failed on this machine; installing the same packages unpinned"
  sed -E 's/[=<>~!].*$//; /^--/d; /^#/d; /^$/d' "$REPO/train/requirements.txt" | xargs "$HOME_DIR/venv/bin/pip" install -q
}

echo "== newest checkpoint of $RUN from $FROM"
NAME=$(grep -o '"--name", "[^"]*"' "$REPO/$JOB" | head -1 | sed 's/.*", "//; s/"//')
CK="$REPO/train/runs/$NAME/checkpoints"; mkdir -p "$CK"
git -C "$REPO" fetch -q origin "$FROM" || true
for f in $(git -C "$REPO" ls-tree --name-only "origin/$FROM" "$FROM_DIR/" 2>/dev/null | grep -E 'policy_(vecnormalize_)?[0-9]+_steps\.(zip|pkl)$'); do
  git -C "$REPO" show "origin/$FROM:$f" > "$CK/$(basename "$f")" && echo "  got $(basename "$f")"
done

echo "== push script"
cat > "$HOME_DIR/push.sh" <<EOF
#!/usr/bin/env bash
# Every 10 min (cron): copy the run's log, newest checkpoint pair and best/ into train/remote/$NAME-oracle, push to $OUT_BRANCH.
cd "$REPO" || exit 0
R=train/runs/$NAME; D=train/remote/$NAME-oracle; mkdir -p "\$D"
cp -f "\$R/train.log" "\$D/" 2>/dev/null
Z=\$(ls -1 "\$R"/checkpoints/policy_*_steps.zip 2>/dev/null | grep -v vecnormalize | sort -t_ -k2 -n | tail -1)
if [ -n "\$Z" ]; then N=\$(basename "\$Z" | sed 's/policy_//; s/_steps.zip//'); rm -f "\$D"/policy_*; cp -f "\$Z" "\$R/checkpoints/policy_vecnormalize_\${N}_steps.pkl" "\$D/"; fi
[ -d "\$R/best" ] && cp -rf "\$R/best" "\$D/"
[ -f "\$R/policy.zip" ] && mkdir -p "\$D/final" && cp -f "\$R/policy.zip" "\$R/vecnormalize.pkl" "\$D/final/"
git add -f "\$D" >/dev/null 2>&1
git diff --cached --quiet || git commit -q -m "oracle $NAME progress \${N:-log}"
git push -q "https://x-access-token:\$(cat $HOME_DIR/.token)@github.com/MrArun005/Cool-Websites" "HEAD:$OUT_BRANCH" 2>/dev/null || true
EOF
chmod +x "$HOME_DIR/push.sh"
cat > "$HOME_DIR/start.sh" <<EOF
#!/usr/bin/env bash
# Start (or keep) the training job; keepalive does nothing if it is already running.
cd "$REPO" && PATH="$HOME_DIR/venv/bin:\$PATH" OMP_NUM_THREADS=1 "$HOME_DIR/venv/bin/python" tools/keepalive/keepalive.py start "$JOB" >> "$HOME_DIR/start.log" 2>&1
EOF
chmod +x "$HOME_DIR/start.sh"

echo "== cron (after a reboot, and a check every 10 min)"
( crontab -l 2>/dev/null | grep -v "$HOME_DIR/" ; echo "@reboot $HOME_DIR/start.sh"; echo "*/10 * * * * $HOME_DIR/start.sh; $HOME_DIR/push.sh" ) | crontab -

echo "== start $RUN"
"$HOME_DIR/start.sh"; sleep 20
tail -3 "$REPO/train/runs/$NAME/train.log" 2>/dev/null || tail -5 "$HOME_DIR/start.log"
"$HOME_DIR/push.sh"
echo "== done: $RUN is training; progress goes to the branch $OUT_BRANCH every 10 minutes"
