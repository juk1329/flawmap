#!/usr/bin/env bash
# Start the inspection service detached from the terminal, so it keeps running after logout.
#   bash scripts/serve_detached.sh [flawmap serve options]      e.g. --host 0.0.0.0 --port 8000
#   bash scripts/serve_detached.sh stop
# Log: runs/logs/serve.log, process id: runs/serve.pid
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${CONDA_BASE:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate env_isaaclab
export PYTHONPATH=
cd "$ROOT"
mkdir -p runs/logs
pidf=runs/serve.pid
running() { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; }
if [ "${1:-}" = "stop" ]; then
  if running; then kill "$(cat "$pidf")" && echo "stopped $(cat "$pidf")"; rm -f "$pidf"; else echo "not running"; fi
  exit 0
fi
if running; then echo "already running (pid $(cat "$pidf")); stop it first: bash scripts/serve_detached.sh stop"; exit 0; fi
setsid nohup flawmap serve "$@" >> runs/logs/serve.log 2>&1 < /dev/null &
echo $! > "$pidf"
sleep 4
if running; then echo "flawmap serve running (pid $(cat "$pidf")), log runs/logs/serve.log"; tail -3 runs/logs/serve.log; else echo "failed to start:"; tail -20 runs/logs/serve.log; exit 1; fi
