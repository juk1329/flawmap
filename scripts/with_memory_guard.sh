#!/usr/bin/env bash
# Run a command in its own process group and stop it when available memory runs low, before the
# kernel's OOM killer acts (on 2026-11-02 it took down systemd and every tmux on this machine).
# Taken from ISCO (scripts/with_memory_guard.sh). The command runs in the background, so its
# stdin is closed: pass a script file, not a heredoc.
#
#   bash scripts/with_memory_guard.sh <command> [args...]
set -u
MIN_AVAILABLE_KB=${FLAWMAP_GUARD_MIN_KB:-8000000}
setsid "$@" &
job=$!
while kill -0 "$job" 2>/dev/null; do
  avail=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
  if (( avail < MIN_AVAILABLE_KB )); then
    echo "[memory guard] ${avail} kB available, stopping the command $(date)"
    kill -TERM -- "-$job"; sleep 5; kill -KILL -- "-$job" 2>/dev/null
    exit 3
  fi
  sleep 2
done
wait "$job"
