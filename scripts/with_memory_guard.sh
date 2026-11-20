#!/usr/bin/env bash
# Run a command in its own session and stop it when available memory runs low, before the
# kernel's OOM killer acts (on 2026-11-02 it took down systemd and every tmux on this machine).
# Taken from ISCO (scripts/with_memory_guard.sh). The command runs in the background, so its
# stdin is closed: pass a script file, not a heredoc.
#
#   bash scripts/with_memory_guard.sh <command> [args...]
#
# The command also runs with a high OOM score (FLAWMAP_OOM_ADJ, default 800), so that when memory runs
# out faster than this guard can act, the kernel kills it before anything of the desktop or the login
# sessions: user services run at 100-200, and on 2026-11-19 (04:27 UTC) a global out-of-memory caused
# by another heavy job killed the user's systemd manager first, and with it every process of
# user@1000.service (430 processes, other terminal sessions among them). Raising one's own OOM score
# needs no privilege.
#
# When memory runs low the guard stops every process of the command's session, not only its process
# group: `timeout` (and anything else that calls setpgid) moves its child into a process group of its
# own, so on 2026-11-19 (06:54 UTC) the guard stopped a job queue but the `timeout flawmap ...` it was
# running went on using memory to the end. A process leaves the session only by calling setsid itself.
set -u
MIN_AVAILABLE_KB=${FLAWMAP_GUARD_MIN_KB:-8000000}
if command -v choom > /dev/null; then
  setsid choom -n "${FLAWMAP_OOM_ADJ:-800}" -- "$@" &
else
  setsid "$@" &
fi
job=$!
while kill -0 "$job" 2>/dev/null; do
  avail=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
  if (( avail < MIN_AVAILABLE_KB )); then
    echo "[memory guard] ${avail} kB available, stopping the command $(date)"
    pkill -TERM -s "$job"; kill -TERM -- "-$job" 2>/dev/null
    sleep 5
    pkill -KILL -s "$job"; kill -KILL -- "-$job" 2>/dev/null
    exit 3
  fi
  sleep 2
done
wait "$job"
