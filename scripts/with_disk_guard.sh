#!/usr/bin/env bash
# Run a (download) command only while the disk keeps enough free space, and stop it when free
# space falls under the floor. The user allows data collection until 50 GB remain; a further
# 30 GB is kept for processing, so the default start threshold is 80 GB and the stop threshold
# 60 GB.
#
#   bash scripts/with_disk_guard.sh <expected_GB> <command> [args...]
#
# <expected_GB> is the size the command will add; the command starts only if
# free - expected >= FLAWMAP_DISK_START_GB (default 80).
set -u
EXPECTED_GB=${1:?expected size in GB}
shift
START_GB=${FLAWMAP_DISK_START_GB:-80}
STOP_GB=${FLAWMAP_DISK_STOP_GB:-60}
TARGET=${FLAWMAP_DISK_PATH:-$(pwd)}

free_gb() { df -B1G --output=avail "$TARGET" | tail -1 | tr -d ' '; }

free=$(free_gb)
if (( free - EXPECTED_GB < START_GB )); then
  echo "[disk guard] ${free} GB free, ${EXPECTED_GB} GB expected: below the ${START_GB} GB start threshold, not starting $(date)"
  exit 4
fi
echo "[disk guard] ${free} GB free before start (expected +${EXPECTED_GB} GB) $(date)"
setsid "$@" &
job=$!
while kill -0 "$job" 2>/dev/null; do
  free=$(free_gb)
  if (( free < STOP_GB )); then
    echo "[disk guard] ${free} GB free, under ${STOP_GB} GB: stopping the command $(date)"
    kill -TERM -- "-$job"; sleep 5; kill -KILL -- "-$job" 2>/dev/null
    exit 3
  fi
  sleep 30
done
wait "$job"
rc=$?
echo "[disk guard] finished rc=${rc}, ${free} GB free $(date)"
exit $rc
