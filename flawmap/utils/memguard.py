"""Run a command and stop it when available memory runs low.

Cross-platform twin of ``scripts/with_memory_guard.sh`` for machines where that script cannot run
(Windows: no bash ``setsid`` or ``/proc/meminfo``). Same threshold variable and exit code:

    python -m flawmap.utils.memguard <command> [args...]

Stops the command and everything it started when available memory falls below
``$FLAWMAP_GUARD_MIN_KB`` (default 8000000 kB), and exits with 3; otherwise exits with the
command's own code.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil


def kill_tree(pid: int, grace_s: float = 5.0) -> None:
    """Terminate a process and its descendants; kill the ones still alive after ``grace_s``."""
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=grace_s)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m flawmap.utils.memguard <command> [args...]", file=sys.stderr)
        return 2
    min_kb = int(os.environ.get("FLAWMAP_GUARD_MIN_KB", "8000000"))
    proc = subprocess.Popen(argv)
    while True:
        try:
            return proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
        avail_kb = psutil.virtual_memory().available // 1024
        if avail_kb < min_kb:
            print(f"[memory guard] {avail_kb} kB available, stopping the command {time.ctime()}", flush=True)
            kill_tree(proc.pid)
            return 3


if __name__ == "__main__":
    sys.exit(main())
