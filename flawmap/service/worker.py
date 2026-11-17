"""The processing queue: one inspection at a time (the server has one GPU).

A thread in the web server takes the oldest queued inspection and runs
``python -m flawmap.service.job`` for it in a new process under ``scripts/with_memory_guard.sh``.
The thread only waits and cleans up: when the process ends without having written ``done`` or
``failed`` (killed, out of memory), the inspection is marked failed here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..utils.io import repo_root
from ..utils.logging import get_logger
from .store import Store

log = get_logger("flawmap.service")


def pid_alive(pid: int) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "nt":   # os.kill(pid, 0) would terminate the process on Windows
        import psutil

        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def memory_guard_cmd() -> list[str]:
    """Prefix that runs a command under the memory guard: the shell script where it works (Linux),
    the Python twin elsewhere (Windows has no bash ``setsid`` or ``/proc/meminfo``)."""
    if os.name == "nt":
        return [sys.executable, "-m", "flawmap.utils.memguard"]
    return ["bash", str(repo_root() / "scripts" / "with_memory_guard.sh")]


class Worker(threading.Thread):
    def __init__(self, store: Store, poll_s: float = 1.0, inline_runner=None, inline_public_runner=None):
        """``inline_runner`` / ``inline_public_runner``: run jobs in this thread with those pipeline
        functions instead of a subprocess (tests)."""
        super().__init__(daemon=True, name="flawmap-worker")
        self.store = store
        self.poll_s = poll_s
        self.inline_runner = inline_runner
        self.inline_public_runner = inline_public_runner
        self._stop = threading.Event()
        self.current: str | None = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        for iid in self.store.recover(pid_alive):
            log.info(f"recovered inspection {iid} after restart")
        while not self._stop.is_set():
            d = self.store.claim_next()
            if d is None:
                self._stop.wait(self.poll_s)
                continue
            self.current = d["id"]
            try:
                self._run_one(d)
            except Exception as e:  # never let the queue thread die
                log.warning(f"job {d['id']}: {type(e).__name__}: {e}")
            finally:
                self.current = None

    def _run_one(self, d: dict) -> None:
        iid = d["id"]
        if self.inline_runner is not None:
            from .job import process

            try:
                process(self.store, iid, runner=self.inline_runner, public_runner=self.inline_public_runner)
            except Exception:
                pass
            return
        jdir = self.store.dir(iid)
        logf = open(jdir / "job.log", "ab")
        cmd = [*memory_guard_cmd(), sys.executable, "-m", "flawmap.service.job", str(self.store.root.resolve()), iid]
        env = dict(os.environ, PYTHONPATH="")
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(repo_root()), env=env)
        self.store.update(iid, pid=p.pid)
        log.info(f"job {iid} started (pid {p.pid})")
        rc = p.wait()
        logf.close()
        after = self.store.get(iid)
        if after and after["status"] == "processing":
            detail = "out of memory: stopped by the memory guard" if rc == 3 else f"processing exited with code {rc}"
            self.store.update(iid, status="failed", finished=time.time(), pid=None,
                              error={"code": "INTERNAL", "message": "처리 중에 서버 오류가 났습니다.",
                                     "hint": "같은 촬영본으로 다시 시도해 주세요.", "detail": detail})
        log.info(f"job {iid} finished rc={rc} in {time.time() - t0:.0f}s -> {after['status'] if after else '?'}")
