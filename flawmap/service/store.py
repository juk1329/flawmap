"""Inspections of the upload service: one SQLite table plus a folder per inspection.

    data/service/
      flawmap.db                    inspections table (WAL; every call opens its own connection)
      inspections/<id>/upload/<face>/   the unpacked Stray Scanner folder of each pass
      inspections/<id>/run/             pipeline output; run/bundle/ is what the viewer reads
      inspections/<id>/job.log          the processing log

Two random tokens per inspection: ``owner`` (the requester's page: progress, sharing, delete) and
``share`` (the read-only result for anyone the requester gives the link to, once ``published``).
``listed`` also puts the result on the public gallery. The inspection id is shown on the result;
the tokens are the only way in.

States: uploaded -> queued -> processing -> done | failed. A superseded inspection (the seller
filmed again) keeps its result but says so.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    owner TEXT UNIQUE NOT NULL,
    share TEXT UNIQUE NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    progress REAL DEFAULT 0,
    device TEXT,
    title TEXT,
    price INTEGER,
    seller TEXT,
    published INTEGER DEFAULT 0,
    listed INTEGER DEFAULT 0,
    superseded_by TEXT,
    replaces TEXT,
    source TEXT DEFAULT 'upload',
    attempts INTEGER DEFAULT 0,
    pid INTEGER,
    started REAL,
    finished REAL,
    n_passes INTEGER DEFAULT 0,
    run_dir TEXT,
    passes TEXT,
    error TEXT,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS ix_status ON inspections(status, created);
"""
JSON_FIELDS = ("passes", "error", "summary")
ACTIVE = ("uploaded", "queued", "processing")


def new_id() -> str:
    return time.strftime("%y%m%d") + "-" + secrets.token_hex(3).upper()


@dataclass
class Store:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        (self.root / "inspections").mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            cols = {r[1] for r in con.execute("PRAGMA table_info(inspections)")}
            if "listed" not in cols:      # databases from before 2026-11-16 evening: shared results were listed
                con.execute("ALTER TABLE inspections ADD COLUMN listed INTEGER DEFAULT 0")
                con.execute("UPDATE inspections SET listed = published")

    # ------------------------------------------------------------------ basics
    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.root / "flawmap.db", timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def dir(self, iid: str) -> Path:
        return self.root / "inspections" / iid

    @staticmethod
    def _row(r: sqlite3.Row | None) -> dict | None:
        if r is None:
            return None
        d = dict(r)
        for k in JSON_FIELDS:
            d[k] = json.loads(d[k]) if d.get(k) else None
        return d

    def create(self, device: str, title: str = "", price: int | None = None, seller: str = "", source: str = "upload",
               status: str = "uploaded", run_dir: str | None = None, published: bool = False, listed: bool = False) -> dict:
        iid, now = new_id(), time.time()
        with self._con() as con:
            con.execute("INSERT INTO inspections (id, owner, share, created, updated, status, device, title, price, seller, "
                        "source, run_dir, published, listed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (iid, secrets.token_urlsafe(16), secrets.token_urlsafe(16), now, now, status, device, title, price,
                         seller, source, run_dir or str(self.dir(iid) / "run"), int(published), int(listed and published)))
        self.dir(iid).mkdir(parents=True, exist_ok=True)
        return self.get(iid)

    def get(self, iid: str) -> dict | None:
        with self._con() as con:
            return self._row(con.execute("SELECT * FROM inspections WHERE id=?", (iid,)).fetchone())

    def by_owner(self, token: str) -> dict | None:
        with self._con() as con:
            return self._row(con.execute("SELECT * FROM inspections WHERE owner=?", (token,)).fetchone())

    def by_share(self, token: str) -> dict | None:
        with self._con() as con:
            return self._row(con.execute("SELECT * FROM inspections WHERE share=?", (token,)).fetchone())

    def update(self, iid: str, **fields) -> None:
        if not fields:
            return
        fields["updated"] = time.time()
        for k in JSON_FIELDS:
            if k in fields and fields[k] is not None and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._con() as con:
            con.execute(f"UPDATE inspections SET {cols} WHERE id=?", (*fields.values(), iid))

    def delete(self, iid: str) -> None:
        with self._con() as con:
            con.execute("DELETE FROM inspections WHERE id=?", (iid,))

    def list(self, status: str | None = None, published: bool | None = None, listed: bool | None = None,
             limit: int = 200) -> list[dict]:
        q, args = "SELECT * FROM inspections WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if published is not None:
            q += " AND published=?"
            args.append(int(published))
        if listed is not None:
            q += " AND listed=?"
            args.append(int(listed))
        q += " ORDER BY created DESC LIMIT ?"
        args.append(limit)
        with self._con() as con:
            return [self._row(r) for r in con.execute(q, args).fetchall()]

    # ------------------------------------------------------------------ queue
    def claim_next(self) -> dict | None:
        """Oldest queued inspection, atomically moved to ``processing``."""
        with self._con() as con:
            con.execute("BEGIN IMMEDIATE")
            r = con.execute("SELECT id FROM inspections WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if r is None:
                con.execute("COMMIT")
                return None
            now = time.time()
            con.execute("UPDATE inspections SET status='processing', stage='start', progress=0, started=?, updated=?, "
                        "attempts=attempts+1, error=NULL WHERE id=?", (now, now, r["id"]))
            con.execute("COMMIT")
        return self.get(r["id"])

    def jobs_ahead(self, iid: str) -> int | None:
        """Jobs that will run before this queued one (including one being processed); None when
        not queued."""
        with self._con() as con:
            r = con.execute("SELECT created, status FROM inspections WHERE id=?", (iid,)).fetchone()
            if r is None or r["status"] != "queued":
                return None
            n = con.execute("SELECT COUNT(*) FROM inspections WHERE status='queued' AND created<?", (r["created"],)).fetchone()[0]
            busy = con.execute("SELECT COUNT(*) FROM inspections WHERE status='processing'").fetchone()[0]
            return int(n) + int(busy)

    def typical_seconds(self, n_passes: int, default_per_pass: float = 150.0) -> float:
        """Median processing time of finished inspections with the same number of passes."""
        with self._con() as con:
            rows = con.execute("SELECT finished - started AS s FROM inspections WHERE status='done' AND source!='demo' AND "
                               "n_passes=? AND started IS NOT NULL AND finished IS NOT NULL ORDER BY finished DESC LIMIT 20",
                               (n_passes,)).fetchall()
        vals = sorted(r["s"] for r in rows if r["s"] and r["s"] > 0)
        return float(vals[len(vals) // 2]) if vals else default_per_pass * max(n_passes, 1)

    def recover(self, pid_alive) -> list[str]:
        """After a server restart: a ``processing`` job whose process is gone goes back to the queue
        once, and fails the second time."""
        changed = []
        for d in self.list(status="processing"):
            if d.get("pid") and pid_alive(d["pid"]):
                continue
            if (d.get("attempts") or 0) < 2:
                self.update(d["id"], status="queued", stage=None, progress=0, pid=None)
            else:
                self.update(d["id"], status="failed", pid=None, finished=time.time(),
                            error={"code": "INTERNAL", "detail": "processing stopped twice (server restart or crash)"})
            changed.append(d["id"])
        return changed
