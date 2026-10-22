"""Twin history for replay: every tick's snapshot stored in SQLite (data/history.sqlite).

SQLite needs no setup. For a real deployment swap this for PostgreSQL +
TimescaleDB; the interface (save / get / times) stays the same.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading


class History:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS ticks (run INTEGER, k INTEGER, t REAL, payload TEXT, "
                        "PRIMARY KEY (run, k))")
        self.db.commit()
        row = self.db.execute("SELECT MAX(run) FROM ticks").fetchone()
        self.run = (row[0] or 0) + 1

    def new_run(self):
        self.run += 1

    def save(self, k: int, t: float, snapshot: dict):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO ticks VALUES (?, ?, ?, ?)",
                            (self.run, k, t, json.dumps(snapshot, separators=(",", ":"))))
            if k % 20 == 0:
                self.db.commit()

    def get(self, k: int) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT payload FROM ticks WHERE run=? AND k=?", (self.run, k)).fetchone()
        return json.loads(row[0]) if row else None

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM ticks WHERE run=?", (self.run,)).fetchone()[0]

    def alert_marks(self) -> list[dict]:
        """Ticks with critical alerts or warnings, for markers on the replay timeline."""
        out = []
        with self.lock:
            rows = self.db.execute("SELECT k, t, payload FROM ticks WHERE run=? ORDER BY k", (self.run,)).fetchall()
        for k, t, p in rows:
            d = json.loads(p)
            lv = {a["level"] for a in d.get("alerts", [])}
            if "critical" in lv:
                out.append({"k": k, "t": t, "level": "critical"})
            elif "warning" in lv:
                out.append({"k": k, "t": t, "level": "warning"})
        return out
