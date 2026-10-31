"""Append-only SQLite audit trail of every scoring decision (compliance / investigation)."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from .config import AUDIT_DB

_SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, sender_id TEXT, receiver_id TEXT, amount REAL,
    channel TEXT, risk_score REAL, decision TEXT, reasons TEXT, model_version TEXT)"""


class AuditLog:
    def __init__(self, path=None):
        self.path = str(path or AUDIT_DB)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute(_SCHEMA)

    def log(self, req: dict, res: dict, model_version: str) -> int:
        with sqlite3.connect(self.path) as c:
            cur = c.execute("INSERT INTO decisions (created_at,sender_id,receiver_id,amount,channel,risk_score,decision,reasons,model_version)"
                            " VALUES (?,?,?,?,?,?,?,?,?)",
                            (datetime.now(timezone.utc).isoformat(), req["sender_id"], req["receiver_id"], req["amount"],
                             req.get("channel"), res["risk_score"], res["decision"], json.dumps(res["reasons"]), model_version))
            return cur.lastrowid

    def recent(self, limit=100):
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))]

    def stats(self):
        with sqlite3.connect(self.path) as c:
            rows = c.execute("SELECT decision, COUNT(*) FROM decisions GROUP BY decision").fetchall()
        return {d: n for d, n in rows}
