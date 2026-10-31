"""Analyst case management: every REVIEW decision an analyst resolves becomes a (delayed) fraud label - the feedback loop."""
import sqlite3
from datetime import datetime, timezone

from .config import CASES_DB, DATA

_SCHEMA = """CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT, txn_id TEXT UNIQUE, created_at TEXT, analyst TEXT, outcome TEXT, notes TEXT,
    risk_score REAL, amount REAL, sender_id TEXT, receiver_id TEXT)"""
OUTCOMES = ("confirmed_fraud", "false_positive", "escalated")


class CaseStore:
    def __init__(self, path=None):
        self.path = str(path or CASES_DB)
        DATA.mkdir(exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute(_SCHEMA)

    def resolve(self, txn_id, outcome, analyst="analyst", notes="", risk_score=None, amount=None, sender_id=None, receiver_id=None):
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT OR REPLACE INTO cases (txn_id,created_at,analyst,outcome,notes,risk_score,amount,sender_id,receiver_id) VALUES (?,?,?,?,?,?,?,?,?)",
                      (txn_id, datetime.now(timezone.utc).isoformat(timespec="seconds"), analyst, outcome, notes, risk_score, amount, sender_id, receiver_id))

    def resolved_ids(self):
        with sqlite3.connect(self.path) as c:
            return {r[0] for r in c.execute("SELECT txn_id FROM cases")}

    def recent(self, limit=100):
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            return [dict(r) for r in c.execute("SELECT * FROM cases ORDER BY id DESC LIMIT ?", (limit,))]

    def stats(self):
        with sqlite3.connect(self.path) as c:
            rows = dict(c.execute("SELECT outcome, COUNT(*) FROM cases GROUP BY outcome").fetchall())
        n = sum(rows.values())
        return dict(resolved=n, **rows, review_precision=round(rows.get("confirmed_fraud", 0) / n, 3) if n else None)
