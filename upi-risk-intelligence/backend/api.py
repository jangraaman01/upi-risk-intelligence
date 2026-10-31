"""FastAPI scoring service.

Optional auth: set env UPI_API_KEY and send header  X-API-Key: <key>.
Every decision is written to an append-only SQLite audit log.
"""
import json
import os
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from src.audit import AuditLog
from src.config import DATA, REPORTS
from src.logger import get_logger
from src.scoring import RiskEngine

log = get_logger("api")
app = FastAPI(title="UPI Behavioural Risk Intelligence API", version="1.1.0",
              description="Real-time transaction risk scoring with explanations and an audit trail.")
engine = RiskEngine()
audit = AuditLog()


def require_key(x_api_key: Optional[str] = Header(default=None)):
    key = os.getenv("UPI_API_KEY")
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class Txn(BaseModel):
    sender_id: str = Field(min_length=2, max_length=32, examples=["U00010"])
    receiver_id: str = Field(min_length=2, max_length=32, examples=["U00500"])
    amount: float = Field(gt=0, le=1_000_000, examples=[45000])
    channel: Literal["p2p", "collect", "qr", "intent"] = "p2p"
    device_id: Optional[str] = Field(default=None, max_length=64)
    timestamp: Optional[str] = Field(default=None, examples=["2026-02-20T02:15:00"])
    city: Optional[str] = Field(default=None, max_length=32, examples=["Mumbai"])


def _score(t: Txn, commit: bool = False, explain: bool = True) -> dict:
    if t.sender_id == t.receiver_id:
        raise HTTPException(422, "sender and receiver must differ")
    try:
        res = engine.score_transaction(**t.model_dump(), commit=commit, explain=explain)
    except ValueError as e:                       # e.g. unparsable timestamp
        raise HTTPException(422, str(e))
    res["audit_id"] = audit.log(t.model_dump(), res, engine.version)
    log.info("scored %s->%s amt=%.0f risk=%.4f %s", t.sender_id, t.receiver_id, t.amount, res["risk_score"], res["decision"])
    return res


@app.get("/health")
def health():
    return {"status": "ok", "model_version": engine.version, "trained_at": engine.trained_at, "history_rows": len(engine.history)}


@app.get("/model-info", dependencies=[Depends(require_key)])
def model_info():
    m = json.load(open(REPORTS / "metrics.json"))
    return {k: m[k] for k in ("model_version", "trained_at", "models", "thresholds", "time_series_cv_pr_auc", "calibration", "business_impact_test")}


@app.post("/score", dependencies=[Depends(require_key)])
def score(t: Txn, commit: bool = Query(False, description="record the payment in live state after scoring"),
          explain: bool = Query(True, description="include SHAP reasons (adds latency)")):
    return _score(t, commit, explain)


@app.post("/score/batch", dependencies=[Depends(require_key)])
def score_batch(txns: List[Txn]):
    if len(txns) > 100:
        raise HTTPException(413, "max 100 transactions per batch")
    return [_score(t) for t in txns]


@app.post("/what-if", dependencies=[Depends(require_key)])
def what_if(t: Txn):
    """Re-score the payment under realistic alternatives (smaller amount, usual device, home city, ...)."""
    return engine.what_if(**t.model_dump())


@app.get("/rings", dependencies=[Depends(require_key)])
def rings():
    return json.load(open(DATA / "rings.json"))


@app.get("/audit", dependencies=[Depends(require_key)])
def audit_log(limit: int = Query(50, ge=1, le=500)):
    return audit.recent(limit)


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return audit.stats()
