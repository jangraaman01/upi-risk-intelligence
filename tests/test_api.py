import pytest
from src.config import MODELS

pytestmark = pytest.mark.skipif(not (MODELS / "artifacts.joblib").exists(), reason="run `python -m src.pipeline` first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["UPI_AUDIT_DB"] = str(tmp_path_factory.mktemp("db") / "audit.db")
    import importlib
    import src.config
    import src.audit
    importlib.reload(src.config); importlib.reload(src.audit)
    from fastapi.testclient import TestClient
    import backend.api as api
    importlib.reload(api)
    return TestClient(api.app)


GOOD = dict(sender_id="U00010", receiver_id="U00500", amount=45000, channel="collect",
            device_id="D_unseen", timestamp="2026-02-20T02:15:00")


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["history_rows"] > 0


def test_score_contract_and_audit(client):
    r = client.post("/score", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["decision"] in {"APPROVE", "STEP_UP_AUTH", "REVIEW", "BLOCK_TRANSACTION"}
    assert len(body["reasons"]) == 5 and "audit_id" in body
    assert any(a["id"] == body["audit_id"] for a in client.get("/audit").json())


def test_risky_vs_benign_ordering(client):
    risky = client.post("/score", json=GOOD).json()["risk_score"]
    benign = client.post("/score", json=dict(sender_id="U00010", receiver_id="M0001", amount=120, channel="qr",
                                             timestamp="2026-02-20T13:00:00")).json()["risk_score"]
    assert risky > benign


@pytest.mark.parametrize("patch", [dict(amount=-5), dict(amount=0), dict(channel="cheque"),
                                   dict(receiver_id="U00010"), dict(timestamp="not-a-date")])
def test_validation_errors(client, patch):
    assert client.post("/score", json={**GOOD, **patch}).status_code == 422


def test_unknown_sender_warns(client):
    r = client.post("/score", json={**GOOD, "sender_id": "U99999"}).json()
    assert any("unknown sender" in w for w in r["warnings"])


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("UPI_API_KEY", "secret")
    assert client.post("/score", json=GOOD).status_code == 401
    assert client.post("/score", json=GOOD, headers={"X-API-Key": "secret"}).status_code == 200


def test_live_mode_commit_and_explain_flag(client):
    base = dict(sender_id="U00020", receiver_id="U00777", amount=900, channel="p2p")
    first = client.post("/score?commit=true&explain=false", json=base).json()
    assert first["mode"] == "live" and first["reasons"] == []
    second = client.post("/score?explain=true", json=base).json()               # state was updated by the committed payment
    assert second["mode"] == "live" and len(second["reasons"]) == 5
    assert second["risk_score"] != first["risk_score"]


def test_what_if_endpoint_lists_alternatives(client):
    r = client.post("/what-if", json=dict(GOOD, city="Delhi")).json()
    assert r["base"]["decision"] in {"APPROVE", "STEP_UP_AUTH", "REVIEW", "BLOCK_TRANSACTION"}
    names = [x["scenario"] for x in r["scenarios"]]
    assert any(n.startswith("Amount x") for n in names) and "Paid at 2 pm instead" in names
    assert r["scenarios"] == sorted(r["scenarios"], key=lambda x: x["risk_score"])


def test_city_changes_the_score(client):
    far = client.post("/score?explain=false", json=dict(GOOD, city="Guwahati", timestamp=None)).json()
    home = client.post("/score?explain=false", json=dict(GOOD, timestamp=None)).json()
    assert far["city"] == "Guwahati" and far["risk_score"] != home["risk_score"]
