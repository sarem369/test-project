import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["HEFAAZ_ADMIN_TOKEN"] = "test-token"
os.environ["HEFAAZ_DATA_DIR"] = str(ROOT / "data-test")

from app.main import app  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": "Bearer test-token"}


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["service"] == "hefaaz"


def test_enroll_and_queue_action():
    enroll = client.post("/api/devices/enroll", headers=AUTH, json={"name": "dev1"})
    assert enroll.status_code == 200
    body = enroll.json()
    device_id = body["device_id"]
    token = body["agent_token"]

    action = client.post(
        "/api/actions",
        headers=AUTH,
        json={"device_id": device_id, "action": "collect_inventory", "params": {}},
    )
    assert action.status_code == 200

    hb = client.post(
        f"/api/devices/{device_id}/heartbeat",
        headers={"X-Device-Id": device_id, "X-Agent-Token": token},
        json={"inventory": {"hostname": "dev1", "findings": []}, "status": "online"},
    )
    assert hb.status_code == 200
    pending = hb.json()["pending_actions"]
    assert len(pending) == 1
    assert pending[0]["action"] == "collect_inventory"


def test_rejects_unknown_action():
    enroll = client.post("/api/devices/enroll", headers=AUTH, json={"name": "dev2"})
    device_id = enroll.json()["device_id"]
    res = client.post(
        "/api/actions",
        headers=AUTH,
        json={"device_id": device_id, "action": "exploit_something", "params": {}},
    )
    assert res.status_code == 400