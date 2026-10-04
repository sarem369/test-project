from __future__ import annotations

import json
import secrets
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STORE_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"devices": {}, "actions": {}, "events": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"devices": {}, "actions": {}, "events": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add_event(self, kind: str, message: str, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.data.setdefault("events", []).append(
                {
                    "id": str(uuid.uuid4()),
                    "kind": kind,
                    "message": message,
                    "meta": meta or {},
                    "created_at": utc_now(),
                }
            )
            self.data["events"] = self.data["events"][-200:]
            self._save()

    def create_device(self, name: str, hostname: str | None, os_name: str | None) -> dict[str, Any]:
        device_id = str(uuid.uuid4())
        agent_token = secrets.token_urlsafe(32)
        device = {
            "id": device_id,
            "name": name,
            "hostname": hostname or "",
            "os_name": os_name or "",
            "agent_token": agent_token,
            "status": "enrolled",
            "inventory": {},
            "created_at": utc_now(),
            "last_seen": None,
        }
        with self._lock:
            self.data.setdefault("devices", {})[device_id] = device
            self._save()
        self.add_event("device_enrolled", f"Device enrolled: {name}", {"device_id": device_id})
        return device

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock:
            devices = list(self.data.get("devices", {}).values())
        # Never expose agent tokens in list views for general UI safety;
        # admin UI still needs enroll token once — return masked copy.
        safe = []
        for d in devices:
            copy = dict(d)
            token = copy.get("agent_token", "")
            copy["agent_token_masked"] = (token[:4] + "…" + token[-4:]) if len(token) > 8 else "••••"
            copy.pop("agent_token", None)
            safe.append(copy)
        return sorted(safe, key=lambda x: x.get("created_at") or "", reverse=True)

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.data.get("devices", {}).get(device_id)

    def auth_device(self, device_id: str, token: str) -> dict[str, Any] | None:
        device = self.get_device(device_id)
        if not device:
            return None
        if not secrets.compare_digest(device.get("agent_token", ""), token):
            return None
        return device

    def heartbeat(self, device_id: str, inventory: dict[str, Any], status: str) -> dict[str, Any]:
        with self._lock:
            device = self.data["devices"][device_id]
            device["inventory"] = inventory
            device["status"] = status
            device["last_seen"] = utc_now()
            if inventory.get("hostname"):
                device["hostname"] = inventory["hostname"]
            if inventory.get("os_name"):
                device["os_name"] = inventory["os_name"]
            self._save()
            return dict(device)

    def create_action(
        self, device_id: str, action: str, params: dict[str, Any], note: str
    ) -> dict[str, Any]:
        action_id = str(uuid.uuid4())
        item = {
            "id": action_id,
            "device_id": device_id,
            "action": action,
            "params": params,
            "note": note,
            "status": "queued",
            "output": "",
            "details": {},
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        with self._lock:
            self.data.setdefault("actions", {})[action_id] = item
            self._save()
        self.add_event(
            "action_queued",
            f"Action queued: {action}",
            {"action_id": action_id, "device_id": device_id},
        )
        return item

    def list_actions(self, device_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            actions = list(self.data.get("actions", {}).values())
        if device_id:
            actions = [a for a in actions if a.get("device_id") == device_id]
        return sorted(actions, key=lambda x: x.get("created_at") or "", reverse=True)

    def pop_queued_actions(self, device_id: str) -> list[dict[str, Any]]:
        with self._lock:
            queued = [
                a
                for a in self.data.get("actions", {}).values()
                if a.get("device_id") == device_id and a.get("status") == "queued"
            ]
            for a in queued:
                a["status"] = "running"
                a["updated_at"] = utc_now()
            self._save()
            return [dict(a) for a in queued]

    def complete_action(
        self, action_id: str, status: str, output: str, details: dict[str, Any]
    ) -> dict[str, Any] | None:
        with self._lock:
            action = self.data.get("actions", {}).get(action_id)
            if not action:
                return None
            action["status"] = status
            action["output"] = output
            action["details"] = details
            action["updated_at"] = utc_now()
            self._save()
            result = dict(action)
        self.add_event(
            "action_finished",
            f"Action {status}: {result.get('action')}",
            {"action_id": action_id, "device_id": result.get("device_id")},
        )
        return result

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self.data.get("events", []))
        return list(reversed(events[-limit:]))


store = Store()