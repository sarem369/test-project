from __future__ import annotations

import secrets
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import ADMIN_TOKEN, ALLOWED_ACTIONS
from . import ollama_client
from .models import (
    ActionCreateRequest,
    ActionResultRequest,
    ChatRequest,
    ChatResponse,
    EnrollRequest,
    EnrollResponse,
    HardenRequest,
    HeartbeatRequest,
    VendorSyncRequest,
)
from .store import store
from .vendors import list_vendors, sync_all, sync_vendor
from .vendors.registry import get_vendor

app = FastAPI(
    title="Hefaaz",
    description="English-first defensive security platform powered by Ollama with multi-vendor update connectors",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_admin(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing admin bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid admin token")


def require_agent(
    x_device_id: str = Header(...),
    x_agent_token: str = Header(...),
) -> dict[str, Any]:
    device = store.auth_device(x_device_id, x_agent_token)
    if not device:
        raise HTTPException(status_code=403, detail="Invalid agent credentials")
    return device


@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    ollama = await ollama_client.health()
    return {
        "ok": True,
        "service": "hefaaz",
        "ollama": ollama,
        "devices": len(store.list_devices()),
        "actions": len(store.list_actions()),
    }


@app.get("/api/meta")
async def api_meta(_: None = Depends(require_admin)) -> dict[str, Any]:
    return {
        "allowed_actions": ALLOWED_ACTIONS,
        "hardening_topics": ollama_client.HARDENING_TOPICS,
        "vendors": list_vendors(),
        "policy": {
            "mode": "defensive_only",
            "language": "english_first",
            "remote_actions": "operator_approved_whitelist",
            "multi_vendor_updates": True,
            "no_offensive_scanning": True,
        },
    }


@app.get("/api/vendors")
async def api_vendors(_: None = Depends(require_admin)) -> dict[str, Any]:
    catalog = list_vendors()
    updates = {u.get("vendor_id"): u for u in store.list_vendor_updates()}
    merged = []
    for vendor in catalog:
        latest = updates.get(vendor["id"])
        merged.append({**vendor, "latest_sync": latest})
    return {"vendors": merged}


@app.post("/api/vendors/sync")
async def api_vendors_sync(body: VendorSyncRequest, _: None = Depends(require_admin)) -> dict[str, Any]:
    results = []
    if body.vendor_ids:
        for vendor_id in body.vendor_ids:
            if not get_vendor(vendor_id):
                raise HTTPException(status_code=404, detail=f"Unknown vendor: {vendor_id}")
            result = await sync_vendor(vendor_id)
            vendor = get_vendor(vendor_id)
            payload = result.to_dict()
            payload["category"] = vendor.category if vendor else None
            payload.setdefault("meta", {})["category"] = vendor.category if vendor else None
            store.save_vendor_update(payload)
            results.append(payload)
    else:
        synced = await sync_all(body.categories or None)
        for result in synced:
            vendor = get_vendor(result.vendor_id)
            payload = result.to_dict()
            payload["category"] = vendor.category if vendor else None
            payload.setdefault("meta", {})["category"] = vendor.category if vendor else None
            store.save_vendor_update(payload)
            results.append(payload)
    return {
        "synced": len(results),
        "ok_count": sum(1 for r in results if r.get("ok")),
        "results": results,
    }


@app.get("/api/vendors/updates")
async def api_vendor_updates(
    category: str | None = None,
    _: None = Depends(require_admin),
) -> dict[str, Any]:
    return {"updates": store.list_vendor_updates(category)}


@app.get("/api/vendors/bundle")
async def api_vendor_bundle(_: None = Depends(require_admin)) -> dict[str, Any]:
    return store.vendor_update_bundle()


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(body: ChatRequest, _: None = Depends(require_admin)) -> ChatResponse:
    messages = [{"role": m.role, "content": m.content} for m in body.messages if m.role != "system"]
    reply, model, offline = await ollama_client.chat(messages, body.model)
    store.add_event("chat", "Security assistant reply generated", {"offline": offline, "model": model})
    return ChatResponse(reply=reply, model=model, offline_fallback=offline)


@app.post("/api/hardening")
async def api_hardening(body: HardenRequest, _: None = Depends(require_admin)) -> dict[str, Any]:
    reply, model, offline = await ollama_client.hardening_playbook(body.topic, body.context, body.language)
    return {"playbook": reply, "model": model, "offline_fallback": offline, "topic": body.topic}


@app.post("/api/devices/enroll", response_model=EnrollResponse)
async def enroll_device(body: EnrollRequest, _: None = Depends(require_admin)) -> EnrollResponse:
    device = store.create_device(body.name, body.hostname, body.os_name)
    return EnrollResponse(device_id=device["id"], agent_token=device["agent_token"], name=device["name"])


@app.get("/api/devices")
async def list_devices(_: None = Depends(require_admin)) -> dict[str, Any]:
    return {"devices": store.list_devices()}


@app.get("/api/devices/{device_id}")
async def get_device(device_id: str, _: None = Depends(require_admin)) -> dict[str, Any]:
    device = store.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    copy = dict(device)
    token = copy.pop("agent_token", "")
    copy["agent_token_masked"] = (token[:4] + "…" + token[-4:]) if len(token) > 8 else "••••"
    copy["actions"] = store.list_actions(device_id)
    return copy


@app.post("/api/devices/{device_id}/heartbeat")
async def device_heartbeat(
    device_id: str,
    body: HeartbeatRequest,
    device: dict[str, Any] = Depends(require_agent),
) -> dict[str, Any]:
    if device["id"] != device_id:
        raise HTTPException(status_code=403, detail="Device mismatch")
    updated = store.heartbeat(device_id, body.inventory, body.status)
    pending = store.pop_queued_actions(device_id)
    safe = dict(updated)
    safe.pop("agent_token", None)
    return {"device": safe, "pending_actions": pending}


@app.get("/api/devices/{device_id}/actions/pull")
async def pull_actions(device_id: str, device: dict[str, Any] = Depends(require_agent)) -> dict[str, Any]:
    if device["id"] != device_id:
        raise HTTPException(status_code=403, detail="Device mismatch")
    return {"actions": store.pop_queued_actions(device_id)}


@app.post("/api/actions")
async def create_action(body: ActionCreateRequest, _: None = Depends(require_admin)) -> dict[str, Any]:
    if body.action not in ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail="Action not in defensive whitelist")
    device = store.get_device(body.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    required = ALLOWED_ACTIONS[body.action].get("requires") or []
    for key in required:
        if not body.params.get(key):
            raise HTTPException(status_code=400, detail=f"Missing param: {key}")
    action = store.create_action(body.device_id, body.action, body.params, body.note)
    return {"action": action}


@app.get("/api/actions")
async def list_actions(device_id: str | None = None, _: None = Depends(require_admin)) -> dict[str, Any]:
    return {"actions": store.list_actions(device_id)}


@app.post("/api/actions/{action_id}/result")
async def action_result(
    action_id: str,
    body: ActionResultRequest,
    device: dict[str, Any] = Depends(require_agent),
) -> dict[str, Any]:
    existing = next((a for a in store.list_actions() if a["id"] == action_id), None)
    if not existing:
        raise HTTPException(status_code=404, detail="Action not found")
    if existing["device_id"] != device["id"]:
        raise HTTPException(status_code=403, detail="Action not owned by this device")
    updated = store.complete_action(action_id, body.status, body.output, body.details)
    return {"action": updated}


@app.get("/api/events")
async def list_events(_: None = Depends(require_admin)) -> dict[str, Any]:
    return {"events": store.list_events()}