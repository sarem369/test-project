from __future__ import annotations

import os
from typing import Any

import httpx

from .adapters import FETCHERS, VENDORS
from .base import VendorDefinition, VendorUpdateResult


def list_vendors() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vendor in VENDORS:
        auth_set = bool(vendor.auth_env and os.getenv(vendor.auth_env, "").strip())
        rows.append(
            {
                "id": vendor.id,
                "name": vendor.name,
                "company": vendor.company,
                "category": vendor.category,
                "description": vendor.description,
                "homepage": vendor.homepage,
                "commercial": vendor.commercial,
                "supports_live_sync": vendor.supports_live_sync,
                "auth_env": vendor.auth_env,
                "auth_configured": auth_set if vendor.auth_env else True,
            }
        )
    return rows


def get_vendor(vendor_id: str) -> VendorDefinition | None:
    return next((v for v in VENDORS if v.id == vendor_id), None)


async def sync_vendor(vendor_id: str) -> VendorUpdateResult:
    if vendor_id not in FETCHERS:
        return VendorUpdateResult(
            vendor_id=vendor_id,
            ok=False,
            source="hefaaz",
            summary="Unknown vendor connector",
            error="Vendor not registered",
        )
    timeout = httpx.Timeout(25.0, connect=8.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        return await FETCHERS[vendor_id](client)


async def sync_all(categories: list[str] | None = None) -> list[VendorUpdateResult]:
    wanted = set(categories or [])
    results: list[VendorUpdateResult] = []
    for vendor in VENDORS:
        if wanted and vendor.category not in wanted:
            continue
        results.append(await sync_vendor(vendor.id))
    return results