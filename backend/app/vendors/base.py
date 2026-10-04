from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


VendorCategory = Literal["antivirus", "network", "malware", "threat_intel"]


@dataclass
class VendorDefinition:
    id: str
    name: str
    company: str
    category: VendorCategory
    description: str
    homepage: str
    auth_env: str | None = None
    supports_live_sync: bool = True
    commercial: bool = False


@dataclass
class VendorUpdateResult:
    vendor_id: str
    ok: bool
    source: str
    summary: str
    items: list[dict[str, Any]] = field(default_factory=list)
    version: str | None = None
    fetched_at: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
