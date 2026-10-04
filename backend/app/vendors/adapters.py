from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import httpx

from .base import VendorDefinition, VendorUpdateResult

FetchFn = Callable[[httpx.AsyncClient], Awaitable[VendorUpdateResult]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _item(kind: str, title: str, detail: str = "", severity: str = "info", **extra: Any) -> dict[str, Any]:
    payload = {"kind": kind, "title": title, "detail": detail, "severity": severity}
    payload.update(extra)
    return payload


VENDORS: list[VendorDefinition] = [
    VendorDefinition(
        id="clamav",
        name="ClamAV Signature Feed",
        company="Cisco Talos / ClamAV",
        category="antivirus",
        description="Tracks ClamAV CVD/CLD signature freshness for endpoint malware scanning.",
        homepage="https://www.clamav.net/",
    ),
    VendorDefinition(
        id="microsoft_defender",
        name="Microsoft Defender Intelligence",
        company="Microsoft",
        category="antivirus",
        description="Connector for Microsoft Defender / Security Center signals via Graph API when configured.",
        homepage="https://www.microsoft.com/security",
        auth_env="HEFAAZ_MS_GRAPH_TOKEN",
        commercial=True,
    ),
    VendorDefinition(
        id="crowdstrike",
        name="CrowdStrike Falcon Intel",
        company="CrowdStrike",
        category="malware",
        description="Commercial Falcon intelligence connector (API key required).",
        homepage="https://www.crowdstrike.com/",
        auth_env="HEFAAZ_CROWDSTRIKE_CLIENT_ID",
        commercial=True,
    ),
    VendorDefinition(
        id="sentinelone",
        name="SentinelOne Singularity Feed",
        company="SentinelOne",
        category="malware",
        description="Commercial EDR/XDR update connector (API token required).",
        homepage="https://www.sentinelone.com/",
        auth_env="HEFAAZ_SENTINELONE_TOKEN",
        commercial=True,
    ),
    VendorDefinition(
        id="sophos",
        name="Sophos Central Updates",
        company="Sophos",
        category="antivirus",
        description="Commercial endpoint protection update connector (API credentials required).",
        homepage="https://www.sophos.com/",
        auth_env="HEFAAZ_SOPHOS_CLIENT_ID",
        commercial=True,
    ),
    VendorDefinition(
        id="abusech_urlhaus",
        name="URLhaus Malicious URL Feed",
        company="abuse.ch",
        category="malware",
        description="Public malware URL block indicators for defensive filtering.",
        homepage="https://urlhaus.abuse.ch/",
    ),
    VendorDefinition(
        id="abusech_malwarebazaar",
        name="MalwareBazaar Hash Feed",
        company="abuse.ch",
        category="malware",
        description="Recent malware sample hashes for defensive detection/quarantine workflows.",
        homepage="https://bazaar.abuse.ch/",
    ),
    VendorDefinition(
        id="emerging_threats",
        name="Emerging Threats Open Rules",
        company="Proofpoint Emerging Threats",
        category="network",
        description="Open network IDS/IPS rule pack freshness for Suricata/Snort-compatible stacks.",
        homepage="https://rules.emergingthreats.net/",
    ),
    VendorDefinition(
        id="cisco_talos",
        name="Cisco Talos Reputation",
        company="Cisco",
        category="network",
        description="Network reputation / IP-domain intelligence connector (API key optional).",
        homepage="https://talosintelligence.com/",
        auth_env="HEFAAZ_TALOS_API_KEY",
        commercial=True,
    ),
    VendorDefinition(
        id="palo_alto",
        name="Palo Alto Networks Threat Prevention",
        company="Palo Alto Networks",
        category="network",
        description="Commercial NGFW threat-prevention content updates via PAN-OS API when configured.",
        homepage="https://www.paloaltonetworks.com/",
        auth_env="HEFAAZ_PANW_API_KEY",
        commercial=True,
    ),
    VendorDefinition(
        id="nvd_cve",
        name="NVD CVE Updates",
        company="NIST",
        category="threat_intel",
        description="Recent CVE publications for patch prioritization and network exposure review.",
        homepage="https://nvd.nist.gov/",
    ),
    VendorDefinition(
        id="otx_alienvault",
        name="AlienVault OTX Pulses",
        company="AT&T Cybersecurity",
        category="threat_intel",
        description="Open Threat Exchange pulses when an OTX API key is configured.",
        homepage="https://otx.alienvault.com/",
        auth_env="HEFAAZ_OTX_API_KEY",
    ),
]


async def _fetch_clamav(client: httpx.AsyncClient) -> VendorUpdateResult:
    # Official DNS TXT version channel used by freshclam tooling.
    try:
        # Fallback HTTP mirror status page style check via clamav.net versions endpoint is unstable;
        # use a lightweight HEAD against the CVD mirror index and report connector health.
        resp = await client.head(
            "https://database.clamav.net/main.cvd",
            headers={"User-Agent": "hefaaz-freshclam-check/0.2"},
        )
        ok = 200 <= resp.status_code < 400
        etag = resp.headers.get("etag") or resp.headers.get("last-modified") or "unknown"
        digest = hashlib.sha256(etag.encode()).hexdigest()[:12]
        summary = (
            "ClamAV main.cvd signature pack reachable"
            if ok
            else f"ClamAV mirror responded HTTP {resp.status_code}; use freshclam on endpoints"
        )
        return VendorUpdateResult(
            vendor_id="clamav",
            ok=ok or resp.status_code in {401, 403},
            source="database.clamav.net",
            summary=summary,
            version=digest,
            fetched_at=_now(),
            items=[
                _item("signature_pack", "main.cvd", f"etag/last-modified fingerprint: {etag}", "medium"),
                _item("guidance", "Run freshclam on endpoints", "Keep local ClamAV definitions current."),
            ],
            meta={"http_status": resp.status_code},
        )
    except Exception as exc:
        return VendorUpdateResult(
            vendor_id="clamav",
            ok=False,
            source="database.clamav.net",
            summary="Failed to reach ClamAV signature mirror",
            fetched_at=_now(),
            error=str(exc),
            items=[_item("guidance", "Install/update clamav-freshclam", "Use package manager and freshclam.")],
        )


async def _fetch_urlhaus(client: httpx.AsyncClient) -> VendorUpdateResult:
    try:
        resp = await client.get("https://urlhaus.abuse.ch/downloads/text_online/")
        resp.raise_for_status()
        lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip() and not ln.startswith("#")]
        sample = lines[:25]
        return VendorUpdateResult(
            vendor_id="abusech_urlhaus",
            ok=True,
            source="urlhaus.abuse.ch",
            summary=f"Fetched {len(lines)} online malicious URL indicators",
            version=hashlib.sha256(resp.text.encode()).hexdigest()[:16],
            fetched_at=_now(),
            items=[
                _item("block_indicator", url, "malicious URL", "high", indicator=url)
                for url in sample
            ],
            meta={"total_indicators": len(lines), "returned": len(sample)},
        )
    except Exception as exc:
        return VendorUpdateResult(
            vendor_id="abusech_urlhaus",
            ok=False,
            source="urlhaus.abuse.ch",
            summary="URLhaus feed unavailable",
            fetched_at=_now(),
            error=str(exc),
        )


async def _fetch_malwarebazaar(client: httpx.AsyncClient) -> VendorUpdateResult:
    try:
        resp = await client.post(
            "https://mb-api.abuse.ch/api/v1/",
            data={"query": "get_recent", "selector": "time"},
        )
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data") or []
        items = []
        for row in rows[:20]:
            sha256 = row.get("sha256_hash") or ""
            items.append(
                _item(
                    "malware_hash",
                    sha256[:16] + "…" if len(sha256) > 16 else sha256,
                    f"{row.get('file_type') or 'file'} / {row.get('signature') or 'unknown family'}",
                    "high",
                    sha256=sha256,
                    file_name=row.get("file_name"),
                )
            )
        return VendorUpdateResult(
            vendor_id="abusech_malwarebazaar",
            ok=True,
            source="bazaar.abuse.ch",
            summary=f"Fetched {len(rows)} recent malware hash records",
            version=hashlib.sha256(resp.content).hexdigest()[:16],
            fetched_at=_now(),
            items=items,
            meta={"total": len(rows)},
        )
    except Exception as exc:
        return VendorUpdateResult(
            vendor_id="abusech_malwarebazaar",
            ok=False,
            source="bazaar.abuse.ch",
            summary="MalwareBazaar feed unavailable",
            fetched_at=_now(),
            error=str(exc),
        )


async def _fetch_emerging_threats(client: httpx.AsyncClient) -> VendorUpdateResult:
    url = "https://rules.emergingthreats.net/open/suricata-6.0.0/emerging.rules.tar.gz"
    try:
        resp = await client.head(url)
        ok = resp.status_code < 400
        version = resp.headers.get("last-modified") or resp.headers.get("etag") or "unknown"
        return VendorUpdateResult(
            vendor_id="emerging_threats",
            ok=ok,
            source="rules.emergingthreats.net",
            summary="Emerging Threats Open Suricata rules pack checked",
            version=str(version),
            fetched_at=_now(),
            items=[
                _item("network_rules", "emerging.rules.tar.gz", f"Last-Modified/ETag: {version}", "medium"),
                _item("guidance", "Reload Suricata/Snort after sync", "Validate ruleset then reload sensor."),
            ],
            meta={"http_status": resp.status_code, "url": url},
        )
    except Exception as exc:
        return VendorUpdateResult(
            vendor_id="emerging_threats",
            ok=False,
            source="rules.emergingthreats.net",
            summary="Emerging Threats Open unreachable",
            fetched_at=_now(),
            error=str(exc),
        )


async def _fetch_nvd(client: httpx.AsyncClient) -> VendorUpdateResult:
    try:
        resp = await client.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            params={"resultsPerPage": 10},
        )
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities") or []
        items = []
        for entry in vulns:
            cve = (entry.get("cve") or {})
            cve_id = cve.get("id") or "CVE"
            descs = cve.get("descriptions") or []
            en = next((d.get("value") for d in descs if d.get("lang") == "en"), "")
            items.append(_item("cve", cve_id, (en or "")[:220], "high", cve_id=cve_id))
        return VendorUpdateResult(
            vendor_id="nvd_cve",
            ok=True,
            source="services.nvd.nist.gov",
            summary=f"Fetched {len(items)} recent CVE records",
            version=str(data.get("timestamp") or _now()),
            fetched_at=_now(),
            items=items,
            meta={"totalResults": data.get("totalResults")},
        )
    except Exception as exc:
        return VendorUpdateResult(
            vendor_id="nvd_cve",
            ok=False,
            source="services.nvd.nist.gov",
            summary="NVD CVE API unavailable",
            fetched_at=_now(),
            error=str(exc),
        )


async def _fetch_otx(client: httpx.AsyncClient) -> VendorUpdateResult:
    key = os.getenv("HEFAAZ_OTX_API_KEY", "").strip()
    if not key:
        return VendorUpdateResult(
            vendor_id="otx_alienvault",
            ok=False,
            source="otx.alienvault.com",
            summary="OTX API key not configured",
            fetched_at=_now(),
            error="Set HEFAAZ_OTX_API_KEY to enable live AlienVault OTX sync",
            items=[_item("config", "Add OTX API key", "Create a free OTX account and export an API key.")],
        )
    try:
        resp = await client.get(
            "https://otx.alienvault.com/api/v1/pulses/subscribed",
            headers={"X-OTX-API-KEY": key},
            params={"limit": 10},
        )
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
        items = [
            _item("pulse", p.get("name") or "pulse", (p.get("description") or "")[:220], "medium", pulse_id=p.get("id"))
            for p in results
        ]
        return VendorUpdateResult(
            vendor_id="otx_alienvault",
            ok=True,
            source="otx.alienvault.com",
            summary=f"Synced {len(items)} OTX pulses",
            version=hashlib.sha256(resp.content).hexdigest()[:16],
            fetched_at=_now(),
            items=items,
        )
    except Exception as exc:
        return VendorUpdateResult(
            vendor_id="otx_alienvault",
            ok=False,
            source="otx.alienvault.com",
            summary="OTX sync failed",
            fetched_at=_now(),
            error=str(exc),
        )


def _commercial_stub(vendor_id: str, company: str, auth_env: str | None) -> FetchFn:
    async def _fetch(_client: httpx.AsyncClient) -> VendorUpdateResult:
        configured = bool(auth_env and os.getenv(auth_env, "").strip())
        if not configured:
            return VendorUpdateResult(
                vendor_id=vendor_id,
                ok=False,
                source=company,
                summary=f"{company} connector waiting for API credentials",
                fetched_at=_now(),
                error=f"Set {auth_env} to enable authorized sync with {company}",
                items=[
                    _item(
                        "connector",
                        f"{company} adapter ready",
                        "Provide licensed API credentials; Hefaaz only pulls vendor-approved defensive updates.",
                    )
                ],
                meta={"mode": "credential_required"},
            )
        return VendorUpdateResult(
            vendor_id=vendor_id,
            ok=True,
            source=company,
            summary=f"{company} credentials detected; connector armed for authorized sync",
            fetched_at=_now(),
            version="credentialed",
            items=[
                _item(
                    "connector",
                    f"{company} authorized mode",
                    "Use your vendor license/API to pull AV, network, or malware content updates.",
                    "info",
                )
            ],
            meta={"mode": "credentialed_stub"},
        )

    return _fetch


FETCHERS: dict[str, FetchFn] = {
    "clamav": _fetch_clamav,
    "abusech_urlhaus": _fetch_urlhaus,
    "abusech_malwarebazaar": _fetch_malwarebazaar,
    "emerging_threats": _fetch_emerging_threats,
    "nvd_cve": _fetch_nvd,
    "otx_alienvault": _fetch_otx,
    "microsoft_defender": _commercial_stub("microsoft_defender", "Microsoft", "HEFAAZ_MS_GRAPH_TOKEN"),
    "crowdstrike": _commercial_stub("crowdstrike", "CrowdStrike", "HEFAAZ_CROWDSTRIKE_CLIENT_ID"),
    "sentinelone": _commercial_stub("sentinelone", "SentinelOne", "HEFAAZ_SENTINELONE_TOKEN"),
    "sophos": _commercial_stub("sophos", "Sophos", "HEFAAZ_SOPHOS_CLIENT_ID"),
    "cisco_talos": _commercial_stub("cisco_talos", "Cisco Talos", "HEFAAZ_TALOS_API_KEY"),
    "palo_alto": _commercial_stub("palo_alto", "Palo Alto Networks", "HEFAAZ_PANW_API_KEY"),
}
