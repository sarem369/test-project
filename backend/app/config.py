from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("HEFAAZ_DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ADMIN_TOKEN = os.getenv("HEFAAZ_ADMIN_TOKEN", "hefaaz-dev-token")
STORE_PATH = DATA_DIR / "store.json"

SYSTEM_PROMPT = """You are Hefaaz, a defensive cybersecurity assistant running on Ollama.
You help authorized operators protect and remediate their own systems.

Rules:
- Only provide defensive guidance: hardening, secure coding, incident response, malware cleanup high-level steps, patching, configuration review, and multi-vendor security update operations.
- Never provide exploit code, attack payloads, penetration-testing attack procedures, or instructions for unauthorized access.
- Prefer concrete, safe checklists and verification steps.
- If the user asks for offensive techniques, refuse and redirect to defensive alternatives.
- Default to clear English unless the operator explicitly asks for another language.
- Be concise and actionable.
"""

# Whitelisted remote remediations the agent may execute after operator approval.
ALLOWED_ACTIONS = {
    "collect_inventory": {
        "title": "Collect security inventory",
        "description": "Gather firewall, update, user, and process posture.",
    },
    "enable_ufw": {
        "title": "Enable UFW firewall (deny incoming)",
        "description": "Enable ufw with default deny incoming / allow outgoing if available.",
    },
    "apt_update_check": {
        "title": "Check pending package updates",
        "description": "Run apt update and list upgradable packages (no install).",
    },
    "clear_temp_caches": {
        "title": "Clear safe temp caches",
        "description": "Remove common temp caches under /tmp/hefaaz-cache if present.",
    },
    "quarantine_path": {
        "title": "Quarantine a path (move only)",
        "description": "Move an operator-specified path into a quarantine folder. No deletion.",
        "requires": ["path"],
    },
    "run_clamav_scan": {
        "title": "Run ClamAV scan if installed",
        "description": "Scan a path with clamscan when available; report findings only.",
        "requires": ["path"],
    },
    "update_clamav_signatures": {
        "title": "Update ClamAV signatures",
        "description": "Run freshclam when available to pull antivirus signature updates.",
    },
    "apply_vendor_indicator_pack": {
        "title": "Apply vendor indicator pack locally",
        "description": "Write the latest synced defensive indicators to a local JSON pack for AV/network tooling.",
    },
}