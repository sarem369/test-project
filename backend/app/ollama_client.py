from __future__ import annotations

import httpx

from .config import OLLAMA_HOST, OLLAMA_MODEL, SYSTEM_PROMPT


OFFLINE_REPLY_EN = (
    "Could not reach Ollama. Offline defensive checklist:\n"
    "1) Install pending OS/app updates.\n"
    "2) Enable the firewall and close unused ports.\n"
    "3) Sync antivirus / malware / network vendor update feeds in Hefaaz.\n"
    "4) Use strong passwords and 2FA on sensitive accounts.\n"
    "5) Quarantine suspicious files before scanning; avoid blind deletion.\n"
    "Retry after Ollama is running for tailored guidance."
)

OFFLINE_REPLY_FA = (
    "Could not reach Ollama. Offline defensive checklist (English-first platform):\n"
    "1) Install pending OS/app updates.\n"
    "2) Enable the firewall and close unused ports.\n"
    "3) Sync antivirus / malware / network vendor update feeds in Hefaaz.\n"
    "4) Use strong passwords and 2FA on sensitive accounts.\n"
    "5) Quarantine suspicious files before scanning; avoid blind deletion."
)


async def chat(messages: list[dict[str, str]], model: str | None = None) -> tuple[str, str, bool]:
    selected = model or OLLAMA_MODEL
    payload = {
        "model": selected,
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("message") or {}).get("content") or ""
            return content.strip() or "No response from model.", selected, False
    except Exception:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        fa = any("\u0600" <= ch <= "\u06FF" for ch in last_user)
        return (OFFLINE_REPLY_FA if fa else OFFLINE_REPLY_EN), selected, True


async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags")
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {"ok": True, "host": OLLAMA_HOST, "models": models, "default_model": OLLAMA_MODEL}
    except Exception as exc:
        return {
            "ok": False,
            "host": OLLAMA_HOST,
            "models": [],
            "default_model": OLLAMA_MODEL,
            "error": str(exc),
        }


HARDENING_TOPICS = {
    "linux_baseline": {
        "fa": "سخت‌سازی پایه لینوکس برای سرور وب",
        "en": "Linux baseline hardening for a web server",
    },
    "web_app": {
        "fa": "سخت‌سازی اپلیکیشن وب (headers، secrets، auth)",
        "en": "Web application hardening (headers, secrets, auth)",
    },
    "malware_cleanup": {
        "fa": "راهنمای سطح‌بالای پاک‌سازی بدافزار و بازیابی امن",
        "en": "High-level malware cleanup and safe recovery",
    },
    "remote_ops": {
        "fa": "عملیات راه دور امن برای پشتیبانی IT مجاز",
        "en": "Secure remote ops for authorized IT support",
    },
}


async def hardening_playbook(topic: str, context: str, language: str) -> tuple[str, str, bool]:
    label = HARDENING_TOPICS.get(topic, {}).get(language) or topic
    prompt = (
        f"Create a defensive hardening playbook for: {label}.\n"
        f"Context: {context or 'general purpose'}\n"
        f"Language: {'Persian' if language == 'fa' else 'English'}\n"
        "Structure with: goals, checks, safe remediations, verification, rollback notes."
    )
    return await chat([{"role": "user", "content": prompt}])