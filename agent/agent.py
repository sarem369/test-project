#!/usr/bin/env python3
"""Hefaaz defensive remote agent.

Only executes operator-approved whitelist remediations from the Hefaaz API.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

API = os.getenv("HEFAAZ_API", "http://127.0.0.1:8080").rstrip("/")
DEVICE_ID = os.getenv("HEFAAZ_DEVICE_ID", "")
AGENT_TOKEN = os.getenv("HEFAAZ_AGENT_TOKEN", "")
ENROLL_TOKEN = os.getenv("HEFAAZ_ENROLL_TOKEN", "")
DEVICE_NAME = os.getenv("HEFAAZ_DEVICE_NAME", socket.gethostname())
POLL_SECONDS = int(os.getenv("HEFAAZ_POLL_SECONDS", "15"))
STATE_FILE = Path(os.getenv("HEFAAZ_STATE_FILE", Path.home() / ".hefaaz-agent.json"))
QUARANTINE_ROOT = Path(os.getenv("HEFAAZ_QUARANTINE", Path.home() / "hefaaz-quarantine"))


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return 1, str(exc)


def collect_inventory() -> dict[str, Any]:
    ufw_code, ufw_out = run(["bash", "-lc", "command -v ufw >/dev/null && ufw status || echo 'ufw not installed'"])
    updates_code, updates_out = run(
        ["bash", "-lc", "command -v apt-get >/dev/null && apt list --upgradable 2>/dev/null | head -n 40 || echo 'apt unavailable'"]
    )
    users_code, users_out = run(["bash", "-lc", "getent passwd | awk -F: '$3>=1000 {print $1}' | head -n 50"])
    procs_code, procs_out = run(["bash", "-lc", "ps -eo pid,user,comm --sort=-%cpu | head -n 20"])

    findings = []
    if "inactive" in ufw_out.lower() or "not installed" in ufw_out.lower():
        findings.append(
            {
                "severity": "medium",
                "title": "Firewall not active",
                "advice": "Enable host firewall and restrict inbound ports.",
            }
        )
    if "upgradable" in updates_out and updates_out.count("\n") > 1:
        findings.append(
            {
                "severity": "medium",
                "title": "Pending package updates",
                "advice": "Review and install security updates during a maintenance window.",
            }
        )

    return {
        "hostname": socket.gethostname(),
        "os_name": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "firewall": {"exit_code": ufw_code, "summary": ufw_out[:2000]},
        "updates": {"exit_code": updates_code, "summary": updates_out[:4000]},
        "local_users": {"exit_code": users_code, "summary": users_out[:2000]},
        "top_processes": {"exit_code": procs_code, "summary": procs_out[:2000]},
        "findings": findings,
        "agent_version": "0.1.0",
        "defensive_mode": True,
    }


def execute_action(action: str, params: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if action == "collect_inventory":
        inv = collect_inventory()
        return "succeeded", json.dumps(inv, ensure_ascii=False, indent=2)[:8000], inv

    if action == "enable_ufw":
        if shutil.which("ufw") is None:
            return "failed", "ufw is not installed", {}
        # Non-interactive best effort; may require root.
        cmds = [
            ["ufw", "default", "deny", "incoming"],
            ["ufw", "default", "allow", "outgoing"],
            ["ufw", "--force", "enable"],
            ["ufw", "status"],
        ]
        logs = []
        for cmd in cmds:
            code, out = run(cmd)
            logs.append(f"$ {' '.join(cmd)}\n{out}")
            if code != 0 and cmd[1] != "status":
                return "failed", "\n\n".join(logs), {"needs_root": True}
        return "succeeded", "\n\n".join(logs), {}

    if action == "apt_update_check":
        code, out = run(["bash", "-lc", "apt-get update -qq && apt list --upgradable 2>/dev/null | head -n 80"])
        return ("succeeded" if code == 0 else "failed"), out, {}

    if action == "clear_temp_caches":
        target = Path("/tmp/hefaaz-cache")
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            return "succeeded", f"Removed {target}", {"path": str(target)}
        return "succeeded", "No /tmp/hefaaz-cache directory present", {}

    if action == "quarantine_path":
        src = Path(params.get("path", "")).expanduser().resolve()
        if not str(src).startswith(("/home/", "/tmp/", "/var/tmp/", str(Path.home()))):
            return "rejected", "Path outside allowed quarantine roots", {}
        if not src.exists():
            return "failed", f"Path not found: {src}", {}
        QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True)
        dest = QUARANTINE_ROOT / f"{int(time.time())}_{src.name}"
        shutil.move(str(src), str(dest))
        return "succeeded", f"Moved {src} -> {dest}", {"from": str(src), "to": str(dest)}

    if action == "run_clamav_scan":
        if shutil.which("clamscan") is None:
            return "failed", "clamscan not installed", {}
        path = Path(params.get("path", ".")).expanduser()
        code, out = run(["clamscan", "-r", "--bell", "-i", str(path)], timeout=300)
        return ("succeeded" if code in (0, 1) else "failed"), out[:8000], {"exit_code": code}

    return "rejected", f"Action not allowed on agent: {action}", {}


def save_state(device_id: str, agent_token: str) -> None:
    STATE_FILE.write_text(
        json.dumps({"device_id": device_id, "agent_token": agent_token}, indent=2),
        encoding="utf-8",
    )


def load_state() -> tuple[str, str]:
    global DEVICE_ID, AGENT_TOKEN
    if DEVICE_ID and AGENT_TOKEN:
        return DEVICE_ID, AGENT_TOKEN
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return data.get("device_id", ""), data.get("agent_token", "")
    return "", ""


def enroll_if_needed(client: httpx.Client) -> tuple[str, str]:
    device_id, token = load_state()
    if device_id and token:
        return device_id, token
    if not ENROLL_TOKEN:
        raise SystemExit("Set HEFAAZ_DEVICE_ID+HEFAAZ_AGENT_TOKEN or HEFAAZ_ENROLL_TOKEN")
    resp = client.post(
        f"{API}/api/devices/enroll",
        headers={"Authorization": f"Bearer {ENROLL_TOKEN}"},
        json={
            "name": DEVICE_NAME,
            "hostname": socket.gethostname(),
            "os_name": f"{platform.system()} {platform.release()}",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    save_state(data["device_id"], data["agent_token"])
    print(f"Enrolled as {data['name']} ({data['device_id']})")
    return data["device_id"], data["agent_token"]


def report_result(client: httpx.Client, device_id: str, token: str, action_id: str, status: str, output: str, details: dict) -> None:
    client.post(
        f"{API}/api/actions/{action_id}/result",
        headers={"X-Device-Id": device_id, "X-Agent-Token": token},
        json={"status": status, "output": output, "details": details},
        timeout=60,
    ).raise_for_status()


def main() -> None:
    print(f"Hefaaz agent starting. API={API}")
    with httpx.Client() as client:
        device_id, token = enroll_if_needed(client)
        while True:
            inventory = collect_inventory()
            try:
                resp = client.post(
                    f"{API}/api/devices/{device_id}/heartbeat",
                    headers={"X-Device-Id": device_id, "X-Agent-Token": token},
                    json={"inventory": inventory, "status": "online"},
                    timeout=60,
                )
                resp.raise_for_status()
                pending = resp.json().get("pending_actions") or []
                for action in pending:
                    print(f"Running approved action: {action.get('action')}")
                    status, output, details = execute_action(action.get("action"), action.get("params") or {})
                    report_result(client, device_id, token, action["id"], status, output, details)
                    print(f"  -> {status}")
            except Exception as exc:
                print(f"Heartbeat error: {exc}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()