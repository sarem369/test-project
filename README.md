# Hefaaz — Defensive Security Platform on Ollama

Hefaaz is an **English-first**, defensive security platform that runs with [Ollama](https://ollama.com) and pulls protective updates from multiple security vendors across:

- Antivirus signature / endpoint protection feeds
- Network security rule and reputation updates
- Malware intelligence / hash / URL indicators

It does **not** include offensive penetration-testing or exploit tooling.

## Components

| Piece | Path | Role |
|---|---|---|
| Backend API | `backend/` | FastAPI + Ollama + multi-vendor sync + action queue |
| Dashboard | `web/` | English operator console |
| Agent | `agent/` | Authorized remote remediation agent |

## Multi-vendor update model

Hefaaz uses pluggable connectors:

**Open / public defensive feeds (live sync):**
- ClamAV signature mirror (Cisco Talos / ClamAV)
- abuse.ch URLhaus
- abuse.ch MalwareBazaar
- Proofpoint Emerging Threats Open
- NIST NVD CVE API
- AlienVault OTX (API key optional)

**Commercial connectors (credential-gated):**
- Microsoft Defender / Graph
- CrowdStrike Falcon
- SentinelOne
- Sophos Central
- Cisco Talos
- Palo Alto Networks Threat Prevention

Set vendor credentials via environment variables, for example:

```bash
export HEFAAZ_OTX_API_KEY=...
export HEFAAZ_MS_GRAPH_TOKEN=...
export HEFAAZ_CROWDSTRIKE_CLIENT_ID=...
export HEFAAZ_SENTINELONE_TOKEN=...
export HEFAAZ_SOPHOS_CLIENT_ID=...
export HEFAAZ_TALOS_API_KEY=...
export HEFAAZ_PANW_API_KEY=...
```

Commercial adapters only activate with **your licensed API access**. Hefaaz does not redistribute proprietary vendor engine code.

## Prerequisites

- Python 3.11+
- Node.js 20+
- Ollama with a model (example: `llama3.2`)

```bash
ollama pull llama3.2
```

## Quick start

### 1) Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.2
export HEFAAZ_ADMIN_TOKEN=change-me
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 2) Dashboard

```bash
cd web
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173` and use the same admin token.

### 3) Agent on an authorized machine

```bash
cd agent
pip install -r requirements.txt
export HEFAAZ_API=http://SERVER_IP:8080
export HEFAAZ_ENROLL_TOKEN=<admin-or-enroll-token>
export HEFAAZ_ADMIN_TOKEN=<admin-token>   # needed to apply vendor packs
python3 agent.py
```

## Operator workflow

1. Open **Vendor updates** and sync open feeds (and commercial vendors if credentials are set).
2. Review synced antivirus / network / malware indicators.
3. Enroll devices and queue approved remediations:
   - collect inventory
   - update ClamAV signatures
   - apply vendor indicator pack locally
   - enable firewall / check package updates
4. Use the Ollama assistant for English defensive guidance and hardening playbooks.

## Security notes

- Deploy only on trusted networks or behind TLS/VPN.
- Use strong admin and agent tokens.
- Run remote actions only on systems you own or are contracted to manage.
- Require operator approval before remediations in production.

## License

MIT
