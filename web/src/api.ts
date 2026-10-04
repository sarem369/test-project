const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8080";

export type Device = {
  id: string;
  name: string;
  hostname?: string;
  os_name?: string;
  status?: string;
  last_seen?: string | null;
  inventory?: {
    findings?: Array<{ severity: string; title: string; advice: string }>;
    firewall?: { summary?: string };
    updates?: { summary?: string };
  };
};

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type Vendor = {
  id: string;
  name: string;
  company: string;
  category: string;
  description: string;
  homepage: string;
  commercial: boolean;
  auth_configured: boolean;
  auth_env?: string | null;
  latest_sync?: {
    ok: boolean;
    summary: string;
    fetched_at?: string;
    version?: string;
    error?: string;
    items?: Array<{ kind: string; title: string; detail?: string; severity?: string }>;
  } | null;
};

function authHeaders(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json() as Promise<T>;
}

export async function getHealth() {
  return parse<{
    ok: boolean;
    ollama: { ok: boolean; models: string[]; default_model: string; error?: string };
    devices: number;
    actions: number;
  }>(await fetch(`${API_BASE}/api/health`));
}

export async function getMeta(token: string) {
  return parse<{
    allowed_actions: Record<string, { title: string; description: string; requires?: string[] }>;
    hardening_topics: Record<string, { fa: string; en: string }>;
    vendors: Vendor[];
  }>(await fetch(`${API_BASE}/api/meta`, { headers: authHeaders(token) }));
}

export async function chat(token: string, messages: ChatMessage[]) {
  return parse<{ reply: string; model: string; offline_fallback: boolean }>(
    await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: authHeaders(token),
      body: JSON.stringify({ messages }),
    }),
  );
}

export async function listDevices(token: string) {
  return parse<{ devices: Device[] }>(
    await fetch(`${API_BASE}/api/devices`, { headers: authHeaders(token) }),
  );
}

export async function enrollDevice(token: string, name: string) {
  return parse<{ device_id: string; agent_token: string; name: string }>(
    await fetch(`${API_BASE}/api/devices/enroll`, {
      method: "POST",
      headers: authHeaders(token),
      body: JSON.stringify({ name }),
    }),
  );
}

export async function createAction(
  token: string,
  deviceId: string,
  action: string,
  params: Record<string, string> = {},
  note = "",
) {
  return parse<{ action: { id: string; status: string } }>(
    await fetch(`${API_BASE}/api/actions`, {
      method: "POST",
      headers: authHeaders(token),
      body: JSON.stringify({ device_id: deviceId, action, params, note }),
    }),
  );
}

export async function listEvents(token: string) {
  return parse<{ events: Array<{ id: string; kind: string; message: string; created_at: string }> }>(
    await fetch(`${API_BASE}/api/events`, { headers: authHeaders(token) }),
  );
}

export async function hardening(token: string, topic: string, context: string, language: "fa" | "en" = "en") {
  return parse<{ playbook: string; offline_fallback: boolean; model: string }>(
    await fetch(`${API_BASE}/api/hardening`, {
      method: "POST",
      headers: authHeaders(token),
      body: JSON.stringify({ topic, context, language }),
    }),
  );
}

export async function listVendors(token: string) {
  return parse<{ vendors: Vendor[] }>(
    await fetch(`${API_BASE}/api/vendors`, { headers: authHeaders(token) }),
  );
}

export async function syncVendors(token: string, vendorIds: string[] = [], categories: string[] = []) {
  return parse<{ synced: number; ok_count: number; results: Array<Record<string, unknown>> }>(
    await fetch(`${API_BASE}/api/vendors/sync`, {
      method: "POST",
      headers: authHeaders(token),
      body: JSON.stringify({ vendor_ids: vendorIds, categories }),
    }),
  );
}

export { API_BASE };
