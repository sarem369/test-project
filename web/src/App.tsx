import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import {
  API_BASE,
  chat,
  createAction,
  enrollDevice,
  getHealth,
  getMeta,
  hardening,
  listDevices,
  listEvents,
} from "./api";
import type { ChatMessage, Device } from "./api";

type Lang = "fa" | "en";
type Tab = "assistant" | "devices" | "hardening";

const copy = {
  fa: {
    tagline: "پلتفرم حفاظت دفاعی روی موتور Ollama",
    heroTitle: "حفاظت، رفع مشکل امنیتی، و پشتیبانی راه دور مجاز",
    heroBody:
      "Hefaaz فقط قابلیت‌های دفاعی دارد: مرور امنیتی، hardening، راهنمای پاک‌سازی بدافزار، و اجرای اقدامات تأیید‌شده روی دستگاه‌های خودتان.",
    token: "توکن ادمین",
    save: "ذخیره",
    refresh: "بروزرسانی",
    assistant: "دستیار امنیتی",
    devices: "دستگاه‌ها",
    hardening: "سخت‌سازی",
    ask: "سؤال دفاعی بپرسید…",
    send: "ارسال",
    enroll: "ثبت دستگاه جدید",
    deviceName: "نام دستگاه",
    enrollHint: "پس از ثبت، توکن ایجنت را روی سیستم مقصد تنظیم کنید.",
    events: "رویدادها",
    ollama: "وضعیت Ollama",
    online: "متصل",
    offline: "قطع / آفلاین",
    noDevices: "هنوز دستگاهی ثبت نشده است.",
    queueInventory: "جمع‌آوری موجودی",
    queueFirewall: "فعال‌سازی فایروال",
    queueUpdates: "بررسی آپدیت‌ها",
    generate: "تولید پلی‌بوک",
    context: "زمینه / محیط",
    language: "زبان",
  },
  en: {
    tagline: "Defensive protection platform on Ollama",
    heroTitle: "Protection, security remediation, and authorized remote support",
    heroBody:
      "Hefaaz is defensive-only: security review, hardening, high-level malware cleanup guidance, and approved remediations on your own machines.",
    token: "Admin token",
    save: "Save",
    refresh: "Refresh",
    assistant: "Security assistant",
    devices: "Devices",
    hardening: "Hardening",
    ask: "Ask a defensive question…",
    send: "Send",
    enroll: "Enroll device",
    deviceName: "Device name",
    enrollHint: "After enrollment, set the agent token on the target machine.",
    events: "Events",
    ollama: "Ollama status",
    online: "Connected",
    offline: "Offline",
    noDevices: "No devices enrolled yet.",
    queueInventory: "Collect inventory",
    queueFirewall: "Enable firewall",
    queueUpdates: "Check updates",
    generate: "Generate playbook",
    context: "Context / environment",
    language: "Language",
  },
} as const;

export default function App() {
  const [lang, setLang] = useState<Lang>("fa");
  const t = copy[lang];
  const [token, setToken] = useState(() => localStorage.getItem("hefaaz_token") || "hefaaz-dev-token");
  const [tokenDraft, setTokenDraft] = useState(token);
  const [tab, setTab] = useState<Tab>("assistant");
  const [health, setHealth] = useState<Awaited<ReturnType<typeof getHealth>> | null>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [events, setEvents] = useState<Array<{ id: string; kind: string; message: string; created_at: string }>>([]);
  const [topics, setTopics] = useState<Record<string, { fa: string; en: string }>>({});
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        lang === "fa"
          ? "سلام، من Hefaaz هستم. فقط راهنمایی دفاعی می‌دهم: hardening، رفع باگ امنیتی، و پاک‌سازی امن."
          : "Hi, I’m Hefaaz. I only provide defensive guidance: hardening, secure fixes, and safe cleanup.",
    },
  ]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [deviceName, setDeviceName] = useState("laptop-1");
  const [enrollResult, setEnrollResult] = useState("");
  const [topic, setTopic] = useState("linux_baseline");
  const [context, setContext] = useState("");
  const [playbook, setPlaybook] = useState("");

  const dir = lang === "fa" ? "rtl" : "ltr";

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = dir;
  }, [lang, dir]);

  async function refresh() {
    setError("");
    try {
      const h = await getHealth();
      setHealth(h);
      const [d, e, m] = await Promise.all([listDevices(token), listEvents(token), getMeta(token)]);
      setDevices(d.devices);
      setEvents(e.events);
      setTopics(m.hardening_topics);
      if (!topic && Object.keys(m.hardening_topics)[0]) {
        setTopic(Object.keys(m.hardening_topics)[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const ollamaOk = Boolean(health?.ollama?.ok);

  const topicOptions = useMemo(() => Object.entries(topics), [topics]);

  function saveToken() {
    localStorage.setItem("hefaaz_token", tokenDraft);
    setToken(tokenDraft);
  }

  async function onSend(e: FormEvent) {
    e.preventDefault();
    if (!prompt.trim() || busy) return;
    const next = [...messages, { role: "user" as const, content: prompt.trim() }];
    setMessages(next);
    setPrompt("");
    setBusy(true);
    setError("");
    try {
      const res = await chat(token, next);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.offline_fallback ? `${res.reply}\n\n(${res.model} · offline fallback)` : res.reply,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onEnroll(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await enrollDevice(token, deviceName.trim());
      setEnrollResult(
        `HEFAAZ_API=${API_BASE}\nHEFAAZ_DEVICE_ID=${res.device_id}\nHEFAAZ_AGENT_TOKEN=${res.agent_token}`,
      );
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function queue(deviceId: string, action: string) {
    setBusy(true);
    setError("");
    try {
      await createAction(token, deviceId, action, {}, "queued from dashboard");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function onHardening(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await hardening(token, topic, context, lang);
      setPlaybook(res.playbook);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <img src="/hefaaz.svg" alt="Hefaaz" />
          <div>
            <h1>Hefaaz</h1>
            <p>{t.tagline}</p>
          </div>
        </div>
        <div className="top-actions">
          <div className="token-box">
            <span className="muted">{t.token}</span>
            <input
              value={tokenDraft}
              onChange={(e) => setTokenDraft(e.target.value)}
              type="password"
              aria-label={t.token}
            />
            <button className="btn btn-ghost" type="button" onClick={saveToken}>
              {t.save}
            </button>
          </div>
          <button className="btn btn-ghost" type="button" onClick={() => setLang(lang === "fa" ? "en" : "fa")}>
            {lang === "fa" ? "EN" : "FA"}
          </button>
          <button className="btn btn-primary" type="button" onClick={() => void refresh()}>
            {t.refresh}
          </button>
        </div>
      </header>

      <section className="hero">
        <div>
          <h2>{t.heroTitle}</h2>
          <p>{t.heroBody}</p>
          {error ? <p style={{ color: "var(--danger)", marginTop: "0.9rem" }}>{error}</p> : null}
        </div>
        <div className="status-grid">
          <div className="status-pill">
            <strong>
              <span className={`dot ${ollamaOk ? "ok" : ""}`} />
              {t.ollama}
            </strong>
            <span>
              {ollamaOk ? t.online : t.offline}
              {health?.ollama?.default_model ? ` · ${health.ollama.default_model}` : ""}
            </span>
          </div>
          <div className="status-pill">
            <strong>{t.devices}</strong>
            <span>{health?.devices ?? devices.length}</span>
          </div>
          <div className="status-pill">
            <strong>API</strong>
            <span>{API_BASE}</span>
          </div>
        </div>
      </section>

      <div className="tabs">
        {(["assistant", "devices", "hardening"] as Tab[]).map((key) => (
          <button key={key} className={`tab ${tab === key ? "active" : ""}`} onClick={() => setTab(key)} type="button">
            {t[key]}
          </button>
        ))}
      </div>

      <div className="layout">
        <div className="stack">
          {tab === "assistant" ? (
            <section className="chat-pane">
              <h3>{t.assistant}</h3>
              <div className="chat-log">
                {messages.map((m, idx) => (
                  <div key={idx} className={`bubble ${m.role}`}>
                    {m.content}
                  </div>
                ))}
              </div>
              <form className="composer" onSubmit={onSend}>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder={t.ask}
                />
                <button className="btn btn-primary" disabled={busy} type="submit">
                  {t.send}
                </button>
              </form>
            </section>
          ) : null}

          {tab === "devices" ? (
            <section className="panel">
              <h3>{t.devices}</h3>
              <form onSubmit={onEnroll} className="composer" style={{ marginBottom: "1rem" }}>
                <div className="field">
                  <label htmlFor="deviceName">{t.deviceName}</label>
                  <input id="deviceName" value={deviceName} onChange={(e) => setDeviceName(e.target.value)} />
                </div>
                <button className="btn btn-primary" disabled={busy} type="submit">
                  {t.enroll}
                </button>
                <p className="muted">{t.enrollHint}</p>
              </form>
              {enrollResult ? (
                <pre className="playbook" style={{ marginBottom: "1rem", padding: "0.8rem", borderRadius: 14 }}>
                  {enrollResult}
                </pre>
              ) : null}
              <div className="device-list">
                {devices.length === 0 ? <p className="muted">{t.noDevices}</p> : null}
                {devices.map((d) => (
                  <article key={d.id}>
                    <h4>{d.name}</h4>
                    <div className="meta">
                      {d.hostname || "—"} · {d.os_name || "—"} · {d.status || "unknown"}
                      {d.last_seen ? ` · ${new Date(d.last_seen).toLocaleString()}` : ""}
                    </div>
                    <div className="findings">
                      {(d.inventory?.findings || []).map((f, i) => (
                        <div className="finding" key={i}>
                          <strong>{f.title}</strong> — {f.advice}
                        </div>
                      ))}
                    </div>
                    <div className="row-actions">
                      <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => void queue(d.id, "collect_inventory")}>
                        {t.queueInventory}
                      </button>
                      <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => void queue(d.id, "enable_ufw")}>
                        {t.queueFirewall}
                      </button>
                      <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => void queue(d.id, "apt_update_check")}>
                        {t.queueUpdates}
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          {tab === "hardening" ? (
            <section className="playbook">
              <h3>{t.hardening}</h3>
              <form onSubmit={onHardening}>
                <div className="field">
                  <label htmlFor="topic">{t.hardening}</label>
                  <select id="topic" value={topic} onChange={(e) => setTopic(e.target.value)}>
                    {topicOptions.map(([key, labels]) => (
                      <option key={key} value={key}>
                        {labels[lang]}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="context">{t.context}</label>
                  <input id="context" value={context} onChange={(e) => setContext(e.target.value)} />
                </div>
                <button className="btn btn-primary" disabled={busy} type="submit">
                  {t.generate}
                </button>
              </form>
              {playbook ? <pre style={{ marginTop: "1rem" }}>{playbook}</pre> : null}
            </section>
          ) : null}
        </div>

        <aside className="panel">
          <h3>{t.events}</h3>
          <div className="events">
            {events.length === 0 ? <p className="muted">—</p> : null}
            {events.map((ev) => (
              <div className="event" key={ev.id}>
                <strong>{ev.kind}</strong>
                <div>{ev.message}</div>
                <div className="meta">{new Date(ev.created_at).toLocaleString()}</div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}