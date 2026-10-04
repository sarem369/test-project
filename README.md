# Hefaaz — پلتفرم حفاظت امنیتی با Ollama

Hefaaz یک پلتفرم **دفاعی** است که روی موتور [Ollama](https://ollama.com) اجرا می‌شود تا برای کامپیوترها و سرویس‌های خودتان:

- مشاوره و مرور امنیتی با هوش مصنوعی محلی
- بررسی وضعیت امنیتی دستگاه‌ها (ایجنت راه دور)
- اجرای اقدامات اصلاحی **تأیید‌شده و سفیدلیست‌شده**
- راهنمای hardening و رفع مشکلات امنیتی رایج

این پروژه عمداً شامل تست نفوذ تهاجمی، اکسپلویت یا اسکن حمله نیست.

## اجزا

| بخش | مسیر | نقش |
|---|---|---|
| Backend API | `backend/` | FastAPI + اتصال به Ollama + صف اقدامات |
| Dashboard | `web/` | داشبورد فارسی/انگلیسی |
| Agent | `agent/` | ایجنت نصب‌شونده روی دستگاه‌های مجاز |

## پیش‌نیاز

- Python 3.11+
- Node.js 20+
- [Ollama](https://ollama.com) در حال اجرا با یک مدل (مثلاً `llama3.2`)

```bash
ollama pull llama3.2
```

## اجرا سریع

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

باز کنید: `http://localhost:5173`  
توکن ادمین پیش‌فرض در UI: همان `HEFAAZ_ADMIN_TOKEN`.

### 3) Agent روی یک دستگاه مجاز

```bash
cd agent
pip install -r requirements.txt
export HEFAAZ_API=http://SERVER_IP:8080
export HEFAAZ_ENROLL_TOKEN=<token-from-dashboard>
python3 agent.py
```

## قابلیت‌های دفاعی

- چت امنیتی با Ollama (مرور، hardening، رفع باگ امنیتی، راهنمای پاک‌سازی بدافزار سطح بالا)
- ثبت و مانیتور دستگاه‌ها از راه دور
- گزارش وضعیت: فایروال، به‌روزرسانی‌ها، کاربران، پروسه‌های مشکوک رایج
- اقدامات اصلاحی فقط با تأیید اپراتور و از لیست سفید (مثلاً فعال‌سازی فایروال، پاک‌سازی کش موقت، اسکن امضای ساده)
- پلی‌بوک‌های hardening آماده

## امنیت استقرار

- فقط روی شبکه‌های مورد اعتماد یا با TLS/VPN
- توکن ادمین و توکن ثبت ایجنت را قوی انتخاب کنید
- اقدامات راه دور فقط روی دستگاه‌هایی که مالک/مجاز هستید اجرا شود
- قبل از اجرای remediation در production، تأیید دستی الزامی است

## لایسنس

MIT
