# Deployment guide

The app is a single container that serves the API. The Streamlit UI is a second
process that talks to it over HTTP, so the two can be deployed together (Docker
Compose) or separately (API on Render, UI on Streamlit Community Cloud).

- [Before you deploy](#before-you-deploy)
- [Option A — Render](#option-a--render-recommended)
- [Option B — Railway](#option-b--railway)
- [Deploying the UI](#deploying-the-ui)
- [Docker Compose](#docker-compose-local-or-vps)
- [Post-deploy verification](#post-deploy-verification)
- [Troubleshooting](#troubleshooting)

---

## Before you deploy

### 1. Get a free LLM key

Neither requires a credit card.

| Provider | Console | Notes |
|---|---|---|
| **Groq** (recommended) | <https://console.groq.com/keys> | Fastest inference; default model `llama-3.3-70b-versatile` |
| **Google Gemini** | <https://aistudio.google.com/apikey> | Default model `gemini-2.0-flash` |

The app deploys and runs without a key — it falls back to the rule-based engine
— but generated drafts and conversational coaching need one.

### 2. Push to GitHub

```bash
git init
git add .
git commit -m "Agentic AI communication training agent"
git branch -M main
git remote add origin https://github.com/<you>/communication-training-agent.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `.venv/` and `logs/`. **Never commit your
API key** — set it in the platform dashboard instead.

---

## Option A — Render (recommended)

Render reads `render.yaml` from the repo root, so most of this is already
configured.

1. Sign in at <https://render.com> with GitHub.
2. **New → Web Service** → select your repository.
3. Render detects `render.yaml` and the Dockerfile. Confirm:
   - **Runtime:** Docker
   - **Plan:** Free
   - **Health check path:** `/health`
4. Under **Environment**, add your key:
   - `GROQ_API_KEY` = `gsk_...` (or `GEMINI_API_KEY`)
   - `LOG_FILE` = *(empty)* — Render's disk is ephemeral, so log to stdout
5. **Create Web Service**. First build takes 3-6 minutes.

Your public URL: `https://<service-name>.onrender.com`

```bash
curl https://<service-name>.onrender.com/health
```

> **Free plan behaviour:** the service sleeps after 15 minutes of inactivity.
> The first request after that takes 30-50s while the container wakes. This is
> the platform, not the app — if you are demoing, hit `/health` a minute
> beforehand to warm it.

### Setting env vars without `render.yaml`

If you create the service manually instead, the only required settings are:

| Setting | Value |
|---|---|
| Runtime | Docker |
| Dockerfile path | `./Dockerfile` |
| Health check path | `/health` |
| `GROQ_API_KEY` *(or `GEMINI_API_KEY`)* | your key |
| `LOG_FILE` | *(empty)* |

Do **not** set `PORT` yourself — Render injects it, and the Dockerfile's
`CMD` already binds to `${PORT:-8000}`.

---

## Option B — Railway

1. Sign in at <https://railway.app> with GitHub.
2. **New Project → Deploy from GitHub repo** → select your repository.
3. Railway detects the Dockerfile automatically.
4. **Variables** → add:
   - `GROQ_API_KEY` = `gsk_...`
   - `LOG_FILE` = *(empty)*
5. **Settings → Networking → Generate Domain** to get a public URL.

Railway also injects `PORT`; the Dockerfile already honours it.

Railway's free trial grants a usage credit rather than an always-free tier —
check your current allowance before relying on it for a long-lived demo.

---

## Deploying the UI

### Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io> and connect the same repository.
2. **Main file path:** `ui/streamlit_app.py`
3. **Advanced settings → Secrets**, point it at your deployed API:

   ```toml
   API_BASE_URL = "https://<service-name>.onrender.com"
   ```

   The UI reads `API_BASE_URL` from the environment, and Streamlit Cloud exposes
   secrets as environment variables.

4. Deploy.

### Or run both on one host

Use Docker Compose (below), or run the UI as a second Render service with:

```
streamlit run ui/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

and `API_BASE_URL` pointing at the API service.

---

## Docker Compose (local or VPS)

```bash
# with an LLM key
echo "GROQ_API_KEY=gsk_..." > .env
docker compose up --build

# or with no key (rule-based engine)
docker compose up --build
```

- API: <http://localhost:8000>
- UI: <http://localhost:8501>

The UI waits for the API's healthcheck before starting. Logs are written to
`./logs` on the host via a bind mount.

```bash
docker compose logs -f api     # follow API logs
docker compose down            # stop
docker compose up --build -d   # rebuild and run detached
```

---

## Post-deploy verification

Replace `$URL` with your public URL.

```bash
# 1. Health — confirms the provider that was picked up
curl $URL/health
# {"status":"ok","version":"1.0.0","provider":"groq","llm_available":true,...}

# 2. Full agentic pipeline
curl -X POST $URL/api/v1/coach \
  -H 'Content-Type: application/json' \
  -d '{"message":"Help me write an email asking my manager for time off"}'

# 3. Session memory — reuse the session_id from step 2
curl -X POST $URL/api/v1/coach \
  -H 'Content-Type: application/json' \
  -d '{"message":"make it more formal","session_id":"<session_id>"}'

# 4. Chat history
curl $URL/api/v1/history/<session_id>

# 5. Analysis
curl -X POST $URL/api/v1/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"i just wanted to maybe check if you could possibly send it"}'

# 6. Improvement
curl -X POST $URL/api/v1/improve \
  -H 'Content-Type: application/json' \
  -d '{"text":"You never send the numbers on time.","target_tone":"diplomatic"}'

# 7. Metrics
curl $URL/metrics

# 8. Interactive docs — open in a browser
open $URL/docs
```

**Checklist**

- [ ] `/health` returns `"status":"ok"`
- [ ] `provider` is `groq` or `gemini`, and `llm_available` is `true`
- [ ] `/api/v1/coach` returns an intent, a plan, tools used and a score
- [ ] A second call with the same `session_id` keeps the session
- [ ] `/api/v1/history/{id}` shows the turns
- [ ] `/docs` renders
- [ ] `/metrics` shows non-zero request counts
- [ ] The UI connects and shows "API online"

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/health` shows `"provider":"heuristic"` | No key was picked up | Check the variable is named exactly `GROQ_API_KEY` or `GEMINI_API_KEY`, and redeploy — env changes need a restart |
| First request takes ~40s | Render free plan cold start | Expected. Hit `/health` to warm it before demoing |
| `502` from a tool, `degraded: true` in the output | Provider rate-limited or down | The tool fell back to rules and the request still succeeded. Check `/metrics` → `recent_errors` |
| UI says "Cannot reach the API" | `API_BASE_URL` wrong or API asleep | Confirm the URL has no trailing slash and `curl $URL/health` works |
| Resume upload returns `422` | The PDF is a scan, not text | Export a text-based PDF; there is no OCR |
| Build fails on `pip install` | Platform pinned to an old Python | The Dockerfile pins `python:3.12-slim`; make sure the platform is building the Dockerfile, not autodetecting a buildpack |
| Logs are empty on Render | `LOG_FILE` still set to a path | Set `LOG_FILE` to empty so logs go to stdout, where Render collects them |

### Reading the logs

Every log line carries a `trace_id`, and so does every response
(`X-Trace-Id` header) and error body. To trace one request end to end:

```bash
docker compose logs api | grep '<trace_id>'
```

That gives you the intent decision, the plan, each tool with its timing, and the
final duration for that single request.
