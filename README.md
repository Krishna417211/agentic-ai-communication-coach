# End-to-End Agentic AI Communication Training Agent

An AI communication coach that reasons about what you need, plans a coaching
workflow, selects its own tools, remembers the conversation, and returns
personalised feedback with a scored improvement.

Every request runs the full agentic pipeline:

```
User Query
    ↓
Intent Detection          heuristic classifier + LLM, with memory carry-over
    ↓
Communication Planner     rule-based workflow per intent, LLM may re-plan
    ↓
Tool Selection            validated against the tool registry
    ↓
Task Execution            tools run in order, each seeing earlier outputs
    ↓
Feedback & Improved Response
```

---

## Contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [The agent pipeline](#the-agent-pipeline)
- [Tools](#tools)
- [Memory](#memory)
- [API reference](#api-reference)
- [User interface](#user-interface)
- [Monitoring and logging](#monitoring-and-logging)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Design decisions](#design-decisions)
- [Project layout](#project-layout)

---

## Quick start

### Run it locally (no API key needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # runtime only (what Docker installs)
# pip install -r requirements-dev.txt    # runtime + pytest, to run the tests

# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

- API: <http://localhost:8000> · interactive docs at `/docs`
- UI: <http://localhost:8501>

**It runs with no API key.** With no key configured the agent uses its
rule-based engine: intent routing, grammar checking, tone analysis, scoring,
rewriting and the knowledge base all work fully. What you lose is generative
output (freely-composed emails, model interview answers, conversational
coaching prose). The app tells you this in the UI and in `/health` rather than
silently degrading.

### Add a free LLM key (recommended)

```bash
cp .env.example .env
```

Then set **one** of these in `.env`:

| Provider | Free tier | Get a key |
|---|---|---|
| **Groq** (default, fastest) | Generous free tier, no card | <https://console.groq.com/keys> |
| **Google Gemini** | Free tier, no card | <https://aistudio.google.com/apikey> |

```dotenv
GEMINI_API_KEY=<your key>
GEMINI_MODEL=gemini-3.6-flash
# or
GROQ_API_KEY=gsk_...
```

`LLM_PROVIDER=auto` (the default) picks whichever key is present, preferring
Groq, and falls back to the rule-based engine if neither is set. Set
`LLM_PROVIDER=gemini` to pin it explicitly.

> **Note on Gemini 3.x:** these models reason before answering, and those
> reasoning tokens are billed against `LLM_MAX_TOKENS`. The default of 3000
> leaves ample room; if you lower it, a long request can exhaust the budget
> during reasoning and return no answer. The provider detects that case and
> reports it rather than failing silently.

### Run it with Docker

```bash
docker compose up --build
```

API on `:8000`, UI on `:8501`. Pass keys through your shell or a `.env` file:

```bash
GROQ_API_KEY=gsk_... docker compose up --build
```

---

## Architecture

```
                          ┌──────────────────────┐
  Streamlit UI  ─────────▶│   FastAPI (app/api)  │
  (ui/, HTTP only)        └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │ CommunicationAgent   │  app/agent/orchestrator.py
                          └──────────┬───────────┘
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
    ┌───────────────┐        ┌──────────────┐        ┌────────────────┐
    │ IntentDetector│        │   Planner    │        │  Synthesizer   │
    └───────┬───────┘        └──────┬───────┘        └───────┬────────┘
            │                       │                        │
            │                ┌──────▼───────┐                │
            │                │ ToolRegistry │                │
            │                └──────┬───────┘                │
            │       ┌───────────────┼───────────────┐        │
            │       ▼               ▼               ▼        │
            │  grammar/tone   email/interview   scoring/RAG  │
            │       └───────────────┬───────────────┘        │
            ▼                       ▼                        ▼
    ┌────────────────────────────────────────────────────────────┐
    │  LLM provider (Groq │ Gemini │ offline)  ·  SessionStore    │
    │  NLP engine (app/nlp)  ·  KnowledgeBase (app/rag)           │
    └────────────────────────────────────────────────────────────┘
```

Two principles run through the design:

1. **The registry is the authority on what can run, never the model.** The
   planner may propose any tool; unknown or unavailable ones are dropped, and an
   empty plan falls back to the rule-based one. A hallucinated tool name cannot
   cause a failure.
2. **Every LLM-backed component has a deterministic fallback.** Providers fail,
   rate-limit and return malformed JSON — especially on free tiers. Each tool
   implements `run_offline()` as well as `run_llm()`, and a provider failure
   degrades that tool rather than the request.

---

## The agent pipeline

### 1. Intent detection — `app/agent/intent.py`

Two classifiers back each other up:

- A **weighted pattern classifier** that always runs. Overlapping patterns
  collapse to the highest-weighted match, so a single signal expressed two ways
  can't outvote two genuinely distinct signals.
- An **LLM classifier** that receives the heuristic result as a prior and may
  override it. If it returns an unrecognised intent, or the provider fails, the
  heuristic result stands.

Two behaviours worth calling out:

- **Follow-ups inherit context.** "make it shorter" carries no topic signal, so
  short follow-ups reuse the session's last intent (`method: memory-carryover`)
  instead of being misrouted to `general_coaching`.
- **The instruction outranks the pasted draft.** In
  `Does this sound too harsh? "You never send the numbers on time."` the draft is
  full of conflict vocabulary, but the user asked a *tone* question. Signals
  inside an extracted draft are weighted at 0.5 against the user's own words.

Recognised intents: `email_writing`, `interview_practice`, `grammar_correction`,
`tone_improvement`, `public_speaking`, `conflict_resolution`,
`customer_communication`, `general_coaching`.

### 2. Planning — `app/agent/planner.py`

Each intent has a default workflow encoding what a coach would actually do — for
conflict resolution: retrieve de-escalation principles → analyse tone → rewrite →
score. Plans adapt to the input: when no draft was supplied, analysis tools are
dropped but scoring stays (it scores the user's own message, which is still real
feedback).

When intent confidence is below 0.85 and an LLM is configured, the planner asks
it to propose a plan, then validates every step against the registry.

### 3. Execution — `app/agent/orchestrator.py`

Tools run in plan order. Each one's output is placed in `ctx.artifacts` so later
steps can build on it — the rewriter sees the tone analysis, the scorer sees the
rewrite and reports a before/after delta. A tool that raises is caught, recorded
as a failed `ToolResult`, and the turn continues.

### 4. Feedback synthesis — `app/agent/synthesis.py`

Tool outputs are collected into a single evidence bundle and turned into one
coaching reply. **Improved text and scores always come from the tools** — the
synthesizer narrates, it never re-decides. Without an LLM it composes a
structured markdown report from the same evidence.

---

## Tools

| Tool | What it does |
|---|---|
| `grammar_correction` | Finds and fixes grammar, spelling, punctuation and word-choice errors. Rule findings are fed to the LLM as grounding and merged back, so mechanical errors are never silently missed. |
| `tone_analysis` | Scores formality, politeness, warmth and assertiveness; names how the message will land; flags escalating phrasing. |
| `email_generation` | Drafts a complete email — subject, greeting, body, one call to action, sign-off. |
| `interview_coaching` | Generates role-specific questions; reviews answers against STAR with a score and a model answer. |
| `conversation_improvement` | Rewrites toward a target tone, with intent-specific playbooks for conflict, customer, and spoken delivery. |
| `communication_scoring` | Scores clarity, tone, grammar, structure and impact 0-100 with the evidence behind each number. |
| `knowledge_lookup` *(bonus)* | BM25 retrieval over the bundled coaching corpus. |
| `resume_analysis` *(bonus)* | Parses a PDF resume, scores its bullets for impact and evidence, generates interview questions from it. |

Tools are self-describing (`name`, `description`, `when_to_use`) and registered in
`app/tools/__init__.py`. Adding one makes it selectable by the planner without
touching the planner or the router. `GET /api/v1/tools` returns the live catalog.

### Scoring is deterministic

Word counts, readability, passive voice, hedges and filler are computed in code,
not asked of the model — LLMs are unreliable at counting, and the same input must
always produce the same score so progress across a session is comparable. The LLM
receives those numbers as grounding facts and adds qualitative commentary on top.

---

## Memory

`app/memory/store.py` keeps short-term conversational memory per session:

- A **bounded window** of recent turns (default 12); overflow is folded into a
  rolling summary rather than dropped.
- **Coaching history** — every score, so the coach can say
  *"improving (+14 since your first message)"*.
- **Recurring issues** — findings are normalised into stable labels and counted,
  so the third hedged message gets *"you've hedged in 3 of your last 4
  messages"* instead of the same generic note a third time.
- **Flow state** for multi-turn features (an interview in progress).
- **TTL eviction** (default 6h) so memory stays flat on a free-tier host.

The store is async-safe and sits behind three methods, so swapping it for Redis
to support multiple workers is a contained change.

---

## API reference

Base path: `/api/v1`. Interactive docs at `/docs`, OpenAPI at `/openapi.json`.

### Required endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/coach` | **Coaching response** — the full agentic pipeline |
| `POST` | `/api/v1/analyze` | **Communication analysis** — score and diagnose, no rewrite |
| `POST` | `/api/v1/improve` | **Communication improvement** — rewrite + score delta |
| `GET` | `/api/v1/history/{session_id}` | **Chat history** |
| `DELETE` | `/api/v1/history/{session_id}` | Clear a session |

### Bonus endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/resume/analyze` | PDF resume analysis (multipart upload) |
| `POST` | `/api/v1/interview/start` | Start an interview simulation |
| `POST` | `/api/v1/interview/answer` | STAR review of an answer |
| `GET` | `/api/v1/knowledge/search` | Search the coaching knowledge base |

### Meta

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, provider, whether an LLM is available |
| `GET` | `/metrics` | Metrics snapshot (JSON) |
| `GET` | `/metrics/prometheus` | Prometheus text format |
| `GET` | `/api/v1/tools` | Tool catalog |
| `GET` | `/api/v1/intents` | Recognised intents |

### Examples

**Coaching response**

```bash
curl -X POST http://localhost:8000/api/v1/coach \
  -H 'Content-Type: application/json' \
  -d '{"message": "I need to tell my client the project will be two weeks late."}'
```

```jsonc
{
  "session_id": "b7579cd45a1b4e0f",
  "trace_id": "042d7118f55f",
  "intent": {
    "intent": "customer_communication",
    "confidence": 0.95,
    "method": "heuristic",
    "rationale": "Keyword score 5.5 for customer_communication, ahead of ...",
    "entities": { "target_text": "", "audience": "my client" }
  },
  "plan": {
    "goal": "Protect the relationship while being straight with the customer.",
    "steps": [
      { "tool": "knowledge_lookup", "objective": "Retrieve customer-communication guidance." },
      { "tool": "tone_analysis", "objective": "Check empathy and blame language." },
      { "tool": "conversation_improvement", "objective": "Rewrite for the customer." },
      { "tool": "communication_scoring", "objective": "Score the response." }
    ],
    "method": "rule-based"
  },
  "tools_used": ["knowledge_lookup", "tone_analysis", "conversation_improvement", "communication_scoring"],
  "coaching_message": "**Coaching focus: customer communication** ...",
  "improved_response": "...",
  "feedback": ["No explicit call to action — the reader won't know what to do."],
  "score": { "clarity": 90, "tone": 55, "grammar": 91, "structure": 60, "impact": 68 },
  "overall_score": 73,
  "tool_results": [ /* per-tool output, timing, ok/error */ ],
  "knowledge_used": ["customer.md — Acknowledge before you explain"],
  "provider": "groq",
  "duration_ms": 1840.2
}
```

Continue the conversation by passing `session_id` back:

```bash
curl -X POST http://localhost:8000/api/v1/coach \
  -H 'Content-Type: application/json' \
  -d '{"message": "make it warmer", "session_id": "b7579cd45a1b4e0f"}'
```

**Improvement**

```bash
curl -X POST http://localhost:8000/api/v1/improve \
  -H 'Content-Type: application/json' \
  -d '{"text": "You never send the numbers on time. This is unacceptable.",
       "target_tone": "diplomatic"}'
```

Returns the rewrite plus `score_before` / `score_after`.

**Resume analysis**

```bash
curl -X POST http://localhost:8000/api/v1/resume/analyze \
  -F 'file=@resume.pdf'
```

### Errors

Every error returns a consistent envelope carrying the trace id, which also
appears in the `X-Trace-Id` response header and in every log line for that
request:

```json
{ "error": "http_404", "detail": "No session 'abc'.", "trace_id": "0f16bd4f3232" }
```

---

## User interface

`streamlit run ui/streamlit_app.py` — six tabs:

- **💬 Coach** — the chat, with an expandable *"How the agent decided"* panel
  showing the detected intent and confidence, the plan, each tool's result and
  timing, and the knowledge base citations.
- **📊 Analyze** — score breakdown, tone dial, grammar issue table.
- **✨ Improve** — before/after side by side with the score delta.
- **🎤 Interview** — question generation and per-answer STAR feedback.
- **📄 Resume** — PDF upload and analysis.
- **📈 Metrics** — live request, latency, tool and intent metrics.

The UI holds no agent logic — it is an HTTP client, so what it shows is exactly
what an API consumer gets. Point it elsewhere with `API_BASE_URL`.

---

## Monitoring and logging

**Structured logging** (`app/observability/logging_config.py`) — JSON by default,
one line per event, with a `trace_id` and `session_id` attached via context vars
so every line from a request correlates. Set `LOG_JSON=false` for readable local
output.

```json
{"timestamp":"2026-08-06T02:55:49Z","level":"INFO","logger":"app.agent.orchestrator",
 "message":"trace=042d71 tool=tone_analysis ok=True duration_ms=4.7",
 "trace_id":"042d7118f55f","session_id":"b7579cd45a1b4e0f"}
```

**Request tracking** (`middleware.py`) — every request gets a trace id (or reuses
an inbound `X-Trace-Id`), is timed, and returns `X-Trace-Id` and
`X-Response-Time-ms`. Requests over `SLOW_REQUEST_MS` are logged at WARNING.
Unhandled exceptions become a traced 500 rather than a stack trace on the wire.

**Metrics** (`metrics.py`) — dependency-free and bounded (latency samples live in
a fixed-size ring buffer per endpoint, so memory stays flat):

- Request counts, error counts and **error rate**, overall and per endpoint
- **Response-time** percentiles (p50/p95/p99, avg, max)
- **API usage** by endpoint and status code
- Tool call counts, failures and latency
- Intent distribution and LLM calls by provider
- The last 25 errors with their trace ids

Available as JSON at `/metrics` and in Prometheus format at
`/metrics/prometheus`.

---

## Configuration

All settings are environment variables (see `.env.example`); `app/config.py` is
the full list.

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `auto`, `groq`, `gemini`, `heuristic` |
| `GROQ_API_KEY` | — | Free tier at console.groq.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | |
| `GEMINI_API_KEY` | — | Free tier at aistudio.google.com |
| `GEMINI_MODEL` | `gemini-3.6-flash` | |
| `LLM_TIMEOUT_SECONDS` | `45` | |
| `LLM_MAX_RETRIES` | `2` | Exponential backoff on 429/5xx |
| `MEMORY_MAX_TURNS` | `12` | Turns kept before summarising |
| `SESSION_TTL_SECONDS` | `21600` | 6 hours |
| `LOG_LEVEL` | `INFO` | |
| `LOG_JSON` | `true` | `false` for human-readable |
| `LOG_FILE` | `logs/app.log` | Empty string = stdout only |
| `SLOW_REQUEST_MS` | `3000` | Threshold for slow-request warnings |
| `ENABLE_RAG` | `true` | |
| `RAG_TOP_K` | `3` | |
| `CORS_ORIGINS` | `*` | Comma-separated |

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                      # 123 tests
pytest -v                   # verbose
pytest tests/test_nlp.py    # one module
```

Coverage spans the deterministic NLP layer (grammar rules, tone lexicon,
scoring bounds and monotonicity, rewrite safety), the agent (intent routing for
all eight intents, entity extraction, planning, memory windowing and summarising,
orchestration), and the API (every endpoint, error envelopes, metrics).

The LLM paths are covered without an API key via a scripted `FakeLLM`
(`tests/conftest.py`), including the failure modes that matter in production:
a provider outage, a hallucinated tool name, an unrecognised intent, and
malformed JSON. Tests assert the system degrades rather than fails.

---

## Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for step-by-step instructions for Render
and Railway, plus the post-deploy verification checklist.

---

## Design decisions

**Why a rule-based engine underneath the LLM?** Free-tier providers rate-limit
and go down. Rather than returning an error, every tool degrades to a
deterministic implementation, and the response says so (`degraded: true`, and a
banner in the UI). The assignment asks for a system that works; this one works
even with no key at all.

**Why is scoring computed in code rather than by the model?** Determinism. The
same message must always score the same, or session progress is meaningless and
the before/after delta on a rewrite is noise. The LLM explains the score; it
doesn't set it.

**Why BM25 instead of embeddings for RAG?** The corpus is small, static and
domain-specific. BM25 over 37 sections retrieves correctly (verified in tests),
adds no dependency, and has no cold-start cost — which matters on a free host
that sleeps. An embedding index would be heavier for no measurable gain here.

**Why in-memory sessions?** The spec asks for short-term conversational memory,
and a free-tier host runs one worker. The store is deliberately narrow — three
methods — so moving to Redis is a contained change rather than a rewrite.

**Free-tier quota is the real constraint.** Gemini's free tier caps requests
*per day per model* (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). A full
coaching turn chains up to five LLM calls, so the daily cap arrives quickly under
testing. The provider distinguishes a burst rate-limit (retried with backoff)
from an exhausted daily cap (fails straight through to the rule engine rather
than burning seconds on a wall it cannot get past). When the cap is hit the app
keeps answering — degraded, and it says so.

**Latency.** With a reasoning model, a turn that runs the full chain
(tone → rewrite → score → synthesis) takes 30-90s; one that skips the rewrite
takes ~10s. `GEMINI_THINKING_LEVEL=low` is the default because it cuts per-call
latency roughly 4x with no quality loss on these tasks. The steps are genuinely
sequential — the rewriter consumes the tone analysis, the scorer consumes the
rewrite — so the remaining win would be running tone analysis and the initial
scoring concurrently.

**Known limitations.** Sessions are per-process, so horizontal scaling needs the
Redis swap. The rule-based rewriter is a transformation, not a generator — with
no LLM key it improves what you wrote and emits a fill-in scaffold for
from-scratch drafts rather than inventing content. The grammar checker is
pattern-based and will miss errors a parser would catch.

---

## Project layout

```
app/
├── main.py                 FastAPI app, lifespan, error handlers
├── config.py               Settings
├── schemas.py              Pydantic models (domain + API)
├── agent/
│   ├── orchestrator.py     The pipeline
│   ├── intent.py           Intent detection (heuristic + LLM)
│   ├── planner.py          Workflow planning + tool selection
│   ├── synthesis.py        Feedback synthesis
│   └── extract.py          Draft / tone / audience / role extraction
├── tools/                  One module per tool + registry
├── nlp/                    Deterministic engine
│   ├── textstats.py        Counts, readability, hedges, passive voice
│   ├── grammar.py          Rule-based grammar checking
│   ├── tone.py             Lexicon-based tone analysis
│   ├── scoring.py          Five-dimension scoring
│   └── rewrite.py          Deterministic rewriting
├── memory/store.py         Session store
├── llm/                    Provider abstraction (groq / gemini / offline)
├── rag/                    Knowledge base + corpus
├── resume/parser.py        PDF extraction
├── observability/          Logging, metrics, middleware
└── api/                    Routes and dependencies

ui/streamlit_app.py         Streamlit UI
tests/                      123 tests
Dockerfile · docker-compose.yml · render.yaml
requirements.txt (runtime) · requirements-dev.txt (+ tests)
```
