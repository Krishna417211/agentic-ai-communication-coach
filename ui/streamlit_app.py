"""Streamlit UI for the Communication Training Agent.

Talks to the FastAPI service over HTTP — it holds no agent logic of its own, so
what you see in the UI is exactly what an API consumer would get.

Two deployment shapes are supported:

* **Split** (docker compose, or UI on Streamlit Cloud + API on Render): set
  `API_BASE_URL` to the API's address and this file is a thin HTTP client.
* **Standalone** (Streamlit Community Cloud on its own): with no
  `API_BASE_URL` set and nothing already serving locally, the UI starts the
  FastAPI app in a background thread and talks to it over loopback. Streamlit
  Cloud only runs one process, so without this the deployed page would have no
  backend to call. The UI still speaks HTTP, so both shapes exercise exactly
  the same API surface.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any

import httpx
import streamlit as st


def _port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@st.cache_resource(show_spinner="Starting the coaching API…")
def _start_embedded_api(port: int = 8000) -> str:
    """Run the FastAPI app in a daemon thread and return its base URL.

    Cached so Streamlit's per-interaction reruns don't spawn a second server.
    """
    import sys
    from pathlib import Path

    # Streamlit Cloud runs this file directly, so the repo root may not be on
    # sys.path even though `app/` sits next to `ui/`.
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import uvicorn

    from app.main import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(100):  # wait up to ~20s for startup
        if _port_is_open("127.0.0.1", port):
            break
        time.sleep(0.2)
    return f"http://127.0.0.1:{port}"


def _resolve_api_base() -> str:
    configured = os.getenv("API_BASE_URL")
    if configured:
        return configured.rstrip("/")
    if _port_is_open("127.0.0.1", 8000):
        return "http://127.0.0.1:8000"  # someone already runs the API locally
    return _start_embedded_api()


# set_page_config must precede every other Streamlit call, including the
# cached starter's spinner — so it comes before the API base is resolved.
st.set_page_config(
    page_title="AI Communication Coach",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = _resolve_api_base()
API = f"{API_BASE}/api/v1"
# A full coaching turn chains up to four LLM calls plus synthesis. On a free
# tier with a reasoning model that legitimately exceeds 90s, so the client
# timeout has to be generous or the UI aborts a request that was going to
# succeed.
TIMEOUT = float(os.getenv("UI_TIMEOUT_SECONDS", "240"))


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def call_api(
    method: str, path: str, **kwargs: Any
) -> tuple[dict | None, str | None]:
    """Return (payload, error_message)."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, f"{API}{path}", **kwargs)
    except httpx.ConnectError:
        return None, (
            f"Cannot reach the API at {API_BASE}. Start it with "
            "`uvicorn app.main:app --reload` or `docker compose up`."
        )
    except httpx.TimeoutException:
        return None, f"The API did not respond within {TIMEOUT:.0f}s."
    except httpx.HTTPError as exc:
        return None, f"Request failed: {exc}"

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"API error {response.status_code}: {detail}"
    return response.json(), None


@st.cache_data(ttl=15)
def get_health() -> dict | None:
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{API_BASE}/health")
        return response.json() if response.status_code == 200 else None
    except httpx.HTTPError:
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SCORE_LABELS = ("clarity", "tone", "grammar", "structure", "impact")


def score_color(value: int) -> str:
    if value >= 75:
        return "🟢"
    return "🟡" if value >= 50 else "🔴"


def render_scores(score: dict, overall: int | None, *, caption: str = "") -> None:
    if overall is not None:
        st.metric("Overall communication score", f"{overall}/100", help=caption or None)
        st.progress(min(max(overall, 0), 100) / 100)
    columns = st.columns(len(SCORE_LABELS))
    for column, label in zip(columns, SCORE_LABELS):
        value = int(score.get(label, 0))
        column.metric(label.capitalize(), f"{score_color(value)} {value}")


def render_coaching(payload: dict) -> None:
    st.markdown(payload["coaching_message"])

    if payload.get("score"):
        st.divider()
        render_scores(payload["score"], payload.get("overall_score"))

    with st.expander("🔍 How the agent decided", expanded=False):
        intent = payload["intent"]
        left, right = st.columns(2)
        with left:
            st.markdown("**Intent detection**")
            st.write(f"- Intent: `{intent['intent']}`")
            st.write(f"- Confidence: {intent['confidence']:.0%}")
            st.write(f"- Method: `{intent['method']}`")
            if intent.get("rationale"):
                st.caption(intent["rationale"])
            if intent.get("entities"):
                extracted = {k: v for k, v in intent["entities"].items() if v}
                if extracted:
                    st.write("**Extracted**")
                    st.json(extracted, expanded=False)
        with right:
            st.markdown("**Plan**")
            st.write(f"_Goal:_ {payload['plan']['goal']}")
            for i, step in enumerate(payload["plan"]["steps"], 1):
                st.write(f"{i}. `{step['tool']}` — {step['objective']}")
            st.caption(f"Planner: {payload['plan']['method']}")

        st.markdown("**Tool execution**")
        for result in payload.get("tool_results", []):
            icon = "✅" if result["ok"] else "❌"
            st.write(
                f"{icon} `{result['tool']}` — {result['summary']} "
                f"({result['duration_ms']:.0f} ms)"
            )
            if result.get("error"):
                st.error(result["error"])

        if payload.get("knowledge_used"):
            st.markdown("**Knowledge base citations**")
            for citation in payload["knowledge_used"]:
                st.write(f"- {citation}")

        st.caption(
            f"provider: {payload['provider']} · trace: {payload['trace_id']} · "
            f"{payload['duration_ms']:.0f} ms total"
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

st.session_state.setdefault("session_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_payload", None)
st.session_state.setdefault("handle", "")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🗣️ Communication Coach")
    st.caption(
        "For early-career professionals who want to stop sounding junior in "
        "writing — and to see it happening."
    )

    health = get_health()
    if health is None:
        st.error("API offline")
        st.caption(f"Expected at {API_BASE}")
    else:
        st.success(f"API online · v{health['version']}")
        if health["llm_available"]:
            st.caption(f"LLM: **{health['provider']}**")
        else:
            st.warning(
                "No LLM key configured — running the rule-based engine. "
                "Analysis and scoring are fully available; generated drafts are not."
            )

    st.divider()

    # Identity is what turns a chat into coaching: without it every visit
    # starts from zero and the agent can only ever react to one message.
    st.caption("Who's practising?")
    handle = st.text_input(
        "Handle",
        value=st.session_state.handle,
        max_chars=32,
        placeholder="e.g. krishna",
        label_visibility="collapsed",
        help=(
            "Pick any name. Your scores and habits are saved under it so the "
            "coach can track whether you're improving across sessions. Only "
            "scores and habit labels are stored — never your messages."
        ),
    )
    if handle != st.session_state.handle:
        st.session_state.handle = handle
        st.rerun()

    if st.session_state.handle:
        progress, _ = call_api("GET", f"/progress/{st.session_state.handle}")
        if progress and progress["turns"]:
            st.caption(f"**{progress['turns']}** turns coached · {progress['trend']}")
        else:
            st.caption("No history yet — your first turn starts the record.")
    else:
        st.info("Add a handle to track your progress over time.", icon="📈")

    st.divider()
    st.caption("This session")
    st.code(st.session_state.session_id or "not started", language=None)

    if st.button("🗑️ New session", width="stretch"):
        if st.session_state.session_id:
            call_api("DELETE", f"/history/{st.session_state.session_id}")
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.last_payload = None
        st.rerun()

    if st.session_state.session_id:
        history, error = call_api("GET", f"/history/{st.session_state.session_id}")
        if history:
            st.metric("Turns", history["turn_count"])
            if history.get("average_score") is not None:
                st.metric("Average score", f"{history['average_score']:.1f}/100")
            if history.get("recurring_issues"):
                st.caption("**Recurring habits**")
                for issue in history["recurring_issues"]:
                    st.caption(f"• {issue}")
            if history.get("summary"):
                with st.expander("Session summary"):
                    st.caption(history["summary"])

    st.divider()
    st.caption(f"API docs: {API_BASE}/docs")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

# Tabs are named after what the user came to do, not after the tools that do
# it. "Analyze" and "Improve" are two halves of one job, as are the interview
# and resume tools, so each pair sits behind a single goal-shaped tab. The
# containers are created here and filled further down, which keeps the section
# bodies flat and in reading order.
chat_tab, progress_tab, toolkit_tab, prep_tab, internals_tab = st.tabs(
    [
        "💬 Coach",
        "📈 Your progress",
        "🔍 Analyze & rewrite",
        "🎯 Interview prep",
        "⚙️ Under the hood",
    ]
)

with toolkit_tab:
    analyze_tab, improve_tab = st.tabs(["📊 Diagnose it", "✨ Rewrite it"])

with prep_tab:
    interview_tab, resume_tab = st.tabs(["🎤 Mock interview", "📄 From your resume"])

metrics_tab = internals_tab


with progress_tab:
    st.subheader("Your progress")
    st.caption(
        "The part a chatbot can't do: what you keep getting wrong, and whether "
        "you're fixing it."
    )

    if not st.session_state.handle:
        st.info(
            "Add a handle in the sidebar to start tracking. Every coached "
            "message then counts towards your score history and habit list.",
            icon="👈",
        )
    else:
        progress, error = call_api("GET", f"/progress/{st.session_state.handle}")
        if error:
            st.error(error)
        elif not progress or not progress["turns"]:
            st.info(
                f"Nothing recorded for **{st.session_state.handle}** yet. "
                "Coach a message and it will show up here.",
                icon="📭",
            )
        else:
            top = st.columns(4)
            top[0].metric("Messages coached", progress["turns"])
            top[1].metric(
                "Latest score",
                f"{progress['latest_score']}/100"
                if progress["latest_score"] is not None
                else "—",
                delta=progress["delta"],
            )
            top[2].metric(
                "Average",
                f"{progress['average_score']}"
                if progress["average_score"] is not None
                else "—",
            )
            top[3].metric(
                "Best",
                f"{progress['best_score']}"
                if progress["best_score"] is not None
                else "—",
            )

            st.caption(f"Overall: **{progress['trend']}**")

            if len(progress["scores"]) >= 2:
                st.markdown("#### Score over time")
                st.line_chart(
                    {"score": progress["scores"]},
                    height=220,
                    y_label="score / 100",
                    x_label="coached message",
                )
            elif progress["scores"]:
                st.caption("One more coached message and a trend line appears here.")

            left, right = st.columns(2)
            with left:
                st.markdown("#### 🔧 Habits to work on")
                if progress["top_habits"]:
                    for habit in progress["top_habits"]:
                        st.write(f"- **{habit['habit']}** — {habit['count']}x")
                else:
                    st.caption(
                        "No repeated habits yet. A weakness has to show up "
                        "twice before it counts as a pattern."
                    )
            with right:
                st.markdown("#### ✅ Habits you're fixing")
                if progress["fixed_habits"]:
                    for habit in progress["fixed_habits"]:
                        st.write(
                            f"- **{habit['habit']}** — {habit['was']}x early, "
                            f"now {habit['now']}x"
                        )
                else:
                    st.caption(
                        "Needs at least 6 coached messages before this can say "
                        "anything honest."
                    )

            if progress["intents"]:
                st.markdown("#### What you practise most")
                st.bar_chart(
                    {
                        i["intent"].replace("_", " "): i["count"]
                        for i in progress["intents"]
                    },
                    height=200,
                    horizontal=True,
                )

            with st.expander("Manage this profile"):
                st.caption(
                    "Stored: scores, intents and habit labels. Never the text "
                    "of your messages."
                )
                if st.button("🗑️ Delete my history", type="secondary"):
                    call_api("DELETE", f"/progress/{st.session_state.handle}")
                    st.rerun()


with chat_tab:
    st.subheader("Coaching conversation")
    st.caption(
        "Paste a message you're about to send, or describe what you need to "
        "say. You get the rewrite, the reasons, and a score you can track."
    )

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        examples = [
            "Help me write an email asking my manager for a raise",
            "Does this sound too harsh? \"You never send the numbers on time.\"",
            "I have a Data Analyst interview next week — let's practise",
            "A customer is angry about an outage and wants a refund",
        ]
        columns = st.columns(2)
        for i, example in enumerate(examples):
            if columns[i % 2].button(example, key=f"ex{i}", width="stretch"):
                st.session_state.pending = example
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_coaching(message["payload"])

    prompt = st.chat_input("What would you like help with?")
    if not prompt and st.session_state.get("pending"):
        prompt = st.session_state.pop("pending")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"), st.spinner("Thinking…"):
            payload, error = call_api(
                "POST",
                "/coach",
                json={
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "handle": st.session_state.handle or None,
                },
            )
            if error:
                st.error(error)
                # Drop the orphaned user turn and do NOT rerun: a rerun would
                # wipe this error off the screen, and the turn would vanish
                # with no explanation of what went wrong.
                st.session_state.messages.pop()
            else:
                st.session_state.session_id = payload["session_id"]
                st.session_state.last_payload = payload
                st.session_state.messages.append(
                    {"role": "assistant", "payload": payload}
                )
                render_coaching(payload)
                st.rerun()


with analyze_tab:
    st.subheader("Communication analysis")
    st.caption("Score and diagnose a message without rewriting it.")

    text = st.text_area("Text to analyse", height=200, key="analyze_text")
    if st.button("Analyse", type="primary", disabled=not text.strip()):
        with st.spinner("Analysing…"):
            payload, error = call_api(
                "POST",
                "/analyze",
                json={"text": text, "session_id": st.session_state.session_id},
            )
        if error:
            st.error(error)
        else:
            render_scores(payload["score"], payload["overall_score"])
            st.divider()

            left, right = st.columns(2)
            with left:
                st.markdown("#### ✅ Strengths")
                for item in payload["strengths"] or ["—"]:
                    st.write(f"- {item}")
            with right:
                st.markdown("#### ⚠️ Weaknesses")
                for item in payload["weaknesses"] or ["—"]:
                    st.write(f"- {item}")

            tone = payload.get("tone", {})
            if tone:
                st.markdown("#### 🎭 Tone")
                columns = st.columns(4)
                for column, dimension in zip(
                    columns, ("formality", "politeness", "warmth", "assertiveness")
                ):
                    column.metric(dimension.capitalize(), f"{tone.get(dimension, 0)}/100")
                st.write(
                    f"**Reads as:** `{tone.get('primary_tone')}`"
                    + (
                        f" (also {', '.join(tone['secondary_tones'])})"
                        if tone.get("secondary_tones")
                        else ""
                    )
                )
                for risk in tone.get("risks", []):
                    st.warning(risk)

            if payload["grammar_issues"]:
                st.markdown("#### ✏️ Grammar and mechanics")
                st.dataframe(
                    [
                        {
                            "Original": i.get("original"),
                            "Suggestion": i.get("suggestion"),
                            "Why": i.get("message"),
                            "Severity": i.get("severity"),
                        }
                        for i in payload["grammar_issues"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                f"provider: {payload['provider']} · {payload['duration_ms']:.0f} ms"
            )


with improve_tab:
    st.subheader("Communication improvement")
    st.caption("Rewrite a message toward a target tone and see the score change.")

    original = st.text_area("Your message", height=200, key="improve_text")
    left, right = st.columns(2)
    target_tone = left.selectbox(
        "Target tone",
        ["professional", "friendly", "assertive", "empathetic", "diplomatic",
         "formal", "concise", "apologetic"],
    )
    audience = right.text_input("Audience (optional)", placeholder="e.g. my manager")

    if st.button("Improve", type="primary", disabled=not original.strip()):
        with st.spinner("Rewriting…"):
            payload, error = call_api(
                "POST",
                "/improve",
                json={
                    "text": original,
                    "target_tone": target_tone,
                    "audience": audience or None,
                    "session_id": st.session_state.session_id,
                },
            )
        if error:
            st.error(error)
        else:
            before, after = st.columns(2)
            with before:
                st.markdown("#### Before")
                st.text_area("original", payload["original"], height=260,
                             disabled=True, label_visibility="collapsed")
                if payload.get("score_before"):
                    st.metric("Score", f"{sum(payload['score_before'].values()) // 5}/100")
            with after:
                st.markdown("#### After")
                st.text_area("improved", payload["improved"], height=260,
                             label_visibility="collapsed")
                if payload.get("score_after"):
                    before_avg = sum(payload["score_before"].values()) // 5
                    after_avg = sum(payload["score_after"].values()) // 5
                    st.metric("Score", f"{after_avg}/100", delta=after_avg - before_avg)

            st.markdown("#### What changed")
            for change in payload["changes"]:
                st.write(f"- {change}")


with interview_tab:
    st.subheader("Interview simulation")
    st.session_state.setdefault("interview", None)

    left, middle, right = st.columns([2, 1, 1])
    role = left.text_input("Role", value="Software Engineer")
    difficulty = middle.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
    count = right.number_input("Questions", 1, 10, 5)

    if st.button("Start interview", type="primary"):
        with st.spinner("Preparing questions…"):
            payload, error = call_api(
                "POST",
                "/interview/start",
                json={
                    "role": role,
                    "difficulty": difficulty,
                    "question_count": int(count),
                    "session_id": st.session_state.session_id,
                },
            )
        if error:
            st.error(error)
        else:
            st.session_state.session_id = payload["session_id"]
            st.session_state.interview = payload
            st.rerun()

    interview = st.session_state.interview
    if interview:
        st.success(f"{interview['role']} · {interview['difficulty']}")
        for question in interview["questions"]:
            with st.expander(f"Q{question['index']}. {question['question']}"):
                st.caption(f"Tests: {question['competency']}")
                answer = st.text_area(
                    "Your answer",
                    key=f"ans{question['index']}",
                    height=170,
                    placeholder="Answer out loud first, then type what you said.",
                )
                if st.button("Get feedback", key=f"sub{question['index']}",
                             disabled=not answer.strip()):
                    with st.spinner("Reviewing…"):
                        payload, error = call_api(
                            "POST",
                            "/interview/answer",
                            json={
                                "session_id": interview["session_id"],
                                "question": question["question"],
                                "answer": answer,
                            },
                        )
                    if error:
                        st.error(error)
                    else:
                        render_scores(payload["score"], payload["overall_score"])
                        st.markdown("**STAR coverage**")
                        columns = st.columns(4)
                        for column, (part, present) in zip(
                            columns, payload["star_coverage"].items()
                        ):
                            column.metric(part.capitalize(), "✅" if present else "❌")
                        st.markdown("**Feedback**")
                        for item in payload["feedback"]:
                            st.write(f"- {item}")
                        if payload.get("model_answer"):
                            st.markdown("**A stronger version**")
                            st.info(payload["model_answer"])


with resume_tab:
    st.subheader("Resume analysis")
    st.caption("Upload a text-based PDF (not a scan). Max 5 MB.")

    uploaded = st.file_uploader("Resume", type=["pdf"])
    if uploaded and st.button("Analyse resume", type="primary"):
        with st.spinner("Reading the PDF…"):
            payload, error = call_api(
                "POST",
                "/resume/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                data={"session_id": st.session_state.session_id or ""},
            )
        if error:
            st.error(error)
        else:
            st.write(payload["summary"])
            render_scores(payload["communication_score"], None)
            st.caption(
                f"{payload['pages']} page(s) · sections found: "
                f"{', '.join(payload['detected_sections']) or 'none'}"
            )
            left, right = st.columns(2)
            with left:
                st.markdown("#### ✅ Strengths")
                for item in payload["strengths"] or ["—"]:
                    st.write(f"- {item}")
            with right:
                st.markdown("#### ⚠️ Weaknesses")
                for item in payload["weaknesses"] or ["—"]:
                    st.write(f"- {item}")
            st.markdown("#### 🎤 Questions an interviewer would ask from this")
            for i, question in enumerate(payload["suggested_interview_questions"], 1):
                st.write(f"{i}. {question}")


with metrics_tab:
    st.subheader("Under the hood")
    st.caption(
        "This app is an agent, not a prompt: it classifies your intent, plans "
        "a workflow, runs independent tools concurrently, and grounds the "
        "advice in a retrieval corpus. Live numbers below."
    )
    if st.button("Refresh"):
        st.rerun()

    try:
        with httpx.Client(timeout=10) as client:
            snapshot = client.get(f"{API_BASE}/metrics").json()
    except httpx.HTTPError as exc:
        st.error(f"Could not load metrics: {exc}")
        snapshot = None

    if snapshot:
        totals = snapshot["totals"]
        latency = snapshot["latency_ms"]
        columns = st.columns(5)
        columns[0].metric("Requests", totals["requests"])
        columns[1].metric("Errors", totals["errors"])
        columns[2].metric("Error rate", f"{totals['error_rate']:.1%}")
        columns[3].metric("p95 latency", f"{latency['p95']:.0f} ms")
        columns[4].metric("Uptime", f"{snapshot['uptime_seconds'] / 60:.1f} min")

        if snapshot["endpoints"]:
            st.markdown("#### Endpoints")
            st.dataframe(
                [
                    {
                        "Endpoint": endpoint,
                        "Requests": data["requests"],
                        "Errors": data["errors"],
                        "p50 ms": data["latency_ms"]["p50"],
                        "p95 ms": data["latency_ms"]["p95"],
                    }
                    for endpoint, data in snapshot["endpoints"].items()
                ],
                width="stretch",
                hide_index=True,
            )

        if snapshot["tools"]:
            st.markdown("#### Tool usage")
            st.dataframe(
                [
                    {
                        "Tool": tool,
                        "Calls": data["calls"],
                        "Failures": data["failures"],
                        "Avg ms": data["latency_ms"]["avg"],
                    }
                    for tool, data in snapshot["tools"].items()
                ],
                width="stretch",
                hide_index=True,
            )

        if snapshot["intents"]:
            st.markdown("#### Detected intents")
            st.bar_chart(snapshot["intents"])

        if snapshot["recent_errors"]:
            st.markdown("#### Recent errors")
            st.dataframe(snapshot["recent_errors"], width="stretch", hide_index=True)
