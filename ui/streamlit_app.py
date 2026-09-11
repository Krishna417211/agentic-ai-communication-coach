"""Streamlit UI for the 4 Core AI Communication Services.

1. 🛠️ Service 1: Message Corrector — Fixes mechanics, grammar, and outputs the Corrected Version.
2. ✉️ Service 2: Professional Email Generator — Generates structured emails with Subject Line & Draft.
3. 🎭 Service 3: Sentence Tone Converter — Converts sentences to any target register with side-by-side diffs.
4. 🎯 Service 4: Interview Question Generator (Domain or PDF) — Generates interview questions for ANY domain or from an uploaded PDF.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any

import httpx
import streamlit as st


def _sync_streamlit_secrets() -> None:
    """Copy Streamlit secrets into os.environ for FastAPI and Pydantic Settings."""
    try:
        if hasattr(st, "secrets"):
            def _inject(d: Any) -> None:
                for k, v in d.items():
                    if isinstance(v, (str, int, float, bool)):
                        str_val = str(v)
                        os.environ[k] = str_val
                        os.environ[k.upper()] = str_val
                        os.environ[k.lower()] = str_val
                    elif hasattr(v, "items"):
                        _inject(v)

            _inject(st.secrets)
    except Exception:
        pass


def _port_is_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@st.cache_resource(show_spinner="Starting Communication Engine…")
def _start_embedded_api(port: int = 8000) -> str:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    _sync_streamlit_secrets()

    import uvicorn
    from app.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(100):
        if _port_is_open("127.0.0.1", port):
            break
        time.sleep(0.2)
    return f"http://127.0.0.1:{port}"


def _resolve_api_base() -> str:
    _sync_streamlit_secrets()
    configured = os.getenv("API_BASE_URL")
    if configured:
        return configured.rstrip("/")
    if _port_is_open("127.0.0.1", 8000):
        return "http://127.0.0.1:8000"
    return _start_embedded_api()


# Page setup
st.set_page_config(
    page_title="AI Communication Suite (4 Services)",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = _resolve_api_base()
API = f"{API_BASE}/api/v1"
TIMEOUT = float(os.getenv("UI_TIMEOUT_SECONDS", "240"))


def call_api(
    method: str, path: str, **kwargs: Any
) -> tuple[dict | None, str | None]:
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.request(method, f"{API}{path}", **kwargs)
    except httpx.ConnectError:
        return None, f"Cannot connect to API backend at {API_BASE}."
    except httpx.TimeoutException:
        return None, f"Request timed out after {TIMEOUT:.0f}s."
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
# UI Helpers
# ---------------------------------------------------------------------------

SCORE_LABELS = ("clarity", "tone", "grammar", "structure", "impact")


def score_badge(value: int) -> str:
    if value >= 80:
        return "🟢"
    elif value >= 60:
        return "🟡"
    return "🔴"


def render_scores(score: dict, overall: int | None = None) -> None:
    if overall is not None:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Overall Score", f"{overall}/100")
        with col2:
            st.progress(min(max(overall, 0), 100) / 100)
    
    cols = st.columns(len(SCORE_LABELS))
    for col, label in zip(cols, SCORE_LABELS):
        val = int(score.get(label, 0))
        col.metric(label.capitalize(), f"{score_badge(val)} {val}")


# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------

st.session_state.setdefault("session_id", None)


# ---------------------------------------------------------------------------
# Sidebar (Settings & Status)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🗣️ Communication Suite")
    st.caption("4 Specialized AI Communication Services")

    health = get_health()
    if health is None:
        st.error("API Backend Offline")
    else:
        st.success(f"Engine Online · v{health['version']}")
        if health.get("llm_available"):
            st.caption(f"Active Provider: **{health['provider']}**")
        else:
            st.warning("Running Rule Engine. Add an API Key below for Generative LLM responses.")

    with st.expander("⚙️ LLM Settings", expanded=False):
        provider_choice = st.selectbox(
            "Provider",
            options=["auto", "gemini", "groq", "openai", "openrouter", "ollama", "heuristic"],
            index=0,
        )
        key_val = st.text_input(
            "API Key",
            value=os.getenv("GEMINI_API_KEY", "") or os.getenv("GROQ_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Enter API key for Google Gemini, Groq, OpenAI, or OpenRouter",
        )
        default_model = "gemini-1.5-flash" if provider_choice in ("auto", "gemini") else ("llama-3.3-70b-versatile" if provider_choice == "groq" else "gpt-4o-mini")
        model_val = st.text_input("Model Name", value=default_model)

        if st.button("Apply Settings", width="stretch"):
            os.environ["LLM_PROVIDER"] = provider_choice
            if key_val:
                os.environ["GEMINI_API_KEY"] = key_val
                os.environ["GROQ_API_KEY"] = key_val
                os.environ["OPENAI_API_KEY"] = key_val
                os.environ["OPENROUTER_API_KEY"] = key_val
            if model_val:
                os.environ["GEMINI_MODEL"] = model_val
                os.environ["GROQ_MODEL"] = model_val
                os.environ["OPENAI_MODEL"] = model_val

            from app.config import get_settings
            get_settings.cache_clear()
            st.success(f"Provider set to {provider_choice}!")
            st.rerun()

    st.divider()
    if st.button("🗑️ Reset Session", width="stretch"):
        if st.session_state.session_id:
            call_api("DELETE", f"/history/{st.session_state.session_id}")
        st.session_state.session_id = None
        st.rerun()


# ---------------------------------------------------------------------------
# Main 4 Services Navigation
# ---------------------------------------------------------------------------

st.title("AI Communication Suite")
st.caption("Select one of the 4 dedicated services below to fix, generate, transform, or practice interview questions.")

service_corrector, service_email, service_tone, service_interview = st.tabs(
    [
        "🛠️ Service 1: Message Corrector",
        "✉️ Service 2: Professional Email Generator",
        "🎭 Service 3: Sentence Tone Converter",
        "🎯 Service 4: Interview Question Generator (Domain or PDF)",
    ]
)

# ===========================================================================
# SERVICE 1: Message Corrector (Grammar & Sentence Fixer)
# ===========================================================================

with service_corrector:
    st.subheader("🛠️ Message Corrector & Grammar Fixer")
    st.caption("Fix spelling errors, bad mechanics, awkward phrasing, and get the Corrected Version.")

    # Quick Example Prompts
    st.markdown("**Try a sample flawed message:**")
    samp_cols = st.columns(3)
    if samp_cols[0].button("Sample 1: Typo & Mechanics", width="stretch"):
        st.session_state.samp_text = "Their going to the meeting tommorow and its starting late."
    if samp_cols[1].button("Sample 2: Accusatory Phrase", width="stretch"):
        st.session_state.samp_text = "You never send the numbers on time and this is obviously your fault."
    if samp_cols[2].button("Sample 3: Filler & Hedges", width="stretch"):
        st.session_state.samp_text = "I just wanted to check if maybe you sort of had time to review this stuff."

    input_msg = st.text_area(
        "Message to Correct",
        value=st.session_state.get("samp_text", ""),
        height=180,
        placeholder="Paste any sentence or paragraph here to check for grammar, spelling, and clarity errors...",
        key="corrector_input",
    )

    if st.button("🛠️ Correct Message", type="primary", width="stretch", disabled=not input_msg.strip()):
        with st.spinner("Analyzing and generating corrected version..."):
            # Call /improve to get the exact corrected version + /analyze for score breakdown
            improve_res, imp_err = call_api(
                "POST",
                "/improve",
                json={"text": input_msg, "target_tone": "professional", "session_id": st.session_state.session_id},
            )
            analyze_res, anz_err = call_api(
                "POST",
                "/analyze",
                json={"text": input_msg, "session_id": st.session_state.session_id},
            )

        if imp_err:
            st.error(imp_err)
        else:
            st.divider()
            
            # Prominent Display of Corrected Version
            st.markdown("### ✨ Corrected Version")
            corrected_text = improve_res.get("improved", input_msg)
            st.success(corrected_text)
            st.code(corrected_text, language="text")

            if analyze_res and not anz_err:
                st.markdown("### 📊 Communication Scores & Diagnostics")
                render_scores(analyze_res["score"], analyze_res["overall_score"])

                issues = analyze_res.get("grammar_issues", [])
                st.markdown("### ✏️ Detailed Corrections Breakdown")
                if issues:
                    st.dataframe(
                        [
                            {
                                "Original Issue": i.get("original"),
                                "Suggested Fix": i.get("suggestion"),
                                "Explanation": i.get("message"),
                                "Severity": i.get("severity", "medium").upper(),
                            }
                            for i in issues
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("No mechanical errors found. The sentence structure was already clean.")

                col_str, col_weak = st.columns(2)
                with col_str:
                    st.markdown("#### ✅ Key Strengths")
                    for item in analyze_res.get("strengths", ["—"]):
                        st.write(f"- {item}")
                with col_weak:
                    st.markdown("#### ⚠️ Areas to Improve")
                    for item in analyze_res.get("weaknesses", ["—"]):
                        st.write(f"- {item}")


# ===========================================================================
# SERVICE 2: Professional Email Generator
# ===========================================================================

with service_email:
    st.subheader("✉️ Professional Email Generator")
    st.caption("Turn rough notes or bullet points into a clean, structured email with Subject Line & Draft.")

    col_recip, col_goal = st.columns(2)
    recipient = col_recip.text_input(
        "Recipient / Audience",
        placeholder="e.g. My Manager, Client, HR Department, Engineering Team",
        key="email_recipient",
    )
    goal = col_goal.text_input(
        "Main Email Goal",
        placeholder="e.g. Requesting approval for $5,000 budget, Updating project timeline",
        key="email_goal",
    )

    notes = st.text_area(
        "Key Points / Rough Notes",
        height=160,
        placeholder="Enter bullet points or rough thoughts...\n- Need $5,000 for software licenses\n- Helps team speed up delivery by 30%\n- Need decision by Friday",
        key="email_notes",
    )

    if st.button("✉️ Generate Professional Email", type="primary", width="stretch", disabled=not notes.strip()):
        prompt_text = f"Help me write a professional email to {recipient or 'the recipient'}.\nGoal: {goal or 'Professional communication'}.\nKey points:\n{notes}"
        with st.spinner("Drafting professional email version..."):
            payload, error = call_api(
                "POST",
                "/coach",
                json={
                    "message": prompt_text,
                    "session_id": st.session_state.session_id,
                    "audience": recipient or None,
                    "goal": goal or None,
                },
            )
        if error:
            st.error(error)
        else:
            st.session_state.session_id = payload["session_id"]
            st.divider()
            
            # Prominent Display of Generated Email Version
            st.markdown("### ✉️ Generated Email Version")
            st.markdown(payload["coaching_message"])

            if payload.get("score"):
                st.divider()
                st.markdown("### 📊 Email Quality Metrics")
                render_scores(payload["score"], payload.get("overall_score"))


# ===========================================================================
# SERVICE 3: Sentence Tone Converter
# ===========================================================================

with service_tone:
    st.subheader("🎭 Sentence Tone Converter")
    st.caption("Convert any sentence or paragraph into your target tone and see the side-by-side diff.")

    tone_input = st.text_area(
        "Original Text to Convert",
        height=160,
        placeholder="Enter sentence or paragraph to convert...\ne.g. 'I need the report today or we are going to fail the presentation.'",
        key="tone_input",
    )

    st.markdown("**Select Target Tone:**")
    t_row1 = st.columns(4)
    t_row2 = st.columns(4)

    tone_selected = "executive"
    if t_row1[0].button("👔 Executive", width="stretch"):
        tone_selected = "executive"
    if t_row1[1].button("🕊️ Diplomatic / Polite & Firm", width="stretch"):
        tone_selected = "polite_firm"
    if t_row1[2].button("💪 Assertive", width="stretch"):
        tone_selected = "assertive"
    if t_row1[3].button("⚡ Concise", width="stretch"):
        tone_selected = "concise"

    if t_row2[0].button("😊 Warm & Friendly", width="stretch"):
        tone_selected = "friendly"
    if t_row2[1].button("💼 Formal", width="stretch"):
        tone_selected = "formal"
    if t_row2[2].button("🤝 Empathetic", width="stretch"):
        tone_selected = "empathetic"
    if t_row2[3].button("🙇 Apologetic", width="stretch"):
        tone_selected = "apologetic"

    chosen_tone = st.selectbox(
        "Active Tone Selection",
        ["executive", "polite_firm", "assertive", "concise", "friendly", "formal", "empathetic", "apologetic"],
        index=["executive", "polite_firm", "assertive", "concise", "friendly", "formal", "empathetic", "apologetic"].index(tone_selected) if tone_selected in ["executive", "polite_firm", "assertive", "concise", "friendly", "formal", "empathetic", "apologetic"] else 0,
        key="chosen_tone_select",
    )

    audience_opt = st.text_input("Audience (Optional)", placeholder="e.g. Executive VP, Direct Report, Client")

    if st.button("🎭 Convert Tone", type="primary", width="stretch", disabled=not tone_input.strip()):
        with st.spinner(f"Converting sentence tone to {chosen_tone.replace('_', ' ').title()}..."):
            payload, error = call_api(
                "POST",
                "/improve",
                json={
                    "text": tone_input,
                    "target_tone": chosen_tone,
                    "audience": audience_opt or None,
                    "session_id": st.session_state.session_id,
                },
            )
        if error:
            st.error(error)
        else:
            st.divider()
            
            # Prominent Display of Converted Version Side-by-Side
            st.markdown("### 🎭 Converted Tone Version")
            col_before, col_after = st.columns(2)

            with col_before:
                st.markdown("#### Original Sentence")
                st.info(payload["original"])
                if payload.get("score_before"):
                    b_score = sum(payload["score_before"].values()) // 5
                    st.metric("Original Score", f"{b_score}/100")

            with col_after:
                st.markdown(f"#### Converted Version ({chosen_tone.replace('_', ' ').title()})")
                st.success(payload["improved"])
                st.code(payload["improved"], language="text")
                if payload.get("score_after") and payload.get("score_before"):
                    b_score = sum(payload["score_before"].values()) // 5
                    a_score = sum(payload["score_after"].values()) // 5
                    st.metric("Converted Score", f"{a_score}/100", delta=a_score - b_score)

            st.markdown("### 📝 Specific Changes Applied")
            for c in payload.get("changes", []):
                st.write(f"- {c}")


# ===========================================================================
# SERVICE 4: Interview Question Generator (Domain or PDF)
# ===========================================================================

with service_interview:
    st.subheader("🎯 Interview Question Generator (Domain or PDF)")
    st.caption("Generate targeted interview questions for ANY domain/job title OR extract questions directly from a PDF resume or job description.")

    mode_domain, mode_pdf = st.tabs(["🌐 Any Custom Domain / Role", "📄 From Uploaded PDF"])

    # -----------------------------------------------------------------------
    # Mode A: Any Custom Domain / Role
    # -----------------------------------------------------------------------
    with mode_domain:
        st.markdown("#### Generate Questions for Any Domain or Job Title")
        col_dom, col_diff, col_cnt = st.columns([2, 1, 1])
        custom_domain = col_dom.text_input(
            "Custom Domain / Role",
            value="Full Stack Developer",
            placeholder="e.g. Data Engineer, DevOps Specialist, Marketing Manager, Financial Analyst, Nurse Practitioner",
            key="custom_domain_input",
        )
        domain_diff = col_diff.selectbox("Difficulty", ["easy", "medium", "hard"], index=1, key="domain_diff")
        domain_cnt = col_cnt.number_input("Question Count", 1, 10, 3, key="domain_cnt")

        if st.button("🎯 Generate Domain Questions", type="primary", width="stretch"):
            with st.spinner(f"Generating questions for {custom_domain}..."):
                payload, error = call_api(
                    "POST",
                    "/interview/start",
                    json={
                        "role": custom_domain,
                        "difficulty": domain_diff,
                        "question_count": int(domain_cnt),
                        "session_id": st.session_state.session_id,
                    },
                )
            if error:
                st.error(error)
            else:
                st.session_state.session_id = payload["session_id"]
                st.session_state.domain_questions = payload
                st.rerun()

        domain_q_data = st.session_state.get("domain_questions")
        if domain_q_data:
            st.divider()
            st.success(f"Generated Questions for Domain: **{domain_q_data['role']}** ({domain_q_data['difficulty'].capitalize()})")
            for q in domain_q_data["questions"]:
                with st.expander(f"Question {q['index']}: {q['question']}", expanded=True):
                    st.caption(f"Competency Tested: **{q['competency']}**")
                    ans_input = st.text_area(
                        "Your Answer",
                        key=f"ans_dom_{q['index']}",
                        height=140,
                        placeholder="Type your response to evaluate against STAR method...",
                    )
                    if st.button("Evaluate Response & Get Correct Version", key=f"eval_dom_{q['index']}", disabled=not ans_input.strip()):
                        with st.spinner("Evaluating STAR coverage and generating model answer..."):
                            res, err = call_api(
                                "POST",
                                "/interview/answer",
                                json={
                                    "session_id": domain_q_data["session_id"],
                                    "question": q["question"],
                                    "answer": ans_input,
                                },
                            )
                        if err:
                            st.error(err)
                        else:
                            render_scores(res["score"], res["overall_score"])
                            
                            st.markdown("#### STAR Framework Coverage")
                            star_cols = st.columns(4)
                            for col, (part, present) in zip(star_cols, res["star_coverage"].items()):
                                col.metric(part.capitalize(), "✅ Present" if present else "❌ Missing")
                            
                            st.markdown("#### 💡 Correct / Exemplary Model Answer Version")
                            if res.get("model_answer"):
                                st.success(res["model_answer"])
                                st.code(res["model_answer"], language="text")
                            else:
                                st.info("Ensure all STAR components (Situation, Task, Action, Result) are covered for a complete model answer.")

                            st.markdown("#### 🔍 Actionable Feedback")
                            for item in res.get("feedback", []):
                                st.write(f"- {item}")

    # -----------------------------------------------------------------------
    # Mode B: From Uploaded PDF
    # -----------------------------------------------------------------------
    with mode_pdf:
        st.markdown("#### Upload PDF (Resume, CV, or Job Description)")
        st.caption("Upload a text-based PDF (Max 5 MB) to extract targeted interview questions.")

        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key="pdf_uploader")
        if uploaded_file and st.button("📄 Extract Questions from PDF", type="primary", width="stretch"):
            with st.spinner("Reading PDF and extracting questions..."):
                payload, error = call_api(
                    "POST",
                    "/resume/analyze",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"session_id": st.session_state.session_id or ""},
                )
            if error:
                st.error(error)
            else:
                st.divider()
                st.markdown("### 📄 PDF Document Summary")
                st.write(payload["summary"])
                st.caption(f"Pages: {payload['pages']} · Detected Sections: {', '.join(payload['detected_sections']) or 'None'}")

                st.markdown("### 🎯 Interview Questions Generated from PDF")
                questions = payload.get("suggested_interview_questions", [])
                for i, q in enumerate(questions, 1):
                    with st.expander(f"Question {i}: {q}", expanded=True):
                        st.markdown(f"**Question:** {q}")
                        st.info(f"**How to Answer (Correct STAR Approach):** Start with a specific situation from your experience, state your exact responsibility, detail the actions you took, and conclude with quantifiable results.")

                col_s, col_w = st.columns(2)
                with col_s:
                    st.markdown("#### ✅ Detected Strengths")
                    for item in payload.get("strengths", ["—"]):
                        st.write(f"- {item}")
                with col_w:
                    st.markdown("#### ⚠️ Weaknesses / Gaps")
                    for item in payload.get("weaknesses", ["—"]):
                        st.write(f"- {item}")
