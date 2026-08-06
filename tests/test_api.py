"""API contract tests."""

from __future__ import annotations

import io

import pytest

from app.observability.metrics import MetricsRegistry
from app.rag.knowledge import get_knowledge_base


class TestMeta:
    def test_root(self, client):
        body = client.get("/").json()
        assert body["endpoints"]["coaching_response"] == "/api/v1/coach"

    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "provider" in body

    def test_openapi_is_valid(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/v1/coach" in spec["paths"]
        assert "/api/v1/analyze" in spec["paths"]
        assert "/api/v1/improve" in spec["paths"]
        assert "/api/v1/history/{session_id}" in spec["paths"]

    def test_trace_header_is_returned(self, client):
        response = client.get("/health")
        assert response.headers["X-Trace-Id"]
        assert float(response.headers["X-Response-Time-ms"]) >= 0

    def test_trace_header_is_echoed(self, client):
        response = client.get("/health", headers={"X-Trace-Id": "my-trace-123"})
        assert response.headers["X-Trace-Id"] == "my-trace-123"


class TestCoach:
    def test_returns_the_full_pipeline(self, client, sample_email):
        body = client.post("/api/v1/coach", json={"message": sample_email}).json()
        assert body["session_id"]
        assert body["intent"]["intent"]
        assert body["plan"]["steps"]
        assert body["tools_used"]
        assert body["coaching_message"]
        assert 0 <= body["overall_score"] <= 100

    def test_session_continuity(self, client):
        first = client.post("/api/v1/coach", json={"message": "help me write an email"}).json()
        second = client.post(
            "/api/v1/coach",
            json={"message": "make it more formal", "session_id": first["session_id"]},
        ).json()
        assert second["session_id"] == first["session_id"]

    @pytest.mark.parametrize("payload", [{}, {"message": ""}, {"message": "x" * 9000}])
    def test_rejects_bad_input(self, client, payload):
        assert client.post("/api/v1/coach", json=payload).status_code == 422


class TestAnalyze:
    def test_analyzes_text(self, client, sample_email):
        body = client.post("/api/v1/analyze", json={"text": sample_email}).json()
        assert set(body["score"]) == {"clarity", "tone", "grammar", "structure", "impact"}
        assert body["grammar_issues"]
        assert body["tone"]["primary_tone"]

    def test_is_deterministic(self, client, sample_email):
        first = client.post("/api/v1/analyze", json={"text": sample_email}).json()
        second = client.post("/api/v1/analyze", json={"text": sample_email}).json()
        assert first["score"] == second["score"]


class TestImprove:
    def test_rewrites_and_scores(self, client, hostile_message):
        body = client.post(
            "/api/v1/improve",
            json={"text": hostile_message, "target_tone": "diplomatic"},
        ).json()
        assert body["improved"] != body["original"]
        assert body["score_after"]["tone"] > body["score_before"]["tone"]
        assert body["changes"]

    def test_defaults_to_professional(self, client):
        body = client.post("/api/v1/improve", json={"text": "hey gonna be late lol"}).json()
        assert body["improved"]


class TestHistory:
    def test_returns_turns(self, client):
        session_id = client.post(
            "/api/v1/coach", json={"message": "help me write an email to my boss"}
        ).json()["session_id"]
        body = client.get(f"/api/v1/history/{session_id}").json()
        assert body["turn_count"] == 2
        assert body["turns"][0]["role"] == "user"

    def test_404_for_unknown_session(self, client):
        response = client.get("/api/v1/history/does-not-exist")
        assert response.status_code == 404
        assert response.json()["trace_id"]

    def test_delete(self, client):
        session_id = client.post("/api/v1/coach", json={"message": "hello"}).json()["session_id"]
        assert client.delete(f"/api/v1/history/{session_id}").status_code == 200
        assert client.get(f"/api/v1/history/{session_id}").status_code == 404


class TestIntrospection:
    def test_lists_tools(self, client):
        tools = client.get("/api/v1/tools").json()["tools"]
        names = {t["name"] for t in tools}
        assert {
            "grammar_correction", "tone_analysis", "email_generation",
            "interview_coaching", "conversation_improvement", "communication_scoring",
        } <= names
        assert all(t["description"] and t["when_to_use"] for t in tools)

    def test_lists_intents(self, client):
        intents = client.get("/api/v1/intents").json()["intents"]
        assert len(intents) == 8


class TestKnowledgeBase:
    def test_corpus_loaded(self):
        assert get_knowledge_base().size > 10

    @pytest.mark.parametrize(
        "query,expected_source",
        [
            ("how do I handle an angry customer", "customer.md"),
            ("STAR method for interview answers", "interview.md"),
            ("my subject line and call to action", "email.md"),
            ("presentation nerves and pacing", "public_speaking.md"),
        ],
    )
    def test_retrieves_relevant_docs(self, query, expected_source):
        results = get_knowledge_base().search(query, top_k=3)
        assert expected_source in {r["source"] for r in results}

    def test_search_endpoint(self, client):
        body = client.get(
            "/api/v1/knowledge/search", params={"q": "de-escalate conflict"}
        ).json()
        assert body["results"]
        assert body["results"][0]["score"] > 0

    def test_empty_query_is_rejected(self, client):
        assert client.get("/api/v1/knowledge/search", params={"q": "a"}).status_code == 422


class TestInterview:
    def test_start_and_answer(self, client):
        start = client.post(
            "/api/v1/interview/start",
            json={"role": "Backend Engineer", "question_count": 3},
        ).json()
        assert len(start["questions"]) == 3

        answer = client.post(
            "/api/v1/interview/answer",
            json={
                "session_id": start["session_id"],
                "question": start["questions"][0]["question"],
                "answer": (
                    "At my last company I owned the payments service. I was asked to "
                    "cut failures. I rewrote the retry logic and as a result failed "
                    "transactions dropped from 4% to 0.6% in two months."
                ),
            },
        ).json()
        assert 0 <= answer["overall_score"] <= 100
        assert set(answer["star_coverage"]) == {"situation", "task", "action", "result"}
        assert all(answer["star_coverage"].values())

    def test_thin_answer_is_criticised(self, client):
        start = client.post("/api/v1/interview/start", json={"question_count": 1}).json()
        answer = client.post(
            "/api/v1/interview/answer",
            json={
                "session_id": start["session_id"],
                "question": start["questions"][0]["question"],
                "answer": "I worked on it and it went fine.",
            },
        ).json()
        assert answer["feedback"]
        assert not all(answer["star_coverage"].values())


class TestResume:
    def test_rejects_non_pdf(self, client):
        response = client.post(
            "/api/v1/resume/analyze",
            files={"file": ("cv.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 415

    def test_rejects_unreadable_pdf(self, client):
        response = client.post(
            "/api/v1/resume/analyze",
            files={"file": ("cv.pdf", io.BytesIO(b"not really a pdf"), "application/pdf")},
        )
        assert response.status_code == 422


class TestMetrics:
    def test_snapshot_shape(self, client):
        client.get("/health")
        body = client.get("/metrics").json()
        assert body["totals"]["requests"] >= 1
        assert set(body["latency_ms"]) >= {"count", "avg", "p50", "p95", "p99"}

    def test_prometheus_format(self, client):
        client.get("/health")
        text = client.get("/metrics/prometheus").text
        assert "# TYPE http_requests_total counter" in text
        assert "app_uptime_seconds" in text

    def test_error_rate_tracking(self):
        registry = MetricsRegistry()
        registry.record_request("GET /x", 200, 10.0)
        registry.record_request("GET /x", 500, 20.0)
        snapshot = registry.snapshot()
        assert snapshot["totals"]["error_rate"] == 0.5
        assert snapshot["latency_ms"]["max"] == 20.0

    def test_tool_metrics(self):
        registry = MetricsRegistry()
        registry.record_tool("grammar_correction", True, 5.0)
        registry.record_tool("grammar_correction", False, 7.0)
        tools = registry.snapshot()["tools"]
        assert tools["grammar_correction"]["calls"] == 2
        assert tools["grammar_correction"]["failures"] == 1

    def test_percentiles_on_empty(self):
        assert MetricsRegistry().snapshot()["latency_ms"]["count"] == 0
