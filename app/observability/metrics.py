"""In-process metrics: request counts, latency percentiles, error rate.

Deliberately dependency-free and bounded — latency samples are kept in a ring
buffer per endpoint so memory stays flat on a long-running free-tier dyno.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any

_MAX_SAMPLES = 500
_MAX_RECENT_ERRORS = 25


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._started_wall = datetime.now(timezone.utc)

        self._requests: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._status: Counter[str] = Counter()
        self._latency: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES)
        )
        self._tool_calls: Counter[str] = Counter()
        self._tool_failures: Counter[str] = Counter()
        self._tool_latency: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES)
        )
        self._intents: Counter[str] = Counter()
        self._llm_calls: Counter[str] = Counter()
        self._recent_errors: deque[dict[str, Any]] = deque(maxlen=_MAX_RECENT_ERRORS)
        self._slow_requests: Counter[str] = Counter()

    # -- recording ---------------------------------------------------------

    def record_request(
        self, endpoint: str, status_code: int, duration_ms: float, *, slow: bool = False
    ) -> None:
        with self._lock:
            self._requests[endpoint] += 1
            self._status[str(status_code)] += 1
            self._latency[endpoint].append(duration_ms)
            if status_code >= 500:
                self._errors[endpoint] += 1
            if slow:
                self._slow_requests[endpoint] += 1

    def record_error(
        self, endpoint: str, error_type: str, message: str, trace_id: str
    ) -> None:
        with self._lock:
            self._errors[endpoint] += 1
            self._recent_errors.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "endpoint": endpoint,
                    "error_type": error_type,
                    "message": message[:300],
                    "trace_id": trace_id,
                }
            )

    def record_tool(self, tool: str, ok: bool, duration_ms: float) -> None:
        with self._lock:
            self._tool_calls[tool] += 1
            self._tool_latency[tool].append(duration_ms)
            if not ok:
                self._tool_failures[tool] += 1

    def record_intent(self, intent: str) -> None:
        with self._lock:
            self._intents[intent] += 1

    def record_llm_call(self, provider: str) -> None:
        with self._lock:
            self._llm_calls[provider] += 1

    # -- reporting ---------------------------------------------------------

    @property
    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._started_at, 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total_requests = sum(self._requests.values())
            total_errors = sum(self._errors.values())
            all_latencies = [v for samples in self._latency.values() for v in samples]

            return {
                "uptime_seconds": self.uptime_seconds,
                "started_at": self._started_wall.isoformat(),
                "totals": {
                    "requests": total_requests,
                    "errors": total_errors,
                    "error_rate": (
                        round(total_errors / total_requests, 4) if total_requests else 0.0
                    ),
                    "slow_requests": sum(self._slow_requests.values()),
                },
                "latency_ms": _percentiles(all_latencies),
                "status_codes": dict(self._status),
                "endpoints": {
                    endpoint: {
                        "requests": count,
                        "errors": self._errors.get(endpoint, 0),
                        "error_rate": round(self._errors.get(endpoint, 0) / count, 4),
                        "latency_ms": _percentiles(list(self._latency[endpoint])),
                    }
                    for endpoint, count in self._requests.most_common()
                },
                "tools": {
                    tool: {
                        "calls": count,
                        "failures": self._tool_failures.get(tool, 0),
                        "latency_ms": _percentiles(list(self._tool_latency[tool])),
                    }
                    for tool, count in self._tool_calls.most_common()
                },
                "intents": dict(self._intents.most_common()),
                "llm_calls": dict(self._llm_calls),
                "recent_errors": list(self._recent_errors),
            }

    def prometheus(self) -> str:
        """Expose the same numbers in Prometheus text format."""
        snap = self.snapshot()
        lines: list[str] = [
            "# HELP app_uptime_seconds Seconds since process start.",
            "# TYPE app_uptime_seconds gauge",
            f"app_uptime_seconds {snap['uptime_seconds']}",
            "# HELP http_requests_total Total HTTP requests by endpoint.",
            "# TYPE http_requests_total counter",
        ]
        for endpoint, data in snap["endpoints"].items():
            label = _escape(endpoint)
            lines.append(f'http_requests_total{{endpoint="{label}"}} {data["requests"]}')
        lines += [
            "# HELP http_request_errors_total Total errored HTTP requests.",
            "# TYPE http_request_errors_total counter",
        ]
        for endpoint, data in snap["endpoints"].items():
            lines.append(
                f'http_request_errors_total{{endpoint="{_escape(endpoint)}"}} {data["errors"]}'
            )
        lines += [
            "# HELP http_request_duration_ms Request latency percentiles.",
            "# TYPE http_request_duration_ms summary",
        ]
        for quantile, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
            lines.append(
                f'http_request_duration_ms{{quantile="{quantile}"}} '
                f'{snap["latency_ms"].get(key, 0)}'
            )
        lines += [
            "# HELP agent_tool_calls_total Tool invocations by tool.",
            "# TYPE agent_tool_calls_total counter",
        ]
        for tool, data in snap["tools"].items():
            lines.append(f'agent_tool_calls_total{{tool="{_escape(tool)}"}} {data["calls"]}')
        lines += [
            "# HELP agent_intents_total Detected intents.",
            "# TYPE agent_intents_total counter",
        ]
        for intent, count in snap["intents"].items():
            lines.append(f'agent_intents_total{{intent="{_escape(intent)}"}} {count}')
        return "\n".join(lines) + "\n"


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    size = len(ordered)

    def at(fraction: float) -> float:
        index = min(size - 1, max(0, int(round(fraction * (size - 1)))))
        return round(ordered[index], 2)

    return {
        "count": size,
        "avg": round(sum(ordered) / size, 2),
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": round(ordered[-1], 2),
    }


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


metrics = MetricsRegistry()
