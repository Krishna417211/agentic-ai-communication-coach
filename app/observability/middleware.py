"""Request tracking middleware: trace ids, access logs, latency, error capture."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.observability.logging_config import session_id_var, trace_id_var
from app.observability.metrics import metrics

logger = logging.getLogger("app.request")

TRACE_HEADER = "X-Trace-Id"


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, slow_request_ms: float = 3000.0) -> None:
        super().__init__(app)
        self._slow_ms = slow_request_ms

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get(TRACE_HEADER) or uuid.uuid4().hex[:12]
        trace_token = trace_id_var.set(trace_id)
        session_token = session_id_var.set(request.headers.get("X-Session-Id", "-"))
        request.state.trace_id = trace_id

        # Route template, not the raw path, so ids don't explode cardinality.
        endpoint = _endpoint_label(request)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - convert to a traced 500
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "unhandled error",
                extra={
                    "endpoint": endpoint,
                    "method": request.method,
                    "duration_ms": round(duration_ms, 2),
                    "error_type": type(exc).__name__,
                },
            )
            metrics.record_error(endpoint, type(exc).__name__, str(exc), trace_id)
            metrics.record_request(endpoint, 500, duration_ms)
            trace_id_var.reset(trace_token)
            session_id_var.reset(session_token)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "detail": "The request could not be completed.",
                    "trace_id": trace_id,
                },
                headers={TRACE_HEADER: trace_id},
            )

        duration_ms = (time.perf_counter() - started) * 1000
        slow = duration_ms > self._slow_ms

        # Route template is only resolvable after routing has happened.
        endpoint = _endpoint_label(request)
        metrics.record_request(endpoint, response.status_code, duration_ms, slow=slow)
        if response.status_code >= 400:
            metrics.record_error(
                endpoint, f"http_{response.status_code}", "", trace_id
            )

        log = logger.warning if slow or response.status_code >= 500 else logger.info
        log(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={
                "endpoint": endpoint,
                "method": request.method,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "slow": slow,
                "client": request.client.host if request.client else None,
            },
        )

        response.headers[TRACE_HEADER] = trace_id
        response.headers["X-Response-Time-ms"] = f"{duration_ms:.1f}"
        trace_id_var.reset(trace_token)
        session_id_var.reset(session_token)
        return response


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None) or request.url.path
    return f"{request.method} {path}"
