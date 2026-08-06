# syntax=docker/dockerfile:1

# ---- builder: install dependencies into a self-contained venv --------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only requirements.txt is copied here, so the (slow) dependency layer is
# cached and reused whenever application code changes but deps do not.
# requirements.txt is runtime-only — pytest lives in requirements-dev.txt and
# never reaches the image.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---- runtime --------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser ui/ ./ui/
COPY --chown=appuser:appuser healthcheck.py ./

# Logs default to stdout in the image; this directory exists so that setting
# LOG_FILE (or bind-mounting ./logs via compose) works without a permission
# error at startup.
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# PORT is injected by Render/Railway. It is deliberately NOT set with ENV here:
# an ENV default would mask the platform's value in some buildpack paths, so
# the fallback lives in the CMD instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "healthcheck.py"]

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
