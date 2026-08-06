"""Container healthcheck: exit 0 if /health responds 200.

A standalone script rather than an inline `python -c` in the Dockerfile — the
nested quoting an inline version needs is easy to break and hard to debug.
"""

from __future__ import annotations

import os
import sys
import urllib.request

PORT = os.getenv("PORT", "8000")
URL = f"http://127.0.0.1:{PORT}/health"

try:
    with urllib.request.urlopen(URL, timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception as exc:  # noqa: BLE001 - any failure is an unhealthy container
    print(f"healthcheck failed: {exc}", file=sys.stderr)
    sys.exit(1)
