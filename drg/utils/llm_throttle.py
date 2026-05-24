"""LLM request throttling utilities.

This module provides an optional, process-local throttle to reduce the likelihood of
HTTP 429 (rate limit) errors when using providers with strict RPM quotas.

Enable by setting `DRG_LLM_MIN_INTERVAL_SECONDS` to a positive number (e.g. 6.5).
"""

from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_last_call_monotonic: float | None = None


def throttle_llm_calls() -> None:
    """Sleep if needed to respect a minimum interval between LLM calls.

    Reads:
      - DRG_LLM_MIN_INTERVAL_SECONDS: float, default 0 (disabled)
      - DRG_LLM_JITTER_SECONDS: float, default 0 (adds +/- jitter to spread bursts)
    """

    try:
        min_interval = float(os.getenv("DRG_LLM_MIN_INTERVAL_SECONDS", "0") or "0")
    except ValueError:
        min_interval = 0.0

    if min_interval <= 0:
        return

    try:
        jitter = float(os.getenv("DRG_LLM_JITTER_SECONDS", "0") or "0")
    except ValueError:
        jitter = 0.0

    # Deterministic "jitter" without random: use fractional part of current time.
    # This avoids importing random and keeps tests deterministic.
    if jitter > 0:
        frac = time.time() % 1.0  # 0..1
        # Map to [-0.5, +0.5]
        jitter_offset = (frac - 0.5) * 2.0 * jitter
    else:
        jitter_offset = 0.0

    target_interval = max(0.0, min_interval + jitter_offset)

    global _last_call_monotonic
    with _lock:
        now = time.monotonic()
        if _last_call_monotonic is None:
            _last_call_monotonic = now
            return

        elapsed = now - _last_call_monotonic
        sleep_s = target_interval - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)
            _last_call_monotonic = time.monotonic()
        else:
            _last_call_monotonic = now
