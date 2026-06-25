"""Tests for drg/utils/llm_throttle.py."""

from __future__ import annotations

import time


def test_throttle_noop_by_default(monkeypatch):
    """When DRG_LLM_MIN_INTERVAL_SECONDS is 0 (default), no sleep occurs."""
    monkeypatch.setenv("DRG_LLM_MIN_INTERVAL_SECONDS", "0")
    from drg.utils import llm_throttle

    # Reset module-level state
    llm_throttle._last_call_monotonic = None

    t0 = time.monotonic()
    llm_throttle.throttle_llm_calls()
    elapsed = time.monotonic() - t0

    # Should return instantly (< 50 ms)
    assert elapsed < 0.05


def test_throttle_noop_invalid_env(monkeypatch):
    """A non-numeric env var falls back to 0 (no throttle)."""
    monkeypatch.setenv("DRG_LLM_MIN_INTERVAL_SECONDS", "not_a_number")
    from drg.utils import llm_throttle

    llm_throttle._last_call_monotonic = None
    t0 = time.monotonic()
    llm_throttle.throttle_llm_calls()
    assert time.monotonic() - t0 < 0.05


def test_throttle_first_call_no_sleep(monkeypatch):
    """The very first call never sleeps regardless of interval setting."""
    monkeypatch.setenv("DRG_LLM_MIN_INTERVAL_SECONDS", "10")
    from drg.utils import llm_throttle

    llm_throttle._last_call_monotonic = None
    t0 = time.monotonic()
    llm_throttle.throttle_llm_calls()
    elapsed = time.monotonic() - t0

    # Must not sleep 10 s; should return in < 50 ms
    assert elapsed < 0.05


def test_throttle_second_call_sleeps(monkeypatch):
    """After the first call, a second call within the interval sleeps."""
    monkeypatch.setenv("DRG_LLM_MIN_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("DRG_LLM_JITTER_SECONDS", "0")
    from drg.utils import llm_throttle

    llm_throttle._last_call_monotonic = None

    llm_throttle.throttle_llm_calls()  # first call — no sleep, records timestamp
    t0 = time.monotonic()
    llm_throttle.throttle_llm_calls()  # second call — should sleep ~0.05 s
    elapsed = time.monotonic() - t0

    # Should have slept at least a small amount (allow 20 ms tolerance for CI timing)
    assert elapsed >= 0.02, f"Expected some sleep but got {elapsed:.3f}s"


def test_throttle_no_sleep_after_sufficient_wait(monkeypatch):
    """No sleep if the caller already waited longer than the interval."""
    monkeypatch.setenv("DRG_LLM_MIN_INTERVAL_SECONDS", "0.01")
    monkeypatch.setenv("DRG_LLM_JITTER_SECONDS", "0")
    from drg.utils import llm_throttle

    llm_throttle._last_call_monotonic = None
    llm_throttle.throttle_llm_calls()  # record first call

    time.sleep(0.05)  # wait longer than the interval

    t0 = time.monotonic()
    llm_throttle.throttle_llm_calls()
    elapsed = time.monotonic() - t0

    # Should return almost immediately (< 30 ms extra)
    assert elapsed < 0.03, f"Expected fast return but got {elapsed:.3f}s"
