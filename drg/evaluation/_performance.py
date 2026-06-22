"""Runtime measurement helpers for benchmark runners."""

from __future__ import annotations

import os
import time
import tracemalloc
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

__all__ = ["measure_call", "summarize_performance"]


def measure_call(func: Callable[[], T]) -> tuple[T, dict[str, float]]:
    """Run ``func`` and return wall-clock and memory measurements.

    The helper intentionally uses only the standard library so performance
    reporting works in the minimal install. ``tracemalloc`` reports Python heap
    allocations; on Unix we also include peak RSS when ``resource`` is present.
    """

    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed_seconds = time.perf_counter() - start
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    metrics = {
        "wall_time_seconds": elapsed_seconds,
        "latency_p50_seconds": elapsed_seconds,
        "latency_p95_seconds": elapsed_seconds,
        "python_heap_current_mb": _bytes_to_mb(current_bytes),
        "python_heap_peak_mb": _bytes_to_mb(peak_bytes),
    }
    peak_rss_mb = _peak_rss_mb()
    if peak_rss_mb is not None:
        metrics["process_peak_rss_mb"] = peak_rss_mb
    return result, metrics


def summarize_performance(dataset_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-dataset runtime metadata into report-level metrics."""

    clean = [m for m in dataset_metrics if m]
    if not clean:
        return {}

    wall_times = [float(m.get("wall_time_seconds", 0.0)) for m in clean]
    heap_peaks = [float(m.get("python_heap_peak_mb", 0.0)) for m in clean]
    rss_peaks = [
        float(m["process_peak_rss_mb"]) for m in clean if m.get("process_peak_rss_mb") is not None
    ]

    summary = {
        "total_wall_time_seconds": sum(wall_times),
        "latency_p50_seconds": _percentile(wall_times, 0.50),
        "latency_p95_seconds": _percentile(wall_times, 0.95),
        "python_heap_peak_mb": max(heap_peaks) if heap_peaks else 0.0,
    }
    if rss_peaks:
        summary["process_peak_rss_mb"] = max(rss_peaks)
    return summary


def _bytes_to_mb(value: int) -> float:
    return value / (1024 * 1024)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _peak_rss_mb() -> float | None:
    if os.name == "nt":
        return None
    try:
        import resource
    except ImportError:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw = float(usage.ru_maxrss)
    if raw <= 0:
        return None
    # macOS reports bytes; Linux reports KiB.
    return raw / (1024 * 1024) if raw > 10_000_000 else raw / 1024
