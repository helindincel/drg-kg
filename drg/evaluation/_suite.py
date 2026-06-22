"""Official benchmark suite helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._reporting import load_benchmark_dataset
from ._types import BenchmarkDataset

__all__ = [
    "BenchmarkSuite",
    "default_benchmark_suite_path",
    "load_benchmark_suite",
    "load_official_benchmark_suite",
]


@dataclass(frozen=True)
class BenchmarkSuite:
    """A named collection of benchmark datasets plus adapter notes."""

    name: str
    datasets: tuple[BenchmarkDataset, ...]
    adapters: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "adapters": list(self.adapters),
            "metadata": dict(self.metadata),
        }


def default_benchmark_suite_path() -> Path:
    """Return the source-checkout official suite manifest path."""
    return Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / "official_suite.json"


def load_official_benchmark_suite() -> BenchmarkSuite:
    """Load DRG's bundled source-checkout benchmark suite."""
    return load_benchmark_suite(default_benchmark_suite_path())


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    """Load a benchmark suite manifest.

    Manifest datasets can be inline dataset objects or JSON file paths relative
    to the manifest directory.
    """
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    datasets: list[BenchmarkDataset] = []

    for item in data.get("datasets", []) or []:
        if isinstance(item, str):
            datasets.append(load_benchmark_dataset(base_dir / item))
        elif isinstance(item, dict) and "path" in item:
            datasets.append(load_benchmark_dataset(base_dir / str(item["path"])))
        elif isinstance(item, dict):
            datasets.append(BenchmarkDataset.from_dict(item))
        else:
            raise ValueError(f"Invalid suite dataset entry: {item!r}")

    return BenchmarkSuite(
        name=str(data.get("name") or manifest_path.stem),
        datasets=tuple(datasets),
        adapters=tuple(str(x) for x in data.get("adapters", []) or ()),
        metadata=dict(data.get("metadata") or {}),
    )
