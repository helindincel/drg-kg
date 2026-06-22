"""Adapter artifact helpers for external benchmark competitors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._types import PipelinePrediction

__all__ = ["load_prediction_artifact"]


def load_prediction_artifact(
    path: str | Path,
) -> tuple[dict[str, PipelinePrediction], dict[str, Any]]:
    """Load predictions produced by DRG or an external competitor adapter.

    Supported JSON shapes:

    - ``{"dataset": "name", "prediction": {...}}``
    - ``{"predictions": {"dataset_name": {...}}, "adapter": "external-baseline"}``
    - ``{"predictions": [{"dataset": "name", ...}]}``
    """

    artifact_path = Path(path)
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = {
        "prediction_artifact": str(artifact_path),
    }
    if isinstance(data, dict):
        for key in ("adapter", "model", "version", "commit"):
            if key in data:
                metadata[key] = data[key]

    return _parse_predictions(data), metadata


def _parse_predictions(data: Any) -> dict[str, PipelinePrediction]:
    if isinstance(data, dict) and "dataset" in data and "prediction" in data:
        return {str(data["dataset"]): PipelinePrediction.from_dict(dict(data["prediction"]))}

    if isinstance(data, dict) and "predictions" in data:
        raw = data["predictions"]
        if isinstance(raw, dict):
            return {
                str(name): PipelinePrediction.from_dict(dict(value)) for name, value in raw.items()
            }
        if isinstance(raw, list):
            parsed: dict[str, PipelinePrediction] = {}
            for item in raw:
                if not isinstance(item, dict) or "dataset" not in item:
                    raise ValueError("Prediction list entries must contain a dataset field")
                payload = dict(item.get("prediction") or item)
                payload.pop("dataset", None)
                parsed[str(item["dataset"])] = PipelinePrediction.from_dict(payload)
            return parsed

    if isinstance(data, dict):
        return {
            str(name): PipelinePrediction.from_dict(dict(value)) for name, value in data.items()
        }

    raise ValueError("Unsupported prediction artifact format")
