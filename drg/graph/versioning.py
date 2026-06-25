"""Snapshot-based graph versioning for persisted EnhancedKG JSON files."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .diff import SnapshotDiff, diff_graph_data
from .kg_core import EnhancedKG

__all__ = [
    "GraphVersion",
    "VersionManifest",
    "create_snapshot",
    "diff_versions",
    "list_versions",
    "rollback_to_version",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _versions_dir(graph_path: str | Path, versions_dir: str | Path | None = None) -> Path:
    if versions_dir is not None:
        return Path(versions_dir)
    path = Path(graph_path)
    return path.parent / f".{path.stem}_versions"


def _manifest_path(graph_path: str | Path, versions_dir: str | Path | None = None) -> Path:
    return _versions_dir(graph_path, versions_dir) / "manifest.json"


@dataclass(frozen=True)
class GraphVersion:
    """A single immutable graph snapshot entry."""

    version_id: str
    parent_version_id: str | None
    created_at: str
    operation: str
    snapshot_path: str
    document_id: str | None = None
    diff_summary: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "created_at": self.created_at,
            "operation": self.operation,
            "snapshot_path": self.snapshot_path,
            "diff_summary": dict(self.diff_summary),
        }
        if self.document_id:
            out["document_id"] = self.document_id
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphVersion":
        return cls(
            version_id=str(data["version_id"]),
            parent_version_id=(
                str(data["parent_version_id"]) if data.get("parent_version_id") else None
            ),
            created_at=str(data["created_at"]),
            operation=str(data.get("operation", "snapshot")),
            snapshot_path=str(data["snapshot_path"]),
            document_id=str(data["document_id"]) if data.get("document_id") else None,
            diff_summary=dict(data.get("diff_summary", {}) or {}),
        )


@dataclass
class VersionManifest:
    """JSON-serialisable manifest of snapshots for one graph file."""

    graph_path: str
    versions: list[GraphVersion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_path": self.graph_path,
            "versions": [v.to_dict() for v in self.versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VersionManifest":
        return cls(
            graph_path=str(data.get("graph_path", "")),
            versions=[GraphVersion.from_dict(v) for v in data.get("versions", [])],
        )

    @classmethod
    def load(cls, graph_path: str | Path, versions_dir: str | Path | None = None) -> "VersionManifest":
        path = _manifest_path(graph_path, versions_dir)
        if not path.exists():
            return cls(graph_path=str(graph_path), versions=[])
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, graph_path: str | Path, versions_dir: str | Path | None = None) -> None:
        path = _manifest_path(graph_path, versions_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, version_id: str) -> GraphVersion:
        for version in self.versions:
            if version.version_id == version_id:
                return version
        raise KeyError(version_id)


def _next_version_id(kg: EnhancedKG, manifest: VersionManifest) -> str:
    base = kg.metadata.get("version")
    if base is not None:
        candidate = f"v{base}"
    else:
        candidate = f"v{len(manifest.versions) + 1}"
    existing = {v.version_id for v in manifest.versions}
    if candidate not in existing:
        return candidate
    idx = 2
    while f"{candidate}-{idx}" in existing:
        idx += 1
    return f"{candidate}-{idx}"


def create_snapshot(
    kg: EnhancedKG,
    graph_path: str | Path,
    *,
    operation: str = "snapshot",
    document_id: str | None = None,
    diff_summary: dict[str, int] | None = None,
    versions_dir: str | Path | None = None,
) -> GraphVersion:
    """Persist ``kg`` as an immutable snapshot and append the manifest."""

    graph_path = Path(graph_path)
    manifest = VersionManifest.load(graph_path, versions_dir)
    version_id = _next_version_id(kg, manifest)
    parent_version_id = manifest.versions[-1].version_id if manifest.versions else None
    created_at = _utc_now_iso()

    snapshots_dir = _versions_dir(graph_path, versions_dir) / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshots_dir / f"{version_id}.json"

    version = GraphVersion(
        version_id=version_id,
        parent_version_id=parent_version_id,
        created_at=created_at,
        operation=operation,
        snapshot_path=str(snapshot_path),
        document_id=document_id,
        diff_summary=dict(diff_summary or {}),
    )
    manifest.versions.append(version)
    manifest.graph_path = str(graph_path)
    manifest.save(graph_path, versions_dir)

    kg.metadata.setdefault("versions", [])
    if version.to_dict() not in kg.metadata["versions"]:
        kg.metadata["versions"].append(version.to_dict())
    snapshot_path.write_text(kg.to_json(), encoding="utf-8")
    return version


def list_versions(
    graph_path: str | Path,
    *,
    versions_dir: str | Path | None = None,
) -> list[GraphVersion]:
    """Return manifest versions in creation order."""

    return VersionManifest.load(graph_path, versions_dir).versions


def _load_snapshot(version: GraphVersion) -> dict[str, Any]:
    return json.loads(Path(version.snapshot_path).read_text(encoding="utf-8"))


def diff_versions(
    graph_path: str | Path,
    old_version_id: str,
    new_version_id: str,
    *,
    versions_dir: str | Path | None = None,
) -> SnapshotDiff:
    """Diff two snapshot versions by id."""

    manifest = VersionManifest.load(graph_path, versions_dir)
    old = manifest.get(old_version_id)
    new = manifest.get(new_version_id)
    return diff_graph_data(_load_snapshot(old), _load_snapshot(new))


def rollback_to_version(
    graph_path: str | Path,
    version_id: str,
    *,
    versions_dir: str | Path | None = None,
) -> GraphVersion:
    """Replace ``graph_path`` with the selected snapshot and record rollback.

    The pre-rollback state is saved to its own snapshot file so the version
    history remains linear and auditable.  The rollback entry's
    ``snapshot_path`` points to this pre-rollback snapshot (not to the
    restored target), preserving all states in the version chain.
    """

    graph_path = Path(graph_path)
    manifest = VersionManifest.load(graph_path, versions_dir)
    target = manifest.get(version_id)
    vdir = _versions_dir(graph_path, versions_dir)
    vdir.mkdir(parents=True, exist_ok=True)

    # Capture the current (pre-rollback) state before overwriting graph_path.
    rollback_seq = len(manifest.versions) + 1
    pre_rollback_id = f"pre-rollback-{version_id}-{rollback_seq}"
    pre_rollback_path = vdir / f"{pre_rollback_id}.json"
    if graph_path.exists():
        shutil.copyfile(graph_path, pre_rollback_path)
    else:
        # Nothing to preserve — point to target as fallback
        pre_rollback_path = Path(target.snapshot_path)

    # Restore the target snapshot.
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target.snapshot_path, graph_path)

    rollback = GraphVersion(
        version_id=f"rollback-{version_id}-{rollback_seq}",
        parent_version_id=manifest.versions[-1].version_id if manifest.versions else None,
        created_at=_utc_now_iso(),
        operation="rollback",
        snapshot_path=str(pre_rollback_path),
        document_id=target.document_id,
        diff_summary={},
    )
    manifest.versions.append(rollback)
    manifest.save(graph_path, versions_dir)
    return rollback
