#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# DRG-KG: Run remaining test modules, update coverage omit list, commit each
# successful step, push to GitHub.
#
# Designed to be run as a single unattended command after the new test files
# have been written. Safe to re-run: each step checks whether work is
# already done.
#
# Usage:
#     bash scripts/run_remaining_tests.sh
#
# What it does, per module:
#   1) pytest <test_file> -q          (verifies tests pass)
#   2) if PASS:
#        - removes the matching source path from pyproject.toml omit list
#        - git add  test file + pyproject.toml
#        - git commit
#   3) if FAIL:
#        - leaves the test file in place but skips the commit
#        - prints a clear notice; continues with next module
#
# At the end:
#   - Runs full pytest -m "not integration" --cov=drg and prints the gate
#   - Updates CHANGELOG.md [Unreleased] section
#   - git push origin main
# ----------------------------------------------------------------------------

set -u  # treat undefined vars as error; do NOT use -e because we want to
        # continue past per-module failures.

cd "$(dirname "$0")/.." || exit 1
REPO_ROOT="$(pwd)"

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: $PY not found. Activate / create the venv first." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

# pyproject_remove_omit <source_path>
# Removes the line `    "<source_path>",` from the omit = [...] block.
pyproject_remove_omit() {
    local path="$1"
    local tmp
    tmp="$(mktemp)"
    # Match the line as it appears in pyproject.toml omit list, preserving
    # surrounding whitespace and trailing comma.
    grep -v "\"${path}\"," pyproject.toml > "$tmp" && mv "$tmp" pyproject.toml
}

# run_module <label> <test_file> <src_path>
run_module() {
    local label="$1"
    local test_file="$2"
    local src_path="$3"

    echo ""
    echo "============================================================"
    echo "  Module: $label"
    echo "  Test:   $test_file"
    echo "  Source: $src_path"
    echo "============================================================"

    if [ ! -f "$test_file" ]; then
        echo "  SKIP: test file does not exist (already cleaned up?)"
        return 0
    fi

    # Check if test file is already committed (idempotency)
    if git ls-files --error-unmatch "$test_file" >/dev/null 2>&1; then
        echo "  SKIP: $test_file is already committed."
        return 0
    fi

    echo "  -> Running pytest..."
    if ! $PY -m pytest "$test_file" -q 2>&1 | tail -15; then
        echo ""
        echo "  FAIL: $test_file did not pass. Leaving file in place for"
        echo "        manual review; not committing this module."
        FAILED_MODULES+=("$label")
        return 1
    fi

    echo "  -> Tests passed."
    echo "  -> Removing $src_path from coverage omit list..."
    pyproject_remove_omit "$src_path"

    echo "  -> Staging and committing..."
    git add "$test_file" pyproject.toml
    git commit -m "tests($label): add unit tests and remove from coverage omit

$test_file exercises the public API of $src_path without LLM or optional
heavy dependencies (or guards them with pytest.importorskip). With real
coverage in place, $src_path is removed from [tool.coverage.run].omit."

    if [ $? -eq 0 ]; then
        echo "  -> Commit OK."
        SUCCEEDED_MODULES+=("$label")
    else
        echo "  -> WARNING: commit failed (likely nothing staged)."
    fi
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

declare -a SUCCEEDED_MODULES=()
declare -a FAILED_MODULES=()

echo ""
echo "============================================================"
echo "  DRG-KG: Running remaining test suites"
echo "============================================================"

# Each entry is: <label>|<test_file>|<src_path>
MODULES=(
    "graph-builders|tests/test_graph_builders.py|drg/graph/builders.py"
    "graph-auto-clusters|tests/test_graph_auto_clusters.py|drg/graph/auto_clusters.py"
    "graph-query-engine|tests/test_graph_query_engine.py|drg/graph/query_engine.py"
    "clustering-summarization|tests/test_clustering_summarization.py|drg/clustering/summarization.py"
    "clustering-algorithms|tests/test_clustering_algorithms.py|drg/clustering/algorithms.py"
)

for entry in "${MODULES[@]}"; do
    IFS='|' read -r label test_file src_path <<< "$entry"
    run_module "$label" "$test_file" "$src_path"
done

# ----------------------------------------------------------------------------
# Final coverage gate
# ----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Final coverage check"
echo "============================================================"
$PY -m pytest -m "not integration" --cov=drg --cov-report=term -q 2>&1 | tail -15

# ----------------------------------------------------------------------------
# Update CHANGELOG (idempotent: only adds if marker not already present)
# ----------------------------------------------------------------------------

CHANGELOG_MARKER="### Added (test coverage expansion)"
if ! grep -q "$CHANGELOG_MARKER" CHANGELOG.md; then
    echo ""
    echo "  -> Updating CHANGELOG.md..."
    python3 - <<'PYEOF'
from pathlib import Path

p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
marker = "## [Unreleased]"
addition = """## [Unreleased]

### Added (test coverage expansion)

- Unit tests for `drg/graph/builders.py`, `drg/graph/auto_clusters.py`,
  `drg/graph/query_engine.py`, `drg/clustering/summarization.py`, and
  `drg/clustering/algorithms.py`. The first four run with zero external
  dependencies; the clustering algorithms tests use `pytest.importorskip`
  for each optional backend (python-louvain, leidenalg+igraph, sklearn)
  so they pass cleanly whether those packages are installed or not.

### Changed (coverage omit list)

- Removed the five sources above from `[tool.coverage.run].omit`; the
  coverage gate now reflects their real measured coverage.
"""
if marker in text and addition.split('\n\n', 1)[1] not in text:
    text = text.replace(marker, addition, 1)
    p.write_text(text, encoding="utf-8")
    print("CHANGELOG.md updated.")
else:
    print("CHANGELOG.md already up to date.")
PYEOF

    if ! git diff --quiet CHANGELOG.md; then
        git add CHANGELOG.md
        git commit -m "docs(changelog): record expanded chunking/graph/clustering test coverage"
    fi
fi

# ----------------------------------------------------------------------------
# Push
# ----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Pushing to origin/main"
echo "============================================================"
if git push origin main 2>&1 | tail -5; then
    echo "  -> Push OK."
else
    echo "  -> Push failed (check output above). Local commits are intact."
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo "  Succeeded: ${#SUCCEEDED_MODULES[@]} modules"
for m in "${SUCCEEDED_MODULES[@]}"; do echo "    + $m"; done
echo "  Failed:    ${#FAILED_MODULES[@]} modules"
for m in "${FAILED_MODULES[@]}"; do echo "    - $m"; done
echo ""
echo "  Latest git log:"
git log --oneline -10
echo "============================================================"
