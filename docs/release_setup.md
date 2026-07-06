# Release Setup — GitHub Secrets and Environments

This guide configures the credentials and review gates required by
[`.github/workflows/release.yml`](../.github/workflows/release.yml).

## 1. Create PyPI API Tokens

### TestPyPI

1. Sign in at [test.pypi.org](https://test.pypi.org/).
2. Open **Account settings → API tokens → Add API token**.
3. Scope: **Entire account** (first upload) or **Project: `drg-kg`** (after the
   project exists).
4. Copy the token (`pypi-…`).

### PyPI (production)

1. Sign in at [pypi.org](https://pypi.org/).
2. Register the project name **`drg-kg`** if it is not claimed yet.
3. Create a **project-scoped** API token for `drg-kg`.
4. Copy the token.

> **Note:** PyPI already has an unrelated package named `drg` (Medicare DRG
> grouper). This project publishes as **`drg-kg`**; users install with
> `pip install drg-kg` and import with `import drg`.

## 2. Add GitHub Secrets

In the repository: **Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|-------------|-------|
| `TEST_PYPI_API_TOKEN` | TestPyPI token from step 1 |
| `PYPI_API_TOKEN` | Production PyPI token from step 1 |

Optional (Codecov upload from CI):

| Secret name | Value |
|-------------|-------|
| `CODECOV_TOKEN` | Codecov upload token |

## 3. Configure GitHub Environments

**Settings → Environments**

### `testpypi`

- No required reviewers (optional: add yourself for audit trail).
- Environment secret override: none needed if repo-level secrets are set.

### `pypi`

- **Required reviewers:** at least one maintainer (recommended before the first
  public release).
- Deployment branches: restrict to tags matching `v*` if desired.

The release workflow references these environment names exactly:

```yaml
environment:
  name: testpypi   # publish-testpypi job
environment:
  name: pypi       # publish-pypi job
```

## 4. Verify Locally Before Tagging

```bash
python -m pip install --upgrade build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
pytest -m "not integration" --cov=drg
```

See also [`docs/launch_checklist.md`](launch_checklist.md).

## 5. Release Tags

| Tag | Upload target |
|-----|---------------|
| `v0.1.1rc1` | TestPyPI only |
| `v0.1.1` | TestPyPI, then PyPI |

```bash
git tag v0.1.1rc1
git push origin v0.1.1rc1
# After smoke test passes:
git tag v0.1.1
git push origin v0.1.1
```

Manual dry-run (no tag):

1. **Actions → Release → Run workflow**
2. Set **target** to `testpypi` or `pypi` (PyPI manual dispatch requires a final
   `vX.Y.Z` ref).

## 6. Post-Release Checklist

- [ ] GitHub Release with notes from [`CHANGELOG.md`](../CHANGELOG.md)
- [ ] Smoke: `pip install drg-kg` from PyPI
- [ ] Smoke: `pip install "drg-kg[extract]"` (DSPy + tiktoken)
- [ ] Update README PyPI badges if not already present

## 7. Workflow stuck on "Waiting"?

If the Release workflow shows **Waiting** after you push a tag, GitHub
Environments (`testpypi`, `pypi`) likely require manual approval:

1. Open **Actions → Release →** the waiting run.
2. Approve the **testpypi** deployment (for `v0.1.1rc1` or any tag).
3. After TestPyPI smoke passes, approve **pypi** (for final `v0.1.1` only).

Without `TEST_PYPI_API_TOKEN` / `PYPI_API_TOKEN` secrets, the publish jobs
fail at upload time — configure secrets in section 2 first.
