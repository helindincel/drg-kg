"""Environment loader utilities.

This module provides a tiny, dependency-free `.env` loader so users can keep API keys
out of code while still using convenient local configuration.

Security:
  - `.env` should remain uncommitted (already ignored via `.gitignore`).
  - This loader does NOT print values.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> bool:
    """Load environment variables from a .env file.

    Supports simple KEY=VALUE lines with optional surrounding quotes.
    Ignores blank lines and comments starting with '#'.

    Args:
        path: Path to .env file (default: ".env" in current working directory).
        override: If True, overwrite existing os.environ keys.

    Returns:
        True if a file was found and parsed; False if file does not exist.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False

    for raw_line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        # Strip surrounding quotes if present.
        if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
            value = value[1:-1]

        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return True
