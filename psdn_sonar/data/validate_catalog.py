"""Offline command-line entry point for benchmark catalog validation."""

from __future__ import annotations

from .catalog import main

if __name__ == "__main__":  # pragma: no cover - exercised as a module
    raise SystemExit(main())
