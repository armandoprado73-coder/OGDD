"""Standalone desktop application for OGDD."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Load the optional GUI dependencies only when the app starts."""

    from .application import main as run_application

    return run_application(argv)


__all__ = ["main"]
