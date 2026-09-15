"""Allow OGDD to start with ``python -m ogdd.app``."""

from __future__ import annotations

import faulthandler
from pathlib import Path
import sys
import traceback


STARTUP_CHECK_ARGUMENT = "--check-startup"
STARTUP_ERROR_LOG = Path.cwd() / "ogdd-startup-error.log"


def run() -> int:
    """Run OGDD and preserve diagnostics for frozen startup checks."""

    check_startup = STARTUP_CHECK_ARGUMENT in sys.argv
    diagnostic_stream = None
    if check_startup:
        diagnostic_stream = STARTUP_ERROR_LOG.open("w", encoding="utf-8")
        faulthandler.enable(file=diagnostic_stream)
        diagnostic_stream.write("OGDD frozen startup check began.\n")
        diagnostic_stream.flush()

    try:
        from ogdd.app.application import main

        exit_code = main()
        if check_startup and exit_code != 0:
            diagnostic_stream.write(
                f"Qt event loop returned exit code {exit_code}.\n"
            )
            diagnostic_stream.flush()
    except Exception:
        if not check_startup:
            raise
        traceback.print_exc(file=diagnostic_stream)
        diagnostic_stream.flush()
        exit_code = 1
    finally:
        if diagnostic_stream is not None:
            faulthandler.disable()
            diagnostic_stream.close()

    if check_startup and exit_code == 0:
        STARTUP_ERROR_LOG.unlink(missing_ok=True)
    return exit_code


raise SystemExit(run())
