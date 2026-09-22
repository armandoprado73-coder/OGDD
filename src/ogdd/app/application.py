"""Application entry point for the standalone OGDD interface."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


APPLICATION_STYLESHEET = """
QMainWindow {
    background: #eef2f4;
}
QMenuBar {
    background: #ffffff;
    border-bottom: 1px solid #d6dee3;
    padding: 3px;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: #dceff0;
    color: #153d40;
}
QMenu {
    background: #ffffff;
    border: 1px solid #cbd5da;
}
QDockWidget {
    color: #20323a;
    font-weight: 600;
}
QDockWidget::title {
    background: #e2eaed;
    border-bottom: 1px solid #c7d2d7;
    padding: 8px;
}
QTreeWidget, QListWidget {
    background: #ffffff;
    border: 0;
    color: #263940;
    outline: 0;
}
QTreeWidget::item, QListWidget::item {
    padding: 7px 5px;
}
QListWidget::item:selected {
    background: #2a777c;
    color: #ffffff;
}
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #d6dee3;
    color: #40545c;
}
"""

STARTUP_CHECK_DELAY_MILLISECONDS = 750
HEADLESS_STARTUP_CHECK_ENVIRONMENT_VARIABLE = (
    "OGDD_HEADLESS_STARTUP_CHECK"
)
APPLICATION_ICON = Path(__file__).with_name("assets") / "ogdd-logo.png"


def headless_startup_check_enabled(
    *,
    check_startup: bool,
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether CI should construct the UI without showing OpenGL."""

    if not check_startup:
        return False
    environment = os.environ if environment is None else environment
    value = environment.get(
        HEADLESS_STARTUP_CHECK_ENVIRONMENT_VARIABLE,
        "",
    )
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main(argv: Sequence[str] | None = None) -> int:
    """Create and run the OGDD desktop application."""

    arguments = list(sys.argv if argv is None else argv)
    check_startup = "--check-startup" in arguments
    if check_startup:
        arguments.remove("--check-startup")
    headless_startup_check = headless_startup_check_enabled(
        check_startup=check_startup
    )
    application = QApplication(arguments)
    application.setApplicationName("OGDD")
    application.setOrganizationName("OGDD")
    application.setApplicationDisplayName(
        "OGDD — Open Geometry for Digital Dentistry"
    )
    application.setWindowIcon(QIcon(str(APPLICATION_ICON)))
    application.setStyle("Fusion")
    application.setStyleSheet(APPLICATION_STYLESHEET)

    window = MainWindow()
    if not headless_startup_check:
        window.show()

    # Packaging jobs use this private flag to prove that the frozen
    # application can construct its complete Qt/VTK interface.  The timer
    # leaves the normal interactive startup path unchanged.
    if check_startup:
        callback = (
            application.quit
            if headless_startup_check
            else window.close
        )
        QTimer.singleShot(STARTUP_CHECK_DELAY_MILLISECONDS, callback)

    return application.exec()
