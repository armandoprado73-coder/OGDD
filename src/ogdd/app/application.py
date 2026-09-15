"""Application entry point for the standalone OGDD interface."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import QTimer
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


def main(argv: Sequence[str] | None = None) -> int:
    """Create and run the OGDD desktop application."""

    arguments = list(sys.argv if argv is None else argv)
    check_startup = "--check-startup" in arguments
    if check_startup:
        arguments.remove("--check-startup")
    application = QApplication(arguments)
    application.setApplicationName("OGDD")
    application.setOrganizationName("OGDD")
    application.setApplicationDisplayName(
        "OGDD — Open Geometry for Digital Dentistry"
    )
    application.setStyle("Fusion")
    application.setStyleSheet(APPLICATION_STYLESHEET)

    window = MainWindow()
    window.show()

    # Packaging jobs use this private flag to prove that the frozen
    # application can construct its complete Qt/VTK interface.  The timer
    # leaves the normal interactive startup path unchanged.
    if check_startup:
        QTimer.singleShot(STARTUP_CHECK_DELAY_MILLISECONDS, window.close)

    return application.exec()
