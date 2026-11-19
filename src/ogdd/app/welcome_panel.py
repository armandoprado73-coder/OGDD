"""Lightweight welcome overlay using the existing OGDD artwork."""

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class WelcomePanel(QWidget):
    """Keep the logo centered as the scene changes size, without a timer."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setObjectName("ogddWelcomePanel")
        self.setStyleSheet(
            "#ogddWelcomePanel { background: qlineargradient("
            "x1: 0, y1: 0, x2: 0, y2: 1, "
            "stop: 0 #101a21, stop: 1 #263944); }"
            "QLabel { background: transparent; }"
        )
        self._logo_source = QPixmap(
            str(Path(__file__).with_name("assets") / "ogdd-logo.png")
        )
        self.logo = QLabel(self)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("Open Geometry for Digital Dentistry", self)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #dce9eb; font-size: 14px;")
        instruction = QLabel("Importe los modelos para comenzar", self)
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instruction.setStyleSheet("color: #9bb4bd; font-size: 12px;")
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addStretch()
        layout.addWidget(self.logo, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        layout.addWidget(instruction)
        layout.addStretch()
        parent.installEventFilter(self)
        self._fit_parent()

    def _fit_parent(self) -> None:
        parent = self.parentWidget()
        self.setGeometry(parent.rect())
        size = max(48, min(280, int(parent.width() * 0.5),
                           int(parent.height() * 0.45)))
        self.logo.setPixmap(self._logo_source.scaled(
            size, size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self._fit_parent()
        return super().eventFilter(watched, event)
