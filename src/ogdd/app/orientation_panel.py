"""Controls for operator-guided anatomical orientation."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OrientationPanel(QWidget):
    """Guide selection of the three mandibular landmarks."""

    pick_requested = Signal(str)
    confirm_requested = Signal()
    reset_requested = Signal()

    LANDMARKS = (
        (
            "DENTAL_MIDLINE",
            "1  Línea media dental",
            "Punto anterior sobre la línea media mandibular",
        ),
        (
            "RIGHT_SECOND_MOLAR",
            "2  Segundo molar derecho",
            "Referencia oclusal posterior derecha del paciente",
        ),
        (
            "LEFT_SECOND_MOLAR",
            "3  Segundo molar izquierdo",
            "Referencia oclusal posterior izquierda del paciente",
        ),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        title = QLabel("Orientación anatómica")
        title.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #24484b;"
        )

        instructions = QLabel(
            "Seleccione los tres puntos sobre la mandíbula. "
            "Después de pulsar un botón, haga clic izquierdo "
            "sobre la superficie del modelo."
        )
        instructions.setWordWrap(True)

        self.availability_label = QLabel(
            "Primero cargue los modelos en RC."
        )
        self.availability_label.setWordWrap(True)
        self.availability_label.setStyleSheet(
            "background: #f0e6c9; color: #604e20; "
            "padding: 8px; border-radius: 3px;"
        )

        self._pick_buttons: dict[str, QPushButton] = {}
        self._coordinate_labels: dict[str, QLabel] = {}
        landmark_layout = QGridLayout()
        landmark_layout.setColumnStretch(0, 1)

        for row, (key, label, tooltip) in enumerate(self.LANDMARKS):
            button = QPushButton(label)
            button.setToolTip(tooltip)
            button.setEnabled(False)
            button.clicked.connect(
                lambda checked=False, landmark_key=key: (
                    self.pick_requested.emit(landmark_key)
                )
            )
            coordinate = QLabel("Pendiente")
            coordinate.setStyleSheet("color: #74858c; padding-left: 7px;")

            self._pick_buttons[key] = button
            self._coordinate_labels[key] = coordinate
            landmark_layout.addWidget(button, row * 2, 0)
            landmark_layout.addWidget(coordinate, row * 2 + 1, 0)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)

        self.confirm_button = QPushButton("Confirmar orientación")
        self.confirm_button.setEnabled(False)
        self.confirm_button.setStyleSheet(
            "QPushButton { background: #2a777c; color: white; "
            "font-weight: 600; padding: 8px; border-radius: 3px; }"
            "QPushButton:disabled { background: #aebabe; color: #edf1f2; }"
        )
        self.confirm_button.clicked.connect(
            lambda checked=False: self.confirm_requested.emit()
        )

        self.reset_button = QPushButton("Restablecer puntos")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(
            lambda checked=False: self.reset_requested.emit()
        )

        self.result_label = QLabel(
            "Los ejes aparecerán aquí después de confirmar."
        )
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "background: #e8eef0; color: #40545c; "
            "padding: 8px; border-radius: 3px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.availability_label)
        layout.addLayout(landmark_layout)
        layout.addWidget(separator)
        layout.addWidget(self.confirm_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.result_label)
        layout.addStretch()

    def set_study_available(self, available: bool) -> None:
        """Enable landmark selection only when a mandible is loaded."""

        self.availability_label.setVisible(not available)
        for button in self._pick_buttons.values():
            button.setEnabled(available)
        if not available:
            self.clear()

    def set_active_landmark(self, landmark_name: str) -> None:
        """Highlight the button whose point is currently requested."""

        for key, button in self._pick_buttons.items():
            if key == landmark_name:
                button.setStyleSheet(
                    "background: #dceff0; color: #153d40; "
                    "font-weight: 600; padding: 6px;"
                )
            else:
                button.setStyleSheet("")

    def set_landmark(
        self,
        landmark_name: str,
        point: Iterable[float],
    ) -> None:
        """Show one selected point and update action availability."""

        x, y, z = point
        self._coordinate_labels[landmark_name].setText(
            f"✓  X {x:.3f}   Y {y:.3f}   Z {z:.3f} mm"
        )
        self._coordinate_labels[landmark_name].setStyleSheet(
            "color: #2a777c; padding-left: 7px; font-weight: 600;"
        )
        self._pick_buttons[landmark_name].setStyleSheet("")
        self._update_buttons()

    def clear(self) -> None:
        """Return every landmark control to its initial state."""

        for key, label, _tooltip in self.LANDMARKS:
            self._pick_buttons[key].setText(label)
            self._pick_buttons[key].setStyleSheet("")
            self._coordinate_labels[key].setText("Pendiente")
            self._coordinate_labels[key].setStyleSheet(
                "color: #74858c; padding-left: 7px;"
            )
        self.confirm_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.result_label.setText(
            "Los ejes aparecerán aquí después de confirmar."
        )
        self.result_label.setStyleSheet(
            "background: #e8eef0; color: #40545c; "
            "padding: 8px; border-radius: 3px;"
        )

    def show_coordinate_system(self, coordinate_system) -> None:
        """Present the resulting anatomical axes to the operator."""

        def vector_text(vector) -> str:
            return "  ".join(f"{value:+.4f}" for value in vector)

        self.result_label.setText(
            "Orientación confirmada\n"
            f"+X  {vector_text(coordinate_system.x_axis)}\n"
            f"+Y  {vector_text(coordinate_system.y_axis)}\n"
            f"+Z  {vector_text(coordinate_system.z_axis)}"
        )
        self.result_label.setStyleSheet(
            "background: #dceff0; color: #153d40; "
            "padding: 8px; border-radius: 3px; font-weight: 600;"
        )

    def _update_buttons(self) -> None:
        selected = sum(
            label.text().startswith("✓")
            for label in self._coordinate_labels.values()
        )
        self.confirm_button.setEnabled(
            selected == len(self.LANDMARKS)
        )
        self.reset_button.setEnabled(selected > 0)
