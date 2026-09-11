"""Status panel for the functional calibration workflow step."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class FunctionalCalibrationPanel(QWidget):
    """Describe calibration readiness before movement controls are connected."""

    open_requested = Signal()
    close_requested = Signal()
    advance_requested = Signal()
    retreat_requested = Signal()
    rc_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        title = QLabel("Calibración funcional")
        title.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #24484b;"
        )

        instructions = QLabel(
            "La posición de relación céntrica será la referencia para "
            "calibrar apertura, protrusión, lateralidad y el cierre "
            "oclusal fino."
        )
        instructions.setWordWrap(True)

        self.availability_label = QLabel(
            "Construya primero el montaje virtual en RC."
        )
        self.availability_label.setWordWrap(True)
        self.availability_label.setStyleSheet(
            "background: #f0e6c9; color: #604e20; "
            "padding: 8px; border-radius: 3px;"
        )

        self.reference_value = QLabel("Pendiente")
        self.opening_value = QLabel("0.0°")
        self.protrusion_value = QLabel("0.0 mm")
        self.lateral_value = QLabel("0.0°")
        self.closure_value = QLabel("0.0°")
        self.path_value = QLabel("—")
        self._mounting_available = False
        self._maximum_opening_degrees = 30.0
        self._maximum_translation_mm: float | None = None
        self._opening_angle_degrees = 0.0
        self._protrusion_distance_mm = 0.0

        form = QFormLayout()
        form.setSpacing(9)
        form.addRow("Referencia", self.reference_value)
        form.addRow("Apertura", self.opening_value)
        form.addRow("Protrusión", self.protrusion_value)
        form.addRow("Lateralidad", self.lateral_value)
        form.addRow("Ajuste oclusal", self.closure_value)
        form.addRow("Recorrido disponible", self.path_value)

        self.close_button = QPushButton("Cerrar 1°")
        self.close_button.clicked.connect(
            lambda checked=False: self.close_requested.emit()
        )

        self.open_button = QPushButton("Abrir 1°")
        self.open_button.clicked.connect(
            lambda checked=False: self.open_requested.emit()
        )

        opening_buttons = QHBoxLayout()
        opening_buttons.setSpacing(8)
        opening_buttons.addWidget(self.close_button)
        opening_buttons.addWidget(self.open_button)

        self.retreat_button = QPushButton("Retroceder 1 mm")
        self.retreat_button.clicked.connect(
            lambda checked=False: self.retreat_requested.emit()
        )

        self.advance_button = QPushButton("Avanzar 1 mm")
        self.advance_button.clicked.connect(
            lambda checked=False: self.advance_requested.emit()
        )

        protrusion_buttons = QHBoxLayout()
        protrusion_buttons.setSpacing(8)
        protrusion_buttons.addWidget(self.retreat_button)
        protrusion_buttons.addWidget(self.advance_button)

        self.rc_button = QPushButton("Volver a relación céntrica (RC)")
        self.rc_button.setStyleSheet(
            "QPushButton { background: #6b4c8a; color: white; "
            "font-weight: 600; padding: 8px; border-radius: 3px; }"
            "QPushButton:disabled { background: #b7aabd; "
            "color: #f1edf3; }"
        )
        self.rc_button.clicked.connect(
            lambda checked=False: self.rc_requested.emit()
        )

        self.state_label = QLabel(
            "Los controles permanecerán protegidos hasta confirmar "
            "el montaje."
        )
        self.state_label.setWordWrap(True)
        self._set_state_style(ready=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.availability_label)
        layout.addLayout(form)
        layout.addLayout(opening_buttons)
        layout.addLayout(protrusion_buttons)
        layout.addWidget(self.rc_button)
        layout.addWidget(self.state_label)
        layout.addStretch()

        self._update_movement_controls()

    def set_mounting_available(
        self,
        available: bool,
        maximum_translation: float | None = None,
        maximum_opening_degrees: float = 30.0,
    ) -> None:
        """Reflect whether functional calibration has a valid RC reference."""

        self._mounting_available = bool(available)
        self._maximum_opening_degrees = float(maximum_opening_degrees)
        self._maximum_translation_mm = (
            None
            if maximum_translation is None
            else float(maximum_translation)
        )
        self.availability_label.setVisible(not available)
        if available:
            self.reference_value.setText("RC confirmada")
            self.opening_value.setText("0.0°")
            self.protrusion_value.setText("0.0 mm")
            self.path_value.setText(
                "—"
                if maximum_translation is None
                else f"{maximum_translation:.1f} mm"
            )
            if maximum_translation is None:
                self.state_label.setText(
                    "Apertura sobre el eje de bisagra disponible. "
                    "La protrusión requiere un recorrido conocido."
                )
            else:
                self.state_label.setText(
                    "Apertura sobre el eje de bisagra y protrusión "
                    "guiada disponibles. Los demás movimientos "
                    "permanecen protegidos."
                )
            self._set_state_style(ready=True)
            self._opening_angle_degrees = 0.0
            self._protrusion_distance_mm = 0.0
            self._update_movement_controls()
            return

        self.clear()

    def clear(self) -> None:
        """Return the panel to its protected initial state."""

        self.availability_label.setVisible(True)
        self.reference_value.setText("Pendiente")
        self.opening_value.setText("0.0°")
        self.protrusion_value.setText("0.0 mm")
        self.lateral_value.setText("0.0°")
        self.closure_value.setText("0.0°")
        self.path_value.setText("—")
        self._maximum_translation_mm = None
        self._opening_angle_degrees = 0.0
        self._protrusion_distance_mm = 0.0
        self.state_label.setText(
            "Los controles permanecerán protegidos hasta confirmar "
            "el montaje."
        )
        self._set_state_style(ready=False)
        self._update_movement_controls()

    def show_opening(self, angle_degrees: float) -> None:
        """Display the current hinge opening and synchronize its controls."""

        angle_degrees = float(angle_degrees)
        self._opening_angle_degrees = angle_degrees
        self.opening_value.setText(f"{angle_degrees:.1f}°")
        self._update_movement_controls()

    def show_protrusion(self, distance_mm: float) -> None:
        """Display guided protrusion and synchronize its controls."""

        distance_mm = float(distance_mm)
        self._protrusion_distance_mm = distance_mm
        self.protrusion_value.setText(f"{distance_mm:.1f} mm")
        self._update_movement_controls()

    def _update_movement_controls(self) -> None:
        can_close = (
            self._mounting_available
            and self._opening_angle_degrees > 0.0
        )
        can_open = (
            self._mounting_available
            and self._opening_angle_degrees
            < self._maximum_opening_degrees
        )
        can_retreat = (
            self._mounting_available
            and self._protrusion_distance_mm > 0.0
        )
        can_advance = (
            self._mounting_available
            and self._maximum_translation_mm is not None
            and self._protrusion_distance_mm
            < self._maximum_translation_mm
        )
        self.close_button.setEnabled(can_close)
        self.open_button.setEnabled(can_open)
        self.retreat_button.setEnabled(can_retreat)
        self.advance_button.setEnabled(can_advance)
        self.rc_button.setEnabled(can_close or can_retreat)

    def _set_state_style(self, *, ready: bool) -> None:
        if ready:
            colors = "background: #dcefea; color: #24594e;"
        else:
            colors = "background: #e8eef0; color: #40545c;"
        self.state_label.setStyleSheet(
            colors + " padding: 8px; border-radius: 3px;"
        )
