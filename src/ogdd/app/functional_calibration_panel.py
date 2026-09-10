"""Status panel for the functional calibration workflow step."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class FunctionalCalibrationPanel(QWidget):
    """Describe calibration readiness before movement controls are connected."""

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

        form = QFormLayout()
        form.setSpacing(9)
        form.addRow("Referencia", self.reference_value)
        form.addRow("Apertura", self.opening_value)
        form.addRow("Protrusión", self.protrusion_value)
        form.addRow("Lateralidad", self.lateral_value)
        form.addRow("Ajuste oclusal", self.closure_value)
        form.addRow("Recorrido disponible", self.path_value)

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
        layout.addWidget(self.state_label)
        layout.addStretch()

    def set_mounting_available(
        self,
        available: bool,
        maximum_translation: float | None = None,
    ) -> None:
        """Reflect whether functional calibration has a valid RC reference."""

        self.availability_label.setVisible(not available)
        if available:
            self.reference_value.setText("RC confirmada")
            self.path_value.setText(
                "—"
                if maximum_translation is None
                else f"{maximum_translation:.1f} mm"
            )
            self.state_label.setText(
                "Montaje disponible. El panel está preparado para "
                "incorporar los controles de movimiento."
            )
            self._set_state_style(ready=True)
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
        self.state_label.setText(
            "Los controles permanecerán protegidos hasta confirmar "
            "el montaje."
        )
        self._set_state_style(ready=False)

    def _set_state_style(self, *, ready: bool) -> None:
        if ready:
            colors = "background: #dcefea; color: #24594e;"
        else:
            colors = "background: #e8eef0; color: #40545c;"
        self.state_label.setStyleSheet(
            colors + " padding: 8px; border-radius: 3px;"
        )
