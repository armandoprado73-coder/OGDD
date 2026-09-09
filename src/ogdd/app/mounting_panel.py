"""Controls for constructing the virtual RC mounting."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ogdd.articulator.configuration import ArticulatorConfiguration


class MountingPanel(QWidget):
    """Configure and request a Bonwill-based virtual mounting."""

    build_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        title = QLabel("Montaje virtual en RC")
        title.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #24484b;"
        )

        instructions = QLabel(
            "Configure las referencias mecánicas del articulador. "
            "El montaje se construirá alrededor del sistema anatómico "
            "confirmado."
        )
        instructions.setWordWrap(True)

        self.availability_label = QLabel(
            "Confirme primero la orientación anatómica."
        )
        self.availability_label.setWordWrap(True)
        self.availability_label.setStyleSheet(
            "background: #f0e6c9; color: #604e20; "
            "padding: 8px; border-radius: 3px;"
        )

        self.preset_combo = QComboBox()
        self.preset_combo.addItem("S — 95 mm", 95.0)
        self.preset_combo.addItem("M — 110 mm", 110.0)
        self.preset_combo.addItem("L — 140 mm", 140.0)
        self.preset_combo.setCurrentIndex(1)

        self.balkwill_angle = self._angle_spinbox(
            minimum=22.0,
            maximum=30.0,
            value=25.0,
        )
        self.right_guidance = self._angle_spinbox(
            minimum=0.0,
            maximum=89.0,
            value=45.0,
        )
        self.left_guidance = self._angle_spinbox(
            minimum=0.0,
            maximum=89.0,
            value=45.0,
        )

        form = QFormLayout()
        form.setSpacing(9)
        form.addRow("Distancia intercondilar", self.preset_combo)
        form.addRow("Ángulo de Balkwill", self.balkwill_angle)
        form.addRow("Guía condilar derecha", self.right_guidance)
        form.addRow("Guía condilar izquierda", self.left_guidance)
        form.addRow("Diámetro condilar", QLabel("6.0 mm"))
        form.addRow("Superficie guía", QLabel("20 × 20 mm"))
        form.addRow("Recorrido funcional", QLabel("17.0 mm"))

        self.build_button = QPushButton("Construir montaje en RC")
        self.build_button.setEnabled(False)
        self.build_button.setStyleSheet(
            "QPushButton { background: #6b4c8a; color: white; "
            "font-weight: 600; padding: 8px; border-radius: 3px; }"
            "QPushButton:disabled { background: #b7aabd; color: #f1edf3; }"
        )
        self.build_button.clicked.connect(
            lambda checked=False: self.build_requested.emit()
        )

        self.reset_button = QPushButton("Retirar montaje")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(
            lambda checked=False: self.reset_requested.emit()
        )

        self.result_label = QLabel(
            "Bonwill, eje de bisagra y guías aparecerán aquí."
        )
        self.result_label.setWordWrap(True)
        self._set_result_style(confirmed=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.availability_label)
        layout.addLayout(form)
        layout.addWidget(self.build_button)
        layout.addWidget(self.reset_button)
        layout.addWidget(self.result_label)
        layout.addStretch()

        self._set_controls_enabled(False)

    @staticmethod
    def _angle_spinbox(
        *,
        minimum: float,
        maximum: float,
        value: float,
    ) -> QDoubleSpinBox:
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setDecimals(1)
        spinbox.setSingleStep(1.0)
        spinbox.setValue(value)
        spinbox.setSuffix("°")
        return spinbox

    def configuration(self) -> ArticulatorConfiguration:
        """Build the validated core configuration selected by the user."""

        return ArticulatorConfiguration(
            intercondylar_width=float(self.preset_combo.currentData()),
            balkwill_angle_degrees=self.balkwill_angle.value(),
            right_condylar_guidance_degrees=self.right_guidance.value(),
            left_condylar_guidance_degrees=self.left_guidance.value(),
        )

    def set_orientation_available(self, available: bool) -> None:
        """Enable mounting only after anatomical orientation."""

        self.availability_label.setVisible(not available)
        self._set_controls_enabled(available)
        if not available:
            self.clear()

    def show_mounting(
        self,
        configuration: ArticulatorConfiguration,
        maximum_translation: float,
    ) -> None:
        """Summarize the mounted configuration."""

        self.result_label.setText(
            "Montaje en RC construido\n"
            f"Intercondilar: {configuration.intercondylar_width:.1f} mm\n"
            f"Balkwill: {configuration.balkwill_angle_degrees:.1f}°\n"
            "Guías D/I: "
            f"{configuration.right_condylar_guidance_degrees:.1f}° / "
            f"{configuration.left_condylar_guidance_degrees:.1f}°\n"
            f"Recorrido disponible: {maximum_translation:.1f} mm"
        )
        self._set_result_style(confirmed=True)
        self.reset_button.setEnabled(True)

    def clear(self) -> None:
        """Clear the result without changing selected parameters."""

        self.result_label.setText(
            "Bonwill, eje de bisagra y guías aparecerán aquí."
        )
        self._set_result_style(confirmed=False)
        self.reset_button.setEnabled(False)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for control in (
            self.preset_combo,
            self.balkwill_angle,
            self.right_guidance,
            self.left_guidance,
            self.build_button,
        ):
            control.setEnabled(enabled)

    def _set_result_style(self, *, confirmed: bool) -> None:
        if confirmed:
            colors = "background: #eadff2; color: #4b3461;"
        else:
            colors = "background: #e8eef0; color: #40545c;"
        self.result_label.setStyleSheet(
            colors + " padding: 8px; border-radius: 3px;"
        )
