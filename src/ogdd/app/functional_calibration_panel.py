"""Status panel for the functional calibration workflow step."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class FunctionalCalibrationPanel(QWidget):
    """Describe calibration readiness before movement controls are connected."""

    MOVEMENT_SCALE = 10

    open_requested = Signal()
    close_requested = Signal()
    advance_requested = Signal()
    retreat_requested = Signal()
    move_left_requested = Signal()
    move_right_requested = Signal()
    lateral_zero_requested = Signal()
    opening_changed = Signal(float)
    protrusion_changed = Signal(float)
    lateral_changed = Signal(float)
    adjust_close_requested = Signal()
    adjust_open_requested = Signal()
    adjustment_zero_requested = Signal()
    save_protrusive_requested = Signal()
    save_right_canine_requested = Signal()
    save_left_canine_requested = Signal()
    go_protrusive_requested = Signal()
    go_right_canine_requested = Signal()
    go_left_canine_requested = Signal()
    clear_limits_requested = Signal()
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
        self._maximum_right_lateral_degrees: float | None = None
        self._maximum_left_lateral_degrees: float | None = None
        self._opening_angle_degrees = 0.0
        self._protrusion_distance_mm = 0.0
        self._lateral_angle_degrees = 0.0
        self._adjustment_angle_degrees = 0.0
        self._protrusive_saved = False
        self._right_canine_saved = False
        self._left_canine_saved = False

        form = QFormLayout()
        form.setSpacing(9)
        form.addRow("Referencia", self.reference_value)
        form.addRow("Apertura", self.opening_value)
        form.addRow("Protrusión", self.protrusion_value)
        form.addRow("Lateralidad", self.lateral_value)
        form.addRow("Ajuste oclusal", self.closure_value)
        form.addRow("Recorrido disponible", self.path_value)

        movement_title = QLabel("Movimiento mandibular base")
        movement_title.setStyleSheet(
            "font-weight: 600; color: #315b5f; margin-top: 4px;"
        )

        self.opening_slider = self._movement_slider()
        self.opening_slider.valueChanged.connect(
            lambda value: self.opening_changed.emit(
                self._from_slider(value)
            )
        )
        self.protrusion_slider = self._movement_slider()
        self.protrusion_slider.valueChanged.connect(
            lambda value: self.protrusion_changed.emit(
                self._from_slider(value)
            )
        )
        self.lateral_slider = self._movement_slider()
        self.lateral_slider.valueChanged.connect(
            lambda value: self.lateral_changed.emit(
                self._from_slider(value)
            )
        )

        movement_sliders = QFormLayout()
        movement_sliders.setSpacing(7)
        movement_sliders.addRow("Apertura", self.opening_slider)
        movement_sliders.addRow("Protrusión", self.protrusion_slider)
        movement_sliders.addRow("Lateralidad", self.lateral_slider)

        self.close_button = QPushButton("Cerrar 0.1°")
        self.close_button.clicked.connect(
            lambda checked=False: self.close_requested.emit()
        )

        self.open_button = QPushButton("Abrir 0.1°")
        self.open_button.clicked.connect(
            lambda checked=False: self.open_requested.emit()
        )

        opening_buttons = QHBoxLayout()
        opening_buttons.setSpacing(8)
        opening_buttons.addWidget(self.close_button)
        opening_buttons.addWidget(self.open_button)

        self.retreat_button = QPushButton("Retroceder 0.1 mm")
        self.retreat_button.clicked.connect(
            lambda checked=False: self.retreat_requested.emit()
        )

        self.advance_button = QPushButton("Avanzar 0.1 mm")
        self.advance_button.clicked.connect(
            lambda checked=False: self.advance_requested.emit()
        )

        protrusion_buttons = QHBoxLayout()
        protrusion_buttons.setSpacing(8)
        protrusion_buttons.addWidget(self.retreat_button)
        protrusion_buttons.addWidget(self.advance_button)

        self.left_button = QPushButton("Izquierda 0.1°")
        self.left_button.clicked.connect(
            lambda checked=False: self.move_left_requested.emit()
        )

        self.right_button = QPushButton("Derecha 0.1°")
        self.right_button.clicked.connect(
            lambda checked=False: self.move_right_requested.emit()
        )

        lateral_buttons = QHBoxLayout()
        lateral_buttons.setSpacing(8)
        lateral_buttons.addWidget(self.left_button)
        lateral_buttons.addWidget(self.right_button)

        adjustment_title = QLabel("Ajuste oclusal fino")
        adjustment_title.setStyleSheet(
            "font-weight: 600; color: #315b5f; margin-top: 4px;"
        )

        self.fine_close_button = QPushButton("Cerrar fino 0.1°")
        self.fine_close_button.clicked.connect(
            lambda checked=False: self.adjust_close_requested.emit()
        )

        self.fine_open_button = QPushButton("Abrir fino 0.1°")
        self.fine_open_button.clicked.connect(
            lambda checked=False: self.adjust_open_requested.emit()
        )

        fine_buttons = QHBoxLayout()
        fine_buttons.setSpacing(8)
        fine_buttons.addWidget(self.fine_close_button)
        fine_buttons.addWidget(self.fine_open_button)

        self.adjustment_zero_button = QPushButton(
            "Restablecer ajuste oclusal (0.0°)"
        )
        self.adjustment_zero_button.clicked.connect(
            lambda checked=False: self.adjustment_zero_requested.emit()
        )

        limits_title = QLabel("Límites funcionales confirmados")
        limits_title.setStyleSheet(
            "font-weight: 600; color: #315b5f; margin-top: 4px;"
        )

        self.protrusive_limit_value = QLabel("No calibrado")
        self.right_canine_limit_value = QLabel("No calibrado")
        self.left_canine_limit_value = QLabel("No calibrado")
        for value_label in (
            self.protrusive_limit_value,
            self.right_canine_limit_value,
            self.left_canine_limit_value,
        ):
            value_label.setWordWrap(True)

        limits_form = QFormLayout()
        limits_form.setSpacing(6)
        limits_form.addRow("Borde a borde", self.protrusive_limit_value)
        limits_form.addRow("Canina derecha", self.right_canine_limit_value)
        limits_form.addRow("Canina izquierda", self.left_canine_limit_value)

        self.save_protrusive_button = QPushButton("Guardar borde a borde")
        self.save_protrusive_button.clicked.connect(
            lambda checked=False: self.save_protrusive_requested.emit()
        )
        self.go_protrusive_button = QPushButton("Ir al borde a borde")
        self.go_protrusive_button.clicked.connect(
            lambda checked=False: self.go_protrusive_requested.emit()
        )
        protrusive_limit_buttons = QHBoxLayout()
        protrusive_limit_buttons.setSpacing(8)
        protrusive_limit_buttons.addWidget(self.save_protrusive_button)
        protrusive_limit_buttons.addWidget(self.go_protrusive_button)

        self.save_right_canine_button = QPushButton("Guardar canina derecha")
        self.save_right_canine_button.clicked.connect(
            lambda checked=False: self.save_right_canine_requested.emit()
        )
        self.go_right_canine_button = QPushButton("Ir a canina derecha")
        self.go_right_canine_button.clicked.connect(
            lambda checked=False: self.go_right_canine_requested.emit()
        )
        right_limit_buttons = QHBoxLayout()
        right_limit_buttons.setSpacing(8)
        right_limit_buttons.addWidget(self.save_right_canine_button)
        right_limit_buttons.addWidget(self.go_right_canine_button)

        self.save_left_canine_button = QPushButton("Guardar canina izquierda")
        self.save_left_canine_button.clicked.connect(
            lambda checked=False: self.save_left_canine_requested.emit()
        )
        self.go_left_canine_button = QPushButton("Ir a canina izquierda")
        self.go_left_canine_button.clicked.connect(
            lambda checked=False: self.go_left_canine_requested.emit()
        )
        left_limit_buttons = QHBoxLayout()
        left_limit_buttons.setSpacing(8)
        left_limit_buttons.addWidget(self.save_left_canine_button)
        left_limit_buttons.addWidget(self.go_left_canine_button)

        self.clear_limits_button = QPushButton(
            "Borrar límites y recalibrar"
        )
        self.clear_limits_button.clicked.connect(
            lambda checked=False: self.clear_limits_requested.emit()
        )

        self.lateral_zero_button = QPushButton(
            "Centrar lateralidad (0°)"
        )
        self.lateral_zero_button.clicked.connect(
            lambda checked=False: self.lateral_zero_requested.emit()
        )

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
        layout.addWidget(movement_title)
        layout.addLayout(movement_sliders)
        layout.addLayout(opening_buttons)
        layout.addLayout(protrusion_buttons)
        layout.addLayout(lateral_buttons)
        layout.addWidget(adjustment_title)
        layout.addLayout(fine_buttons)
        layout.addWidget(self.adjustment_zero_button)
        layout.addWidget(limits_title)
        layout.addLayout(limits_form)
        layout.addLayout(protrusive_limit_buttons)
        layout.addLayout(right_limit_buttons)
        layout.addLayout(left_limit_buttons)
        layout.addWidget(self.clear_limits_button)
        layout.addWidget(self.lateral_zero_button)
        layout.addWidget(self.rc_button)
        layout.addWidget(self.state_label)
        layout.addStretch()

        self._update_movement_controls()

    @classmethod
    def _to_slider(cls, value: float) -> int:
        """Represent one decimal movement value with an integer slider."""

        return int(round(float(value) * cls.MOVEMENT_SCALE))

    @classmethod
    def _from_slider(cls, value: int) -> float:
        """Return the decimal movement represented by an integer slider."""

        return float(value) / cls.MOVEMENT_SCALE

    @staticmethod
    def _movement_slider() -> QSlider:
        """Build one horizontal slider used for operator positioning."""

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setTracking(True)
        return slider

    @classmethod
    def _set_slider_value(cls, slider: QSlider, value: float) -> None:
        """Synchronize a slider without requesting another movement."""

        blocker = QSignalBlocker(slider)
        slider.setValue(cls._to_slider(value))
        del blocker

    def _configure_slider_ranges(self) -> None:
        """Reflect the current asymmetric mechanical movement limits."""

        blockers = (
            QSignalBlocker(self.opening_slider),
            QSignalBlocker(self.protrusion_slider),
            QSignalBlocker(self.lateral_slider),
        )
        self.opening_slider.setRange(
            0,
            self._to_slider(self._maximum_opening_degrees),
        )
        maximum_translation = self._maximum_translation_mm or 0.0
        self.protrusion_slider.setRange(
            0,
            self._to_slider(maximum_translation),
        )
        maximum_left = self._maximum_left_lateral_degrees or 0.0
        maximum_right = self._maximum_right_lateral_degrees or 0.0
        self.lateral_slider.setRange(
            -self._to_slider(maximum_left),
            self._to_slider(maximum_right),
        )
        del blockers

    def set_mounting_available(
        self,
        available: bool,
        maximum_translation: float | None = None,
        maximum_opening_degrees: float = 30.0,
        maximum_right_lateral_degrees: float | None = None,
        maximum_left_lateral_degrees: float | None = None,
    ) -> None:
        """Reflect whether functional calibration has a valid RC reference."""

        self._mounting_available = bool(available)
        self._maximum_opening_degrees = float(maximum_opening_degrees)
        self._maximum_translation_mm = (
            None
            if maximum_translation is None
            else float(maximum_translation)
        )
        self._maximum_right_lateral_degrees = (
            None
            if maximum_right_lateral_degrees is None
            else float(maximum_right_lateral_degrees)
        )
        self._maximum_left_lateral_degrees = (
            None
            if maximum_left_lateral_degrees is None
            else float(maximum_left_lateral_degrees)
        )
        self._configure_slider_ranges()
        self.availability_label.setVisible(not available)
        if available:
            self.reference_value.setText("RC confirmada")
            self.opening_value.setText("0.0°")
            self.protrusion_value.setText("0.0 mm")
            self.lateral_value.setText("0.0°")
            self.closure_value.setText("0.0°")
            self.path_value.setText(
                "—"
                if maximum_translation is None
                else f"{maximum_translation:.1f} mm"
            )
            lateral_available = (
                maximum_right_lateral_degrees is not None
                and maximum_left_lateral_degrees is not None
            )
            if maximum_translation is None:
                self.state_label.setText(
                    "Apertura sobre el eje de bisagra disponible. "
                    "La protrusión requiere un recorrido conocido."
                )
            elif lateral_available:
                self.state_label.setText(
                    "Apertura, protrusión y lateralidad guiadas "
                    "disponibles. El cierre oclusal fino está "
                    "disponible en pasos de 0.1°."
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
            self._lateral_angle_degrees = 0.0
            self._adjustment_angle_degrees = 0.0
            self._protrusive_saved = False
            self._right_canine_saved = False
            self._left_canine_saved = False
            self._clear_limit_labels()
            self._set_slider_value(self.opening_slider, 0.0)
            self._set_slider_value(self.protrusion_slider, 0.0)
            self._set_slider_value(self.lateral_slider, 0.0)
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
        self._maximum_right_lateral_degrees = None
        self._maximum_left_lateral_degrees = None
        self._opening_angle_degrees = 0.0
        self._protrusion_distance_mm = 0.0
        self._lateral_angle_degrees = 0.0
        self._adjustment_angle_degrees = 0.0
        self._protrusive_saved = False
        self._right_canine_saved = False
        self._left_canine_saved = False
        self._clear_limit_labels()
        self._configure_slider_ranges()
        self._set_slider_value(self.opening_slider, 0.0)
        self._set_slider_value(self.protrusion_slider, 0.0)
        self._set_slider_value(self.lateral_slider, 0.0)
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
        self._set_slider_value(self.opening_slider, angle_degrees)
        self._update_movement_controls()

    def show_protrusion(self, distance_mm: float) -> None:
        """Display guided protrusion and synchronize its controls."""

        distance_mm = float(distance_mm)
        self._protrusion_distance_mm = distance_mm
        self.protrusion_value.setText(f"{distance_mm:.1f} mm")
        self._set_slider_value(self.protrusion_slider, distance_mm)
        self._update_movement_controls()

    def show_lateral(self, angle_degrees: float) -> None:
        """Display signed lateral movement and synchronize its controls."""

        angle_degrees = float(angle_degrees)
        self._lateral_angle_degrees = angle_degrees
        if angle_degrees > 0.0:
            text = f"+{angle_degrees:.1f}° (derecha)"
        elif angle_degrees < 0.0:
            text = f"{angle_degrees:.1f}° (izquierda)"
        else:
            text = "0.0°"
        self.lateral_value.setText(text)
        self._set_slider_value(self.lateral_slider, angle_degrees)
        self._update_movement_controls()

    def show_adjustment(self, angle_degrees: float) -> None:
        """Display the independent signed occlusal fine adjustment."""

        angle_degrees = float(angle_degrees)
        self._adjustment_angle_degrees = angle_degrees
        if angle_degrees > 0.0:
            text = f"+{angle_degrees:.1f}° (apertura)"
        elif angle_degrees < 0.0:
            text = f"{angle_degrees:.1f}° (cierre)"
        else:
            text = "0.0°"
        self.closure_value.setText(text)
        self._update_movement_controls()

    @staticmethod
    def _format_limit(limit) -> str:
        """Format one complete saved calibration without storing geometry."""

        if limit is None:
            return "No calibrado"
        return (
            f"A {limit.base_opening_angle_degrees:.1f}° | "
            f"P {limit.protrusion_distance_mm:.1f} mm | "
            f"L {limit.lateral_angle_degrees:+.1f}° | "
            f"ajuste {limit.adjustment_angle_degrees:+.1f}°"
        )

    def _clear_limit_labels(self) -> None:
        """Return every saved-limit label to its initial state."""

        self.protrusive_limit_value.setText("No calibrado")
        self.right_canine_limit_value.setText("No calibrado")
        self.left_canine_limit_value.setText("No calibrado")

    def show_limits(
        self,
        *,
        protrusive_limit=None,
        right_canine_limit=None,
        left_canine_limit=None,
    ) -> None:
        """Display the three immutable numeric functional endpoints."""

        self._protrusive_saved = protrusive_limit is not None
        self._right_canine_saved = right_canine_limit is not None
        self._left_canine_saved = left_canine_limit is not None
        self.protrusive_limit_value.setText(
            self._format_limit(protrusive_limit)
        )
        self.right_canine_limit_value.setText(
            self._format_limit(right_canine_limit)
        )
        self.left_canine_limit_value.setText(
            self._format_limit(left_canine_limit)
        )
        self._update_movement_controls()

    def show_position(
        self,
        *,
        opening_angle_degrees: float,
        protrusion_distance_mm: float,
        lateral_angle_degrees: float,
        maximum_translation_mm: float,
        maximum_right_lateral_degrees: float,
        maximum_left_lateral_degrees: float,
        adjustment_angle_degrees: float = 0.0,
    ) -> None:
        """Synchronize one complete movement state and its dynamic limits."""

        self._maximum_translation_mm = float(maximum_translation_mm)
        self._maximum_right_lateral_degrees = float(
            maximum_right_lateral_degrees
        )
        self._maximum_left_lateral_degrees = float(
            maximum_left_lateral_degrees
        )
        self._configure_slider_ranges()
        self.path_value.setText(f"{maximum_translation_mm:.1f} mm")
        self.show_opening(opening_angle_degrees)
        self.show_protrusion(protrusion_distance_mm)
        self.show_lateral(lateral_angle_degrees)
        self.show_adjustment(adjustment_angle_degrees)

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
        can_move_left = (
            self._mounting_available
            and self._maximum_left_lateral_degrees is not None
            and self._lateral_angle_degrees
            > -self._maximum_left_lateral_degrees
        )
        can_move_right = (
            self._mounting_available
            and self._maximum_right_lateral_degrees is not None
            and self._lateral_angle_degrees
            < self._maximum_right_lateral_degrees
        )
        can_center_lateral = (
            self._mounting_available
            and self._lateral_angle_degrees != 0.0
        )
        has_adjustment = (
            self._mounting_available
            and self._adjustment_angle_degrees != 0.0
        )
        can_save_protrusive = (
            self._mounting_available
            and not self._protrusive_saved
            and self._protrusion_distance_mm > 0.0
            and self._lateral_angle_degrees == 0.0
        )
        can_save_right_canine = (
            self._mounting_available
            and not self._right_canine_saved
            and self._lateral_angle_degrees > 0.0
        )
        can_save_left_canine = (
            self._mounting_available
            and not self._left_canine_saved
            and self._lateral_angle_degrees < 0.0
        )
        has_limits = (
            self._protrusive_saved
            or self._right_canine_saved
            or self._left_canine_saved
        )
        self.opening_slider.setEnabled(self._mounting_available)
        self.protrusion_slider.setEnabled(
            self._mounting_available
            and self._maximum_translation_mm is not None
        )
        self.lateral_slider.setEnabled(
            self._mounting_available
            and self._maximum_right_lateral_degrees is not None
            and self._maximum_left_lateral_degrees is not None
        )
        self.close_button.setEnabled(can_close)
        self.open_button.setEnabled(can_open)
        self.retreat_button.setEnabled(can_retreat)
        self.advance_button.setEnabled(can_advance)
        self.left_button.setEnabled(can_move_left)
        self.right_button.setEnabled(can_move_right)
        self.lateral_zero_button.setEnabled(can_center_lateral)
        self.fine_close_button.setEnabled(self._mounting_available)
        self.fine_open_button.setEnabled(self._mounting_available)
        self.adjustment_zero_button.setEnabled(has_adjustment)
        self.save_protrusive_button.setEnabled(can_save_protrusive)
        self.save_right_canine_button.setEnabled(can_save_right_canine)
        self.save_left_canine_button.setEnabled(can_save_left_canine)
        self.go_protrusive_button.setEnabled(self._protrusive_saved)
        self.go_right_canine_button.setEnabled(self._right_canine_saved)
        self.go_left_canine_button.setEnabled(self._left_canine_saved)
        self.clear_limits_button.setEnabled(
            self._mounting_available and has_limits
        )
        self.rc_button.setEnabled(
            can_close or can_retreat or can_center_lateral or has_adjustment
        )

    def _set_state_style(self, *, ready: bool) -> None:
        if ready:
            colors = "background: #dcefea; color: #24594e;"
        else:
            colors = "background: #e8eef0; color: #40545c;"
        self.state_label.setStyleSheet(
            colors + " padding: 8px; border-radius: 3px;"
        )
