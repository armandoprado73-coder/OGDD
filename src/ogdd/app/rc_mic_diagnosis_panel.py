"""Controls and results for the RC-to-MIC diagnostic workflow."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RcMicDiagnosisPanel(QWidget):
    """Collect two MIC seeds and report bilateral condylar displacement."""

    select_maxillary_seed_requested = Signal()
    select_mandibular_seed_requested = Signal()
    diagnose_requested = Signal()
    reset_requested = Signal()
    view_rc_requested = Signal()
    view_mic_requested = Signal()
    view_overlay_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._mic_available = False
        self._mounting_available = False
        self._maxillary_seed_vertex: int | None = None
        self._mandibular_seed_vertex: int | None = None

        title = QLabel("Diagnóstico RC–MIC")
        title.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #24484b;"
        )

        instructions = QLabel(
            "Seleccione una semilla sobre cada arcada del registro MIC. "
            "OGDD separará las superficies, anclará el maxilar al montaje "
            "en RC y calculará el desplazamiento mandibular y condilar."
        )
        instructions.setWordWrap(True)

        self.availability_label = QLabel(
            "Cargue un registro MIC y construya el montaje en RC."
        )
        self.availability_label.setWordWrap(True)
        self.availability_label.setStyleSheet(
            "background: #f0e6c9; color: #604e20; "
            "padding: 8px; border-radius: 3px;"
        )

        self.maxillary_seed_value = QLabel("Pendiente")
        self.mandibular_seed_value = QLabel("Pendiente")

        seed_form = QFormLayout()
        seed_form.setSpacing(9)
        seed_form.addRow("Semilla maxilar", self.maxillary_seed_value)
        seed_form.addRow("Semilla mandibular", self.mandibular_seed_value)

        self.maxillary_seed_button = QPushButton(
            "Seleccionar sobre maxilar MIC"
        )
        self.maxillary_seed_button.clicked.connect(
            lambda checked=False: (
                self.select_maxillary_seed_requested.emit()
            )
        )
        self.mandibular_seed_button = QPushButton(
            "Seleccionar sobre mandíbula MIC"
        )
        self.mandibular_seed_button.clicked.connect(
            lambda checked=False: (
                self.select_mandibular_seed_requested.emit()
            )
        )

        seed_buttons = QVBoxLayout()
        seed_buttons.setSpacing(7)
        seed_buttons.addWidget(self.maxillary_seed_button)
        seed_buttons.addWidget(self.mandibular_seed_button)

        self.diagnose_button = QPushButton("Calcular diagnóstico RC–MIC")
        self.diagnose_button.setStyleSheet(
            "QPushButton { background: #6b4c8a; color: white; "
            "font-weight: 600; padding: 8px; border-radius: 3px; }"
            "QPushButton:disabled { background: #b7aabd; color: #f1edf3; }"
        )
        self.diagnose_button.clicked.connect(
            lambda checked=False: self.diagnose_requested.emit()
        )

        self.reset_button = QPushButton("Reiniciar diagnóstico")
        self.reset_button.clicked.connect(
            lambda checked=False: self.reset_requested.emit()
        )

        action_buttons = QHBoxLayout()
        action_buttons.setSpacing(8)
        action_buttons.addWidget(self.diagnose_button)
        action_buttons.addWidget(self.reset_button)

        results_title = QLabel("Desplazamiento condilar RC → MIC")
        results_title.setStyleSheet(
            "font-weight: 600; color: #315b5f; margin-top: 6px;"
        )

        self.convergence_value = QLabel("Pendiente")
        self.maxillary_rmse_value = QLabel("—")
        self.mandibular_rmse_value = QLabel("—")
        self.right_distance_value = QLabel("—")
        self.right_vector_value = QLabel("X —  |  Y —  |  Z —")
        self.left_distance_value = QLabel("—")
        self.left_vector_value = QLabel("X —  |  Y —  |  Z —")
        self.right_vector_value.setWordWrap(True)
        self.left_vector_value.setWordWrap(True)

        results_form = QFormLayout()
        results_form.setSpacing(9)
        results_form.addRow("Registro", self.convergence_value)
        results_form.addRow("RMSE maxilar", self.maxillary_rmse_value)
        results_form.addRow("RMSE mandibular", self.mandibular_rmse_value)
        results_form.addRow("Cóndilo derecho", self.right_distance_value)
        results_form.addRow("Vector derecho", self.right_vector_value)
        results_form.addRow("Cóndilo izquierdo", self.left_distance_value)
        results_form.addRow("Vector izquierdo", self.left_vector_value)

        comparison_title = QLabel("Comparación visual RC–MIC")
        comparison_title.setStyleSheet(
            "font-weight: 600; color: #315b5f; margin-top: 6px;"
        )
        comparison_help = QLabel(
            "Cambie entre ambas posiciones sin modificar el montaje ni "
            "la cámara actual."
        )
        comparison_help.setWordWrap(True)

        self.view_rc_button = QPushButton("Mostrar RC")
        self.view_mic_button = QPushButton("Mostrar MIC")
        self.view_overlay_button = QPushButton("Superponer RC/MIC")
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        for button in (
            self.view_rc_button,
            self.view_mic_button,
            self.view_overlay_button,
        ):
            button.setCheckable(True)
            self.view_group.addButton(button)
        self.view_overlay_button.setChecked(True)
        self.view_rc_button.clicked.connect(
            lambda checked=False: self.view_rc_requested.emit()
        )
        self.view_mic_button.clicked.connect(
            lambda checked=False: self.view_mic_requested.emit()
        )
        self.view_overlay_button.clicked.connect(
            lambda checked=False: self.view_overlay_requested.emit()
        )

        comparison_buttons = QVBoxLayout()
        comparison_buttons.setSpacing(7)
        comparison_buttons.addWidget(self.view_rc_button)
        comparison_buttons.addWidget(self.view_mic_button)
        comparison_buttons.addWidget(self.view_overlay_button)

        self.state_label = QLabel(
            "El diagnóstico permanecerá protegido hasta confirmar "
            "sus referencias."
        )
        self.state_label.setWordWrap(True)
        self._set_state_style(ready=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.availability_label)
        layout.addLayout(seed_form)
        layout.addLayout(seed_buttons)
        layout.addLayout(action_buttons)
        layout.addWidget(results_title)
        layout.addLayout(results_form)
        layout.addWidget(comparison_title)
        layout.addWidget(comparison_help)
        layout.addLayout(comparison_buttons)
        layout.addWidget(self.state_label)
        layout.addStretch()

        self._update_controls()

    @property
    def seeds_complete(self) -> bool:
        """Whether the operator confirmed one vertex on each MIC arch."""

        return (
            self._maxillary_seed_vertex is not None
            and self._mandibular_seed_vertex is not None
        )

    def set_prerequisites(
        self,
        *,
        mic_available: bool,
        mounting_available: bool,
    ) -> None:
        """Protect diagnosis until both the record and RC mount exist."""

        self._mic_available = bool(mic_available)
        self._mounting_available = bool(mounting_available)
        ready = self._mic_available and self._mounting_available
        self.availability_label.setVisible(not ready)
        if not self._mic_available:
            self.availability_label.setText(
                "Cargue primero un registro MIC combinado."
            )
        elif not self._mounting_available:
            self.availability_label.setText(
                "Construya primero el montaje virtual en RC."
            )
        else:
            self.state_label.setText(
                "Seleccione manualmente una semilla sobre cada arcada MIC."
            )
            self._set_state_style(ready=False)
        if not ready:
            self.clear()
        self._update_controls()

    def show_seed(
        self,
        arch: str,
        vertex_index: int,
        point,
    ) -> None:
        """Display one operator-selected MIC seed vertex."""

        point = np.asarray(point, dtype=float)
        text = (
            f"Vértice {int(vertex_index):,} — "
            f"X {point[0]:.2f}, Y {point[1]:.2f}, Z {point[2]:.2f} mm"
        )
        if arch == "maxillary":
            self._maxillary_seed_vertex = int(vertex_index)
            self.maxillary_seed_value.setText(text)
        elif arch == "mandibular":
            self._mandibular_seed_vertex = int(vertex_index)
            self.mandibular_seed_value.setText(text)
        else:
            raise ValueError("Arch must be 'maxillary' or 'mandibular'.")

        if self.seeds_complete:
            self.state_label.setText(
                "Ambas arcadas están identificadas. El diagnóstico está listo."
            )
            self._set_state_style(ready=True)
        self._clear_results()
        self._update_controls()

    def show_running(self) -> None:
        """Report that registration is executing synchronously."""

        self.state_label.setText(
            "Separando y registrando las superficies MIC contra RC…"
        )
        self._set_state_style(ready=False)
        self.diagnose_button.setEnabled(False)

    def show_result(
        self,
        *,
        registration,
        right_vector,
        left_vector,
    ) -> None:
        """Display registration quality and local condylar vectors."""

        right_vector = np.asarray(right_vector, dtype=float)
        left_vector = np.asarray(left_vector, dtype=float)
        self.convergence_value.setText(
            "Convergente" if registration.converged else "No convergente"
        )
        self.maxillary_rmse_value.setText(
            f"{registration.maxillary_registration.root_mean_square_error:.6f} mm"
        )
        self.mandibular_rmse_value.setText(
            f"{registration.mandibular_registration.root_mean_square_error:.6f} mm"
        )
        self.right_distance_value.setText(
            f"{np.linalg.norm(right_vector):.4f} mm"
        )
        self.right_vector_value.setText(self._format_vector(right_vector))
        self.left_distance_value.setText(
            f"{np.linalg.norm(left_vector):.4f} mm"
        )
        self.left_vector_value.setText(self._format_vector(left_vector))
        self.state_label.setText(
            "Diagnóstico calculado en ejes anatómicos: "
            "+X derecha, +Y anterior, +Z superior."
        )
        self._set_state_style(ready=registration.converged)
        self.view_overlay_button.setChecked(True)
        self._update_controls()

    def show_error(self, message: str) -> None:
        """Expose one rejected separation or registration without stale data."""

        self._clear_results()
        self.state_label.setText(f"Diagnóstico no calculado — {message}")
        self._set_state_style(ready=False)
        self._update_controls()

    def clear(self) -> None:
        """Discard seeds and results while preserving prerequisite flags."""

        self._maxillary_seed_vertex = None
        self._mandibular_seed_vertex = None
        self.maxillary_seed_value.setText("Pendiente")
        self.mandibular_seed_value.setText("Pendiente")
        self._clear_results()
        if self._mic_available and self._mounting_available:
            self.state_label.setText(
                "Seleccione manualmente una semilla sobre cada arcada MIC."
            )
        else:
            self.state_label.setText(
                "El diagnóstico permanecerá protegido hasta confirmar "
                "sus referencias."
            )
        self._set_state_style(ready=False)
        self._update_controls()

    def _clear_results(self) -> None:
        self.convergence_value.setText("Pendiente")
        self.maxillary_rmse_value.setText("—")
        self.mandibular_rmse_value.setText("—")
        self.right_distance_value.setText("—")
        self.right_vector_value.setText("X —  |  Y —  |  Z —")
        self.left_distance_value.setText("—")
        self.left_vector_value.setText("X —  |  Y —  |  Z —")

    @property
    def diagnostic_view(self) -> str:
        """Return the operator-selected comparison mode."""

        if self.view_rc_button.isChecked():
            return "rc"
        if self.view_mic_button.isChecked():
            return "mic"
        return "overlay"

    def set_diagnostic_view(self, mode: str) -> None:
        """Synchronize the selected comparison button without emitting."""

        buttons = {
            "rc": self.view_rc_button,
            "mic": self.view_mic_button,
            "overlay": self.view_overlay_button,
        }
        try:
            buttons[mode].setChecked(True)
        except KeyError as error:
            raise ValueError("Diagnostic view must be RC, MIC, or overlay.") from error

    @staticmethod
    def _format_vector(vector: np.ndarray) -> str:
        return (
            f"X {vector[0]:+.4f}  |  "
            f"Y {vector[1]:+.4f}  |  "
            f"Z {vector[2]:+.4f} mm"
        )

    def _update_controls(self) -> None:
        ready = self._mic_available and self._mounting_available
        self.maxillary_seed_button.setEnabled(ready)
        self.mandibular_seed_button.setEnabled(ready)
        self.diagnose_button.setEnabled(ready and self.seeds_complete)
        result_available = self.convergence_value.text() != "Pendiente"
        self.view_rc_button.setEnabled(result_available)
        self.view_mic_button.setEnabled(result_available)
        self.view_overlay_button.setEnabled(result_available)
        self.reset_button.setEnabled(
            ready
            and (
                self._maxillary_seed_vertex is not None
                or self._mandibular_seed_vertex is not None
                or self.convergence_value.text() != "Pendiente"
            )
        )

    def _set_state_style(self, *, ready: bool) -> None:
        if ready:
            colors = "background: #dcefea; color: #24594e;"
        else:
            colors = "background: #e8eef0; color: #40545c;"
        self.state_label.setStyleSheet(
            colors + " padding: 8px; border-radius: 3px;"
        )
