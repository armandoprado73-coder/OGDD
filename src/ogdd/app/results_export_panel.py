"""Point-7 controls for native study storage and result exports."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .results_export import StudyResultsSnapshot


class ResultsExportPanel(QWidget):
    """Summarize available results and request explicit file exports."""

    save_study_requested = Signal()
    export_pdf_requested = Signal()
    export_csv_requested = Signal()
    export_png_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._study_available = False
        self._results_available = False

        title = QLabel("Resultados y exportación")
        title.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #24484b;"
        )

        instructions = QLabel(
            "Guarde el estudio completo como .ogdd o exporte sus "
            "resultados clínicos en formatos independientes."
        )
        instructions.setWordWrap(True)

        self.availability_label = QLabel(
            "Importe primero los modelos del estudio."
        )
        self.availability_label.setWordWrap(True)
        self.availability_label.setStyleSheet(
            "background: #f0e6c9; color: #604e20; "
            "padding: 8px; border-radius: 3px;"
        )

        self.study_value = QLabel("Pendiente")
        self.mounting_value = QLabel("No construido")
        self.limits_value = QLabel("0 de 3")
        self.diagnosis_value = QLabel("No calculado")

        form = QFormLayout()
        form.setSpacing(9)
        form.addRow("Estudio", self.study_value)
        form.addRow("Montaje en RC", self.mounting_value)
        form.addRow("Límites funcionales", self.limits_value)
        form.addRow("Diagnóstico RC–MIC", self.diagnosis_value)

        self.save_study_button = QPushButton("Guardar estudio .ogdd")
        self.save_study_button.setStyleSheet(
            "QPushButton { background: #2a777c; color: white; "
            "font-weight: 600; padding: 8px; border-radius: 3px; }"
            "QPushButton:disabled { background: #9eb7b9; color: #eef5f5; }"
        )
        self.save_study_button.clicked.connect(
            lambda checked=False: self.save_study_requested.emit()
        )

        self.export_pdf_button = QPushButton("Exportar informe PDF")
        self.export_pdf_button.clicked.connect(
            lambda checked=False: self.export_pdf_requested.emit()
        )
        self.export_csv_button = QPushButton("Exportar datos CSV")
        self.export_csv_button.clicked.connect(
            lambda checked=False: self.export_csv_requested.emit()
        )
        self.export_png_button = QPushButton("Guardar captura PNG")
        self.export_png_button.clicked.connect(
            lambda checked=False: self.export_png_requested.emit()
        )

        self.state_label = QLabel(
            "El archivo .ogdd contendrá las mallas y el estado clínico."
        )
        self.state_label.setWordWrap(True)
        self.state_label.setStyleSheet(
            "background: #e8eef0; color: #40545c; "
            "padding: 8px; border-radius: 3px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(instructions)
        layout.addWidget(self.availability_label)
        layout.addLayout(form)
        layout.addWidget(self.save_study_button)
        layout.addWidget(self.export_pdf_button)
        layout.addWidget(self.export_csv_button)
        layout.addWidget(self.export_png_button)
        layout.addWidget(self.state_label)
        layout.addStretch()
        self._update_controls()

    def set_study_available(self, available: bool) -> None:
        """Protect every output until a study exists."""

        self._study_available = bool(available)
        self.availability_label.setVisible(not available)
        if not available:
            self.clear()
        self._update_controls()

    def show_snapshot(self, snapshot: StudyResultsSnapshot) -> None:
        """Display which parts of the study will be exported."""

        self._study_available = True
        self.study_value.setText(snapshot.study_name)
        self.mounting_value.setText(
            "Construido" if snapshot.mounting is not None else "No construido"
        )
        self.limits_value.setText(f"{len(snapshot.functional_limits)} de 3")
        self.diagnosis_value.setText(
            "Convergente"
            if snapshot.rc_mic is not None and snapshot.rc_mic["converged"]
            else "No calculado"
        )
        self._results_available = (
            snapshot.mounting is not None
            or bool(snapshot.functional_limits)
            or snapshot.rc_mic is not None
        )
        if self._results_available:
            self.state_label.setText(
                "Resultados disponibles para PDF y CSV. La captura PNG "
                "conservará la vista 3D actual."
            )
        else:
            self.state_label.setText(
                "El estudio puede guardarse. Complete el montaje para "
                "generar resultados clínicos."
            )
        self._update_controls()

    def clear(self) -> None:
        """Return the panel to its protected initial state."""

        self._results_available = False
        self.study_value.setText("Pendiente")
        self.mounting_value.setText("No construido")
        self.limits_value.setText("0 de 3")
        self.diagnosis_value.setText("No calculado")
        self.state_label.setText(
            "El archivo .ogdd contendrá las mallas y el estado clínico."
        )
        self._update_controls()

    def _update_controls(self) -> None:
        self.save_study_button.setEnabled(self._study_available)
        self.export_pdf_button.setEnabled(
            self._study_available and self._results_available
        )
        self.export_csv_button.setEnabled(
            self._study_available and self._results_available
        )
        self.export_png_button.setEnabled(self._study_available)
