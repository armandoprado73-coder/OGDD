"""Dialog for selecting the files that make up an OGDD study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


@dataclass(frozen=True)
class StudyFileSelection:
    """Paths selected for one RC–MIC study."""

    maxillary_rc: Path
    mandibular_rc: Path
    mic_record: Path | None = None


class StudyImportDialog(QDialog):
    """Collect the two RC models and an optional combined MIC record."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Importar archivos del estudio")
        self.setModal(True)
        self.setMinimumWidth(720)

        self._last_directory = Path.home()
        self._fields: dict[str, QLineEdit] = {}

        description = QLabel(
            "Seleccione los modelos montados en relación céntrica. "
            "El registro MIC combinado puede agregarse ahora o después."
        )
        description.setWordWrap(True)

        files_layout = QGridLayout()
        files_layout.setColumnStretch(1, 1)

        rows = (
            ("maxillary_rc", "Maxilar en RC *"),
            ("mandibular_rc", "Mandíbula en RC *"),
            ("mic_record", "Registro MIC combinado"),
        )
        for row, (key, label) in enumerate(rows):
            field = QLineEdit()
            field.setReadOnly(True)
            field.setPlaceholderText("Ningún archivo seleccionado")
            field.setClearButtonEnabled(True)
            self._fields[key] = field

            browse_button = QPushButton("Examinar…")
            browse_button.clicked.connect(
                lambda checked=False, field_key=key: (
                    self._browse(field_key)
                )
            )

            files_layout.addWidget(QLabel(label), row, 0)
            files_layout.addWidget(field, row, 1)
            files_layout.addWidget(browse_button, row, 2)

        note = QLabel("* Archivos obligatorios")
        note.setStyleSheet("color: #64777f;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open
            | QDialogButtonBox.StandardButton.Cancel
        )
        open_button = buttons.button(
            QDialogButtonBox.StandardButton.Open
        )
        open_button.setText("Cargar estudio")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(description)
        layout.addLayout(files_layout)
        layout.addWidget(note)
        layout.addStretch()
        layout.addWidget(buttons)

    @property
    def selection(self) -> StudyFileSelection:
        """Return the paths after the dialog has been accepted."""

        mic_text = self._fields["mic_record"].text()
        return StudyFileSelection(
            maxillary_rc=Path(self._fields["maxillary_rc"].text()),
            mandibular_rc=Path(
                self._fields["mandibular_rc"].text()
            ),
            mic_record=Path(mic_text) if mic_text else None,
        )

    def _browse(self, field_key: str) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo STL",
            str(self._last_directory),
            "Modelos STL (*.stl);;Todos los archivos (*)",
        )
        if not path_text:
            return

        selected_path = Path(path_text)
        self._last_directory = selected_path.parent
        self._fields[field_key].setText(str(selected_path))
        self._fields[field_key].setToolTip(str(selected_path))

    def _validate_and_accept(self) -> None:
        missing = [
            label
            for key, label in (
                ("maxillary_rc", "Maxilar en RC"),
                ("mandibular_rc", "Mandíbula en RC"),
            )
            if not self._fields[key].text()
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Faltan archivos",
                "Seleccione los archivos obligatorios:\n\n"
                + "\n".join(f"• {label}" for label in missing),
            )
            return

        self.accept()
