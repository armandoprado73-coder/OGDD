"""Main window for the standalone OGDD application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QFileDialog,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.dental_model import DentalModel
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.mandibular_assembly import MandibularAssembly
from ogdd.articulator.bonwill_builder import BonwillBuilder
from ogdd.articulator.combined_controller import CombinedController
from ogdd.articulator.combined_movement import CombinedMovement
from ogdd.articulator.condylar_guide_builder import CondylarGuideBuilder
from ogdd.articulator.configuration import ArticulatorConfiguration
from ogdd.articulator.functional_calibration_controller import (
    FunctionalCalibrationController,
)
from ogdd.articulator.functional_limits import FunctionalLimitKind
from ogdd.articulator.guided_lateral_excursion import (
    GuidedLateralExcursion,
)
from ogdd.articulator.guided_protrusion import GuidedProtrusion
from ogdd.articulator.lateral_excursion import LateralSide
from ogdd.articulator.occlusal_closure import OcclusalClosure
from ogdd.articulator.occlusal_closure_controller import (
    OcclusalClosureController,
)
from ogdd.io.stl import STLReader
from ogdd.registration.centric_relation_registration import (
    CentricRelationRegistration,
)
from ogdd.registration.occlusal_record_builder import OcclusalRecordBuilder

from .functional_calibration_panel import FunctionalCalibrationPanel
from .mounting_panel import MountingPanel
from .orientation_panel import OrientationPanel
from .rc_mic_diagnosis_panel import RcMicDiagnosisPanel
from .results_export import ResultsExporter, StudyResultsSnapshot
from .results_export_panel import ResultsExportPanel
from .scene_view import SceneView
from .study_archive import StudyArchive
from .study_import_dialog import StudyFileSelection, StudyImportDialog


class MainWindow(QMainWindow):
    """OGDD shell: menus, study tree, 3D scene and clinical flow."""

    WINDOW_TITLE = "OGDD — Open Geometry for Digital Dentistry"
    REPOSITORY_URL = "https://github.com/armandoprado73-coder/OGDD"

    LAYERS = (
        ("maxillary_rc", "Maxilar en RC"),
        ("mandibular_rc", "Mandíbula en RC"),
        ("mic_record", "Registro MIC"),
        ("landmarks", "Landmarks"),
        ("balkwill", "Triángulo de Balkwill"),
        ("bonwill", "Triángulo de Bonwill"),
        ("virtual_condyles", "Cóndilos virtuales"),
        ("hinge_axis", "Eje de bisagra"),
        ("axes", "Ejes anatómicos"),
        ("condylar_guides", "Guías condilares"),
        ("functional_limits", "Límites funcionales"),
        ("mic_seeds", "Semillas del registro MIC"),
        ("mandibular_mic", "Mandíbula en MIC"),
        ("condylar_displacement", "Desplazamiento condilar RC–MIC"),
    )

    CLINICAL_STEPS = (
        "1  Archivos del estudio",
        "2  Orientación anatómica",
        "3  Montaje en RC",
        "4  Calibración funcional",
        "5  Registro MIC",
        "6  Diagnóstico RC–MIC",
        "7  Resultados y exportación",
    )

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self.setDockNestingEnabled(True)

        self._study_files: StudyFileSelection | None = None
        self._study_meshes: dict[str, Any] = {}
        self._landmark_points: dict[str, np.ndarray] = {}
        self._coordinate_system = None
        self._visibility_before_pick: dict[str, bool] = {}
        self._articulator_configuration = None
        self._bonwill = None
        self._hinge_axis = None
        self._condylar_guides = None
        self._functional_calibration_controller = None
        self._mic_seed_vertices: dict[str, int] = {}
        self._rc_mic_registration = None
        self._condylar_displacement = None
        self._study_path: Path | None = None
        self._study_dirty = False
        self._restoring_study = False

        self.scene = SceneView(self)
        self.setCentralWidget(self.scene)

        self._create_study_dock()
        self._create_workflow_dock()
        self._create_actions()
        self._create_menus()
        self._create_status_bar()

    def _create_study_dock(self) -> None:
        self.study_dock = QDockWidget("Estudio", self)
        self.study_dock.setObjectName("study_dock")
        self.study_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.study_dock.setMinimumWidth(235)

        self.study_tree = QTreeWidget()
        self.study_tree.setHeaderHidden(True)
        self.study_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.study_sections = {
            key: QTreeWidgetItem(self.study_tree, [label])
            for key, label in (
                ("models", "Modelos"),
                ("records", "Registros"),
                ("landmarks", "Referencias anatómicas"),
                ("articulator", "Articulador"),
                ("results", "Resultados"),
            )
        }

        self.study_dock.setWidget(self.study_tree)
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.study_dock,
        )

    def _create_workflow_dock(self) -> None:
        self.workflow_dock = QDockWidget("Flujo clínico", self)
        self.workflow_dock.setObjectName("workflow_dock")
        self.workflow_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.workflow_dock.setMinimumWidth(285)

        self.workflow_list = QListWidget()
        self.workflow_list.setSpacing(2)
        self.workflow_list.setMaximumHeight(245)
        for index, label in enumerate(self.CLINICAL_STEPS):
            item = QListWidgetItem(label)
            item.setToolTip("Paso disponible próximamente")
            self.workflow_list.addItem(item)
            if index == 0:
                self.workflow_list.setCurrentItem(item)

        self.orientation_panel = OrientationPanel()
        self.orientation_panel.setVisible(False)
        self.orientation_panel.pick_requested.connect(
            self._start_landmark_pick
        )
        self.orientation_panel.confirm_requested.connect(
            self._confirm_orientation
        )
        self.orientation_panel.reset_requested.connect(
            self._reset_orientation
        )

        self.mounting_panel = MountingPanel()
        self.mounting_panel.setVisible(False)
        self.mounting_panel.build_requested.connect(
            self._build_rc_mounting
        )
        self.mounting_panel.reset_requested.connect(
            self._clear_mounting_state
        )

        self.functional_calibration_panel = FunctionalCalibrationPanel()
        self.functional_calibration_panel.setVisible(False)
        self.functional_calibration_scroll = QScrollArea()
        self.functional_calibration_scroll.setWidgetResizable(True)
        self.functional_calibration_scroll.setFrameShape(
            QFrame.Shape.NoFrame
        )
        self.functional_calibration_scroll.setWidget(
            self.functional_calibration_panel
        )
        self.functional_calibration_scroll.setVisible(False)
        self.functional_calibration_panel.open_requested.connect(
            self._open_mandible
        )
        self.functional_calibration_panel.close_requested.connect(
            self._close_mandible
        )
        self.functional_calibration_panel.advance_requested.connect(
            self._advance_mandible
        )
        self.functional_calibration_panel.retreat_requested.connect(
            self._retreat_mandible
        )
        self.functional_calibration_panel.move_left_requested.connect(
            self._move_mandible_left
        )
        self.functional_calibration_panel.move_right_requested.connect(
            self._move_mandible_right
        )
        self.functional_calibration_panel.lateral_zero_requested.connect(
            self._zero_lateral
        )
        self.functional_calibration_panel.opening_changed.connect(
            self._set_mandibular_opening
        )
        self.functional_calibration_panel.protrusion_changed.connect(
            self._set_mandibular_protrusion
        )
        self.functional_calibration_panel.lateral_changed.connect(
            self._set_mandibular_lateral
        )
        self.functional_calibration_panel.adjust_close_requested.connect(
            self._adjust_occlusion_close
        )
        self.functional_calibration_panel.adjust_open_requested.connect(
            self._adjust_occlusion_open
        )
        self.functional_calibration_panel.adjustment_zero_requested.connect(
            self._reset_occlusal_adjustment
        )
        self.functional_calibration_panel.save_protrusive_requested.connect(
            self._save_protrusive_limit
        )
        self.functional_calibration_panel.save_right_canine_requested.connect(
            self._save_right_canine_limit
        )
        self.functional_calibration_panel.save_left_canine_requested.connect(
            self._save_left_canine_limit
        )
        self.functional_calibration_panel.go_protrusive_requested.connect(
            self._go_to_protrusive_limit
        )
        self.functional_calibration_panel.go_right_canine_requested.connect(
            self._go_to_right_canine_limit
        )
        self.functional_calibration_panel.go_left_canine_requested.connect(
            self._go_to_left_canine_limit
        )
        self.functional_calibration_panel.clear_limits_requested.connect(
            self._clear_functional_limits
        )
        self.functional_calibration_panel.rc_requested.connect(
            self._return_to_rc
        )

        self.rc_mic_diagnosis_panel = RcMicDiagnosisPanel()
        self.rc_mic_diagnosis_panel.setVisible(False)
        self.rc_mic_diagnosis_scroll = QScrollArea()
        self.rc_mic_diagnosis_scroll.setWidgetResizable(True)
        self.rc_mic_diagnosis_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.rc_mic_diagnosis_scroll.setWidget(
            self.rc_mic_diagnosis_panel
        )
        self.rc_mic_diagnosis_scroll.setVisible(False)
        self.rc_mic_diagnosis_panel.select_maxillary_seed_requested.connect(
            lambda: self._start_mic_seed_pick("maxillary")
        )
        self.rc_mic_diagnosis_panel.select_mandibular_seed_requested.connect(
            lambda: self._start_mic_seed_pick("mandibular")
        )
        self.rc_mic_diagnosis_panel.diagnose_requested.connect(
            self._diagnose_rc_mic
        )
        self.rc_mic_diagnosis_panel.reset_requested.connect(
            self._reset_rc_mic_diagnosis
        )
        self.rc_mic_diagnosis_panel.view_rc_requested.connect(
            lambda: self._set_rc_mic_diagnostic_view("rc")
        )
        self.rc_mic_diagnosis_panel.view_mic_requested.connect(
            lambda: self._set_rc_mic_diagnostic_view("mic")
        )
        self.rc_mic_diagnosis_panel.view_overlay_requested.connect(
            lambda: self._set_rc_mic_diagnostic_view("overlay")
        )

        self.results_export_panel = ResultsExportPanel()
        self.results_export_panel.setVisible(False)
        self.results_export_scroll = QScrollArea()
        self.results_export_scroll.setWidgetResizable(True)
        self.results_export_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.results_export_scroll.setWidget(self.results_export_panel)
        self.results_export_scroll.setVisible(False)
        self.results_export_panel.save_study_requested.connect(
            self._save_study_as
        )
        self.results_export_panel.export_pdf_requested.connect(
            self._export_results_pdf
        )
        self.results_export_panel.export_csv_requested.connect(
            self._export_results_csv
        )
        self.results_export_panel.export_png_requested.connect(
            self._export_scene_png
        )

        self.workflow_message = QLabel(
            "Importe los modelos y el registro que forman el estudio."
        )
        self.workflow_message.setWordWrap(True)
        self.workflow_message.setStyleSheet(
            "color: #52666e; padding: 10px;"
        )

        workflow_widget = QWidget()
        workflow_layout = QVBoxLayout(workflow_widget)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.addWidget(self.workflow_list)
        workflow_layout.addWidget(self.workflow_message)
        workflow_layout.addWidget(self.orientation_panel, 1)
        workflow_layout.addWidget(self.mounting_panel, 1)
        workflow_layout.addWidget(self.functional_calibration_scroll, 1)
        workflow_layout.addWidget(self.rc_mic_diagnosis_scroll, 1)
        workflow_layout.addWidget(self.results_export_scroll, 1)

        self.workflow_list.currentRowChanged.connect(
            self._workflow_step_changed
        )
        self.workflow_dock.setWidget(workflow_widget)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.workflow_dock,
        )

    def _action(
        self,
        text: str,
        slot: Callable[..., Any] | None = None,
        *,
        shortcut: QKeySequence.StandardKey | str | None = None,
        status_tip: str = "",
        enabled: bool = True,
        checkable: bool = False,
        checked: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        if slot is not None:
            action.triggered.connect(slot)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.setStatusTip(status_tip)
        action.setEnabled(enabled)
        action.setCheckable(checkable)
        action.setChecked(checked)
        return action

    def _create_actions(self) -> None:
        self.new_action = self._action(
            "&Nuevo estudio",
            self._new_study,
            shortcut=QKeySequence.StandardKey.New,
            status_tip="Crear un estudio de OGDD",
        )
        self.open_action = self._action(
            "&Abrir estudio…",
            self._open_study,
            shortcut=QKeySequence.StandardKey.Open,
            status_tip="Abrir un estudio existente",
        )
        self.save_action = self._action(
            "&Guardar",
            self._save_study,
            shortcut=QKeySequence.StandardKey.Save,
            status_tip="Guardar el estudio actual",
            enabled=False,
        )
        self.save_as_action = self._action(
            "Guardar &como…",
            self._save_study_as,
            shortcut=QKeySequence.StandardKey.SaveAs,
            enabled=False,
        )
        self.import_action = self._action(
            "&Importar modelos…",
            self._import_study_files,
            shortcut="Ctrl+I",
            status_tip="Cargar modelos y registros dentales",
        )
        self.export_action = self._action(
            "&Exportar resultados…",
            lambda: self.workflow_list.setCurrentRow(6),
            enabled=False,
        )
        self.exit_action = self._action(
            "&Salir",
            self.close,
            shortcut=QKeySequence.StandardKey.Quit,
        )

        self.undo_action = self._action(
            "&Deshacer",
            shortcut=QKeySequence.StandardKey.Undo,
            enabled=False,
        )
        self.redo_action = self._action(
            "&Rehacer",
            shortcut=QKeySequence.StandardKey.Redo,
            enabled=False,
        )
        self.clear_selection_action = self._action(
            "Eliminar selección",
            enabled=False,
        )
        self.repeat_selection_action = self._action(
            "Repetir selección",
            enabled=False,
        )
        self.reset_step_action = self._action(
            "Restablecer paso actual",
            enabled=False,
        )

        view_labels = (
            ("front", "Frontal"),
            ("back", "Posterior"),
            ("right", "Derecha"),
            ("left", "Izquierda"),
            ("top", "Superior"),
            ("bottom", "Inferior"),
        )
        self.camera_actions = {
            name: self._action(
                label,
                lambda checked=False, key=name: (
                    self.scene.set_anatomical_view(key)
                ),
            )
            for name, label in view_labels
        }
        self.reset_camera_action = self._action(
            "Centrar modelos",
            self.scene.reset_camera,
            shortcut="Home",
        )

        self.layer_actions = {}
        for key, label in self.LAYERS:
            action = self._action(
                label,
                lambda checked, layer=key: (
                    self.scene.set_layer_visible(layer, checked)
                ),
                checkable=True,
                checked=False,
                enabled=False,
            )
            self.layer_actions[key] = action

        self.validate_action = self._action(
            "Validar archivos del estudio",
            self._show_next_step,
        )
        self.measure_action = self._action(
            "Medir distancia",
            enabled=False,
        )
        self.capture_action = self._action(
            "Capturar imagen",
            self._export_scene_png,
            enabled=False,
        )
        self.preferences_action = self._action(
            "Preferencias…",
            enabled=False,
        )

        self.repository_action = self._action(
            "Repositorio de OGDD",
            lambda: QDesktopServices.openUrl(QUrl(self.REPOSITORY_URL)),
        )
        self.about_action = self._action(
            "Acerca de OGDD",
            self._show_about,
        )

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&Archivo")
        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("&Editar")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.clear_selection_action)
        edit_menu.addAction(self.repeat_selection_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.reset_step_action)

        view_menu = self.menuBar().addMenu("&Ver")
        cameras_menu = view_menu.addMenu("Vistas anatómicas")
        for action in self.camera_actions.values():
            cameras_menu.addAction(action)
        cameras_menu.addSeparator()
        cameras_menu.addAction(self.reset_camera_action)

        layers_menu = view_menu.addMenu("Capas")
        for action in self.layer_actions.values():
            layers_menu.addAction(action)

        panels_menu = view_menu.addMenu("Paneles")
        panels_menu.addAction(self.study_dock.toggleViewAction())
        panels_menu.addAction(self.workflow_dock.toggleViewAction())

        tools_menu = self.menuBar().addMenu("&Herramientas")
        tools_menu.addAction(self.validate_action)
        tools_menu.addAction(self.measure_action)
        tools_menu.addAction(self.capture_action)
        tools_menu.addSeparator()
        tools_menu.addAction(self.preferences_action)

        help_menu = self.menuBar().addMenu("A&yuda")
        help_menu.addAction(self.repository_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)

    def _create_status_bar(self) -> None:
        self.statusBar().showMessage(
            "Listo — cree, abra o importe un estudio"
        )
        units = QLabel("mm  |  grados")
        units.setContentsMargins(8, 0, 8, 0)
        self.statusBar().addPermanentWidget(units)

    def _show_next_step(self) -> None:
        self.statusBar().showMessage(
            "La gestión de estudios se conectará en el siguiente bloque.",
            5000,
        )

    def _study_display_name(self) -> str:
        """Return the portable study name used by panels and exports."""

        if self._study_path is not None:
            return self._study_path.stem
        if self._study_files is not None:
            return self._study_files.mandibular_rc.stem
        return "estudio_ogdd"

    def _set_study_actions_enabled(self, enabled: bool) -> None:
        """Synchronize every action that requires a loaded study."""

        self.save_action.setEnabled(enabled)
        self.save_as_action.setEnabled(enabled)
        self.export_action.setEnabled(enabled)
        self.capture_action.setEnabled(enabled)
        self.results_export_panel.set_study_available(enabled)

    def _mark_study_dirty(self) -> None:
        """Record one user-visible state change outside archive restoration."""

        if self._restoring_study or self._study_files is None:
            return
        self._study_dirty = True
        self._update_window_title()

    def _state_changed(self) -> None:
        """Mark and redisplay one accepted clinical state change."""

        self._mark_study_dirty()
        self._refresh_results_panel()

    def _mark_study_clean(self, path: Path | None = None) -> None:
        """Record that the current state is safely represented on disk."""

        if path is not None:
            self._study_path = path
        self._study_dirty = False
        self._update_window_title()

    def _update_window_title(self) -> None:
        """Show the native filename and unsaved-change marker."""

        if self._study_path is None:
            name = "Sin guardar" if self._study_files is not None else ""
        else:
            name = self._study_path.name
        dirty = " *" if self._study_dirty else ""
        prefix = f"{name}{dirty} — " if name else ""
        self.setWindowTitle(prefix + self.WINDOW_TITLE)

    def _confirm_discard_changes(self, action: str) -> bool:
        """Protect unsaved studies before replacing or closing them."""

        if not self._study_dirty:
            return True
        answer = QMessageBox.warning(
            self,
            "Cambios sin guardar",
            f"El estudio tiene cambios sin guardar antes de {action}.",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save_study()
        return answer == QMessageBox.StandardButton.Discard

    def _study_filenames(self) -> dict[str, str]:
        """Return source basenames without persisting machine-specific paths."""

        if self._study_files is None:
            return {}
        filenames = {
            "maxillary_rc": self._study_files.maxillary_rc.name,
            "mandibular_rc": self._study_files.mandibular_rc.name,
        }
        if self._study_files.mic_record is not None:
            filenames["mic_record"] = self._study_files.mic_record.name
        return filenames

    def _study_state(self) -> dict[str, Any]:
        """Capture the reproducible numeric state stored inside ``.ogdd``."""

        configuration = self._articulator_configuration
        mounting = None
        if configuration is not None:
            mounting = {
                "intercondylar_width": configuration.intercondylar_width,
                "balkwill_angle_degrees": (
                    configuration.balkwill_angle_degrees
                ),
                "right_condylar_guidance_degrees": (
                    configuration.right_condylar_guidance_degrees
                ),
                "left_condylar_guidance_degrees": (
                    configuration.left_condylar_guidance_degrees
                ),
            }

        controller = self._functional_calibration_controller
        limits = []
        position = None
        if controller is not None:
            limits = [
                {
                    "kind": limit.kind.value,
                    "base_opening_angle_degrees": (
                        limit.base_opening_angle_degrees
                    ),
                    "adjustment_angle_degrees": (
                        limit.adjustment_angle_degrees
                    ),
                    "total_opening_angle_degrees": (
                        limit.total_opening_angle_degrees
                    ),
                    "lateral_angle_degrees": limit.lateral_angle_degrees,
                    "protrusion_distance_mm": limit.protrusion_distance_mm,
                    "working_side": (
                        None
                        if limit.working_side is None
                        else limit.working_side.value
                    ),
                }
                for limit in controller.limits.values
            ]
            position = {
                "opening_angle_degrees": controller.opening_angle_degrees,
                "lateral_angle_degrees": controller.lateral_angle_degrees,
                "protrusion_distance_mm": controller.protrusion_distance_mm,
                "adjustment_angle_degrees": controller.adjustment_angle_degrees,
            }

        return {
            "landmarks": {
                name: point.tolist()
                for name, point in self._landmark_points.items()
            },
            "mounting": mounting,
            "functional_limits": limits,
            "current_position": position,
            "mic_seed_vertices": dict(self._mic_seed_vertices),
            "diagnosis_calculated": self._rc_mic_registration is not None,
            "diagnostic_view": (
                self.rc_mic_diagnosis_panel.diagnostic_view
            ),
            "workflow_step": self.workflow_list.currentRow(),
            "visible_layers": {
                name: action.isChecked()
                for name, action in self.layer_actions.items()
                if action.isEnabled()
            },
        }

    def _save_study(self) -> bool:
        """Save to the current native path or request one when necessary."""

        if self._study_files is None:
            return False
        if self._study_path is None:
            return self._save_study_as()
        return self._write_study(self._study_path)

    def _save_study_as(self) -> bool:
        """Request and save one native ``.ogdd`` archive path."""

        if self._study_files is None:
            return False
        suggested = self._study_path or Path.cwd() / (
            f"{self._study_display_name()}.ogdd"
        )
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar estudio OGDD",
            str(suggested),
            "Estudios OGDD (*.ogdd)",
        )
        if not path_text:
            return False
        path = Path(path_text)
        if path.suffix.lower() != ".ogdd":
            path = path.with_suffix(".ogdd")
        return self._write_study(path)

    def _write_study(self, path: Path) -> bool:
        """Serialize the complete current study and report failures safely."""

        self.statusBar().showMessage("Guardando estudio OGDD…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            StudyArchive.save(
                path,
                meshes=self._study_meshes,
                filenames=self._study_filenames(),
                state=self._study_state(),
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "No fue posible guardar el estudio",
                f"{type(error).__name__}: {error}",
            )
            self.statusBar().showMessage("El estudio no fue guardado.")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._mark_study_clean(path)
        self._refresh_results_panel()
        self.statusBar().showMessage(f"Estudio guardado — {path.name}")
        return True

    def _open_study(self) -> None:
        """Load a complete native study after explicit file selection."""

        if not self._confirm_discard_changes("abrir otro estudio"):
            return
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir estudio OGDD",
            str(self._study_path.parent if self._study_path else Path.cwd()),
            "Estudios OGDD (*.ogdd)",
        )
        if not path_text:
            return
        path = Path(path_text)
        self.statusBar().showMessage("Abriendo estudio OGDD…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            archive = StudyArchive.load(path)
            self._restore_study(archive, path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "No fue posible abrir el estudio",
                f"{type(error).__name__}: {error}",
            )
            self.statusBar().showMessage("La apertura fue cancelada.")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"Estudio abierto — {path.name}")

    def _restore_study(self, archive, path: Path) -> None:
        """Reconstruct all saved geometry and operator-confirmed state."""

        filenames = archive.filenames
        selection = StudyFileSelection(
            maxillary_rc=Path(
                filenames.get("maxillary_rc", "maxillary_rc.stl")
            ),
            mandibular_rc=Path(
                filenames.get("mandibular_rc", "mandibular_rc.stl")
            ),
            mic_record=(
                None
                if "mic_record" not in archive.meshes
                else Path(filenames.get("mic_record", "mic_record.stl"))
            ),
        )
        self._restoring_study = True
        try:
            self._display_loaded_study(selection, archive.meshes)
            state = archive.state
            landmarks = state.get("landmarks", {})
            if landmarks:
                self._landmark_points = {
                    name: np.asarray(point, dtype=float)
                    for name, point in landmarks.items()
                }
                self._confirm_orientation()

            mounting = state.get("mounting")
            if mounting is not None:
                configuration = ArticulatorConfiguration(
                    intercondylar_width=mounting["intercondylar_width"],
                    balkwill_angle_degrees=(
                        mounting["balkwill_angle_degrees"]
                    ),
                    right_condylar_guidance_degrees=(
                        mounting["right_condylar_guidance_degrees"]
                    ),
                    left_condylar_guidance_degrees=(
                        mounting["left_condylar_guidance_degrees"]
                    ),
                )
                self.mounting_panel.set_configuration(configuration)
                self._build_rc_mounting()
                self._restore_functional_state(state)
                self._restore_mic_diagnosis(state)
                if self._rc_mic_registration is not None:
                    self._set_rc_mic_diagnostic_view(
                        state.get("diagnostic_view", "overlay")
                    )

            visible_layers = state.get("visible_layers", {})
            for name, visible in visible_layers.items():
                action = self.layer_actions.get(name)
                if action is not None and action.isEnabled():
                    self._set_layer_visibility(name, bool(visible))
            self.workflow_list.setCurrentRow(
                int(state.get("workflow_step", 0))
            )
            self._study_path = path
        finally:
            self._restoring_study = False
        self._set_study_actions_enabled(True)
        self._mark_study_clean(path)
        self._refresh_results_panel()

    def _restore_functional_state(self, state: dict[str, Any]) -> None:
        """Recreate saved endpoints and the current mandibular position."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        controller.clear_limits()
        save_actions = {
            FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE.value: (
                controller.save_protrusive_limit
            ),
            FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP.value: (
                controller.save_right_canine_limit
            ),
            FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP.value: (
                controller.save_left_canine_limit
            ),
        }
        for saved in state.get("functional_limits", []):
            controller.reset_movement()
            controller.reset_adjustment()
            controller.set_position(
                opening_angle_degrees=saved["base_opening_angle_degrees"],
                lateral_angle_degrees=saved["lateral_angle_degrees"],
                protrusion_distance_mm=saved["protrusion_distance_mm"],
            )
            controller.set_adjustment(saved["adjustment_angle_degrees"])
            save_actions[saved["kind"]]()

        self._apply_saved_position(state.get("current_position"))

    def _apply_saved_position(self, saved: dict[str, Any] | None) -> None:
        """Restore one complete operator position after other reconstruction."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        controller.reset_movement()
        controller.reset_adjustment()
        if saved is not None:
            controller.set_position(
                opening_angle_degrees=saved["opening_angle_degrees"],
                lateral_angle_degrees=saved["lateral_angle_degrees"],
                protrusion_distance_mm=saved["protrusion_distance_mm"],
            )
            controller.set_adjustment(saved["adjustment_angle_degrees"])
        self._show_functional_position(controller.position)

    def _restore_mic_diagnosis(self, state: dict[str, Any]) -> None:
        """Restore manual MIC seeds and deterministically recalculate diagnosis."""

        seeds = state.get("mic_seed_vertices", {})
        if not seeds or "mic_record" not in self._study_meshes:
            return
        mic_mesh = self._study_meshes["mic_record"]
        coordinate_system = self._coordinate_system
        if coordinate_system is None:
            return
        for arch in ("maxillary", "mandibular"):
            if arch not in seeds:
                continue
            index = int(seeds[arch])
            if index < 0 or index >= mic_mesh.vertex_count:
                raise ValueError(f"Saved MIC {arch} seed is outside the mesh.")
            self._mic_seed_vertices[arch] = index
            local_point = coordinate_system.to_local(mic_mesh.vertices[index])
            self.rc_mic_diagnosis_panel.show_seed(arch, index, local_point)
        selected_points = {
            arch: coordinate_system.to_local(mic_mesh.vertices[index])
            for arch, index in self._mic_seed_vertices.items()
        }
        self.scene.show_mic_seeds(selected_points)
        self.layer_actions["mic_seeds"].setEnabled(True)
        self.layer_actions["mic_seeds"].setChecked(True)
        if state.get("diagnosis_calculated"):
            current_position = state.get("current_position")
            self._diagnose_rc_mic()
            self._apply_saved_position(current_position)

    def _new_study(self) -> None:
        """Clear the currently loaded study after user confirmation."""

        if not self._confirm_discard_changes("crear un estudio nuevo"):
            return

        for layer_name in (
            "maxillary_rc",
            "mandibular_rc",
            "mic_record",
            "mic_seeds",
            "mandibular_mic",
            "condylar_displacement",
        ):
            self.scene.clear_layer(layer_name, render=False)
            action = self.layer_actions[layer_name]
            action.setChecked(False)
            action.setEnabled(False)

        for section_name in ("models", "records"):
            section = self.study_sections[section_name]
            section.takeChildren()

        self._study_files = None
        self._study_meshes.clear()
        self._study_path = None
        self._study_dirty = False
        self._clear_anatomical_state()
        self.orientation_panel.set_study_available(False)
        self._set_study_actions_enabled(False)
        self._update_window_title()
        self.scene.show_welcome()
        self.scene.plotter.render()
        self.statusBar().showMessage(
            "Nuevo estudio — importe los modelos en RC"
        )

    def _import_study_files(self) -> None:
        """Select, read and display the files of an RC–MIC study."""

        if not self._confirm_discard_changes("importar otro estudio"):
            return

        dialog = StudyImportDialog(self)
        if dialog.exec() != StudyImportDialog.DialogCode.Accepted:
            return

        selection = dialog.selection
        paths = {
            "maxillary_rc": selection.maxillary_rc,
            "mandibular_rc": selection.mandibular_rc,
        }
        if selection.mic_record is not None:
            paths["mic_record"] = selection.mic_record

        self.statusBar().showMessage("Cargando archivos del estudio…")
        QApplication.setOverrideCursor(
            Qt.CursorShape.WaitCursor
        )
        QApplication.processEvents()
        try:
            loaded_meshes = {
                layer_name: STLReader.read(path)
                for layer_name, path in paths.items()
            }
        except Exception as error:
            QMessageBox.critical(
                self,
                "No fue posible cargar el estudio",
                "OGDD no pudo leer uno de los archivos seleccionados.\n\n"
                f"{type(error).__name__}: {error}",
            )
            self.statusBar().showMessage(
                "La carga fue cancelada; el estudio anterior no cambió."
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._display_loaded_study(selection, loaded_meshes)

    def _display_loaded_study(
        self,
        selection: StudyFileSelection,
        meshes: dict[str, Any],
    ) -> None:
        """Commit validated meshes to the scene and study tree."""

        visual_styles = {
            "maxillary_rc": {
                "color": "mistyrose",
                "opacity": 0.72,
                "visible": True,
            },
            "mandibular_rc": {
                "color": "lightblue",
                "opacity": 1.0,
                "visible": True,
            },
            "mic_record": {
                "color": "orange",
                "opacity": 0.42,
                "visible": False,
            },
        }
        for layer_name, mesh in meshes.items():
            style = visual_styles[layer_name]
            self.scene.add_dental_mesh(
                layer_name,
                mesh,
                color=style["color"],
                opacity=style["opacity"],
                visible=style["visible"],
            )
            action = self.layer_actions[layer_name]
            action.setEnabled(True)
            action.setChecked(style["visible"])

        if "mic_record" not in meshes:
            self.scene.clear_layer("mic_record", render=False)
            self.layer_actions["mic_record"].setChecked(False)
            self.layer_actions["mic_record"].setEnabled(False)

        self._study_files = selection
        self._study_meshes = meshes
        self._reset_rc_mic_diagnosis(mark_dirty=False)
        self._clear_anatomical_state()
        self.orientation_panel.set_study_available(True)
        self._update_study_tree(selection, meshes)
        self.scene.finish_study_load()
        self.workflow_list.setCurrentRow(0)
        self._set_study_actions_enabled(True)
        if not self._restoring_study:
            self._study_path = None
            self._mark_study_dirty()
        self._refresh_results_panel()
        self.statusBar().showMessage(
            f"Estudio cargado — {len(meshes)} archivos"
        )

    def _update_study_tree(
        self,
        selection: StudyFileSelection,
        meshes: dict[str, Any],
    ) -> None:
        """Describe loaded files and mesh sizes in the study panel."""

        models_section = self.study_sections["models"]
        records_section = self.study_sections["records"]
        models_section.takeChildren()
        records_section.takeChildren()

        entries = (
            (
                "maxillary_rc",
                "Maxilar RC",
                selection.maxillary_rc,
                models_section,
            ),
            (
                "mandibular_rc",
                "Mandíbula RC",
                selection.mandibular_rc,
                models_section,
            ),
            (
                "mic_record",
                "Registro MIC",
                selection.mic_record,
                records_section,
            ),
        )
        for key, label, path, parent in entries:
            if path is None or key not in meshes:
                continue
            mesh = meshes[key]
            item = QTreeWidgetItem(parent, [f"{label} — {path.name}"])
            item.setToolTip(
                0,
                f"{path}\n"
                f"Vértices: {mesh.vertex_count:,}\n"
                f"Caras: {len(mesh.faces):,}",
            )

        models_section.setExpanded(True)
        records_section.setExpanded(True)

    def _workflow_step_changed(self, row: int) -> None:
        """Show contextual controls for the selected clinical step."""

        orientation_selected = row == 1
        mounting_selected = row == 2
        calibration_selected = row == 3
        diagnosis_selected = row == 5
        results_selected = row == 6
        self.orientation_panel.setVisible(orientation_selected)
        self.mounting_panel.setVisible(mounting_selected)
        self.functional_calibration_panel.setVisible(calibration_selected)
        self.functional_calibration_scroll.setVisible(calibration_selected)
        self.rc_mic_diagnosis_panel.setVisible(diagnosis_selected)
        self.rc_mic_diagnosis_scroll.setVisible(diagnosis_selected)
        self.results_export_panel.setVisible(results_selected)
        self.results_export_scroll.setVisible(results_selected)
        self.workflow_message.setVisible(
            not orientation_selected
            and not mounting_selected
            and not calibration_selected
            and not diagnosis_selected
            and not results_selected
        )

        messages = {
            0: "Importe los modelos y el registro que forman el estudio.",
            2: "El montaje en RC se habilitará después de la orientación.",
            3: "La calibración funcional se conectará al articulador.",
            4: "El registro MIC se separará y registrará en este paso.",
            5: "Aquí aparecerá el diagnóstico de desplazamiento RC–MIC.",
            6: "Los resultados podrán revisarse y exportarse aquí.",
        }
        if (
            not orientation_selected
            and not mounting_selected
            and not calibration_selected
            and not diagnosis_selected
            and not results_selected
        ):
            self.workflow_message.setText(
                messages.get(row, "Paso clínico en preparación.")
            )
        if results_selected:
            self._refresh_results_panel()
        if not orientation_selected:
            self._finish_landmark_pick()

    def _start_landmark_pick(self, landmark_name: str) -> None:
        """Enter one-shot picking mode for a mandibular landmark."""

        if "mandibular_rc" not in self._study_meshes:
            QMessageBox.information(
                self,
                "Mandíbula no disponible",
                "Cargue primero los modelos del estudio.",
            )
            return

        self._finish_landmark_pick()
        self._visibility_before_pick = {
            layer: self.layer_actions[layer].isChecked()
            for layer in (
                "maxillary_rc",
                "mandibular_rc",
                "mic_record",
            )
        }
        self._set_layer_visibility("maxillary_rc", False)
        self._set_layer_visibility("mic_record", False)
        self._set_layer_visibility("mandibular_rc", True)

        for layer in (
            "maxillary_rc",
            "mandibular_rc",
            "mic_record",
            "landmarks",
            "balkwill",
            "bonwill",
            "virtual_condyles",
            "hinge_axis",
            "axes",
            "condylar_guides",
            "mic_seeds",
            "mandibular_mic",
            "condylar_displacement",
        ):
            self.scene.set_layer_pickable(
                layer,
                layer == "mandibular_rc",
            )

        self.orientation_panel.set_active_landmark(landmark_name)
        self.statusBar().showMessage(
            "Selección activa — haga clic izquierdo sobre la mandíbula"
        )
        self.scene.enable_surface_pick(
            lambda point, name=landmark_name: (
                self._landmark_picked(name, point)
            )
        )

    def _landmark_picked(
        self,
        landmark_name: str,
        displayed_point,
    ) -> None:
        """Store a picked point in original scanner coordinates."""

        point = np.asarray(displayed_point, dtype=float)
        if point.shape != (3,):
            return

        if self._coordinate_system is None:
            world_point = point
        else:
            world_point = self._coordinate_system.to_world(point)

        self._landmark_points[landmark_name] = world_point
        self.orientation_panel.set_landmark(
            landmark_name,
            world_point,
        )
        self._refresh_landmark_visuals()
        self._finish_landmark_pick()
        self.statusBar().showMessage(
            "Punto guardado — puede seleccionarlo nuevamente para corregirlo"
        )
        self._state_changed()

    def _finish_landmark_pick(self) -> None:
        """Leave picking mode and restore previous layer visibility."""

        self.scene.disable_surface_pick()
        for layer in (
            "maxillary_rc",
            "mandibular_rc",
            "mic_record",
            "landmarks",
            "balkwill",
            "bonwill",
            "virtual_condyles",
            "hinge_axis",
            "axes",
            "condylar_guides",
            "mic_seeds",
            "mandibular_mic",
            "condylar_displacement",
        ):
            self.scene.set_layer_pickable(layer, True)

        for layer, visible in self._visibility_before_pick.items():
            self._set_layer_visibility(layer, visible)
        self._visibility_before_pick.clear()

    def _start_mic_seed_pick(self, arch: str) -> None:
        """Request one maxillary or mandibular seed on the combined MIC mesh."""

        if arch not in ("maxillary", "mandibular"):
            raise ValueError("MIC seed arch must be maxillary or mandibular.")
        if "mic_record" not in self._study_meshes:
            QMessageBox.information(
                self,
                "Registro MIC no disponible",
                "Cargue primero un registro MIC combinado.",
            )
            return
        if self._coordinate_system is None or self._hinge_axis is None:
            QMessageBox.information(
                self,
                "Montaje en RC pendiente",
                "Confirme la orientación y construya el montaje en RC.",
            )
            return

        self._finish_landmark_pick()
        selectable_layers = (
            "maxillary_rc",
            "mandibular_rc",
            "mic_record",
            "landmarks",
            "balkwill",
            "bonwill",
            "virtual_condyles",
            "hinge_axis",
            "axes",
            "condylar_guides",
            "mic_seeds",
            "mandibular_mic",
            "condylar_displacement",
        )
        self._visibility_before_pick = {
            layer: self.layer_actions[layer].isChecked()
            for layer in selectable_layers
        }
        for layer in selectable_layers:
            self._set_layer_visibility(layer, layer == "mic_record")
            self.scene.set_layer_pickable(
                layer,
                layer == "mic_record",
            )

        name = "maxilar" if arch == "maxillary" else "mandíbula"
        self.statusBar().showMessage(
            f"Selección MIC activa — haga clic sobre el {name}"
        )
        self.scene.enable_surface_pick(
            lambda point, selected_arch=arch: (
                self._mic_seed_picked(selected_arch, point)
            )
        )

    def _mic_seed_picked(self, arch: str, displayed_point) -> None:
        """Resolve a displayed MIC point to its nearest original vertex."""

        point = np.asarray(displayed_point, dtype=float)
        if point.shape != (3,):
            return
        coordinate_system = self._coordinate_system
        if coordinate_system is None:
            return

        world_point = coordinate_system.to_world(point)
        mic_mesh = self._study_meshes["mic_record"]
        distances_squared = np.sum(
            (mic_mesh.vertices - world_point) ** 2,
            axis=1,
        )
        vertex_index = int(np.argmin(distances_squared))
        self._mic_seed_vertices[arch] = vertex_index

        selected_points = {
            name: coordinate_system.to_local(mic_mesh.vertices[index])
            for name, index in self._mic_seed_vertices.items()
        }
        selected_point = selected_points[arch]
        self.rc_mic_diagnosis_panel.show_seed(
            arch,
            vertex_index,
            selected_point,
        )
        self.scene.show_mic_seeds(selected_points, render=False)
        self._finish_landmark_pick()
        self.layer_actions["mic_seeds"].setEnabled(True)
        self.layer_actions["mic_seeds"].setChecked(True)
        self.scene.set_layer_visible("mic_seeds", True)
        name = "maxilar" if arch == "maxillary" else "mandíbula"
        self.statusBar().showMessage(
            f"Semilla del {name} MIC guardada — vértice {vertex_index:,}"
        )
        self._state_changed()

    def _diagnose_rc_mic(self) -> None:
        """Separate the combined record and calculate RC-to-MIC displacement."""

        required_seeds = {"maxillary", "mandibular"}
        if not required_seeds.issubset(self._mic_seed_vertices):
            return
        if (
            self._coordinate_system is None
            or self._bonwill is None
            or "mic_record" not in self._study_meshes
        ):
            return

        controller = self._functional_calibration_controller
        if controller is not None:
            controller.reset_movement()
            position = controller.reset_adjustment()
            self._show_functional_position(position)

        self.rc_mic_diagnosis_panel.show_running()
        self.statusBar().showMessage("Calculando diagnóstico RC–MIC…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            mic_record = OcclusalRecordBuilder.from_seed_vertices(
                mesh=self._study_meshes["mic_record"],
                maxillary_seed_vertex=(
                    self._mic_seed_vertices["maxillary"]
                ),
                mandibular_seed_vertex=(
                    self._mic_seed_vertices["mandibular"]
                ),
            )
            registration = CentricRelationRegistration.register(
                maxillary_mesh=self._study_meshes["maxillary_rc"],
                mandibular_rc_mesh=self._study_meshes["mandibular_rc"],
                mic_record=mic_record,
                maximum_iterations=50,
                tolerance=1e-6,
                trim_fraction=0.90,
                sample_size=10000,
            )
            if not registration.converged:
                raise ValueError(
                    "el registro iterativo no alcanzó convergencia."
                )

            displacement = registration.condylar_displacement(
                right_condyle_point=self._bonwill.right_condyle.point,
                left_condyle_point=self._bonwill.left_condyle.point,
            )
            coordinate_system = self._coordinate_system
            right_vector = (
                coordinate_system.to_local(displacement.right_mic_point)
                - coordinate_system.to_local(displacement.right_rc_point)
            )
            left_vector = (
                coordinate_system.to_local(displacement.left_mic_point)
                - coordinate_system.to_local(displacement.left_rc_point)
            )
            mandibular_mic_points = (
                registration.mandibular_rc_to_mic_transform.apply(
                    self._study_meshes["mandibular_rc"].vertices
                )
            )

            self._rc_mic_registration = registration
            self._condylar_displacement = displacement
            self.scene.show_rc_mic_diagnosis(
                mandibular_rc_mesh=self._study_meshes["mandibular_rc"],
                mandibular_mic_points=mandibular_mic_points,
                displacement=displacement,
                coordinate_system=coordinate_system,
            )
            self._set_layer_visibility("mic_record", False)
            for layer_name in (
                "mandibular_mic",
                "condylar_displacement",
            ):
                action = self.layer_actions[layer_name]
                action.setEnabled(True)
                action.setChecked(True)
                self.scene.set_layer_visible(layer_name, True, render=False)

            self.rc_mic_diagnosis_panel.show_result(
                registration=registration,
                right_vector=right_vector,
                left_vector=left_vector,
            )
            self._update_rc_mic_tree(right_vector, left_vector)
            self._set_rc_mic_diagnostic_view("overlay")
            self.scene.plotter.render()
            self.statusBar().showMessage(
                "Diagnóstico RC–MIC calculado — vectores expresados en X/Y/Z"
            )
            self._state_changed()
        except (TypeError, ValueError) as error:
            self.rc_mic_diagnosis_panel.show_error(str(error))
            self.statusBar().showMessage(
                f"Diagnóstico RC–MIC no calculado — {error}"
            )
            QMessageBox.warning(
                self,
                "Diagnóstico RC–MIC no calculado",
                str(error),
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _set_rc_mic_diagnostic_view(self, mode: str) -> None:
        """Switch between RC, MIC, and their overlay without moving camera."""

        if self._rc_mic_registration is None:
            return
        self.rc_mic_diagnosis_panel.set_diagnostic_view(mode)
        self.scene.set_rc_mic_view(mode)
        labels = {
            "rc": "Relación céntrica",
            "mic": "Máxima intercuspidación",
            "overlay": "Superposición RC/MIC",
        }
        self.statusBar().showMessage(
            f"Vista diagnóstica — {labels[mode]}"
        )

    def _reset_rc_mic_diagnosis(self, *, mark_dirty: bool = True) -> None:
        """Remove MIC seeds and computed diagnostic actors."""

        if hasattr(self, "scene"):
            self._finish_landmark_pick()
        self._mic_seed_vertices.clear()
        self._rc_mic_registration = None
        self._condylar_displacement = None
        for layer_name in (
            "mic_seeds",
            "mandibular_mic",
            "condylar_displacement",
        ):
            if hasattr(self, "scene"):
                self.scene.clear_layer(layer_name, render=False)
            action = getattr(self, "layer_actions", {}).get(layer_name)
            if action is not None:
                action.setChecked(False)
                action.setEnabled(False)
        if hasattr(self, "rc_mic_diagnosis_panel"):
            self.rc_mic_diagnosis_panel.clear()
        section = getattr(self, "study_sections", {}).get("results")
        if section is not None:
            section.takeChildren()
        if hasattr(self, "scene"):
            self.scene.plotter.render()
        if mark_dirty:
            self._state_changed()

    def _update_rc_mic_tree(
        self,
        right_vector: np.ndarray,
        left_vector: np.ndarray,
    ) -> None:
        """Expose the current diagnosis in the study results tree."""

        section = self.study_sections["results"]
        section.takeChildren()
        registration = self._rc_mic_registration
        entries = (
            (
                "Registro RC–MIC convergente",
                "RMSE maxilar "
                f"{registration.maxillary_registration.root_mean_square_error:.6f} "
                "mm | "
                "RMSE mandibular "
                f"{registration.mandibular_registration.root_mean_square_error:.6f} mm",
            ),
            (
                f"Cóndilo derecho — {np.linalg.norm(right_vector):.4f} mm",
                self._format_diagnostic_vector(right_vector),
            ),
            (
                f"Cóndilo izquierdo — {np.linalg.norm(left_vector):.4f} mm",
                self._format_diagnostic_vector(left_vector),
            ),
        )
        for label, tooltip in entries:
            item = QTreeWidgetItem(section, [label])
            item.setToolTip(0, tooltip)
        section.setExpanded(True)

    @staticmethod
    def _format_diagnostic_vector(vector: np.ndarray) -> str:
        """Format one local RC-to-MIC vector for the results tree."""

        return (
            f"X {vector[0]:+.4f} mm | "
            f"Y {vector[1]:+.4f} mm | "
            f"Z {vector[2]:+.4f} mm"
        )

    def _results_snapshot(self) -> StudyResultsSnapshot:
        """Build one immutable view of every currently available result."""

        mounting = None
        configuration = self._articulator_configuration
        if configuration is not None:
            functional_path = 17.0
            if self._condylar_guides is not None:
                functional_path = min(
                    self._condylar_guides.right_guide.maximum_translation,
                    self._condylar_guides.left_guide.maximum_translation,
                )
            mounting = {
                "intercondylar_width": configuration.intercondylar_width,
                "balkwill_angle_degrees": (
                    configuration.balkwill_angle_degrees
                ),
                "right_condylar_guidance_degrees": (
                    configuration.right_condylar_guidance_degrees
                ),
                "left_condylar_guidance_degrees": (
                    configuration.left_condylar_guidance_degrees
                ),
                "functional_path_mm": functional_path,
            }

        controller = self._functional_calibration_controller
        limits: tuple[dict[str, Any], ...] = ()
        if controller is not None:
            limits = tuple(
                {
                    "kind": limit.kind.value,
                    "base_opening_angle_degrees": (
                        limit.base_opening_angle_degrees
                    ),
                    "adjustment_angle_degrees": (
                        limit.adjustment_angle_degrees
                    ),
                    "total_opening_angle_degrees": (
                        limit.total_opening_angle_degrees
                    ),
                    "lateral_angle_degrees": limit.lateral_angle_degrees,
                    "protrusion_distance_mm": limit.protrusion_distance_mm,
                }
                for limit in controller.limits.values
            )

        diagnosis = None
        registration = self._rc_mic_registration
        displacement = self._condylar_displacement
        coordinate_system = self._coordinate_system
        if (
            registration is not None
            and displacement is not None
            and coordinate_system is not None
        ):
            right_vector = (
                coordinate_system.to_local(displacement.right_mic_point)
                - coordinate_system.to_local(displacement.right_rc_point)
            )
            left_vector = (
                coordinate_system.to_local(displacement.left_mic_point)
                - coordinate_system.to_local(displacement.left_rc_point)
            )
            diagnosis = {
                "converged": registration.converged,
                "maxillary_rmse_mm": (
                    registration.maxillary_registration.root_mean_square_error
                ),
                "mandibular_rmse_mm": (
                    registration.mandibular_registration.root_mean_square_error
                ),
                "right_vector_mm": right_vector.tolist(),
                "right_distance_mm": float(np.linalg.norm(right_vector)),
                "left_vector_mm": left_vector.tolist(),
                "left_distance_mm": float(np.linalg.norm(left_vector)),
            }

        return StudyResultsSnapshot(
            study_name=self._study_display_name(),
            source_files=self._study_filenames(),
            mounting=mounting,
            functional_limits=limits,
            rc_mic=diagnosis,
        )

    def _refresh_results_panel(self) -> None:
        """Synchronize point 7 with the complete current study state."""

        if self._study_files is None:
            self.results_export_panel.set_study_available(False)
            return
        self.results_export_panel.set_study_available(True)
        self.results_export_panel.show_snapshot(self._results_snapshot())

    def _suggested_output_path(self, suffix: str) -> Path:
        """Return a stable default beside the native study when possible."""

        directory = (
            self._study_path.parent
            if self._study_path is not None
            else Path.cwd()
        )
        return directory / f"{self._study_display_name()}{suffix}"

    def _export_results_pdf(self) -> None:
        """Request and create one clinical PDF report."""

        if self._study_files is None:
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar informe clínico",
            str(self._suggested_output_path("_resultados.pdf")),
            "Documentos PDF (*.pdf)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")
        self._run_export(
            lambda: self._create_pdf_report(path),
            path,
            "Informe PDF exportado",
        )

    def _create_pdf_report(self, path: Path) -> Path:
        """Capture standardized clinical views and write the PDF report."""

        with TemporaryDirectory(prefix="ogdd-report-") as directory:
            captures = self._capture_report_visuals(Path(directory))
            return ResultsExporter.export_pdf(
                path,
                self._results_snapshot(),
                visual_captures=captures,
            )

    def _capture_report_visuals(
        self,
        directory: Path,
    ) -> dict[str, Path | None]:
        """Create six reproducible views while preserving interactive state."""

        keys = (
            "overlay_right",
            "overlay_front",
            "overlay_left",
            "right_canine",
            "protrusive",
            "left_canine",
        )
        captures: dict[str, Path | None] = {key: None for key in keys}
        controller = self._functional_calibration_controller
        if controller is None or self._coordinate_system is None:
            return captures

        camera_state = self.scene.camera_state()
        layer_state = {
            name: action.isChecked()
            for name, action in self.layer_actions.items()
        }
        position_state = (
            controller.opening_angle_degrees,
            controller.lateral_angle_degrees,
            controller.protrusion_distance_mm,
            controller.adjustment_angle_degrees,
        )
        previous_view = self.rc_mic_diagnosis_panel.diagnostic_view
        previous_restoring = self._restoring_study
        previous_dirty = self._study_dirty
        self._restoring_study = True

        try:
            controller.reset_movement()
            position = controller.reset_adjustment()
            self._show_functional_position(position)
            if self._rc_mic_registration is not None:
                self._set_report_layers(
                    {
                        "maxillary_rc",
                        "mandibular_rc",
                        "mandibular_mic",
                        "virtual_condyles",
                        "hinge_axis",
                        "condylar_displacement",
                    }
                )
                self.scene.set_rc_mic_view("overlay", render=False)
                for key, view in (
                    ("overlay_right", "right"),
                    ("overlay_front", "front"),
                    ("overlay_left", "left"),
                ):
                    captures[key] = self._capture_report_view(
                        directory,
                        key,
                        view,
                    )

            self._set_report_layers({"maxillary_rc", "mandibular_rc"})
            for key, kind, view in (
                (
                    "right_canine",
                    FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP,
                    "right",
                ),
                (
                    "protrusive",
                    FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE,
                    "front",
                ),
                (
                    "left_canine",
                    FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP,
                    "left",
                ),
            ):
                if controller.limits.get(kind) is None:
                    continue
                position = controller.go_to_limit(kind)
                self._show_functional_position(position)
                self._set_report_layers({"maxillary_rc", "mandibular_rc"})
                captures[key] = self._capture_report_view(
                    directory,
                    key,
                    view,
                )
        finally:
            controller.reset_movement()
            controller.reset_adjustment()
            restored = controller.set_position(
                opening_angle_degrees=position_state[0],
                lateral_angle_degrees=position_state[1],
                protrusion_distance_mm=position_state[2],
            )
            restored = controller.set_adjustment(position_state[3])
            self._show_functional_position(restored)
            for name, visible in layer_state.items():
                self.scene.set_layer_visible(name, visible, render=False)
            if self._rc_mic_registration is not None:
                self.rc_mic_diagnosis_panel.set_diagnostic_view(previous_view)
                self.scene.set_rc_mic_view(previous_view, render=False)
            self.scene.restore_camera_state(camera_state)
            self._restoring_study = previous_restoring
            self._study_dirty = previous_dirty
            self._update_window_title()
            self._refresh_results_panel()
        return captures

    def _set_report_layers(self, visible_layers: set[str]) -> None:
        """Prepare a clean scene without changing the layer menu state."""

        for name in self.layer_actions:
            self.scene.set_layer_visible(
                name,
                name in visible_layers,
                render=False,
            )

    def _capture_report_view(
        self,
        directory: Path,
        key: str,
        anatomical_view: str,
    ) -> Path:
        """Frame and save one standardized clinical screenshot."""

        path = directory / f"{key}.png"
        self.scene.set_anatomical_view(anatomical_view)
        self.scene.save_screenshot(str(path))
        return path

    def _export_results_csv(self) -> None:
        """Request and create one long-form numeric CSV export."""

        if self._study_files is None:
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar datos clínicos",
            str(self._suggested_output_path("_resultados.csv")),
            "Datos CSV (*.csv)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        self._run_export(
            lambda: ResultsExporter.export_csv(path, self._results_snapshot()),
            path,
            "Datos CSV exportados",
        )

    def _export_scene_png(self) -> None:
        """Request and save the current 3D scene as a PNG image."""

        if self._study_files is None:
            return
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar captura de la escena",
            str(self._suggested_output_path("_escena.png")),
            "Imágenes PNG (*.png)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        self._run_export(
            lambda: self.scene.save_screenshot(str(path)),
            path,
            "Captura PNG guardada",
        )

    def _run_export(
        self,
        action: Callable[[], Any],
        path: Path,
        success_message: str,
    ) -> None:
        """Run one explicit export with consistent feedback and protection."""

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            action()
        except Exception as error:
            QMessageBox.critical(
                self,
                "No fue posible exportar",
                f"{type(error).__name__}: {error}",
            )
            self.statusBar().showMessage("La exportación no fue creada.")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"{success_message} — {path.name}"
        )

    def _set_layer_visibility(
        self,
        layer_name: str,
        visible: bool,
    ) -> None:
        """Synchronize one layer actor and its menu action."""

        self.scene.set_layer_visible(layer_name, visible)
        action = self.layer_actions.get(layer_name)
        if action is not None:
            action.setChecked(visible)

    def _refresh_landmark_visuals(self) -> None:
        """Draw stored landmarks in the coordinate frame on screen."""

        if self._coordinate_system is None:
            displayed = self._landmark_points
        else:
            displayed = {
                name: self._coordinate_system.to_local(point)
                for name, point in self._landmark_points.items()
            }
        self.scene.show_landmarks(displayed)
        landmarks_action = self.layer_actions["landmarks"]
        landmarks_action.setEnabled(bool(displayed))
        landmarks_action.setChecked(bool(displayed))

    def _confirm_orientation(self) -> None:
        """Build the anatomical system and orient every study mesh."""

        required = {
            "DENTAL_MIDLINE",
            "RIGHT_SECOND_MOLAR",
            "LEFT_SECOND_MOLAR",
        }
        if not required.issubset(self._landmark_points):
            return

        model = DentalModel(
            mesh=self._study_meshes["mandibular_rc"]
        )
        references = {
            "DENTAL_MIDLINE": "Dental midline",
            "RIGHT_SECOND_MOLAR": "Right second molar cusp",
            "LEFT_SECOND_MOLAR": "Left second molar cusp",
        }
        for name in required:
            model.add_landmark(
                Landmark(
                    name=name,
                    point=self._landmark_points[name].copy(),
                    reference_used=references[name],
                )
            )

        try:
            coordinate_system = model.coordinate_system
        except ValueError as error:
            QMessageBox.warning(
                self,
                "Orientación no válida",
                "Los puntos seleccionados no permiten construir "
                f"el sistema anatómico.\n\n{error}",
            )
            return

        self._coordinate_system = coordinate_system
        self._clear_mounting_state()
        for layer_name, mesh in self._study_meshes.items():
            self.scene.update_dental_mesh_points(
                layer_name,
                coordinate_system.to_local(mesh.vertices),
            )

        local_landmarks = {
            name: coordinate_system.to_local(point)
            for name, point in self._landmark_points.items()
        }
        self.scene.show_landmarks(local_landmarks)
        self.scene.show_balkwill(local_landmarks)
        self.scene.show_anatomical_axes()

        for layer_name in ("landmarks", "balkwill", "axes"):
            action = self.layer_actions[layer_name]
            action.setEnabled(True)
            action.setChecked(True)
            self.scene.set_layer_visible(layer_name, True)

        self._set_layer_visibility("maxillary_rc", True)
        self._set_layer_visibility("mandibular_rc", True)
        self._set_layer_visibility("mic_record", False)
        self.orientation_panel.show_coordinate_system(coordinate_system)
        self.mounting_panel.set_orientation_available(True)
        self._update_anatomical_tree(coordinate_system)
        self.scene.reset_camera()
        self.statusBar().showMessage(
            "Orientación anatómica confirmada — "
            "+X derecha, +Y anterior, +Z superior"
        )
        self._state_changed()

    def _reset_orientation(self) -> None:
        """Discard landmarks and return meshes to scanner coordinates."""

        self._finish_landmark_pick()
        for layer_name, mesh in self._study_meshes.items():
            self.scene.update_dental_mesh_points(
                layer_name,
                mesh.vertices,
            )
        self._clear_anatomical_state()
        self.scene.reset_camera()
        self.statusBar().showMessage(
            "Orientación restablecida — coordenadas originales del escáner"
        )
        self._state_changed()

    def _clear_anatomical_state(self) -> None:
        """Clear anatomical state, actors and study-tree entries."""

        self._coordinate_system = None
        self._landmark_points.clear()
        self._clear_mounting_state()
        for layer_name in ("landmarks", "balkwill", "axes"):
            self.scene.clear_layer(layer_name, render=False)
            action = self.layer_actions.get(layer_name)
            if action is not None:
                action.setChecked(False)
                action.setEnabled(False)
        if hasattr(self, "orientation_panel"):
            self.orientation_panel.clear()
        section = self.study_sections.get("landmarks")
        if section is not None:
            section.takeChildren()
        self.scene.plotter.render()

    def _update_anatomical_tree(self, coordinate_system) -> None:
        """List confirmed landmarks and axes in the study tree."""

        section = self.study_sections["landmarks"]
        section.takeChildren()
        labels = {
            "DENTAL_MIDLINE": "Línea media dental",
            "RIGHT_SECOND_MOLAR": "Segundo molar derecho",
            "LEFT_SECOND_MOLAR": "Segundo molar izquierdo",
        }
        for name in (
            "DENTAL_MIDLINE",
            "RIGHT_SECOND_MOLAR",
            "LEFT_SECOND_MOLAR",
        ):
            point = self._landmark_points[name]
            child = QTreeWidgetItem(section, [labels[name]])
            child.setToolTip(
                0,
                f"X {point[0]:.3f}  Y {point[1]:.3f}  "
                f"Z {point[2]:.3f} mm",
            )
        QTreeWidgetItem(section, ["Sistema anatómico confirmado"])
        section.setExpanded(True)

    def _build_rc_mounting(self) -> None:
        """Construct Bonwill, hinge axis and guides from current settings."""

        if self._coordinate_system is None:
            QMessageBox.information(
                self,
                "Orientación pendiente",
                "Confirme primero la orientación anatómica.",
            )
            return

        configuration = self.mounting_panel.configuration()
        dental_midline = Landmark(
            name="DENTAL_MIDLINE",
            point=self._landmark_points["DENTAL_MIDLINE"].copy(),
            reference_used="Dental midline",
        )
        bonwill = BonwillBuilder.build(
            coordinate_system=self._coordinate_system,
            dental_midline=dental_midline,
            configuration=configuration,
        )
        hinge_axis = HingeAxis(
            left_condyle=bonwill.left_condyle,
            right_condyle=bonwill.right_condyle,
        )
        guide_pair = CondylarGuideBuilder.build(
            hinge_axis=hinge_axis,
            coordinate_system=self._coordinate_system,
            configuration=configuration,
        )

        right_posterior = Landmark(
            name="RIGHT_SECOND_MOLAR",
            point=self._landmark_points["RIGHT_SECOND_MOLAR"].copy(),
            reference_used="Right second molar cusp",
        )
        left_posterior = Landmark(
            name="LEFT_SECOND_MOLAR",
            point=self._landmark_points["LEFT_SECOND_MOLAR"].copy(),
            reference_used="Left second molar cusp",
        )
        balkwill = BalkwillTriangle(
            right_posterior=right_posterior,
            left_posterior=left_posterior,
            dental_midline=dental_midline,
        )
        assembly = MandibularAssembly(
            mesh=self._study_meshes["mandibular_rc"],
            balkwill=balkwill,
            bonwill=bonwill,
            hinge_axis=hinge_axis,
        )
        right_excursion = GuidedLateralExcursion(
            hinge_axis=hinge_axis,
            superior_direction=self._coordinate_system.z_axis,
            working_side=LateralSide.RIGHT,
            balancing_guide=guide_pair.left_guide,
        )
        left_excursion = GuidedLateralExcursion(
            hinge_axis=hinge_axis,
            superior_direction=self._coordinate_system.z_axis,
            working_side=LateralSide.LEFT,
            balancing_guide=guide_pair.right_guide,
        )
        maximum_lateral_angle = min(
            10.0,
            right_excursion.maximum_angle_degrees,
            left_excursion.maximum_angle_degrees,
        )
        protrusion = GuidedProtrusion(
            hinge_axis=hinge_axis,
            right_guide=guide_pair.right_guide,
            left_guide=guide_pair.left_guide,
        )
        combined_movement = CombinedMovement(
            assembly=assembly,
            right_excursion=right_excursion,
            left_excursion=left_excursion,
            protrusion=protrusion,
        )
        combined_controller = CombinedController(
            movement=combined_movement,
            maximum_opening_angle_degrees=30.0,
            maximum_lateral_angle_degrees=maximum_lateral_angle,
            maximum_protrusion_distance_mm=protrusion.maximum_translation,
            opening_step_degrees=0.1,
            lateral_step_degrees=0.1,
            protrusion_step_mm=0.1,
        )
        closure_controller = OcclusalClosureController(
            closure=OcclusalClosure(),
            base_position=combined_controller.position,
            step_degrees=0.1,
        )
        functional_controller = FunctionalCalibrationController(
            combined=combined_controller,
            closure=closure_controller,
        )

        self._articulator_configuration = configuration
        self._bonwill = bonwill
        self._hinge_axis = hinge_axis
        self._condylar_guides = guide_pair
        self._functional_calibration_controller = functional_controller

        self.scene.show_virtual_bonwill(
            bonwill,
            self._coordinate_system,
        )
        self.scene.show_hinge_axis(
            hinge_axis,
            self._coordinate_system,
        )
        self.scene.show_condylar_guides(
            guide_pair,
            self._coordinate_system,
        )

        for layer_name in (
            "bonwill",
            "virtual_condyles",
            "hinge_axis",
            "condylar_guides",
        ):
            action = self.layer_actions[layer_name]
            action.setEnabled(True)
            action.setChecked(True)
            self.scene.set_layer_visible(layer_name, True)

        self.mounting_panel.show_mounting(
            configuration,
            guide_pair.right_guide.maximum_translation,
        )
        self.functional_calibration_panel.set_mounting_available(
            True,
            maximum_translation=(
                functional_controller.maximum_protrusion_distance_mm
            ),
            maximum_opening_degrees=(
                combined_controller.maximum_opening_angle_degrees
            ),
            maximum_right_lateral_degrees=(
                functional_controller
                .maximum_right_lateral_angle_degrees
            ),
            maximum_left_lateral_degrees=(
                functional_controller
                .maximum_left_lateral_angle_degrees
            ),
        )
        self._show_functional_position(functional_controller.position)
        self.rc_mic_diagnosis_panel.set_prerequisites(
            mic_available="mic_record" in self._study_meshes,
            mounting_available=True,
        )
        self._update_articulator_tree()
        self.scene.reset_camera()
        self.statusBar().showMessage(
            "Montaje en RC construido — Bonwill, eje y guías confirmados"
        )

    def _clear_mounting_state(self) -> None:
        """Remove the virtual mounting while preserving orientation."""

        if (
            self._functional_calibration_controller is not None
            and self._coordinate_system is not None
        ):
            self._functional_calibration_controller.reset_movement()
            position = (
                self._functional_calibration_controller.reset_adjustment()
            )
            self._show_functional_position(position)

        self._articulator_configuration = None
        self._bonwill = None
        self._hinge_axis = None
        self._condylar_guides = None
        self._functional_calibration_controller = None
        self._reset_rc_mic_diagnosis(mark_dirty=False)
        for layer_name in (
            "bonwill",
            "virtual_condyles",
            "hinge_axis",
            "condylar_guides",
        ):
            self.scene.clear_layer(layer_name, render=False)
            action = self.layer_actions.get(layer_name)
            if action is not None:
                action.setChecked(False)
                action.setEnabled(False)
        if hasattr(self, "mounting_panel"):
            self.mounting_panel.clear()
            self.mounting_panel.set_orientation_available(
                self._coordinate_system is not None
            )
        if hasattr(self, "functional_calibration_panel"):
            self.functional_calibration_panel.set_mounting_available(False)
        if hasattr(self, "rc_mic_diagnosis_panel"):
            self.rc_mic_diagnosis_panel.set_prerequisites(
                mic_available="mic_record" in self._study_meshes,
                mounting_available=False,
            )
        section = self.study_sections.get("articulator")
        if section is not None:
            section.takeChildren()
        self.scene.plotter.render()
        self._state_changed()

    def _open_mandible(self) -> None:
        """Open the mounted mandible by one validated hinge step."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.open_mandible,
            lambda: (
                "Apertura mandibular — "
                f"{controller.opening_angle_degrees:.1f}°"
            ),
        )

    def _set_mandibular_opening(self, angle_degrees: float) -> None:
        """Set one exact decimal hinge opening from the panel slider."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            lambda: controller.set_opening(angle_degrees),
            lambda: (
                "Apertura mandibular — "
                f"{controller.opening_angle_degrees:.1f}°"
            ),
        )

    def _close_mandible(self) -> None:
        """Close the mounted mandible by one validated hinge step."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.close_mandible,
            lambda: (
                "Apertura mandibular — "
                f"{controller.opening_angle_degrees:.1f}°"
            ),
        )

    def _advance_mandible(self) -> None:
        """Advance the mounted mandible by one guided step."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.advance,
            lambda: (
                "Protrusión mandibular — "
                f"{controller.protrusion_distance_mm:.1f} mm"
            ),
        )

    def _set_mandibular_protrusion(self, distance_mm: float) -> None:
        """Set one exact decimal protrusive distance from the panel slider."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            lambda: controller.set_protrusion(distance_mm),
            lambda: (
                "Protrusión mandibular — "
                f"{controller.protrusion_distance_mm:.1f} mm"
            ),
        )

    def _retreat_mandible(self) -> None:
        """Retreat the mounted mandible one guided step toward RC."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.retreat,
            lambda: (
                "Protrusión mandibular — "
                f"{controller.protrusion_distance_mm:.1f} mm"
            ),
        )

    def _move_mandible_left(self) -> None:
        """Move one guided lateral step toward the patient's left."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.move_left,
            lambda: self._lateral_status_text(
                controller.lateral_angle_degrees
            ),
        )

    def _move_mandible_right(self) -> None:
        """Move one guided lateral step toward the patient's right."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.move_right,
            lambda: self._lateral_status_text(
                controller.lateral_angle_degrees
            ),
        )

    def _set_mandibular_lateral(self, angle_degrees: float) -> None:
        """Set one exact signed lateral angle from the panel slider."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            lambda: controller.set_lateral(angle_degrees),
            lambda: self._lateral_status_text(
                controller.lateral_angle_degrees
            ),
        )

    def _adjust_occlusion_close(self) -> None:
        """Apply one independent negative occlusal adjustment step."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.adjust_close,
            lambda: (
                "Ajuste oclusal fino — "
                f"{controller.adjustment_angle_degrees:.1f}° cierre"
            ),
        )

    def _adjust_occlusion_open(self) -> None:
        """Apply one independent positive occlusal adjustment step."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.adjust_open,
            lambda: (
                "Ajuste oclusal fino — "
                f"+{controller.adjustment_angle_degrees:.1f}° apertura"
            ),
        )

    def _reset_occlusal_adjustment(self) -> None:
        """Remove only the independent occlusal fine adjustment."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            controller.reset_adjustment,
            lambda: "Ajuste oclusal fino restablecido — 0.0°",
        )

    def _save_protrusive_limit(self) -> None:
        """Save the operator-confirmed protrusive edge-to-edge position."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._save_functional_limit(
            controller.save_protrusive_limit,
            lambda limit: (
                "Borde a borde guardado — "
                f"{limit.protrusion_distance_mm:.1f} mm | "
                f"ajuste {limit.adjustment_angle_degrees:+.1f}°"
            ),
        )

    def _save_right_canine_limit(self) -> None:
        """Save the operator-confirmed right canine protection."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._save_functional_limit(
            controller.save_right_canine_limit,
            lambda limit: (
                "Protección canina derecha guardada — "
                f"{limit.lateral_angle_degrees:+.1f}° | "
                f"ajuste {limit.adjustment_angle_degrees:+.1f}°"
            ),
        )

    def _save_left_canine_limit(self) -> None:
        """Save the operator-confirmed left canine protection."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._save_functional_limit(
            controller.save_left_canine_limit,
            lambda limit: (
                "Protección canina izquierda guardada — "
                f"{limit.lateral_angle_degrees:+.1f}° | "
                f"ajuste {limit.adjustment_angle_degrees:+.1f}°"
            ),
        )

    def _save_functional_limit(
        self,
        action: Callable[[], Any],
        message: Callable[[Any], str],
    ) -> None:
        """Save one valid endpoint and refresh its effective movement range."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        try:
            limit = action()
        except ValueError as error:
            self.statusBar().showMessage(
                f"Límite no guardado — {error}"
            )
            return
        self._show_functional_position(controller.position)
        self.statusBar().showMessage(message(limit))

    def _go_to_protrusive_limit(self) -> None:
        """Return to the saved protrusive edge-to-edge position."""

        self._go_to_functional_limit(
            FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE,
            "Borde a borde protrusivo reproducido",
        )

    def _go_to_right_canine_limit(self) -> None:
        """Return to the saved right canine protection."""

        self._go_to_functional_limit(
            FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP,
            "Protección canina derecha reproducida",
        )

    def _go_to_left_canine_limit(self) -> None:
        """Return to the saved left canine protection."""

        self._go_to_functional_limit(
            FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP,
            "Protección canina izquierda reproducida",
        )

    def _go_to_functional_limit(
        self,
        kind: FunctionalLimitKind,
        message: str,
    ) -> None:
        """Reproduce one complete saved movement and fine adjustment."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            lambda: controller.go_to_limit(kind),
            lambda: message,
        )

    def _clear_functional_limits(self) -> None:
        """Remove all endpoints so the operator can recalibrate them."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        controller.clear_limits()
        controller.reset_movement()
        position = controller.reset_adjustment()
        self._show_functional_position(position)
        self.statusBar().showMessage(
            "Límites funcionales borrados — recalibración disponible"
        )

    def _run_functional_movement(
        self,
        action: Callable[[], Any],
        message: Callable[[], str],
    ) -> None:
        """Render one accepted movement and visibly reject invalid geometry."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        try:
            position = action()
        except ValueError as error:
            self._show_functional_position(controller.position)
            self.statusBar().showMessage(
                f"Movimiento no aceptado — {error}"
            )
            return
        self._show_functional_position(position)
        self.statusBar().showMessage(message())

    def _zero_lateral(self) -> None:
        """Center lateral movement while preserving opening and protrusion."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        self._run_functional_movement(
            lambda: controller.set_lateral(0.0),
            lambda: self._lateral_status_text(
                controller.lateral_angle_degrees
            ),
        )

    def _show_lateral_status(self, angle_degrees: float) -> None:
        """Describe the current signed lateral position."""

        self.statusBar().showMessage(
            self._lateral_status_text(angle_degrees)
        )

    @staticmethod
    def _lateral_status_text(angle_degrees: float) -> str:
        """Format one signed lateral value for the status bar."""

        if angle_degrees > 0.0:
            value = f"+{angle_degrees:.1f}° derecha"
        elif angle_degrees < 0.0:
            value = f"{angle_degrees:.1f}° izquierda"
        else:
            value = "0.0°"
        return f"Lateralidad mandibular — {value}"

    def _return_to_rc(self) -> None:
        """Return every calibrated movement component exactly to RC."""

        controller = self._functional_calibration_controller
        if controller is None:
            return
        controller.reset_movement()
        position = controller.reset_adjustment()
        self._show_functional_position(position)
        self.statusBar().showMessage("Mandíbula en relación céntrica — 0.0°")

    def _show_functional_position(self, position) -> None:
        """Display one absolute mandibular position without moving the maxilla."""

        if self._coordinate_system is None:
            return

        coordinate_system = self._coordinate_system
        self.scene.update_dental_mesh_points(
            "mandibular_rc",
            coordinate_system.to_local(position.mesh.vertices),
        )
        moving_landmarks = {
            "DENTAL_MIDLINE": coordinate_system.to_local(
                position.balkwill.dental_midline.point
            ),
            "RIGHT_SECOND_MOLAR": coordinate_system.to_local(
                position.balkwill.right_posterior.point
            ),
            "LEFT_SECOND_MOLAR": coordinate_system.to_local(
                position.balkwill.left_posterior.point
            ),
        }
        self.scene.show_landmarks(moving_landmarks, render=False)
        self.scene.show_balkwill(moving_landmarks)
        self.scene.show_virtual_bonwill(
            position.bonwill,
            coordinate_system,
        )

        for layer_name in (
            "landmarks",
            "balkwill",
            "bonwill",
            "virtual_condyles",
        ):
            self.scene.set_layer_visible(
                layer_name,
                self.layer_actions[layer_name].isChecked(),
                render=False,
            )

        controller = self._functional_calibration_controller
        if controller is not None:
            self.functional_calibration_panel.show_position(
                opening_angle_degrees=(
                    controller.opening_angle_degrees
                ),
                protrusion_distance_mm=(
                    controller.protrusion_distance_mm
                ),
                lateral_angle_degrees=(
                    controller.lateral_angle_degrees
                ),
                maximum_translation_mm=(
                    controller.maximum_protrusion_distance_mm
                ),
                maximum_right_lateral_degrees=(
                    controller.maximum_right_lateral_angle_degrees
                ),
                maximum_left_lateral_degrees=(
                    controller.maximum_left_lateral_angle_degrees
                ),
                adjustment_angle_degrees=(
                    controller.adjustment_angle_degrees
                ),
            )
            self.functional_calibration_panel.show_limits(
                protrusive_limit=controller.limits.protrusive,
                right_canine_limit=controller.limits.right_canine,
                left_canine_limit=controller.limits.left_canine,
            )
        self.scene.plotter.render()
        self._state_changed()

    def _update_articulator_tree(self) -> None:
        """Describe the active virtual mounting in the study tree."""

        section = self.study_sections["articulator"]
        section.takeChildren()
        configuration = self._articulator_configuration
        guide_pair = self._condylar_guides

        entries = (
            (
                f"Bonwill — {configuration.intercondylar_width:.1f} mm",
                f"Ángulo de Balkwill: "
                f"{configuration.balkwill_angle_degrees:.1f}°",
            ),
            (
                f"Eje de bisagra — {self._hinge_axis.length:.1f} mm",
                "Dirección intercondilar de izquierda a derecha",
            ),
            (
                "Guía condilar derecha",
                f"{configuration.right_condylar_guidance_degrees:.1f}° | "
                f"recorrido {guide_pair.right_guide.maximum_translation:.1f} mm",
            ),
            (
                "Guía condilar izquierda",
                f"{configuration.left_condylar_guidance_degrees:.1f}° | "
                f"recorrido {guide_pair.left_guide.maximum_translation:.1f} mm",
            ),
        )
        for label, tooltip in entries:
            item = QTreeWidgetItem(section, [label])
            item.setToolTip(0, tooltip)
        section.setExpanded(True)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Acerca de OGDD",
            "<h3>OGDD</h3>"
            "<p><b>Open Geometry for Digital Dentistry</b></p>"
            "<p>Geometría computacional abierta, reproducible y "
            "orientada a la enseñanza y la práctica odontológica.</p>"
            "<p>Autor: Armando Prado<br>Licencia MIT</p>",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard_changes("cerrar OGDD"):
            event.ignore()
            return
        self.scene.close_scene()
        super().closeEvent(event)
