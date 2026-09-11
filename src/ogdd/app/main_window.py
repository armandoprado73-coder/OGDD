"""Main window for the standalone OGDD application."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
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

from .functional_calibration_panel import FunctionalCalibrationPanel
from .mounting_panel import MountingPanel
from .orientation_panel import OrientationPanel
from .scene_view import SceneView
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
            self._show_next_step,
            shortcut=QKeySequence.StandardKey.Open,
            status_tip="Abrir un estudio existente",
        )
        self.save_action = self._action(
            "&Guardar",
            shortcut=QKeySequence.StandardKey.Save,
            status_tip="Guardar el estudio actual",
            enabled=False,
        )
        self.save_as_action = self._action(
            "Guardar &como…",
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

    def _new_study(self) -> None:
        """Clear the currently loaded study after user confirmation."""

        if self._study_files is not None:
            answer = QMessageBox.question(
                self,
                "Nuevo estudio",
                "Se cerrará el estudio cargado. ¿Desea continuar?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        for layer_name in (
            "maxillary_rc",
            "mandibular_rc",
            "mic_record",
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
        self._clear_anatomical_state()
        self.orientation_panel.set_study_available(False)
        self.scene.show_welcome()
        self.scene.plotter.render()
        self.statusBar().showMessage(
            "Nuevo estudio — importe los modelos en RC"
        )

    def _import_study_files(self) -> None:
        """Select, read and display the files of an RC–MIC study."""

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
        self._clear_anatomical_state()
        self.orientation_panel.set_study_available(True)
        self._update_study_tree(selection, meshes)
        self.scene.finish_study_load()
        self.workflow_list.setCurrentRow(0)
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
        self.orientation_panel.setVisible(orientation_selected)
        self.mounting_panel.setVisible(mounting_selected)
        self.functional_calibration_panel.setVisible(calibration_selected)
        self.functional_calibration_scroll.setVisible(calibration_selected)
        self.workflow_message.setVisible(
            not orientation_selected
            and not mounting_selected
            and not calibration_selected
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
        ):
            self.workflow_message.setText(
                messages.get(row, "Paso clínico en preparación.")
            )
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
        ):
            self.scene.set_layer_pickable(layer, True)

        for layer, visible in self._visibility_before_pick.items():
            self._set_layer_visibility(layer, visible)
        self._visibility_before_pick.clear()

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
            "Orientación anatómica confirmada — +X derecha, +Y anterior, +Z superior"
        )

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
        section = self.study_sections.get("articulator")
        if section is not None:
            section.takeChildren()
        self.scene.plotter.render()

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
        self.scene.close_scene()
        super().closeEvent(event)
