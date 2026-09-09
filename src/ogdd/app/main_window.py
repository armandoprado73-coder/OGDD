"""Main window for the standalone OGDD application."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from ogdd.io.stl import STLReader

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
        for index, label in enumerate(self.CLINICAL_STEPS):
            item = QListWidgetItem(label)
            item.setToolTip("Paso disponible próximamente")
            self.workflow_list.addItem(item)
            if index == 0:
                self.workflow_list.setCurrentItem(item)

        self.workflow_dock.setWidget(self.workflow_list)
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
