"""Qt widget that owns the embedded PyVista scene."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QFrame, QVBoxLayout
from pyvistaqt import QtInteractor


class SceneView(QFrame):
    """Central 3D view with named, independently visible layers."""

    _CAMERA_VECTORS = {
        "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        "back": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "left": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "bottom": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    }

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self._layer_actors: dict[str, list[Any]] = defaultdict(list)
        self._prepare_empty_scene()

    def _prepare_empty_scene(self) -> None:
        self.plotter.set_background(
            "#263944",
            top="#101a21",
        )
        self.show_welcome()

    def show_welcome(self) -> None:
        """Show the OGDD title while no study is loaded."""

        self.plotter.add_text(
            "OGDD\nOpen Geometry for Digital Dentistry",
            position="upper_left",
            color="#dce9eb",
            font_size=13,
            name="welcome",
        )

    @staticmethod
    def _to_polydata(mesh: Any) -> pv.PolyData:
        """Convert an OGDD triangular mesh to PyVista PolyData."""

        face_sizes = np.full(
            (len(mesh.faces), 1),
            3,
            dtype=np.int64,
        )
        faces = np.hstack((face_sizes, mesh.faces)).ravel()
        return pv.PolyData(mesh.vertices, faces)

    def add_dental_mesh(
        self,
        layer_name: str,
        mesh: Any,
        *,
        color: str,
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Display an OGDD mesh and register its actor by layer."""

        self.clear_layer(layer_name, render=False)
        surface = self._to_polydata(mesh)
        actor = self.plotter.add_mesh(
            surface,
            color=color,
            opacity=opacity,
            show_edges=False,
            smooth_shading=True,
            name=layer_name,
            reset_camera=False,
        )
        actor.SetVisibility(visible)
        self.register_actor(layer_name, actor)
        return actor

    def register_actor(self, layer_name: str, actor: Any) -> None:
        """Associate a PyVista actor with a visibility layer."""

        self._layer_actors[layer_name].append(actor)

    def clear_layer(
        self,
        layer_name: str,
        *,
        render: bool = True,
    ) -> None:
        """Remove all actors registered in one layer."""

        for actor in self._layer_actors.pop(layer_name, []):
            self.plotter.remove_actor(actor, render=False)
        if render:
            self.plotter.render()

    def finish_study_load(self) -> None:
        """Frame newly loaded models and remove the welcome message."""

        self.plotter.remove_actor("welcome", render=False)
        self.reset_camera()

    def set_layer_visible(
        self,
        layer_name: str,
        visible: bool,
    ) -> None:
        """Show or hide every actor registered in a layer."""

        for actor in self._layer_actors.get(layer_name, []):
            actor.SetVisibility(visible)
        self.plotter.render()

    def set_anatomical_view(self, view_name: str) -> None:
        """Move the camera to a predefined anatomical view."""

        direction, view_up = self._CAMERA_VECTORS[view_name]
        self.plotter.view_vector(direction, viewup=view_up)
        self.plotter.reset_camera()
        self.plotter.render()

    def reset_camera(self) -> None:
        """Return to an isometric view containing the whole scene."""

        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    def close_scene(self) -> None:
        """Release VTK resources before closing the Qt window."""

        self.plotter.close()
