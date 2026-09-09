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
        self._layer_surfaces: dict[str, pv.PolyData] = {}
        self._surface_pick_enabled = False
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
        self._layer_surfaces[layer_name] = surface
        return actor

    def update_dental_mesh_points(
        self,
        layer_name: str,
        points: np.ndarray,
    ) -> None:
        """Update a displayed dental mesh without rebuilding its faces."""

        surface = self._layer_surfaces.get(layer_name)
        if surface is None:
            return
        surface.points = np.asarray(points, dtype=float)

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
        self._layer_surfaces.pop(layer_name, None)
        if render:
            self.plotter.render()

    def set_layer_pickable(
        self,
        layer_name: str,
        pickable: bool,
    ) -> None:
        """Control whether actors in one layer can receive a pick."""

        for actor in self._layer_actors.get(layer_name, []):
            actor.SetPickable(pickable)

    def enable_surface_pick(self, callback) -> None:
        """Request one surface point using PyVista 0.48 picking."""

        self.disable_surface_pick()
        self.plotter.enable_surface_point_picking(
            callback=callback,
            show_message=False,
            show_point=False,
            left_clicking=True,
            clear_on_no_selection=False,
        )
        self._surface_pick_enabled = True

    def disable_surface_pick(self) -> None:
        """End point-picking mode and restore ordinary interaction."""

        if not self._surface_pick_enabled:
            return
        self.plotter.disable_picking()
        self._surface_pick_enabled = False

    def show_landmarks(
        self,
        points: dict[str, np.ndarray],
    ) -> None:
        """Rebuild named anatomical landmark markers."""

        self.clear_layer("landmarks", render=False)
        labels = {
            "DENTAL_MIDLINE": "LÍNEA MEDIA",
            "RIGHT_SECOND_MOLAR": "MOLAR DERECHO",
            "LEFT_SECOND_MOLAR": "MOLAR IZQUIERDO",
        }
        colors = {
            "DENTAL_MIDLINE": "gold",
            "RIGHT_SECOND_MOLAR": "tomato",
            "LEFT_SECOND_MOLAR": "deepskyblue",
        }

        for landmark_name, point in points.items():
            marker = pv.Sphere(
                radius=0.8,
                center=np.asarray(point, dtype=float),
                theta_resolution=24,
                phi_resolution=24,
            )
            marker_actor = self.plotter.add_mesh(
                marker,
                color=colors[landmark_name],
                name=f"{landmark_name}_marker",
                reset_camera=False,
            )
            label_actor = self.plotter.add_point_labels(
                np.asarray([point], dtype=float),
                [labels[landmark_name]],
                show_points=False,
                always_visible=True,
                font_size=11,
                text_color=colors[landmark_name],
                shape=None,
                name=f"{landmark_name}_label",
            )
            self.register_actor("landmarks", marker_actor)
            self.register_actor("landmarks", label_actor)

        self.plotter.render()

    def show_balkwill(self, points: dict[str, np.ndarray]) -> None:
        """Draw the closed Balkwill triangle from local landmarks."""

        self.clear_layer("balkwill", render=False)
        ordered = np.asarray(
            [
                points["RIGHT_SECOND_MOLAR"],
                points["LEFT_SECOND_MOLAR"],
                points["DENTAL_MIDLINE"],
            ],
            dtype=float,
        )
        line = pv.lines_from_points(ordered, close=True)
        actor = self.plotter.add_mesh(
            line,
            color="gold",
            line_width=4,
            render_lines_as_tubes=True,
            name="balkwill",
            reset_camera=False,
        )
        self.register_actor("balkwill", actor)

    def show_anatomical_axes(self, length: float = 20.0) -> None:
        """Draw the standard local +X, +Y and +Z axes."""

        self.clear_layer("axes", render=False)
        definitions = (
            ("X", (1.0, 0.0, 0.0), "tomato"),
            ("Y", (0.0, 1.0, 0.0), "limegreen"),
            ("Z", (0.0, 0.0, 1.0), "deepskyblue"),
        )
        endpoints = []
        labels = []
        for label, direction, color in definitions:
            arrow = pv.Arrow(
                start=(0.0, 0.0, 0.0),
                direction=direction,
                scale=length,
            )
            actor = self.plotter.add_mesh(
                arrow,
                color=color,
                name=f"anatomical_axis_{label}",
                reset_camera=False,
            )
            self.register_actor("axes", actor)
            endpoints.append(np.asarray(direction) * length)
            labels.append(f"+{label}")

        label_actor = self.plotter.add_point_labels(
            np.asarray(endpoints),
            labels,
            show_points=False,
            always_visible=True,
            font_size=12,
            text_color="white",
            shape=None,
            name="anatomical_axis_labels",
        )
        self.register_actor("axes", label_actor)
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
