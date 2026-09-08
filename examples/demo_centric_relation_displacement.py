"""
OGDD - Centric Relation Displacement Demo

Registers one combined MIC record to an RC mounting
and visualizes the resulting mandibular and condylar
displacement.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pyvista as pv

from ogdd.anatomy.dental_model import DentalModel
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.bonwill_builder import (
    BonwillBuilder,
)
from ogdd.articulator.configuration import (
    ArticulatorConfiguration,
)
from ogdd.io.stl import STLReader
from ogdd.registration.centric_relation_registration import (
    CentricRelationRegistration,
)
from ogdd.registration.occlusal_record_builder import (
    OcclusalRecordBuilder,
)


MAXILLARY_SEED_VERTEX = 28810
MANDIBULAR_SEED_VERTEX = 74585

CONDYLE_RADIUS_MM = 3.0
VECTOR_DISPLAY_MAGNIFICATION = 8.0


def _require_file(
    path: Path,
) -> Path:
    """
    Return one required demo file or fail clearly.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Required demo file was not found: {path}"
        )

    return path


def _surface(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> pv.PolyData:
    """
    Build a triangular PyVista surface.
    """

    pyvista_faces = np.hstack(
        [
            np.full(
                (len(faces), 1),
                3,
                dtype=int,
            ),
            faces,
        ]
    ).ravel()

    return pv.PolyData(
        vertices,
        pyvista_faces,
    )


def _movement_direction(
    value: float,
    positive_name: str,
    negative_name: str,
) -> str:
    """
    Describe the anatomical direction of one component.
    """

    if np.isclose(
        value,
        0.0,
        atol=1e-6,
    ):
        return "none"

    if value > 0.0:
        return positive_name

    return negative_name


def _formatted_condyle(
    name: str,
    vector: np.ndarray,
) -> str:
    """
    Format one condylar displacement in local axes.
    """

    distance = float(
        np.linalg.norm(vector)
    )

    x_direction = _movement_direction(
        vector[0],
        "right",
        "left",
    )

    y_direction = _movement_direction(
        vector[1],
        "anterior",
        "posterior",
    )

    z_direction = _movement_direction(
        vector[2],
        "superior",
        "inferior",
    )

    return "\n".join(
        [
            f"{name}: {distance:.4f} mm",
            (
                f"  X {vector[0]:+.4f} mm "
                f"({x_direction})"
            ),
            (
                f"  Y {vector[1]:+.4f} mm "
                f"({y_direction})"
            ),
            (
                f"  Z {vector[2]:+.4f} mm "
                f"({z_direction})"
            ),
        ]
    )


def _add_action_button(
    plotter: pv.Plotter,
    label: str,
    action,
    position: tuple[int, int],
    color: str,
) -> None:
    """
    Add one neutral push-like action button.
    """

    plotter.add_checkbox_button_widget(
        callback=lambda _checked: action(),
        value=False,
        position=position,
        size=24,
        border_size=2,
        color_on=color,
        color_off=color,
        background_color="dimgray",
    )

    plotter.add_text(
        label,
        position=(
            position[0] + 32,
            position[1] + 4,
        ),
        font_size=9,
    )


def main() -> None:
    print("=" * 64)
    print("OGDD - CENTRIC RELATION DISPLACEMENT DEMO")
    print("=" * 64)

    maxillary_path = _require_file(
        Path(
            "data/modelos/Maxillary Anatomy.stl"
        )
    )

    mandibular_rc_path = _require_file(
        Path(
            "data/modelos/"
            "Mandibular Anatomy_Relacion centrica.stl"
        )
    )

    mic_record_path = _require_file(
        Path(
            "data/registro/registro.stl"
        )
    )

    print("\nLoading RC mounting and MIC record...")

    maxillary_mesh = STLReader.read(
        maxillary_path
    )

    mandibular_rc_mesh = STLReader.read(
        mandibular_rc_path
    )

    mic_combined_mesh = STLReader.read(
        mic_record_path
    )

    print(
        "Maxillary vertices    : "
        f"{maxillary_mesh.vertex_count}"
    )

    print(
        "Mandibular RC vertices: "
        f"{mandibular_rc_mesh.vertex_count}"
    )

    print(
        "MIC record vertices   : "
        f"{mic_combined_mesh.vertex_count}"
    )

    print("\nSeparating MIC record regions...")

    builder_started = perf_counter()

    mic_record = (
        OcclusalRecordBuilder
        .from_seed_vertices(
            mesh=mic_combined_mesh,
            maxillary_seed_vertex=(
                MAXILLARY_SEED_VERTEX
            ),
            mandibular_seed_vertex=(
                MANDIBULAR_SEED_VERTEX
            ),
        )
    )

    builder_elapsed = (
        perf_counter()
        - builder_started
    )

    print(
        "Maxillary registration points: "
        f"{len(mic_record.maxillary_vertex_indices)}"
    )

    print(
        "Mandibular registration points: "
        f"{len(mic_record.mandibular_vertex_indices)}"
    )

    print(
        "Record separation time        : "
        f"{builder_elapsed:.3f} s"
    )

    print("\nRegistering MIC record to RC mount...")

    registration_started = perf_counter()

    registration = (
        CentricRelationRegistration
        .register(
            maxillary_mesh=maxillary_mesh,
            mandibular_rc_mesh=(
                mandibular_rc_mesh
            ),
            mic_record=mic_record,
            maximum_iterations=50,
            tolerance=1e-6,
            trim_fraction=0.90,
            sample_size=10000,
        )
    )

    registration_elapsed = (
        perf_counter()
        - registration_started
    )

    print(
        "Registration converged: "
        f"{registration.converged}"
    )

    print(
        "Maxillary RMSE        : "
        f"{registration.maxillary_registration.root_mean_square_error:.8f} mm"
    )

    print(
        "Mandibular RMSE       : "
        f"{registration.mandibular_registration.root_mean_square_error:.8f} mm"
    )

    print(
        "Registration time     : "
        f"{registration_elapsed:.3f} s"
    )

    if not registration.converged:
        raise RuntimeError(
            "RC-MIC registration did not converge."
        )

    # These points were digitized on the complete MIC
    # mandibular model used in the original articulator
    # demo. They are first anchored to the maxillary
    # mount and then returned to RC with the mandibular
    # registration transform.
    mic_landmark_points = np.array(
        [
            [
                -2.764405,
                -23.366814,
                3.742300,
            ],
            [
                -26.484565,
                13.030479,
                0.946036,
            ],
            [
                29.139037,
                13.553044,
                0.271681,
            ],
        ]
    )

    positioned_mic_landmarks = (
        registration
        .record_to_mount_transform
        .apply(mic_landmark_points)
    )

    rc_landmark_points = (
        registration
        .mandibular_mic_to_rc_transform
        .apply(positioned_mic_landmarks)
    )

    model = DentalModel(
        mandibular_rc_mesh
    )

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=rc_landmark_points[0],
        reference_used=(
            "MIC landmark transformed to RC"
        ),
    )

    right_second_molar = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=rc_landmark_points[1],
        reference_used=(
            "MIC landmark transformed to RC"
        ),
    )

    left_second_molar = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=rc_landmark_points[2],
        reference_used=(
            "MIC landmark transformed to RC"
        ),
    )

    model.add_landmark(midline)
    model.add_landmark(right_second_molar)
    model.add_landmark(left_second_molar)

    coordinate_system = model.coordinate_system

    virtual_bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=ArticulatorConfiguration(),
    )

    displacement = (
        registration
        .condylar_displacement(
            right_condyle_point=(
                virtual_bonwill
                .right_condyle
                .point
            ),
            left_condyle_point=(
                virtual_bonwill
                .left_condyle
                .point
            ),
        )
    )

    mandibular_mic_vertices = (
        registration
        .mandibular_rc_to_mic_transform
        .apply(
            mandibular_rc_mesh.vertices
        )
    )

    maxillary_local = coordinate_system.to_local(
        maxillary_mesh.vertices
    )

    mandibular_rc_local = (
        coordinate_system.to_local(
            mandibular_rc_mesh.vertices
        )
    )

    mandibular_mic_local = (
        coordinate_system.to_local(
            mandibular_mic_vertices
        )
    )

    right_rc_local = coordinate_system.to_local(
        displacement.right_rc_point
    )

    right_mic_local = coordinate_system.to_local(
        displacement.right_mic_point
    )

    left_rc_local = coordinate_system.to_local(
        displacement.left_rc_point
    )

    left_mic_local = coordinate_system.to_local(
        displacement.left_mic_point
    )

    right_vector_local = (
        right_mic_local
        - right_rc_local
    )

    left_vector_local = (
        left_mic_local
        - left_rc_local
    )

    print("\nCONDYLAR DISPLACEMENT RC -> MIC")
    print("Local axes: +X right | +Y anterior | +Z superior")
    print(
        _formatted_condyle(
            "RIGHT CONDYLE",
            right_vector_local,
        )
    )
    print(
        _formatted_condyle(
            "LEFT CONDYLE",
            left_vector_local,
        )
    )

    maxillary_surface = _surface(
        vertices=maxillary_local,
        faces=maxillary_mesh.faces,
    )

    mandibular_rc_surface = _surface(
        vertices=mandibular_rc_local,
        faces=mandibular_rc_mesh.faces,
    )

    mandibular_mic_surface = _surface(
        vertices=mandibular_mic_local,
        faces=mandibular_rc_mesh.faces,
    )

    right_rc_condyle = pv.Sphere(
        radius=CONDYLE_RADIUS_MM,
        center=right_rc_local,
        theta_resolution=48,
        phi_resolution=48,
    )

    left_rc_condyle = pv.Sphere(
        radius=CONDYLE_RADIUS_MM,
        center=left_rc_local,
        theta_resolution=48,
        phi_resolution=48,
    )

    right_mic_condyle = pv.Sphere(
        radius=CONDYLE_RADIUS_MM,
        center=right_mic_local,
        theta_resolution=48,
        phi_resolution=48,
    )

    left_mic_condyle = pv.Sphere(
        radius=CONDYLE_RADIUS_MM,
        center=left_mic_local,
        theta_resolution=48,
        phi_resolution=48,
    )

    rc_hinge_line = pv.Line(
        right_rc_local,
        left_rc_local,
    )

    mic_hinge_line = pv.Line(
        right_mic_local,
        left_mic_local,
    )

    right_arrow = pv.Arrow(
        start=right_rc_local,
        direction=right_vector_local,
        scale=(
            np.linalg.norm(right_vector_local)
            * VECTOR_DISPLAY_MAGNIFICATION
        ),
        tip_length=0.18,
        tip_radius=0.04,
        shaft_radius=0.015,
    )

    left_arrow = pv.Arrow(
        start=left_rc_local,
        direction=left_vector_local,
        scale=(
            np.linalg.norm(left_vector_local)
            * VECTOR_DISPLAY_MAGNIFICATION
        ),
        tip_length=0.18,
        tip_radius=0.04,
        shaft_radius=0.015,
    )

    plotter = pv.Plotter(
        window_size=(1600, 900)
    )

    plotter.add_mesh(
        maxillary_surface,
        color="mistyrose",
        opacity=0.55,
        show_edges=False,
    )

    rc_actor = plotter.add_mesh(
        mandibular_rc_surface,
        color="lightblue",
        opacity=0.70,
        show_edges=False,
    )

    mic_actor = plotter.add_mesh(
        mandibular_mic_surface,
        color="gold",
        opacity=0.55,
        show_edges=False,
    )

    right_rc_condyle_actor = plotter.add_mesh(
        right_rc_condyle,
        color="cornflowerblue",
        opacity=0.85,
        smooth_shading=True,
        label="Condyles in RC",
    )

    left_rc_condyle_actor = plotter.add_mesh(
        left_rc_condyle,
        color="cornflowerblue",
        opacity=0.85,
        smooth_shading=True,
    )

    right_mic_condyle_actor = plotter.add_mesh(
        right_mic_condyle,
        color="gold",
        opacity=0.70,
        smooth_shading=True,
        label="Condyles in MIC",
    )

    left_mic_condyle_actor = plotter.add_mesh(
        left_mic_condyle,
        color="gold",
        opacity=0.70,
        smooth_shading=True,
    )

    rc_hinge_actor = plotter.add_mesh(
        rc_hinge_line,
        color="cornflowerblue",
        line_width=5,
        render_lines_as_tubes=True,
    )

    mic_hinge_actor = plotter.add_mesh(
        mic_hinge_line,
        color="goldenrod",
        line_width=5,
        render_lines_as_tubes=True,
    )

    right_arrow_actor = plotter.add_mesh(
        right_arrow,
        color="deepskyblue",
        label=(
            "Right displacement direction"
        ),
    )

    left_arrow_actor = plotter.add_mesh(
        left_arrow,
        color="magenta",
        label=(
            "Left displacement direction"
        ),
    )

    result_text = "\n".join(
        [
            "RC -> MIC CONDYLAR DISPLACEMENT",
            "Local axes: +X right | +Y anterior | +Z superior",
            "",
            _formatted_condyle(
                "RIGHT CONDYLE",
                right_vector_local,
            ),
            "",
            _formatted_condyle(
                "LEFT CONDYLE",
                left_vector_local,
            ),
            "",
            (
                "Direction arrows enlarged "
                f"x{VECTOR_DISPLAY_MAGNIFICATION:.0f}"
            ),
            "Distances and mandibular positions use true scale.",
        ]
    )

    results_actor = plotter.add_text(
        result_text,
        position=(930, 570),
        font_size=9,
        name="registration_results",
    )

    results_actor.SetVisibility(False)
    right_arrow_actor.SetVisibility(False)
    left_arrow_actor.SetVisibility(False)

    plotter.add_text(
        (
            "RC-MIC displacement | R: RC | M: MIC | "
            "O: overlay | I: ICP result"
        ),
        position="upper_left",
        font_size=12,
    )

    plotter.add_legend(
        bcolor="white",
        face=None,
        size=(0.24, 0.18),
    )

    plotter.show_axes()

    rc_condyle_actors = (
        right_rc_condyle_actor,
        left_rc_condyle_actor,
        rc_hinge_actor,
    )

    mic_condyle_actors = (
        right_mic_condyle_actor,
        left_mic_condyle_actor,
        mic_hinge_actor,
    )

    def set_actor_visibility(
        actors,
        visible: bool,
    ) -> None:
        """
        Set one group of related actors visible or hidden.
        """

        for actor in actors:
            actor.SetVisibility(
                visible
            )

    def show_rc() -> None:
        """
        Show the mandibular RC position.
        """

        rc_actor.SetVisibility(True)
        mic_actor.SetVisibility(False)
        set_actor_visibility(
            rc_condyle_actors,
            True,
        )
        set_actor_visibility(
            mic_condyle_actors,
            False,
        )
        plotter.render()

    def show_mic() -> None:
        """
        Show the mandibular MIC position.
        """

        rc_actor.SetVisibility(False)
        mic_actor.SetVisibility(True)
        set_actor_visibility(
            rc_condyle_actors,
            False,
        )
        set_actor_visibility(
            mic_condyle_actors,
            True,
        )
        plotter.render()

    def show_overlay() -> None:
        """
        Superimpose mandibular RC and MIC positions.
        """

        rc_actor.SetVisibility(True)
        mic_actor.SetVisibility(True)
        set_actor_visibility(
            rc_condyle_actors,
            True,
        )
        set_actor_visibility(
            mic_condyle_actors,
            True,
        )
        plotter.render()

    results_visible = {
        "value": False,
    }

    def toggle_icp_results() -> None:
        """
        Show or hide numeric ICP results and arrows.
        """

        results_visible["value"] = not (
            results_visible["value"]
        )

        visible = results_visible["value"]

        results_actor.SetVisibility(
            visible
        )

        right_arrow_actor.SetVisibility(
            visible
        )

        left_arrow_actor.SetVisibility(
            visible
        )

        plotter.render()

    _add_action_button(
        plotter=plotter,
        label="SHOW RC",
        action=show_rc,
        position=(1030, 95),
        color="lightblue",
    )

    _add_action_button(
        plotter=plotter,
        label="SHOW MIC",
        action=show_mic,
        position=(1210, 95),
        color="gold",
    )

    _add_action_button(
        plotter=plotter,
        label="OVERLAY RC / MIC",
        action=show_overlay,
        position=(1380, 95),
        color="mediumorchid",
    )

    _add_action_button(
        plotter=plotter,
        label="ICP RESULT X / Y / Z",
        action=toggle_icp_results,
        position=(1210, 50),
        color="limegreen",
    )

    plotter.add_key_event(
        "r",
        show_rc,
    )

    plotter.add_key_event(
        "m",
        show_mic,
    )

    plotter.add_key_event(
        "o",
        show_overlay,
    )

    plotter.add_key_event(
        "i",
        toggle_icp_results,
    )

    plotter.view_isometric()
    plotter.reset_camera()

    print("\nOpening RC-MIC displacement viewer...")

    plotter.show()


if __name__ == "__main__":
    main()
