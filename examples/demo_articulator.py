"""
OGDD - Interactive Articulator Demo

Builds a mandibular assembly from a real dental
model and prepares it for interactive movement.
"""

from pathlib import Path
import pyvista as pv
import numpy as np

from ogdd.anatomy.dental_model import DentalModel
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.bonwill_builder import (
    BonwillBuilder,
)
from ogdd.articulator.configuration import (
    ArticulatorConfiguration,
)
from ogdd.articulator.mandibular_controller import (
    MandibularController,
)
from ogdd.io.stl import STLReader


def main() -> None:
    print("=" * 50)
    print("OGDD - INTERACTIVE ARTICULATOR DEMO")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Load real mandibular mesh
    # --------------------------------------------------

    mandibular_stl = Path(
        "data/stl/Mandibular Anatomy_Mordida normal.stl"
    )

    maxillary_stl = Path(
        "data/stl/Maxillary Anatomy.stl"
    )

    print("\nLoading dental meshes...")

    mesh = STLReader.read(
        mandibular_stl
    )

    maxillary_mesh = STLReader.read(
        maxillary_stl
    )

    model = DentalModel(mesh)
    # --------------------------------------------------
    # 2. Define real anatomical landmarks
    # --------------------------------------------------

    midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([
            -2.764405,
            -23.366814,
            3.742300,
        ]),
        reference_used="manual",
    )

    right_second_molar = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array([
            -26.484565,
            13.030479,
            0.946036,
        ]),
        reference_used="manual",
    )

    left_second_molar = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array([
            29.139037,
            13.553044,
            0.271681,
        ]),
        reference_used="manual",
    )

    model.add_landmark(midline)
    model.add_landmark(right_second_molar)
    model.add_landmark(left_second_molar)

    # --------------------------------------------------
    # 3. Build anatomical structures
    # --------------------------------------------------

    coordinate_system = model.coordinate_system

    balkwill = model.balkwill_triangle

    configuration = ArticulatorConfiguration()

    virtual_bonwill = BonwillBuilder.build(
        coordinate_system=coordinate_system,
        dental_midline=midline,
        configuration=configuration,
    )

    model.add_landmark(
        virtual_bonwill.right_condyle
    )

    model.add_landmark(
        virtual_bonwill.left_condyle
    )

    bonwill = model.bonwill_triangle

    hinge_axis = HingeAxis(
        left_condyle=bonwill.left_condyle,
        right_condyle=bonwill.right_condyle,
    )

    # --------------------------------------------------
    # 4. Build movable mandibular assembly
    # --------------------------------------------------

    assembly = MandibularAssembly(
        mesh=model.mesh,
        balkwill=balkwill,
        bonwill=bonwill,
        hinge_axis=hinge_axis,
    )

    controller = MandibularController(
        assembly=assembly,
        maximum_angle_degrees=30.0,
        step_degrees=1.0,
    )

    # --------------------------------------------------
    # 5. Verify movement with the real mesh
    # --------------------------------------------------

    closed_position = controller.position

    opened_position = controller.open()

    mesh_changed = not np.allclose(
        closed_position.mesh.vertices,
        opened_position.mesh.vertices,
    )

    controller.reset()

    print("\nArticulator ready!")
    print(f"Mesh vertices   : {model.mesh.vertex_count}")
    print(f"Hinge length    : {hinge_axis.length:.2f} mm")
    print(
        f"Maximum opening : "
        f"{controller.maximum_angle_degrees:.2f} deg"
    )
    print(
        f"Opening step    : "
        f"{controller.step_degrees:.2f} deg"
    )
    print(f"Real mesh moved : {mesh_changed}")
    print(
        f"Current angle   : "
        f"{controller.angle_degrees:.2f} deg"
    )

    # --------------------------------------------------
    # 6. Build interactive visualization
    # --------------------------------------------------

    def triangle_points_local(
        triangle,
    ) -> np.ndarray:
        """
        Returns a closed triangle in anatomical
        coordinates for interactive visualization.
        """

        world_points = np.array([
            triangle.a,
            triangle.b,
            triangle.c,
            triangle.a,
        ])

        return coordinate_system.to_local(
            world_points
        )

    def bonwill_sides_local(
        bonwill_triangle,
    ) -> np.ndarray:
        """
        Returns only the two anterior Bonwill sides.

        The condylar side is represented separately
        by the hinge axis.
        """

        triangle = bonwill_triangle.triangle

        world_points = np.array([
            triangle.a,
            triangle.c,
            triangle.b,
        ])

        return coordinate_system.to_local(
            world_points
        )

    def moving_landmarks_local(
        position,
    ) -> np.ndarray:
        """
        Returns the moving mandibular landmarks in
        anatomical coordinates.
        """

        world_points = np.array([
            position.balkwill.dental_midline.point,
            position.balkwill.right_posterior.point,
            position.balkwill.left_posterior.point,
        ])

        return coordinate_system.to_local(
            world_points
        )

    initial_position = controller.position

    faces = np.hstack(
        [
            np.full(
                (len(initial_position.mesh.faces), 1),
                3,
                dtype=int,
            ),
            initial_position.mesh.faces,
        ]
    ).ravel()

    mandibular_surface = pv.PolyData(
        coordinate_system.to_local(
            initial_position.mesh.vertices
        ),
        faces,
    )

    maxillary_faces = np.hstack(
        [
            np.full(
                (len(maxillary_mesh.faces), 1),
                3,
                dtype=int,
            ),
            maxillary_mesh.faces,
        ]
    ).ravel()

    maxillary_surface = pv.PolyData(
        coordinate_system.to_local(
            maxillary_mesh.vertices
        ),
        maxillary_faces,
    )

    balkwill_line = pv.lines_from_points(
        triangle_points_local(
            initial_position.balkwill.triangle
        )
    )

    bonwill_line = pv.lines_from_points(
        bonwill_sides_local(
            initial_position.bonwill
        )
    )

    landmark_cloud = pv.PolyData(
        moving_landmarks_local(
            initial_position
        )
    )

    hinge_points_local = coordinate_system.to_local(
        np.array([
            hinge_axis.right_condyle.point,
            hinge_axis.left_condyle.point,
        ])
    )

    hinge_line = pv.lines_from_points(
        hinge_points_local
    )

    plotter = pv.Plotter()

    plotter.add_mesh(
        maxillary_surface,
        color="mistyrose",
        opacity=0.55,
        show_edges=False,
    )

    plotter.add_mesh(
        mandibular_surface,
        color="lightblue",
        show_edges=False,
    )

    plotter.add_mesh(
        balkwill_line,
        color="yellow",
        line_width=5,
    )

    plotter.add_mesh(
        bonwill_line,
        color="red",
        line_width=5,
    )

    plotter.add_mesh(
        hinge_line,
        color="black",
        line_width=8,
        render_lines_as_tubes=True,
    )

    plotter.add_mesh(
        landmark_cloud,
        color="yellow",
        point_size=14,
        render_points_as_spheres=True,
    )

    plotter.add_point_labels(
        hinge_points_local,
        [
            "RIGHT CONDYLE",
            "LEFT CONDYLE",
        ],
        point_size=12,
        font_size=14,
        render_points_as_spheres=True,
    )

    plotter.add_text(
        "Move the slider to open or close",
        position="upper_left",
        font_size=12,
    )

    plotter.show_axes()

    # --------------------------------------------------
    # 7. Connect slider to mandibular controller
    # --------------------------------------------------

    def update_opening(
        angle_degrees: float,
    ) -> None:
        position = controller.set_angle(
            angle_degrees
        )

        mandibular_surface.points = (
            coordinate_system.to_local(
                position.mesh.vertices
            )
        )

        balkwill_line.points = (
            triangle_points_local(
                position.balkwill.triangle
            )
        )

        bonwill_line.points = (
            bonwill_sides_local(
                position.bonwill
            )
        )

        landmark_cloud.points = (
            moving_landmarks_local(position)
        )

        plotter.render()

    plotter.add_slider_widget(
        callback=update_opening,
        rng=[
            0.0,
            controller.maximum_angle_degrees,
        ],
        value=controller.angle_degrees,
        title="Opening angle (degrees)",
        pointa=(0.20, 0.10),
        pointb=(0.80, 0.10),
        interaction_event="always",
    )

    plotter.view_isometric()

    print("\nOpening interactive viewer...")

    plotter.show()
if __name__ == "__main__":
    main()