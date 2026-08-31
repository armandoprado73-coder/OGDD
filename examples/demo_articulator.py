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
from ogdd.articulator.lateral_controller import (
    LateralController,
)
from ogdd.articulator.guided_lateral_excursion import (
    GuidedLateralExcursion,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)
from ogdd.articulator.guided_protrusion import (
    GuidedProtrusion,
)
from ogdd.articulator.protrusion_controller import (
    ProtrusionController,
)
from ogdd.articulator.combined_movement import (
    CombinedMovement,
)
from ogdd.articulator.combined_controller import (
    CombinedController,
)
from ogdd.io.stl import STLReader
from ogdd.articulator.condylar_guide_builder import (
    CondylarGuideBuilder,
)

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

    condylar_guides = CondylarGuideBuilder.build(
        hinge_axis=hinge_axis,
        coordinate_system=coordinate_system,
        configuration=configuration,
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

    right_excursion = GuidedLateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=coordinate_system.z_axis,
        working_side=LateralSide.RIGHT,
        balancing_guide=condylar_guides.left_guide,
    )

    left_excursion = GuidedLateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=coordinate_system.z_axis,
        working_side=LateralSide.LEFT,
        balancing_guide=condylar_guides.right_guide,
    )

    maximum_lateral_angle = min(
        10.0,
        right_excursion.maximum_angle_degrees,
        left_excursion.maximum_angle_degrees,
    )

    lateral_controller = LateralController(
        assembly=assembly,
        right_excursion=right_excursion,
        left_excursion=left_excursion,
        maximum_angle_degrees=maximum_lateral_angle,
        step_degrees=1.0,
    )

    protrusion = GuidedProtrusion(
        hinge_axis=hinge_axis,
        right_guide=condylar_guides.right_guide,
        left_guide=condylar_guides.left_guide,
    )

    protrusion_controller = ProtrusionController(
        assembly=assembly,
        protrusion=protrusion,
        maximum_distance_mm=(
            protrusion.maximum_translation
        ),
        step_mm=1.0,
    )

    combined_movement = CombinedMovement(
        assembly=assembly,
        right_excursion=right_excursion,
        left_excursion=left_excursion,
        protrusion=protrusion,
    )

    combined_controller = CombinedController(
        movement=combined_movement,
        maximum_opening_angle_degrees=(
            controller.maximum_angle_degrees
        ),
        maximum_lateral_angle_degrees=(
            lateral_controller.maximum_angle_degrees
        ),
        maximum_protrusion_distance_mm=(
            protrusion_controller.maximum_distance_mm
        ),
        opening_step_degrees=(
            controller.step_degrees
        ),
        lateral_step_degrees=(
            lateral_controller.step_degrees
        ),
        protrusion_step_mm=(
            protrusion_controller.step_mm
        ),
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

    centric_lateral_position = (
        lateral_controller.position
    )

    right_lateral_position = (
        lateral_controller.move_right()
    )

    right_lateral_mesh_changed = (
        not np.allclose(
            centric_lateral_position.mesh.vertices,
            right_lateral_position.mesh.vertices,
        )
    )

    right_working_condyle_fixed = np.allclose(
        right_lateral_position
        .bonwill
        .right_condyle
        .point,
        bonwill.right_condyle.point,
    )

    lateral_controller.reset()

    centric_combined_position = (
        combined_controller.position
    )

    combined_position = (
        combined_controller.set_position(
            opening_angle_degrees=10.0,
            lateral_angle_degrees=3.0,
            protrusion_distance_mm=5.0,
        )
    )

    combined_mesh_changed = (
        not np.allclose(
            centric_combined_position.mesh.vertices,
            combined_position.mesh.vertices,
        )
    )

    combined_controller.reset()

    centric_protrusive_position = (
        protrusion_controller.position
    )

    protrusive_position = (
        protrusion_controller.advance()
    )

    protrusive_mesh_changed = (
        not np.allclose(
            centric_protrusive_position.mesh.vertices,
            protrusive_position.mesh.vertices,
        )
    )

    right_condyle_follows_guide = np.allclose(
        protrusive_position
        .bonwill
        .right_condyle
        .point,
        protrusion.right_target_at(
            protrusion_controller.distance_mm
        ),
    )

    left_condyle_follows_guide = np.allclose(
        protrusive_position
        .bonwill
        .left_condyle
        .point,
        protrusion.left_target_at(
            protrusion_controller.distance_mm
        ),
    )

    protrusion_controller.reset()

    left_lateral_position = (
        lateral_controller.move_left()
    )

    left_lateral_mesh_changed = (
        not np.allclose(
            centric_lateral_position.mesh.vertices,
            left_lateral_position.mesh.vertices,
        )
    )

    left_working_condyle_fixed = np.allclose(
        left_lateral_position
        .bonwill
        .left_condyle
        .point,
        bonwill.left_condyle.point,
    )

    lateral_controller.reset()

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
    print(
        f"Maximum lateral : "
        f"{lateral_controller.maximum_angle_degrees:.2f} deg"
    )

    print(
        f"Lateral step    : "
        f"{lateral_controller.step_degrees:.2f} deg"
    )

    print(
        f"Right mesh moved: "
        f"{right_lateral_mesh_changed}"
    )

    print(
        f"Right work fixed: "
        f"{right_working_condyle_fixed}"
    )

    print(
        f"Left mesh moved : "
        f"{left_lateral_mesh_changed}"
    )

    print(
        f"Left work fixed : "
        f"{left_working_condyle_fixed}"
    )
    print(
        f"Maximum protrus.: "
        f"{protrusion_controller.maximum_distance_mm:.2f} mm"
    )

    print(
        f"Protrusion step : "
        f"{protrusion_controller.step_mm:.2f} mm"
    )

    print(
        f"Protrusive mesh : "
        f"{protrusive_mesh_changed}"
    )

    print(
        f"Right on guide  : "
        f"{right_condyle_follows_guide}"
    )

    print(
        f"Left on guide   : "
        f"{left_condyle_follows_guide}"
    )
    print(
        f"Combined mesh   : "
        f"{combined_mesh_changed}"
    )
    print(
        f"Right guidance : "
        f"{condylar_guides.right_guide.angle_degrees:.2f} deg"
    )

    print(
        f"Left guidance  : "
        f"{condylar_guides.left_guide.angle_degrees:.2f} deg"
    )

    print(
        f"Functional path: "
        f"{condylar_guides.right_guide.maximum_translation:.2f} mm"
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

    def quad_surface_local(
        world_vertices: np.ndarray,
    ) -> pv.PolyData:
        """
        Builds one quadrilateral surface in local
        anatomical coordinates.
        """

        local_vertices = coordinate_system.to_local(
            world_vertices
        )

        quad_face = np.array([
            4,
            0,
            1,
            2,
            3,
        ])

        return pv.PolyData(
            local_vertices,
            quad_face,
        )

    initial_position = combined_controller.position

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

    right_main_guide = quad_surface_local(
        condylar_guides
        .right_guide
        .main_surface_vertices
    )

    right_posterior_stop = quad_surface_local(
        condylar_guides
        .right_guide
        .posterior_stop_vertices
    )

    left_main_guide = quad_surface_local(
        condylar_guides
        .left_guide
        .main_surface_vertices
    )

    left_posterior_stop = quad_surface_local(
        condylar_guides
        .left_guide
        .posterior_stop_vertices
    )

    right_condyle_center = (
        coordinate_system.to_local(
            condylar_guides
            .right_guide
            .condyle_center
        )
    )

    left_condyle_center = (
        coordinate_system.to_local(
            condylar_guides
            .left_guide
            .condyle_center
        )
    )

    right_condyle_sphere = pv.Sphere(
        radius=(
            condylar_guides
            .right_guide
            .condyle_radius
        ),
        center=right_condyle_center,
        theta_resolution=48,
        phi_resolution=48,
    )

    left_condyle_sphere = pv.Sphere(
        radius=(
            condylar_guides
            .left_guide
            .condyle_radius
        ),
        center=left_condyle_center,
        theta_resolution=48,
        phi_resolution=48,
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
            "RIGHT CONDYLE (RC)",
            "LEFT CONDYLE (RC)",
        ],
        point_size=12,
        font_size=14,
        render_points_as_spheres=True,
    )

    plotter.add_text(
        "Combined movement: opening | lateral | protrusion",
        position="upper_left",
        font_size=12,
    )

    plotter.show_axes()

    plotter.add_mesh(
        right_main_guide,
        color="orange",
        opacity=0.70,
        show_edges=True,
        edge_color="darkorange",
        line_width=2,
    )

    plotter.add_mesh(
        right_posterior_stop,
        color="tomato",
        opacity=0.85,
        show_edges=True,
        edge_color="darkred",
        line_width=2,
    )

    plotter.add_mesh(
        left_main_guide,
        color="orange",
        opacity=0.70,
        show_edges=True,
        edge_color="darkorange",
        line_width=2,
    )

    plotter.add_mesh(
        left_posterior_stop,
        color="tomato",
        opacity=0.85,
        show_edges=True,
        edge_color="darkred",
        line_width=2,
    )

    plotter.add_mesh(
        right_condyle_sphere,
        color="silver",
        smooth_shading=True,
    )

    plotter.add_mesh(
        left_condyle_sphere,
        color="silver",
        smooth_shading=True,
    )

    right_condyle_sphere_initial_points = (
        right_condyle_sphere.points.copy()
    )

    left_condyle_sphere_initial_points = (
        left_condyle_sphere.points.copy()
    )

    initial_right_condyle_local = (
        coordinate_system.to_local(
            bonwill.right_condyle.point
        )
    )

    initial_left_condyle_local = (
        coordinate_system.to_local(
            bonwill.left_condyle.point
        )
    )

    # --------------------------------------------------
    # 7. Connect movement controllers
    # --------------------------------------------------

    def update_mandibular_visuals(
        position,
    ) -> None:
        """
        Update every movable mandibular structure.
        """

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

        current_hinge_points_local = (
            coordinate_system.to_local(
                np.array([
                    position
                    .bonwill
                    .right_condyle
                    .point,
                    position
                    .bonwill
                    .left_condyle
                    .point,
                ])
            )
        )

        hinge_line.points = (
            current_hinge_points_local
        )

        right_condyle_displacement = (
            current_hinge_points_local[0]
            - initial_right_condyle_local
        )

        left_condyle_displacement = (
            current_hinge_points_local[1]
            - initial_left_condyle_local
        )

        right_condyle_sphere.points = (
            right_condyle_sphere_initial_points
            + right_condyle_displacement
        )

        left_condyle_sphere.points = (
            left_condyle_sphere_initial_points
            + left_condyle_displacement
        )

        plotter.render()

    slider_widgets = {}

    def set_slider_value(
        name: str,
        value: float,
    ) -> None:
        """
        Update another slider without changing geometry.
        """

        if name not in slider_widgets:
            return

        representation = (
            slider_widgets[name]
            .GetRepresentation()
        )

        if not np.isclose(
            representation.GetValue(),
            value,
        ):
            representation.SetValue(
                value
            )

    def update_opening(
        angle_degrees: float,
    ) -> None:
        """
        Combine opening with the current movements.
        """

        position = combined_controller.set_opening(
            float(
                np.clip(
                    angle_degrees,
                    0.0,
                    combined_controller
                    .maximum_opening_angle_degrees,
                )
            )
        )

        update_mandibular_visuals(
            position
        )

    def update_lateral(
        angle_degrees: float,
    ) -> None:
        """
        Combine lateral movement with the current state.
        """

        clamped_angle = float(
            np.clip(
                angle_degrees,
                -combined_controller
                .maximum_left_lateral_angle_degrees,
                combined_controller
                .maximum_right_lateral_angle_degrees,
            )
        )

        set_slider_value(
            name="lateral",
            value=clamped_angle,
        )

        position = combined_controller.set_lateral(
            clamped_angle
        )

        update_mandibular_visuals(
            position
        )

    def update_protrusion(
        distance_mm: float,
    ) -> None:
        """
        Combine protrusion with the current state.
        """

        clamped_distance = float(
            np.clip(
                distance_mm,
                0.0,
                combined_controller
                .maximum_current_protrusion_distance_mm,
            )
        )

        set_slider_value(
            name="protrusion",
            value=clamped_distance,
        )

        position = combined_controller.set_protrusion(
            clamped_distance
        )

        update_mandibular_visuals(
            position
        )

    opening_slider = plotter.add_slider_widget(
        callback=update_opening,
        rng=[
            0.0,
            combined_controller
            .maximum_opening_angle_degrees,
        ],
        value=combined_controller.opening_angle_degrees,
        title="Opening angle",
        pointa=(0.08, 0.20),
        pointb=(0.08, 0.80),
        interaction_event="always",
    )

    slider_widgets["opening"] = (
        opening_slider
    )

    lateral_slider = plotter.add_slider_widget(
        callback=update_lateral,
        rng=[
            -combined_controller
            .maximum_lateral_angle_degrees,
            combined_controller
            .maximum_lateral_angle_degrees,
        ],
        value=combined_controller.lateral_angle_degrees,
        title="LEFT   -   RC   -   RIGHT",
        pointa=(0.25, 0.10),
        pointb=(0.80, 0.10),
        interaction_event="always",
    )

    slider_widgets["lateral"] = (
        lateral_slider
    )

    protrusion_slider = plotter.add_slider_widget(
        callback=update_protrusion,
        rng=[
            0.0,
            combined_controller
            .maximum_protrusion_distance_mm,
        ],
        value=combined_controller.protrusion_distance_mm,
        title="PROTRUSION (mm)",
        pointa=(0.25, 0.90),
        pointb=(0.80, 0.90),
        interaction_event="always",
    )

    slider_widgets["protrusion"] = (
        protrusion_slider
    )

    print("\nOpening interactive viewer...")

    plotter.show()
if __name__ == "__main__":
    main()