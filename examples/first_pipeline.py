"""
OGDD - First Pipeline

Primer ejemplo completo del flujo de trabajo de OGDD.
"""

from pathlib import Path
import pyvista as pv
import numpy as np
from ogdd.anatomy.dental_model import DentalModel
from ogdd.anatomy.landmark import Landmark
from ogdd.io.stl import STLReader


def main():
    print("=" * 50)
    print("OGDD - FIRST PIPELINE")
    print("=" * 50)

    # --------------------------------------------------
    # 1. Load dental mesh
    # --------------------------------------------------

    stl = Path(
        "data/stl/Mandibular Anatomy_Mordida normal.stl"
    )

    print("\nLoading mesh...")

    mesh = STLReader.read(stl)

    # --------------------------------------------------
    # 2. Create DentalModel
    # --------------------------------------------------

    model = DentalModel(mesh)

    print("\nDentalModel created successfully!")
    print(f"Mesh vertices : {len(model.mesh.vertices)}")
    print(f"Mesh faces    : {len(model.mesh.faces)}")
    print(f"Landmarks     : {model.landmark_count}")

    # --------------------------------------------------
    # 3. Define anatomical landmarks
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

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([100.0, 0.0, 20.0]),
        reference_used="manual",
    )

    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-100.0, 0.0, 20.0]),
        reference_used="manual",
    )

    model.add_landmark(midline)

    print("\nLandmark added successfully!")
    print(f"Landmarks     : {model.landmark_count}")
    print(
        f"Midline exists: "
        f"{model.get_landmark('DENTAL_MIDLINE') is not None}"
    )
    print(f"Balkwill ready : {model.is_balkwill_ready}")
    print(f"Bonwill ready  : {model.is_bonwill_ready}")

    model.add_landmark(right_second_molar)
    model.add_landmark(left_second_molar)
    model.add_landmark(right_condyle)
    model.add_landmark(left_condyle)

    print(f"\nLandmarks     : {model.landmark_count}")
    print(f"Balkwill ready : {model.is_balkwill_ready}")
    print(f"Bonwill ready  : {model.is_bonwill_ready}")

    # --------------------------------------------------
    # 4. Build anatomical coordinate system
    # --------------------------------------------------

    print("\nAnatomical Coordinate System")
    print("-" * 30)

    coordinate_system = model.coordinate_system

    print(f"Origin : {coordinate_system.origin}")
    print(f"X axis : {coordinate_system.x_axis}")
    print(f"Y axis : {coordinate_system.y_axis}")
    print(f"Z axis : {coordinate_system.z_axis}")

    # --------------------------------------------------
    # 5. Transform mesh to anatomical coordinates
    # --------------------------------------------------

    local_vertices = coordinate_system.to_local(
        model.mesh.vertices
    )

    print("\nMesh in Anatomical Coordinates")
    print("-" * 30)

    print(f"Vertices : {len(local_vertices)}")

    print(
        f"X range  : "
        f"{local_vertices[:, 0].min()} "
        f"to {local_vertices[:, 0].max()}"
    )

    print(
        f"Y range  : "
        f"{local_vertices[:, 1].min()} "
        f"to {local_vertices[:, 1].max()}"
    )

    print(
        f"Z range  : "
        f"{local_vertices[:, 2].min()} "
        f"to {local_vertices[:, 2].max()}"
    )

    # --------------------------------------------------
    # 6. Validate anatomical landmarks
    # --------------------------------------------------

    landmark_points = np.array(
        [
            midline.point,
            right_second_molar.point,
            left_second_molar.point,
        ]
    )

    local_landmarks = coordinate_system.to_local(
        landmark_points
    )

    print("\nLandmarks in Anatomical Coordinates")
    print("-" * 30)

    print(f"Midline     : {local_landmarks[0]}")
    print(f"Right molar : {local_landmarks[1]}")
    print(f"Left molar  : {local_landmarks[2]}")

    # --------------------------------------------------
    # 7. Visualize anatomical mesh
    # --------------------------------------------------

    faces = np.hstack(
        [
            np.full(
                (len(model.mesh.faces), 1),
                3,
                dtype=int,
            ),
            model.mesh.faces,
        ]
    ).ravel()

    anatomical_mesh = pv.PolyData(
        local_vertices,
        faces,
    )

    plotter = pv.Plotter()

    plotter.add_mesh(
        anatomical_mesh,
        show_edges=False,
    )

    plotter.show_axes()

    # --------------------------------------------------
    # Anatomical landmarks
    # --------------------------------------------------

    plotter.add_point_labels(
        local_landmarks,
        [
            "DENTAL MIDLINE",
            "RIGHT MOLAR",
            "LEFT MOLAR",
        ],
        point_size=12,
        font_size=14,
        render_points_as_spheres=True,
    )

    # --------------------------------------------------
    # OGDD anatomical axes
    # --------------------------------------------------

    origin = np.array(
        [[0.0, 0.0, 0.0]]
    )

    axis_length = 15.0

    plotter.add_arrows(
        origin,
        np.array([[1.0, 0.0, 0.0]]),
        mag=axis_length,
        color="red",
    )

    plotter.add_arrows(
        origin,
        np.array([[0.0, 1.0, 0.0]]),
        mag=axis_length,
        color="green",
    )

    plotter.add_arrows(
        origin,
        np.array([[0.0, 0.0, 1.0]]),
        mag=axis_length,
        color="blue",
    )
    print("\nOpening anatomical mesh viewer...")

    plotter.show()
    # --------------------------------------------------
    # 8. Build Balkwill triangle
    # --------------------------------------------------

    if not model.is_balkwill_ready:
        raise RuntimeError(
            "DentalModel is not ready for Balkwill triangle."
        )

    balkwill = model.balkwill_triangle

    # --------------------------------------------------
    # 9. Display Balkwill geometry
    # --------------------------------------------------

    print("\nBalkwill Triangle")
    print("-" * 30)

    print(f"Right side           : {balkwill.right_side}")
    print(f"Left side            : {balkwill.left_side}")
    print(f"Intermolar width     : {balkwill.intermolar_width}")
    print(
        f"Symmetry difference  : "
        f"{balkwill.symmetry_difference}"
    )

    # --------------------------------------------------
    # 10. Build Bonwill triangle
    # --------------------------------------------------

    if not model.is_bonwill_ready:
        raise RuntimeError(
            "DentalModel is not ready for Bonwill triangle."
        )

    bonwill = model.bonwill_triangle

    # --------------------------------------------------
    # 11. Display Bonwill geometry
    # --------------------------------------------------

    print("\nBonwill Triangle")
    print("-" * 30)

    print(f"Right side           : {bonwill.right_side}")
    print(f"Left side            : {bonwill.left_side}")
    print(f"Condylar width       : {bonwill.condylar_width}")
    print(
        f"Symmetry difference  : "
        f"{bonwill.symmetry_difference}"
    )

if __name__ == "__main__":
    main()