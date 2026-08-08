
"""
OGDD - First Pipeline

Primer ejemplo completo del flujo de trabajo de OGDD.
"""

from pathlib import Path

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

    stl = Path("data/stl/Mandibular Anatomy_Mordida normal.stl")

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
        point=np.array([0.0, 0.0, 0.0]),
        reference_used="manual",
    )

    right_second_molar = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="manual",
    )

    left_second_molar = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array([-50.0, 0.0, 0.0]),
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

    model.add_landmark(right_second_molar)
    model.add_landmark(left_second_molar)

    print(f"\nLandmarks     : {model.landmark_count}")
    print(f"Balkwill ready : {model.is_balkwill_ready}")

    # --------------------------------------------------
    # 4. Build Balkwill triangle
    # --------------------------------------------------

    if not model.is_balkwill_ready:
        raise RuntimeError(
            "DentalModel is not ready for Balkwill triangle."
        )

    triangle = model.balkwill_triangle

    # --------------------------------------------------
    # 5. Display Balkwill geometry
    # --------------------------------------------------

    print("\nBalkwill Triangle")
    print("-" * 30)

    print(f"Right side           : {triangle.right_side}")
    print(f"Left side            : {triangle.left_side}")
    print(f"Intermolar width     : {triangle.intermolar_width}")
    print(f"Symmetry difference  : {triangle.symmetry_difference}")


if __name__ == "__main__":
    main()