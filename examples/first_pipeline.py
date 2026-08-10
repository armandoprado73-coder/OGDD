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
    # 4. Build Balkwill triangle
    # --------------------------------------------------

    if not model.is_balkwill_ready:
        raise RuntimeError(
            "DentalModel is not ready for Balkwill triangle."
        )

    balkwill = model.balkwill_triangle

    # --------------------------------------------------
    # 5. Display Balkwill geometry
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
    # 6. Build Bonwill triangle
    # --------------------------------------------------

    if not model.is_bonwill_ready:
        raise RuntimeError(
            "DentalModel is not ready for Bonwill triangle."
        )

    bonwill = model.bonwill_triangle

    # --------------------------------------------------
    # 7. Display Bonwill geometry
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