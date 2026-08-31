import math

import numpy as np
import pytest

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.combined_movement import (
    CombinedMovement,
    MandibularCombinedPosition,
)
from ogdd.articulator.condylar_guide import (
    CondylarGuide,
)
from ogdd.articulator.guided_lateral_excursion import (
    GuidedLateralExcursion,
)
from ogdd.articulator.guided_protrusion import (
    GuidedProtrusion,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)
from ogdd.geometry.coordinate_system import (
    CoordinateSystem,
)
from ogdd.mesh import Mesh


@pytest.fixture
def coordinate_system() -> CoordinateSystem:
    return CoordinateSystem.identity()


@pytest.fixture
def hinge_axis() -> HingeAxis:
    return HingeAxis(
        left_condyle=Landmark(
            name="LEFT_CONDYLE",
            point=np.array([-55.0, 0.0, 0.0]),
            reference_used="condylar center",
        ),
        right_condyle=Landmark(
            name="RIGHT_CONDYLE",
            point=np.array([55.0, 0.0, 0.0]),
            reference_used="condylar center",
        ),
    )


@pytest.fixture
def left_guide(
    coordinate_system: CoordinateSystem,
) -> CondylarGuide:
    return CondylarGuide(
        condyle_center=np.array([-55.0, 0.0, 0.0]),
        coordinate_system=coordinate_system,
        angle_degrees=45.0,
    )


@pytest.fixture
def right_guide(
    coordinate_system: CoordinateSystem,
) -> CondylarGuide:
    return CondylarGuide(
        condyle_center=np.array([55.0, 0.0, 0.0]),
        coordinate_system=coordinate_system,
        angle_degrees=45.0,
    )


@pytest.fixture
def right_excursion(
    hinge_axis: HingeAxis,
    left_guide: CondylarGuide,
) -> GuidedLateralExcursion:
    return GuidedLateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=np.array([0.0, 0.0, 1.0]),
        working_side=LateralSide.RIGHT,
        balancing_guide=left_guide,
    )


@pytest.fixture
def left_excursion(
    hinge_axis: HingeAxis,
    right_guide: CondylarGuide,
) -> GuidedLateralExcursion:
    return GuidedLateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=np.array([0.0, 0.0, 1.0]),
        working_side=LateralSide.LEFT,
        balancing_guide=right_guide,
    )


@pytest.fixture
def protrusion(
    hinge_axis: HingeAxis,
    right_guide: CondylarGuide,
    left_guide: CondylarGuide,
) -> GuidedProtrusion:
    return GuidedProtrusion(
        hinge_axis=hinge_axis,
        right_guide=right_guide,
        left_guide=left_guide,
    )


@pytest.fixture
def assembly(
    hinge_axis: HingeAxis,
) -> MandibularAssembly:
    dental_midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
    )

    balkwill = BalkwillTriangle(
        left_posterior=Landmark(
            name="LEFT_SECOND_MOLAR",
            point=np.array([-50.0, 20.0, 0.0]),
            reference_used="distobuccal cusp",
        ),
        right_posterior=Landmark(
            name="RIGHT_SECOND_MOLAR",
            point=np.array([50.0, 20.0, 0.0]),
            reference_used="distobuccal cusp",
        ),
        dental_midline=dental_midline,
    )

    bonwill = BonwillTriangle(
        left_condyle=hinge_axis.left_condyle,
        right_condyle=hinge_axis.right_condyle,
        dental_midline=dental_midline,
    )

    mesh = Mesh(
        vertices=np.array([
            [-50.0, 20.0, 0.0],
            [50.0, 20.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 60.0, -20.0],
        ]),
        faces=np.array([
            [0, 1, 2],
            [0, 2, 3],
        ]),
        normals=np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        attributes={
            "test_value": np.array([
                1.0,
                2.0,
                3.0,
                4.0,
            ]),
        },
        metadata={
            "source": "synthetic",
        },
    )

    return MandibularAssembly(
        mesh=mesh,
        balkwill=balkwill,
        bonwill=bonwill,
        hinge_axis=hinge_axis,
    )


@pytest.fixture
def combined_movement(
    assembly: MandibularAssembly,
    right_excursion: GuidedLateralExcursion,
    left_excursion: GuidedLateralExcursion,
    protrusion: GuidedProtrusion,
) -> CombinedMovement:
    return CombinedMovement(
        assembly=assembly,
        right_excursion=right_excursion,
        left_excursion=left_excursion,
        protrusion=protrusion,
    )


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        points[:, None, :]
        - points[None, :, :],
        axis=2,
    )


def test_zero_returns_original_geometry(
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=0.0,
        lateral_angle_degrees=0.0,
        protrusion_distance_mm=0.0,
    )

    assert np.allclose(
        position.mesh.vertices,
        assembly.mesh.vertices,
    )

    assert position.working_side is None


def test_pure_opening_matches_assembly(
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
) -> None:
    combined = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=0.0,
        protrusion_distance_mm=0.0,
    )

    opening = assembly.position_at(
        angle_degrees=12.0
    )

    assert np.allclose(
        combined.mesh.vertices,
        opening.mesh.vertices,
    )


def test_pure_protrusion_matches_existing_movement(
    combined_movement: CombinedMovement,
    protrusion: GuidedProtrusion,
    assembly: MandibularAssembly,
) -> None:
    combined = combined_movement.position_at(
        opening_angle_degrees=0.0,
        lateral_angle_degrees=0.0,
        protrusion_distance_mm=5.0,
    )

    protrusive = protrusion.position_at(
        assembly=assembly,
        distance=5.0,
    )

    assert np.allclose(
        combined.mesh.vertices,
        protrusive.mesh.vertices,
    )


@pytest.mark.parametrize(
    (
        "lateral_angle_degrees",
        "fixture_name",
        "working_side",
    ),
    [
        (3.0, "right_excursion", LateralSide.RIGHT),
        (-3.0, "left_excursion", LateralSide.LEFT),
    ],
)
def test_pure_lateral_matches_existing_movement(
    lateral_angle_degrees: float,
    fixture_name: str,
    working_side: LateralSide,
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
    request,
) -> None:
    excursion = request.getfixturevalue(
        fixture_name
    )

    combined = combined_movement.position_at(
        opening_angle_degrees=0.0,
        lateral_angle_degrees=(
            lateral_angle_degrees
        ),
        protrusion_distance_mm=0.0,
    )

    lateral = excursion.position_at(
        assembly=assembly,
        angle_degrees=abs(
            lateral_angle_degrees
        ),
    )

    assert np.allclose(
        combined.mesh.vertices,
        lateral.mesh.vertices,
    )

    assert combined.working_side is working_side


def test_combined_position_records_all_components(
    combined_movement: CombinedMovement,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=-3.0,
        protrusion_distance_mm=5.0,
    )

    assert isinstance(
        position,
        MandibularCombinedPosition,
    )

    assert position.opening_angle_degrees == (
        pytest.approx(12.0)
    )
    assert position.lateral_angle_degrees == (
        pytest.approx(-3.0)
    )
    assert position.protrusion_distance_mm == (
        pytest.approx(5.0)
    )
    assert position.working_side is LateralSide.LEFT


def test_opening_uses_mobile_hinge_axis(
    combined_movement: CombinedMovement,
) -> None:
    preopening = combined_movement.position_at(
        opening_angle_degrees=0.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    combined = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    expected_vertices = (
        preopening.hinge_axis.rotate_points(
            points=preopening.mesh.vertices,
            angle_degrees=12.0,
        )
    )

    assert np.allclose(
        combined.mesh.vertices,
        expected_vertices,
    )


def test_opening_does_not_move_repositioned_condyles(
    combined_movement: CombinedMovement,
) -> None:
    preopening = combined_movement.position_at(
        opening_angle_degrees=0.0,
        lateral_angle_degrees=-3.0,
        protrusion_distance_mm=5.0,
    )

    opened = combined_movement.position_at(
        opening_angle_degrees=15.0,
        lateral_angle_degrees=-3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.allclose(
        opened.bonwill.left_condyle.point,
        preopening.bonwill.left_condyle.point,
    )
    assert np.allclose(
        opened.bonwill.right_condyle.point,
        preopening.bonwill.right_condyle.point,
    )


def test_combined_movement_is_rigid(
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.allclose(
        pairwise_distances(position.mesh.vertices),
        pairwise_distances(assembly.mesh.vertices),
    )


def test_combined_movement_preserves_triangles(
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=-3.0,
        protrusion_distance_mm=5.0,
    )

    assert position.balkwill.intermolar_width == (
        pytest.approx(
            assembly.balkwill.intermolar_width
        )
    )
    assert position.bonwill.condylar_width == (
        pytest.approx(
            assembly.bonwill.condylar_width
        )
    )


def test_all_structures_move_together(
    combined_movement: CombinedMovement,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.allclose(
        position.mesh.vertices[2],
        position.balkwill.dental_midline.point,
    )
    assert np.allclose(
        position.balkwill.dental_midline.point,
        position.bonwill.dental_midline.point,
    )


def test_hinge_axis_matches_transformed_bonwill(
    combined_movement: CombinedMovement,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.allclose(
        position.hinge_axis.left_condyle.point,
        position.bonwill.left_condyle.point,
    )
    assert np.allclose(
        position.hinge_axis.right_condyle.point,
        position.bonwill.right_condyle.point,
    )


def test_mesh_data_and_normal_lengths_are_preserved(
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
) -> None:
    position = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=-3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.array_equal(
        position.mesh.faces,
        assembly.mesh.faces,
    )
    assert np.array_equal(
        position.mesh.attributes["test_value"],
        assembly.mesh.attributes["test_value"],
    )
    assert position.mesh.metadata == (
        assembly.mesh.metadata
    )
    assert np.allclose(
        np.linalg.norm(position.mesh.normals, axis=1),
        np.linalg.norm(assembly.mesh.normals, axis=1),
    )


def test_positions_do_not_accumulate_transformations(
    combined_movement: CombinedMovement,
) -> None:
    combined_movement.position_at(
        opening_angle_degrees=5.0,
        lateral_angle_degrees=-2.0,
        protrusion_distance_mm=2.0,
    )

    result = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    direct = combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.allclose(
        result.mesh.vertices,
        direct.mesh.vertices,
    )


def test_combined_movement_does_not_modify_original(
    combined_movement: CombinedMovement,
    assembly: MandibularAssembly,
) -> None:
    original_vertices = assembly.mesh.vertices.copy()

    combined_movement.position_at(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    assert np.allclose(
        assembly.mesh.vertices,
        original_vertices,
    )


@pytest.mark.parametrize(
    "opening_angle_degrees",
    [
        -1.0,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_opening(
    combined_movement: CombinedMovement,
    opening_angle_degrees: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="opening angle",
    ):
        combined_movement.position_at(
            opening_angle_degrees=(
                opening_angle_degrees
            ),
            lateral_angle_degrees=0.0,
            protrusion_distance_mm=0.0,
        )


@pytest.mark.parametrize(
    "lateral_angle_degrees",
    [
        -9.0,
        9.0,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_lateral_angle(
    combined_movement: CombinedMovement,
    lateral_angle_degrees: float,
) -> None:
    with pytest.raises(ValueError):
        combined_movement.position_at(
            opening_angle_degrees=0.0,
            lateral_angle_degrees=(
                lateral_angle_degrees
            ),
            protrusion_distance_mm=0.0,
        )


@pytest.mark.parametrize(
    "protrusion_distance_mm",
    [
        -1.0,
        17.1,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_protrusion(
    combined_movement: CombinedMovement,
    protrusion_distance_mm: float,
) -> None:
    with pytest.raises(ValueError):
        combined_movement.position_at(
            opening_angle_degrees=0.0,
            lateral_angle_degrees=0.0,
            protrusion_distance_mm=(
                protrusion_distance_mm
            ),
        )


def test_rejects_combined_balancing_guide_overflow(
    combined_movement: CombinedMovement,
) -> None:
    with pytest.raises(
        ValueError,
        match="combined movement",
    ):
        combined_movement.position_at(
            opening_angle_degrees=0.0,
            lateral_angle_degrees=8.0,
            protrusion_distance_mm=5.0,
        )


def test_accepts_exact_combined_guide_limit(
    combined_movement: CombinedMovement,
    right_excursion: GuidedLateralExcursion,
) -> None:
    lateral_angle = 5.0

    protrusion_distance = (
        right_excursion
        .balancing_guide
        .maximum_translation
        - right_excursion.guide_distance_at(
            lateral_angle
        )
    )

    position = combined_movement.position_at(
        opening_angle_degrees=10.0,
        lateral_angle_degrees=lateral_angle,
        protrusion_distance_mm=(
            protrusion_distance
        ),
    )

    assert position.protrusion_distance_mm == (
        pytest.approx(protrusion_distance)
    )


def test_rejects_reversed_excursions(
    assembly: MandibularAssembly,
    right_excursion: GuidedLateralExcursion,
    left_excursion: GuidedLateralExcursion,
    protrusion: GuidedProtrusion,
) -> None:
    with pytest.raises(ValueError):
        CombinedMovement(
            assembly=assembly,
            right_excursion=left_excursion,
            left_excursion=right_excursion,
            protrusion=protrusion,
        )
