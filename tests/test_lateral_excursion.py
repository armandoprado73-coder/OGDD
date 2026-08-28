import numpy as np
import pytest

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.lateral_excursion import (
    LateralExcursion,
    LateralSide,
)
from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.mesh import Mesh
from ogdd.anatomy.mandibular_assembly import MandibularAssembly
from ogdd.articulator.lateral_excursion import (
    LateralExcursion,
    LateralSide,
    MandibularLateralPosition,
)

@pytest.fixture
def hinge_axis() -> HingeAxis:
    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-55.0, 0.0, 0.0]),
        reference_used="condylar center",
    )

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([55.0, 0.0, 0.0]),
        reference_used="condylar center",
    )

    return HingeAxis(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
    )


@pytest.fixture
def right_excursion(
    hinge_axis: HingeAxis,
) -> LateralExcursion:
    return LateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=np.array([0.0, 0.0, 1.0]),
        working_side=LateralSide.RIGHT,
    )


@pytest.fixture
def left_excursion(
    hinge_axis: HingeAxis,
) -> LateralExcursion:
    return LateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=np.array([0.0, 0.0, 1.0]),
        working_side=LateralSide.LEFT,
    )


def test_superior_direction_is_normalized(
    hinge_axis: HingeAxis,
) -> None:
    excursion = LateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=np.array([0.0, 0.0, 10.0]),
        working_side=LateralSide.RIGHT,
    )

    assert np.allclose(
        excursion.superior_direction,
        np.array([0.0, 0.0, 1.0]),
    )


def test_right_working_side_selects_condyles(
    right_excursion: LateralExcursion,
    hinge_axis: HingeAxis,
) -> None:
    assert (
        right_excursion.working_condyle
        is hinge_axis.right_condyle
    )

    assert (
        right_excursion.balancing_condyle
        is hinge_axis.left_condyle
    )


def test_left_working_side_selects_condyles(
    left_excursion: LateralExcursion,
    hinge_axis: HingeAxis,
) -> None:
    assert (
        left_excursion.working_condyle
        is hinge_axis.left_condyle
    )

    assert (
        left_excursion.balancing_condyle
        is hinge_axis.right_condyle
    )


def test_zero_superior_direction_is_rejected(
    hinge_axis: HingeAxis,
) -> None:
    with pytest.raises(ValueError):

        LateralExcursion(
            hinge_axis=hinge_axis,
            superior_direction=np.zeros(3),
            working_side=LateralSide.RIGHT,
        )


def test_invalid_working_side_is_rejected(
    hinge_axis: HingeAxis,
) -> None:
    with pytest.raises(ValueError):

        LateralExcursion(
            hinge_axis=hinge_axis,
            superior_direction=np.array([0.0, 0.0, 1.0]),
            working_side="right",
        )


def test_negative_excursion_is_rejected(
    right_excursion: LateralExcursion,
) -> None:
    with pytest.raises(ValueError):

        right_excursion.transform_at(
            angle_degrees=-1.0
        )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "right_excursion",
        "left_excursion",
    ],
)
def test_working_condyle_remains_fixed(
    fixture_name: str,
    request,
) -> None:
    excursion = request.getfixturevalue(
        fixture_name
    )

    result = excursion.working_condyle_at(
        angle_degrees=10.0
    )

    assert np.allclose(
        result,
        excursion.working_condyle.point,
    )


def test_right_balancing_condyle_moves_anterior_and_medial(
    right_excursion: LateralExcursion,
) -> None:
    original = (
        right_excursion.balancing_condyle.point
    )

    moved = right_excursion.balancing_condyle_at(
        angle_degrees=10.0
    )

    assert moved[0] > original[0]
    assert moved[1] > original[1]


def test_left_balancing_condyle_moves_anterior_and_medial(
    left_excursion: LateralExcursion,
) -> None:
    original = (
        left_excursion.balancing_condyle.point
    )

    moved = left_excursion.balancing_condyle_at(
        angle_degrees=10.0
    )

    assert moved[0] < original[0]
    assert moved[1] > original[1]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "right_excursion",
        "left_excursion",
    ],
)
def test_intercondylar_distance_is_preserved(
    fixture_name: str,
    request,
) -> None:
    excursion = request.getfixturevalue(
        fixture_name
    )

    working = excursion.working_condyle_at(
        angle_degrees=10.0
    )

    balancing = excursion.balancing_condyle_at(
        angle_degrees=10.0
    )

    distance = np.linalg.norm(
        balancing - working
    )

    assert distance == pytest.approx(
        excursion.hinge_axis.length
    )


def test_zero_excursion_returns_original_balance_position(
    right_excursion: LateralExcursion,
) -> None:
    result = right_excursion.balancing_condyle_at(
        angle_degrees=0.0
    )

    assert np.allclose(
        result,
        right_excursion.balancing_condyle.point,
    )
@pytest.fixture
def balkwill_triangle() -> BalkwillTriangle:
    return BalkwillTriangle(
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
        dental_midline=Landmark(
            name="DENTAL_MIDLINE",
            point=np.array([0.0, 100.0, 0.0]),
            reference_used="mandibular dental midline",
        ),
    )


@pytest.fixture
def bonwill_triangle(
    hinge_axis: HingeAxis,
) -> BonwillTriangle:
    return BonwillTriangle(
        left_condyle=hinge_axis.left_condyle,
        right_condyle=hinge_axis.right_condyle,
        dental_midline=Landmark(
            name="DENTAL_MIDLINE",
            point=np.array([0.0, 100.0, 0.0]),
            reference_used="mandibular dental midline",
        ),
    )


@pytest.fixture
def mandibular_mesh() -> Mesh:
    return Mesh(
        vertices=np.array(
            [
                [-50.0, 20.0, 0.0],
                [50.0, 20.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 60.0, -20.0],
            ]
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [1, 3, 2],
                [0, 3, 1],
            ]
        ),
        normals=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        attributes={
            "test_value": np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
        },
        metadata={
            "source": "synthetic",
        },
    )


def test_rotate_landmark_preserves_metadata(
    right_excursion: LateralExcursion,
) -> None:
    landmark = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
    )

    rotated = right_excursion.rotate_landmark(
        landmark=landmark,
        angle_degrees=10.0,
    )

    assert rotated.name == landmark.name
    assert rotated.reference_used == landmark.reference_used
    assert rotated.confidence == landmark.confidence
    assert rotated.created_by == landmark.created_by


def test_rotate_balkwill_preserves_distances(
    right_excursion: LateralExcursion,
    balkwill_triangle: BalkwillTriangle,
) -> None:
    rotated = right_excursion.rotate_balkwill(
        balkwill=balkwill_triangle,
        angle_degrees=10.0,
    )

    assert rotated.intermolar_width == pytest.approx(
        balkwill_triangle.intermolar_width
    )

    assert rotated.right_side == pytest.approx(
        balkwill_triangle.right_side
    )

    assert rotated.left_side == pytest.approx(
        balkwill_triangle.left_side
    )


def test_rotate_bonwill_keeps_working_condyle_fixed(
    right_excursion: LateralExcursion,
    bonwill_triangle: BonwillTriangle,
) -> None:
    rotated = right_excursion.rotate_bonwill(
        bonwill=bonwill_triangle,
        angle_degrees=10.0,
    )

    assert np.allclose(
        rotated.right_condyle.point,
        bonwill_triangle.right_condyle.point,
    )


def test_rotate_bonwill_moves_balancing_condyle(
    right_excursion: LateralExcursion,
    bonwill_triangle: BonwillTriangle,
) -> None:
    rotated = right_excursion.rotate_bonwill(
        bonwill=bonwill_triangle,
        angle_degrees=10.0,
    )

    assert np.allclose(
        rotated.left_condyle.point,
        right_excursion.balancing_condyle_at(
            angle_degrees=10.0
        ),
    )


def test_rotate_mesh_vertices(
    right_excursion: LateralExcursion,
    mandibular_mesh: Mesh,
) -> None:
    rotated = right_excursion.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=10.0,
    )

    expected = right_excursion.rotate_points(
        points=mandibular_mesh.vertices,
        angle_degrees=10.0,
    )

    assert np.allclose(
        rotated.vertices,
        expected,
    )


def test_rotate_mesh_rotates_normals(
    right_excursion: LateralExcursion,
    mandibular_mesh: Mesh,
) -> None:
    rotated = right_excursion.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=90.0,
    )

    assert rotated.normals is not None

    assert np.allclose(
        rotated.normals[0],
        np.array([0.0, -1.0, 0.0]),
        atol=1e-8,
    )


def test_rotate_mesh_preserves_data(
    right_excursion: LateralExcursion,
    mandibular_mesh: Mesh,
) -> None:
    rotated = right_excursion.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=10.0,
    )

    assert np.array_equal(
        rotated.faces,
        mandibular_mesh.faces,
    )

    assert np.array_equal(
        rotated.attributes["test_value"],
        mandibular_mesh.attributes["test_value"],
    )

    assert rotated.metadata == mandibular_mesh.metadata


def test_rotate_mesh_does_not_modify_original(
    right_excursion: LateralExcursion,
    mandibular_mesh: Mesh,
) -> None:
    original_vertices = (
        mandibular_mesh.vertices.copy()
    )

    right_excursion.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=10.0,
    )

    assert np.allclose(
        mandibular_mesh.vertices,
        original_vertices,
    )

@pytest.fixture
def mandibular_assembly(
    mandibular_mesh: Mesh,
    balkwill_triangle: BalkwillTriangle,
    bonwill_triangle: BonwillTriangle,
    hinge_axis: HingeAxis,
) -> MandibularAssembly:
    return MandibularAssembly(
        mesh=mandibular_mesh,
        balkwill=balkwill_triangle,
        bonwill=bonwill_triangle,
        hinge_axis=hinge_axis,
    )


def test_position_at_returns_complete_lateral_position(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert isinstance(
        position,
        MandibularLateralPosition,
    )

    assert (
        position.working_side
        is LateralSide.RIGHT
    )

    assert position.angle_degrees == pytest.approx(
        10.0
    )


def test_lateral_position_moves_structures_together(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert np.allclose(
        position.mesh.vertices[2],
        position.balkwill.dental_midline.point,
    )

    assert np.allclose(
        position.balkwill.dental_midline.point,
        position.bonwill.dental_midline.point,
    )


def test_lateral_position_keeps_working_condyle_fixed(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert np.allclose(
        position.bonwill.right_condyle.point,
        mandibular_assembly.bonwill.right_condyle.point,
    )


def test_lateral_position_moves_balancing_condyle(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert not np.allclose(
        position.bonwill.left_condyle.point,
        mandibular_assembly.bonwill.left_condyle.point,
    )


def test_lateral_position_updates_hinge_axis(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert np.allclose(
        position.hinge_axis.left_condyle.point,
        position.bonwill.left_condyle.point,
    )

    assert np.allclose(
        position.hinge_axis.right_condyle.point,
        position.bonwill.right_condyle.point,
    )

    assert position.hinge_axis.length == pytest.approx(
        mandibular_assembly.hinge_axis.length
    )


def test_lateral_positions_do_not_accumulate_rotation(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position_5 = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=5.0,
    )

    position_10 = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    direct_position_10 = right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert not np.allclose(
        position_5.mesh.vertices,
        position_10.mesh.vertices,
    )

    assert np.allclose(
        position_10.mesh.vertices,
        direct_position_10.mesh.vertices,
    )


def test_lateral_position_does_not_modify_original(
    right_excursion: LateralExcursion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    original_vertices = (
        mandibular_assembly.mesh.vertices.copy()
    )

    original_left_condyle = (
        mandibular_assembly.bonwill.left_condyle.point.copy()
    )

    right_excursion.position_at(
        assembly=mandibular_assembly,
        angle_degrees=10.0,
    )

    assert np.allclose(
        mandibular_assembly.mesh.vertices,
        original_vertices,
    )

    assert np.allclose(
        mandibular_assembly.bonwill.left_condyle.point,
        original_left_condyle,
    )