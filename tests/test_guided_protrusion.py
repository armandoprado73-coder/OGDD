import math

import numpy as np
import pytest

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.condylar_guide import (
    CondylarGuide,
)
from ogdd.articulator.guided_protrusion import (
    GuidedProtrusion,
    MandibularProtrusivePosition,
)
from ogdd.geometry.coordinate_system import (
    CoordinateSystem,
)
from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
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
        condyle_center=np.array(
            [-55.0, 0.0, 0.0]
        ),
        coordinate_system=coordinate_system,
        angle_degrees=45.0,
    )


@pytest.fixture
def right_guide(
    coordinate_system: CoordinateSystem,
) -> CondylarGuide:
    return CondylarGuide(
        condyle_center=np.array(
            [55.0, 0.0, 0.0]
        ),
        coordinate_system=coordinate_system,
        angle_degrees=45.0,
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
def mandibular_assembly(
    hinge_axis: HingeAxis,
) -> MandibularAssembly:
    left_posterior = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array([-50.0, 20.0, 0.0]),
        reference_used="distobuccal cusp",
    )

    right_posterior = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array([50.0, 20.0, 0.0]),
        reference_used="distobuccal cusp",
    )

    dental_midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
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
    )

    balkwill = BalkwillTriangle(
        left_posterior=left_posterior,
        right_posterior=right_posterior,
        dental_midline=dental_midline,
    )

    bonwill = BonwillTriangle(
        left_condyle=hinge_axis.left_condyle,
        right_condyle=hinge_axis.right_condyle,
        dental_midline=dental_midline,
    )

    return MandibularAssembly(
        mesh=mesh,
        balkwill=balkwill,
        bonwill=bonwill,
        hinge_axis=hinge_axis,
    )

def test_protrusion_uses_both_guides(
    protrusion: GuidedProtrusion,
    right_guide: CondylarGuide,
    left_guide: CondylarGuide,
) -> None:
    assert protrusion.right_guide is right_guide
    assert protrusion.left_guide is left_guide


def test_rejects_right_guide_from_left_condyle(
    hinge_axis: HingeAxis,
    left_guide: CondylarGuide,
) -> None:
    with pytest.raises(
        ValueError,
        match="right condyle",
    ):
        GuidedProtrusion(
            hinge_axis=hinge_axis,
            right_guide=left_guide,
            left_guide=left_guide,
        )


def test_rejects_left_guide_from_right_condyle(
    hinge_axis: HingeAxis,
    right_guide: CondylarGuide,
) -> None:
    with pytest.raises(
        ValueError,
        match="left condyle",
    ):
        GuidedProtrusion(
            hinge_axis=hinge_axis,
            right_guide=right_guide,
            left_guide=right_guide,
        )


def test_rejects_nonparallel_trajectories(
    hinge_axis: HingeAxis,
    coordinate_system: CoordinateSystem,
    right_guide: CondylarGuide,
) -> None:
    left_guide = CondylarGuide(
        condyle_center=np.array(
            [-55.0, 0.0, 0.0]
        ),
        coordinate_system=coordinate_system,
        angle_degrees=35.0,
    )

    with pytest.raises(
        ValueError,
        match="parallel",
    ):
        GuidedProtrusion(
            hinge_axis=hinge_axis,
            right_guide=right_guide,
            left_guide=left_guide,
        )


def test_maximum_translation_is_seventeen_mm(
    protrusion: GuidedProtrusion,
) -> None:
    assert (
        protrusion.maximum_translation
        == pytest.approx(17.0)
    )


def test_shorter_guide_determines_limit(
    hinge_axis: HingeAxis,
    coordinate_system: CoordinateSystem,
    left_guide: CondylarGuide,
) -> None:
    shorter_right_guide = CondylarGuide(
        condyle_center=np.array(
            [55.0, 0.0, 0.0]
        ),
        coordinate_system=coordinate_system,
        angle_degrees=45.0,
        length=13.0,
    )

    protrusion = GuidedProtrusion(
        hinge_axis=hinge_axis,
        right_guide=shorter_right_guide,
        left_guide=left_guide,
    )

    assert (
        protrusion.maximum_translation
        == pytest.approx(10.0)
    )


def test_trajectory_direction_is_shared_unit_vector(
    protrusion: GuidedProtrusion,
) -> None:
    direction = protrusion.trajectory_direction

    assert np.linalg.norm(
        direction
    ) == pytest.approx(1.0)

    assert direction[0] == pytest.approx(0.0)
    assert direction[1] > 0.0
    assert direction[2] < 0.0


def test_translation_vector_follows_guides(
    protrusion: GuidedProtrusion,
) -> None:
    distance = 10.0

    expected = (
        distance
        * protrusion.trajectory_direction
    )

    result = protrusion.translation_vector_at(
        distance
    )

    assert np.allclose(
        result,
        expected,
    )


def test_targets_follow_both_guides(
    protrusion: GuidedProtrusion,
) -> None:
    distance = 10.0

    assert np.allclose(
        protrusion.right_target_at(distance),
        protrusion.right_guide.center_at(
            distance
        ),
    )

    assert np.allclose(
        protrusion.left_target_at(distance),
        protrusion.left_guide.center_at(
            distance
        ),
    )


def test_both_condyles_move_anterior_inferior(
    protrusion: GuidedProtrusion,
) -> None:
    distance = 10.0

    right_target = protrusion.right_target_at(
        distance
    )

    left_target = protrusion.left_target_at(
        distance
    )

    right_original = (
        protrusion.hinge_axis.right_condyle.point
    )

    left_original = (
        protrusion.hinge_axis.left_condyle.point
    )

    assert right_target[1] > right_original[1]
    assert right_target[2] < right_original[2]

    assert left_target[1] > left_original[1]
    assert left_target[2] < left_original[2]


def test_intercondylar_distance_is_preserved(
    protrusion: GuidedProtrusion,
) -> None:
    distance = 10.0

    right_target = protrusion.right_target_at(
        distance
    )

    left_target = protrusion.left_target_at(
        distance
    )

    result = np.linalg.norm(
        right_target - left_target
    )

    assert result == pytest.approx(
        protrusion.hinge_axis.length
    )


def test_transform_moves_all_points_rigidly(
    protrusion: GuidedProtrusion,
) -> None:
    points = np.array([
        [-55.0, 0.0, 0.0],
        [55.0, 0.0, 0.0],
        [0.0, 50.0, -20.0],
    ])

    distance = 10.0

    result = protrusion.transform_at(
        distance
    ).apply(points)

    expected = (
        points
        + protrusion.translation_vector_at(
            distance
        )
    )

    assert np.allclose(
        result,
        expected,
    )


def test_zero_returns_identity_transform(
    protrusion: GuidedProtrusion,
) -> None:
    points = np.array([
        [-55.0, 0.0, 0.0],
        [55.0, 0.0, 0.0],
        [0.0, 50.0, -20.0],
    ])

    result = protrusion.transform_at(
        0.0
    ).apply(points)

    assert np.allclose(
        result,
        points,
    )


@pytest.mark.parametrize(
    "distance",
    [
        -1.0,
        17.1,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_distance(
    protrusion: GuidedProtrusion,
    distance: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="common guide limit",
    ):
        protrusion.transform_at(
            distance
        )
def test_position_at_returns_protrusive_position(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    assert isinstance(
        position,
        MandibularProtrusivePosition,
    )

    assert position.distance_mm == pytest.approx(
        10.0
    )


def test_position_moves_all_structures_together(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    mesh_midline = position.mesh.vertices[2]

    balkwill_midline = (
        position.balkwill.dental_midline.point
    )

    bonwill_midline = (
        position.bonwill.dental_midline.point
    )

    assert np.allclose(
        mesh_midline,
        balkwill_midline,
    )

    assert np.allclose(
        balkwill_midline,
        bonwill_midline,
    )


def test_position_condyles_follow_guides(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    distance = 10.0

    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=distance,
    )

    assert np.allclose(
        position.hinge_axis.right_condyle.point,
        protrusion.right_target_at(distance),
    )

    assert np.allclose(
        position.hinge_axis.left_condyle.point,
        protrusion.left_target_at(distance),
    )


def test_position_preserves_intercondylar_width(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    assert position.hinge_axis.length == pytest.approx(
        mandibular_assembly.hinge_axis.length
    )


def test_position_preserves_mesh_geometry(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    original_distances = np.linalg.norm(
        mandibular_assembly.mesh.vertices[:, None, :]
        - mandibular_assembly.mesh.vertices[None, :, :],
        axis=2,
    )

    moved_distances = np.linalg.norm(
        position.mesh.vertices[:, None, :]
        - position.mesh.vertices[None, :, :],
        axis=2,
    )

    assert np.allclose(
        moved_distances,
        original_distances,
    )


def test_translation_preserves_mesh_normals(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    assert np.allclose(
        position.mesh.normals,
        mandibular_assembly.mesh.normals,
    )


def test_zero_returns_original_geometry(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=0.0,
    )

    assert np.allclose(
        position.mesh.vertices,
        mandibular_assembly.mesh.vertices,
    )

    assert np.allclose(
        position.bonwill.left_condyle.point,
        mandibular_assembly.bonwill.left_condyle.point,
    )

    assert np.allclose(
        position.bonwill.right_condyle.point,
        mandibular_assembly.bonwill.right_condyle.point,
    )


def test_positions_do_not_accumulate_translation(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    position_5 = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=5.0,
    )

    position_10 = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    direct_position_10 = protrusion.position_at(
        assembly=mandibular_assembly,
        distance=10.0,
    )

    assert not np.allclose(
        position_5.mesh.vertices,
        position_10.mesh.vertices,
    )

    assert np.allclose(
        position_10.mesh.vertices,
        direct_position_10.mesh.vertices,
    )


def test_position_rejects_different_assembly(
    protrusion: GuidedProtrusion,
    mandibular_assembly: MandibularAssembly,
) -> None:
    other_left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-60.0, 0.0, 0.0]),
        reference_used="condylar center",
    )

    other_right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([50.0, 0.0, 0.0]),
        reference_used="condylar center",
    )

    other_hinge_axis = HingeAxis(
        left_condyle=other_left_condyle,
        right_condyle=other_right_condyle,
    )

    other_bonwill = BonwillTriangle(
        left_condyle=other_left_condyle,
        right_condyle=other_right_condyle,
        dental_midline=(
            mandibular_assembly.bonwill.dental_midline
        ),
    )

    other_assembly = MandibularAssembly(
        mesh=mandibular_assembly.mesh,
        balkwill=mandibular_assembly.balkwill,
        bonwill=other_bonwill,
        hinge_axis=other_hinge_axis,
    )

    with pytest.raises(
        ValueError,
        match="same left condyle",
    ):
        protrusion.position_at(
            assembly=other_assembly,
            distance=10.0,
        )