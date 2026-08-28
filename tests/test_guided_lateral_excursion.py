import math

import numpy as np
import pytest

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.condylar_guide import (
    CondylarGuide,
)
from ogdd.articulator.guided_lateral_excursion import (
    GuidedLateralExcursion,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)
from ogdd.geometry.coordinate_system import (
    CoordinateSystem,
)


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
def right_excursion(
    hinge_axis: HingeAxis,
    coordinate_system: CoordinateSystem,
    left_guide: CondylarGuide,
) -> GuidedLateralExcursion:
    return GuidedLateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=coordinate_system.z_axis,
        working_side=LateralSide.RIGHT,
        balancing_guide=left_guide,
    )


@pytest.fixture
def left_excursion(
    hinge_axis: HingeAxis,
    coordinate_system: CoordinateSystem,
    right_guide: CondylarGuide,
) -> GuidedLateralExcursion:
    return GuidedLateralExcursion(
        hinge_axis=hinge_axis,
        superior_direction=coordinate_system.z_axis,
        working_side=LateralSide.LEFT,
        balancing_guide=right_guide,
    )


def test_guided_excursion_uses_balancing_guide(
    right_excursion: GuidedLateralExcursion,
    left_guide: CondylarGuide,
) -> None:
    assert (
        right_excursion.balancing_guide
        is left_guide
    )


def test_rejects_guide_from_working_condyle(
    hinge_axis: HingeAxis,
    coordinate_system: CoordinateSystem,
    right_guide: CondylarGuide,
) -> None:
    with pytest.raises(
        ValueError,
        match="balancing condyle",
    ):
        GuidedLateralExcursion(
            hinge_axis=hinge_axis,
            superior_direction=coordinate_system.z_axis,
            working_side=LateralSide.RIGHT,
            balancing_guide=right_guide,
        )


def test_maximum_angle_comes_from_guide_length(
    right_excursion: GuidedLateralExcursion,
) -> None:
    expected = math.degrees(
        math.asin(17.0 / 110.0)
    )

    assert (
        right_excursion.maximum_angle_degrees
        == pytest.approx(expected)
    )


def test_angle_converts_to_guide_distance(
    right_excursion: GuidedLateralExcursion,
) -> None:
    angle = math.degrees(
        math.asin(10.0 / 110.0)
    )

    assert (
        right_excursion.guide_distance_at(angle)
        == pytest.approx(10.0)
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
        angle_degrees=5.0
    )

    assert np.allclose(
        result,
        excursion.working_condyle.point,
    )


def test_right_balance_moves_anterior_inferior_medial(
    right_excursion: GuidedLateralExcursion,
) -> None:
    original = (
        right_excursion.balancing_condyle.point
    )

    moved = right_excursion.balancing_condyle_at(
        angle_degrees=5.0
    )

    assert moved[0] > original[0]
    assert moved[1] > original[1]
    assert moved[2] < original[2]


def test_left_balance_moves_anterior_inferior_medial(
    left_excursion: GuidedLateralExcursion,
) -> None:
    original = (
        left_excursion.balancing_condyle.point
    )

    moved = left_excursion.balancing_condyle_at(
        angle_degrees=5.0
    )

    assert moved[0] < original[0]
    assert moved[1] > original[1]
    assert moved[2] < original[2]


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
        angle_degrees=5.0
    )

    balancing = excursion.balancing_condyle_at(
        angle_degrees=5.0
    )

    distance = np.linalg.norm(
        balancing - working
    )

    assert distance == pytest.approx(
        excursion.hinge_axis.length
    )


def test_balance_remains_tangent_to_guide_plane(
    right_excursion: GuidedLateralExcursion,
) -> None:
    angle = 5.0

    moved = right_excursion.balancing_condyle_at(
        angle_degrees=angle
    )

    guide_center = (
        right_excursion
        .balancing_guide
        .center_at(
            right_excursion.guide_distance_at(
                angle
            )
        )
    )

    transverse_difference = (
        moved - guide_center
    )

    assert np.dot(
        transverse_difference,
        right_excursion
        .balancing_guide
        .surface_normal,
    ) == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_rejects_angle_beyond_guide_limit(
    right_excursion: GuidedLateralExcursion,
) -> None:
    with pytest.raises(
        ValueError,
        match="guide limit",
    ):
        right_excursion.transform_at(
            right_excursion.maximum_angle_degrees
            + 0.1
        )


def test_zero_returns_original_balance_position(
    right_excursion: GuidedLateralExcursion,
) -> None:
    result = right_excursion.balancing_condyle_at(
        angle_degrees=0.0
    )

    assert np.allclose(
        result,
        right_excursion.balancing_condyle.point,
    )