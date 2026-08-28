import numpy as np
import pytest

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.lateral_excursion import (
    LateralExcursion,
    LateralSide,
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