from unittest.mock import MagicMock, sentinel

import pytest

from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.lateral_controller import (
    LateralController,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)


@pytest.fixture
def assembly() -> MagicMock:
    return MagicMock(
        spec=MandibularAssembly,
    )


@pytest.fixture
def right_excursion() -> MagicMock:
    excursion = MagicMock()

    excursion.working_side = LateralSide.RIGHT

    excursion.position_at.return_value = (
        sentinel.right_position
    )

    return excursion


@pytest.fixture
def left_excursion() -> MagicMock:
    excursion = MagicMock()

    excursion.working_side = LateralSide.LEFT

    excursion.position_at.return_value = (
        sentinel.left_position
    )

    return excursion


@pytest.fixture
def controller(
    assembly: MagicMock,
    right_excursion: MagicMock,
    left_excursion: MagicMock,
) -> LateralController:
    return LateralController(
        assembly=assembly,
        right_excursion=right_excursion,
        left_excursion=left_excursion,
        maximum_angle_degrees=10.0,
        step_degrees=2.0,
    )


def test_controller_starts_centered(
    controller: LateralController,
) -> None:
    assert controller.angle_degrees == pytest.approx(
        0.0
    )

    assert controller.working_side is None
    assert controller.is_centered
    assert not controller.is_at_right_limit
    assert not controller.is_at_left_limit


def test_centered_position_uses_zero_geometry(
    controller: LateralController,
    assembly: MagicMock,
    right_excursion: MagicMock,
) -> None:
    position = controller.position

    right_excursion.position_at.assert_called_once_with(
        assembly=assembly,
        angle_degrees=0.0,
    )

    assert position is sentinel.right_position


def test_positive_angle_uses_right_excursion(
    controller: LateralController,
    assembly: MagicMock,
    right_excursion: MagicMock,
) -> None:
    position = controller.set_angle(
        4.0
    )

    assert (
        controller.working_side
        is LateralSide.RIGHT
    )

    right_excursion.position_at.assert_called_once_with(
        assembly=assembly,
        angle_degrees=4.0,
    )

    assert position is sentinel.right_position


def test_negative_angle_uses_left_excursion(
    controller: LateralController,
    assembly: MagicMock,
    left_excursion: MagicMock,
) -> None:
    position = controller.set_angle(
        -4.0
    )

    assert (
        controller.working_side
        is LateralSide.LEFT
    )

    left_excursion.position_at.assert_called_once_with(
        assembly=assembly,
        angle_degrees=4.0,
    )

    assert position is sentinel.left_position


def test_move_right_advances_one_step(
    controller: LateralController,
) -> None:
    position = controller.move_right()

    assert controller.angle_degrees == pytest.approx(
        2.0
    )

    assert position is sentinel.right_position


def test_move_right_stops_at_limit(
    controller: LateralController,
) -> None:
    for _ in range(20):
        controller.move_right()

    assert controller.angle_degrees == pytest.approx(
        10.0
    )

    assert controller.is_at_right_limit


def test_move_left_advances_one_step(
    controller: LateralController,
) -> None:
    position = controller.move_left()

    assert controller.angle_degrees == pytest.approx(
        -2.0
    )

    assert position is sentinel.left_position


def test_move_left_stops_at_limit(
    controller: LateralController,
) -> None:
    for _ in range(20):
        controller.move_left()

    assert controller.angle_degrees == pytest.approx(
        -10.0
    )

    assert controller.is_at_left_limit


def test_reset_returns_to_center(
    controller: LateralController,
) -> None:
    controller.set_angle(
        -6.0
    )

    position = controller.reset()

    assert controller.angle_degrees == pytest.approx(
        0.0
    )

    assert controller.working_side is None
    assert controller.is_centered
    assert position is sentinel.right_position


def test_set_angle_rejects_out_of_range_values(
    controller: LateralController,
) -> None:
    with pytest.raises(ValueError):
        controller.set_angle(-11.0)

    with pytest.raises(ValueError):
        controller.set_angle(11.0)


@pytest.mark.parametrize(
    (
        "maximum_angle_degrees",
        "step_degrees",
    ),
    [
        (0.0, 2.0),
        (-10.0, 2.0),
        (10.0, 0.0),
        (10.0, -2.0),
        (10.0, 11.0),
        (float("inf"), 2.0),
        (10.0, float("nan")),
    ],
)
def test_controller_rejects_invalid_configuration(
    assembly: MagicMock,
    right_excursion: MagicMock,
    left_excursion: MagicMock,
    maximum_angle_degrees: float,
    step_degrees: float,
) -> None:
    with pytest.raises(ValueError):
        LateralController(
            assembly=assembly,
            right_excursion=right_excursion,
            left_excursion=left_excursion,
            maximum_angle_degrees=(
                maximum_angle_degrees
            ),
            step_degrees=step_degrees,
        )


def test_controller_rejects_reversed_excursions(
    assembly: MagicMock,
    right_excursion: MagicMock,
    left_excursion: MagicMock,
) -> None:
    with pytest.raises(ValueError):
        LateralController(
            assembly=assembly,
            right_excursion=left_excursion,
            left_excursion=right_excursion,
            maximum_angle_degrees=10.0,
            step_degrees=2.0,
        )