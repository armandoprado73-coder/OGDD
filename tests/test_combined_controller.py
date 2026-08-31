import math
from unittest.mock import MagicMock, sentinel

import pytest

from ogdd.articulator.combined_controller import (
    CombinedController,
)
from ogdd.articulator.combined_movement import (
    CombinedMovement,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)


@pytest.fixture
def movement() -> MagicMock:
    movement = MagicMock(
        spec=CombinedMovement,
    )

    right_excursion = MagicMock()
    right_excursion.working_side = LateralSide.RIGHT
    right_excursion.maximum_angle_degrees = 9.0
    right_excursion.hinge_axis.length = 110.0
    right_excursion.balancing_guide.maximum_translation = (
        17.0
    )
    right_excursion.guide_distance_at.side_effect = (
        lambda angle: 110.0
        * math.sin(math.radians(angle))
    )

    left_excursion = MagicMock()
    left_excursion.working_side = LateralSide.LEFT
    left_excursion.maximum_angle_degrees = 9.0
    left_excursion.hinge_axis.length = 110.0
    left_excursion.balancing_guide.maximum_translation = (
        17.0
    )
    left_excursion.guide_distance_at.side_effect = (
        lambda angle: 110.0
        * math.sin(math.radians(angle))
    )

    movement.right_excursion = right_excursion
    movement.left_excursion = left_excursion
    movement.protrusion = MagicMock()
    movement.protrusion.maximum_translation = 17.0
    movement.position_at.return_value = (
        sentinel.combined_position
    )

    return movement


@pytest.fixture
def controller(
    movement: MagicMock,
) -> CombinedController:
    return CombinedController(
        movement=movement,
        maximum_opening_angle_degrees=30.0,
        maximum_lateral_angle_degrees=8.0,
        maximum_protrusion_distance_mm=17.0,
        opening_step_degrees=1.0,
        lateral_step_degrees=1.0,
        protrusion_step_mm=1.0,
    )


def test_controller_starts_centered(
    controller: CombinedController,
) -> None:
    assert controller.opening_angle_degrees == (
        pytest.approx(0.0)
    )
    assert controller.lateral_angle_degrees == (
        pytest.approx(0.0)
    )
    assert controller.protrusion_distance_mm == (
        pytest.approx(0.0)
    )
    assert controller.working_side is None
    assert controller.is_centered


def test_position_uses_all_current_components(
    controller: CombinedController,
    movement: MagicMock,
) -> None:
    position = controller.position

    movement.position_at.assert_called_once_with(
        opening_angle_degrees=0.0,
        lateral_angle_degrees=0.0,
        protrusion_distance_mm=0.0,
    )

    assert position is sentinel.combined_position


def test_set_position_updates_atomically(
    controller: CombinedController,
    movement: MagicMock,
) -> None:
    position = controller.set_position(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )

    assert controller.opening_angle_degrees == (
        pytest.approx(12.0)
    )
    assert controller.lateral_angle_degrees == (
        pytest.approx(3.0)
    )
    assert controller.protrusion_distance_mm == (
        pytest.approx(5.0)
    )
    assert controller.working_side is LateralSide.RIGHT
    assert position is sentinel.combined_position

    movement.position_at.assert_called_once_with(
        opening_angle_degrees=12.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
    )


def test_component_setters_preserve_other_values(
    controller: CombinedController,
) -> None:
    controller.set_position(10.0, -2.0, 3.0)
    controller.set_opening(12.0)
    controller.set_lateral(-3.0)
    controller.set_protrusion(4.0)

    assert controller.opening_angle_degrees == (
        pytest.approx(12.0)
    )
    assert controller.lateral_angle_degrees == (
        pytest.approx(-3.0)
    )
    assert controller.protrusion_distance_mm == (
        pytest.approx(4.0)
    )
    assert controller.working_side is LateralSide.LEFT


def test_open_and_close_use_configured_step(
    controller: CombinedController,
) -> None:
    controller.open()
    controller.open()
    controller.close()

    assert controller.opening_angle_degrees == (
        pytest.approx(1.0)
    )


def test_opening_stops_at_limits(
    controller: CombinedController,
) -> None:
    for _ in range(40):
        controller.open()

    assert controller.opening_angle_degrees == (
        pytest.approx(30.0)
    )

    for _ in range(40):
        controller.close()

    assert controller.opening_angle_degrees == (
        pytest.approx(0.0)
    )


def test_move_right_and_left_use_signed_angle(
    controller: CombinedController,
) -> None:
    controller.move_right()
    controller.move_right()
    assert controller.lateral_angle_degrees == (
        pytest.approx(2.0)
    )

    controller.move_left()
    controller.move_left()
    controller.move_left()
    assert controller.lateral_angle_degrees == (
        pytest.approx(-1.0)
    )


def test_lateral_steps_stop_at_dynamic_limits(
    controller: CombinedController,
) -> None:
    controller.set_protrusion(10.0)

    expected_limit = math.degrees(
        math.asin(7.0 / 110.0)
    )

    for _ in range(20):
        controller.move_right()

    assert controller.lateral_angle_degrees == (
        pytest.approx(expected_limit)
    )

    controller.set_lateral(0.0)

    for _ in range(20):
        controller.move_left()

    assert controller.lateral_angle_degrees == (
        pytest.approx(-expected_limit)
    )


def test_advance_and_retreat_use_configured_step(
    controller: CombinedController,
) -> None:
    controller.advance()
    controller.advance()
    controller.retreat()

    assert controller.protrusion_distance_mm == (
        pytest.approx(1.0)
    )


def test_protrusion_steps_stop_at_dynamic_limit(
    controller: CombinedController,
) -> None:
    controller.set_lateral(5.0)

    expected_limit = (
        17.0
        - 110.0 * math.sin(math.radians(5.0))
    )

    for _ in range(30):
        controller.advance()

    assert controller.protrusion_distance_mm == (
        pytest.approx(expected_limit)
    )

    for _ in range(30):
        controller.retreat()

    assert controller.protrusion_distance_mm == (
        pytest.approx(0.0)
    )


def test_dynamic_lateral_limits_shrink_with_protrusion(
    controller: CombinedController,
) -> None:
    controller.set_protrusion(10.0)

    expected = math.degrees(
        math.asin(7.0 / 110.0)
    )

    assert (
        controller.maximum_right_lateral_angle_degrees
        == pytest.approx(expected)
    )
    assert (
        controller.maximum_left_lateral_angle_degrees
        == pytest.approx(expected)
    )


def test_dynamic_protrusion_limit_shrinks_with_lateral(
    controller: CombinedController,
) -> None:
    controller.set_lateral(-5.0)

    expected = (
        17.0
        - 110.0 * math.sin(math.radians(5.0))
    )

    assert (
        controller.maximum_current_protrusion_distance_mm
        == pytest.approx(expected)
    )


@pytest.mark.parametrize(
    "side",
    [
        LateralSide.RIGHT,
        LateralSide.LEFT,
    ],
)
def test_zero_protrusion_keeps_configured_lateral_limit(
    controller: CombinedController,
    side: LateralSide,
) -> None:
    assert controller.maximum_lateral_angle_for(side) == (
        pytest.approx(8.0)
    )


def test_zero_lateral_keeps_configured_protrusion_limit(
    controller: CombinedController,
) -> None:
    assert controller.maximum_current_protrusion_distance_mm == (
        pytest.approx(17.0)
    )


@pytest.mark.parametrize(
    (
        "opening",
        "lateral",
        "protrusion",
    ),
    [
        (-1.0, 0.0, 0.0),
        (31.0, 0.0, 0.0),
        (0.0, -9.0, 0.0),
        (0.0, 9.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 18.0),
        (math.nan, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        (0.0, 0.0, math.nan),
    ],
)
def test_set_position_rejects_invalid_values(
    controller: CombinedController,
    opening: float,
    lateral: float,
    protrusion: float,
) -> None:
    with pytest.raises(ValueError):
        controller.set_position(
            opening_angle_degrees=opening,
            lateral_angle_degrees=lateral,
            protrusion_distance_mm=protrusion,
        )


def test_rejects_combined_guide_overflow(
    controller: CombinedController,
) -> None:
    with pytest.raises(
        ValueError,
        match="combined movement",
    ):
        controller.set_position(
            opening_angle_degrees=10.0,
            lateral_angle_degrees=8.0,
            protrusion_distance_mm=5.0,
        )


def test_rejected_state_is_not_partially_applied(
    controller: CombinedController,
) -> None:
    controller.set_position(10.0, 2.0, 3.0)

    with pytest.raises(ValueError):
        controller.set_position(20.0, 8.0, 5.0)

    assert controller.opening_angle_degrees == (
        pytest.approx(10.0)
    )
    assert controller.lateral_angle_degrees == (
        pytest.approx(2.0)
    )
    assert controller.protrusion_distance_mm == (
        pytest.approx(3.0)
    )


def test_reset_returns_every_component_to_zero(
    controller: CombinedController,
) -> None:
    controller.set_position(12.0, -3.0, 5.0)

    position = controller.reset()

    assert controller.is_centered
    assert controller.working_side is None
    assert position is sentinel.combined_position


@pytest.mark.parametrize(
    (
        "maximum_opening",
        "maximum_lateral",
        "maximum_protrusion",
        "opening_step",
        "lateral_step",
        "protrusion_step",
    ),
    [
        (0.0, 8.0, 17.0, 1.0, 1.0, 1.0),
        (30.0, 0.0, 17.0, 1.0, 1.0, 1.0),
        (30.0, 8.0, 0.0, 1.0, 1.0, 1.0),
        (30.0, 8.0, 17.0, 0.0, 1.0, 1.0),
        (30.0, 8.0, 17.0, 1.0, 0.0, 1.0),
        (30.0, 8.0, 17.0, 1.0, 1.0, 0.0),
        (30.0, 8.0, 17.0, 31.0, 1.0, 1.0),
        (30.0, 8.0, 17.0, 1.0, 9.0, 1.0),
        (30.0, 8.0, 17.0, 1.0, 1.0, 18.0),
        (math.inf, 8.0, 17.0, 1.0, 1.0, 1.0),
        (30.0, math.nan, 17.0, 1.0, 1.0, 1.0),
    ],
)
def test_rejects_invalid_configuration(
    movement: MagicMock,
    maximum_opening: float,
    maximum_lateral: float,
    maximum_protrusion: float,
    opening_step: float,
    lateral_step: float,
    protrusion_step: float,
) -> None:
    with pytest.raises(ValueError):
        CombinedController(
            movement=movement,
            maximum_opening_angle_degrees=(
                maximum_opening
            ),
            maximum_lateral_angle_degrees=(
                maximum_lateral
            ),
            maximum_protrusion_distance_mm=(
                maximum_protrusion
            ),
            opening_step_degrees=opening_step,
            lateral_step_degrees=lateral_step,
            protrusion_step_mm=protrusion_step,
        )


@pytest.mark.parametrize(
    (
        "maximum_lateral",
        "maximum_protrusion",
    ),
    [
        (9.1, 17.0),
        (8.0, 17.1),
    ],
)
def test_rejects_limits_beyond_movement(
    movement: MagicMock,
    maximum_lateral: float,
    maximum_protrusion: float,
) -> None:
    with pytest.raises(ValueError):
        CombinedController(
            movement=movement,
            maximum_opening_angle_degrees=30.0,
            maximum_lateral_angle_degrees=(
                maximum_lateral
            ),
            maximum_protrusion_distance_mm=(
                maximum_protrusion
            ),
        )
