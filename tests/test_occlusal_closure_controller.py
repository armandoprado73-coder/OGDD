import math
from unittest.mock import MagicMock, sentinel

import pytest

from ogdd.articulator.combined_movement import (
    MandibularCombinedPosition,
)
from ogdd.articulator.occlusal_closure import (
    OcclusalClosure,
)
from ogdd.articulator.occlusal_closure_controller import (
    OcclusalClosureController,
)


@pytest.fixture
def closure() -> MagicMock:
    closure = MagicMock(
        spec=OcclusalClosure,
    )
    closure.position_at.return_value = (
        sentinel.occlusal_position
    )

    return closure


@pytest.fixture
def base_position() -> MagicMock:
    position = MagicMock(
        spec=MandibularCombinedPosition,
    )
    position.opening_angle_degrees = 5.0

    return position


@pytest.fixture
def controller(
    closure: MagicMock,
    base_position: MagicMock,
) -> OcclusalClosureController:
    return OcclusalClosureController(
        closure=closure,
        base_position=base_position,
        step_degrees=0.1,
    )


def test_controller_starts_unadjusted(
    controller: OcclusalClosureController,
) -> None:
    assert controller.adjustment_angle_degrees == (
        pytest.approx(0.0)
    )
    assert controller.total_opening_angle_degrees == (
        pytest.approx(5.0)
    )
    assert controller.is_unadjusted
    assert not controller.is_opened
    assert not controller.is_closed


def test_position_uses_base_and_current_adjustment(
    controller: OcclusalClosureController,
    closure: MagicMock,
    base_position: MagicMock,
) -> None:
    position = controller.position

    closure.position_at.assert_called_once_with(
        position=base_position,
        adjustment_angle_degrees=0.0,
    )

    assert position is sentinel.occlusal_position


def test_open_adds_one_tenth_degree(
    controller: OcclusalClosureController,
) -> None:
    position = controller.open()

    assert controller.adjustment_angle_degrees == (
        pytest.approx(0.1)
    )
    assert controller.total_opening_angle_degrees == (
        pytest.approx(5.1)
    )
    assert controller.is_opened
    assert not controller.is_closed
    assert position is sentinel.occlusal_position


def test_close_subtracts_one_tenth_degree(
    controller: OcclusalClosureController,
) -> None:
    position = controller.close()

    assert controller.adjustment_angle_degrees == (
        pytest.approx(-0.1)
    )
    assert controller.total_opening_angle_degrees == (
        pytest.approx(4.9)
    )
    assert controller.is_closed
    assert not controller.is_opened
    assert position is sentinel.occlusal_position


def test_open_and_close_are_reversible(
    controller: OcclusalClosureController,
) -> None:
    controller.close()
    controller.open()

    assert controller.adjustment_angle_degrees == (
        pytest.approx(0.0)
    )
    assert controller.is_unadjusted


def test_repeated_steps_are_numerically_stable(
    controller: OcclusalClosureController,
) -> None:
    for _ in range(13):
        controller.close()

    assert controller.adjustment_angle_degrees == (
        pytest.approx(-1.3)
    )

    for _ in range(13):
        controller.open()

    assert controller.adjustment_angle_degrees == (
        pytest.approx(0.0)
    )


@pytest.mark.parametrize(
    "angle_degrees",
    [
        -2.3,
        -0.1,
        0.0,
        0.1,
        2.3,
    ],
)
def test_set_adjustment_accepts_signed_values(
    controller: OcclusalClosureController,
    angle_degrees: float,
) -> None:
    controller.set_adjustment(
        angle_degrees
    )

    assert controller.adjustment_angle_degrees == (
        pytest.approx(angle_degrees)
    )


def test_negative_zero_is_normalized(
    controller: OcclusalClosureController,
) -> None:
    controller.set_adjustment(-0.0)

    assert controller.adjustment_angle_degrees == 0.0
    assert math.copysign(
        1.0,
        controller.adjustment_angle_degrees,
    ) == 1.0


def test_reset_returns_to_base_position(
    controller: OcclusalClosureController,
) -> None:
    controller.set_adjustment(-1.3)

    position = controller.reset()

    assert controller.adjustment_angle_degrees == (
        pytest.approx(0.0)
    )
    assert controller.is_unadjusted
    assert position is sentinel.occlusal_position


def test_new_base_resets_adjustment(
    controller: OcclusalClosureController,
    closure: MagicMock,
) -> None:
    controller.set_adjustment(-1.3)

    new_base = MagicMock(
        spec=MandibularCombinedPosition,
    )
    new_base.opening_angle_degrees = 2.0

    closure.reset_mock()

    position = controller.set_base_position(
        new_base
    )

    assert controller.base_position is new_base
    assert controller.adjustment_angle_degrees == (
        pytest.approx(0.0)
    )
    assert controller.total_opening_angle_degrees == (
        pytest.approx(2.0)
    )
    closure.position_at.assert_called_once_with(
        position=new_base,
        adjustment_angle_degrees=0.0,
    )
    assert position is sentinel.occlusal_position


def test_geometry_is_always_calculated_from_base(
    controller: OcclusalClosureController,
    closure: MagicMock,
    base_position: MagicMock,
) -> None:
    controller.close()
    controller.close()
    controller.open()

    assert closure.position_at.call_count == 3

    for call in closure.position_at.call_args_list:
        assert call.kwargs["position"] is base_position


@pytest.mark.parametrize(
    "angle_degrees",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_nonfinite_adjustment_without_change(
    controller: OcclusalClosureController,
    angle_degrees: float,
) -> None:
    controller.set_adjustment(-1.3)

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        controller.set_adjustment(
            angle_degrees
        )

    assert controller.adjustment_angle_degrees == (
        pytest.approx(-1.3)
    )


@pytest.mark.parametrize(
    "step_degrees",
    [
        0.0,
        -0.1,
        math.nan,
        math.inf,
    ],
)
def test_rejects_invalid_step(
    closure: MagicMock,
    base_position: MagicMock,
    step_degrees: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="step",
    ):
        OcclusalClosureController(
            closure=closure,
            base_position=base_position,
            step_degrees=step_degrees,
        )
