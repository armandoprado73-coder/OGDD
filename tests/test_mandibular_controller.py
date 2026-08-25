from unittest.mock import MagicMock, sentinel

import pytest

from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.mandibular_controller import (
    MandibularController,
)


@pytest.fixture
def assembly() -> MagicMock:
    mandibular_assembly = MagicMock(
        spec=MandibularAssembly,
    )

    mandibular_assembly.position_at.return_value = (
        sentinel.position
    )

    return mandibular_assembly


@pytest.fixture
def controller(
    assembly: MagicMock,
) -> MandibularController:
    return MandibularController(
        assembly=assembly,
        maximum_angle_degrees=30.0,
        step_degrees=5.0,
    )


def test_controller_starts_closed(
    controller: MandibularController,
) -> None:
    assert controller.angle_degrees == pytest.approx(
        0.0
    )

    assert controller.is_closed
    assert not controller.is_fully_open


def test_position_uses_current_angle(
    controller: MandibularController,
    assembly: MagicMock,
) -> None:
    position = controller.position

    assembly.position_at.assert_called_once_with(
        angle_degrees=0.0,
    )

    assert position is sentinel.position


def test_open_advances_one_step(
    controller: MandibularController,
) -> None:
    position = controller.open()

    assert controller.angle_degrees == pytest.approx(
        5.0
    )

    assert position is sentinel.position


def test_open_never_exceeds_maximum(
    controller: MandibularController,
) -> None:
    for _ in range(10):
        controller.open()

    assert controller.angle_degrees == pytest.approx(
        30.0
    )

    assert controller.is_fully_open


def test_close_reduces_one_step(
    controller: MandibularController,
) -> None:
    controller.set_angle(20.0)

    position = controller.close()

    assert controller.angle_degrees == pytest.approx(
        15.0
    )

    assert position is sentinel.position


def test_close_never_goes_below_zero(
    controller: MandibularController,
) -> None:
    controller.close()

    assert controller.angle_degrees == pytest.approx(
        0.0
    )

    assert controller.is_closed


def test_set_angle_rejects_out_of_range_values(
    controller: MandibularController,
) -> None:
    with pytest.raises(ValueError):
        controller.set_angle(-1.0)

    with pytest.raises(ValueError):
        controller.set_angle(31.0)


def test_reset_returns_to_closed_position(
    controller: MandibularController,
) -> None:
    controller.set_angle(20.0)

    position = controller.reset()

    assert controller.angle_degrees == pytest.approx(
        0.0
    )

    assert controller.is_closed
    assert position is sentinel.position


@pytest.mark.parametrize(
    (
        "maximum_angle_degrees",
        "step_degrees",
    ),
    [
        (0.0, 5.0),
        (-30.0, 5.0),
        (30.0, 0.0),
        (30.0, -5.0),
        (30.0, 31.0),
    ],
)
def test_controller_rejects_invalid_configuration(
    assembly: MagicMock,
    maximum_angle_degrees: float,
    step_degrees: float,
) -> None:
    with pytest.raises(ValueError):
        MandibularController(
            assembly=assembly,
            maximum_angle_degrees=(
                maximum_angle_degrees
            ),
            step_degrees=step_degrees,
        )