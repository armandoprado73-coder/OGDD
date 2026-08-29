from unittest.mock import MagicMock, sentinel

import pytest

from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
)
from ogdd.articulator.guided_protrusion import (
    GuidedProtrusion,
)
from ogdd.articulator.protrusion_controller import (
    ProtrusionController,
)


@pytest.fixture
def assembly() -> MagicMock:
    return MagicMock(
        spec=MandibularAssembly,
    )


@pytest.fixture
def protrusion() -> MagicMock:
    protrusion = MagicMock(
        spec=GuidedProtrusion,
    )

    protrusion.maximum_translation = 17.0

    protrusion.position_at.return_value = (
        sentinel.protrusive_position
    )

    return protrusion


@pytest.fixture
def controller(
    assembly: MagicMock,
    protrusion: MagicMock,
) -> ProtrusionController:
    return ProtrusionController(
        assembly=assembly,
        protrusion=protrusion,
        maximum_distance_mm=10.0,
        step_mm=2.0,
    )


def test_controller_starts_centered(
    controller: ProtrusionController,
) -> None:
    assert controller.distance_mm == pytest.approx(
        0.0
    )

    assert controller.is_centered
    assert not controller.is_at_limit


def test_centered_position_uses_zero_distance(
    controller: ProtrusionController,
    assembly: MagicMock,
    protrusion: MagicMock,
) -> None:
    position = controller.position

    protrusion.position_at.assert_called_once_with(
        assembly=assembly,
        distance=0.0,
    )

    assert position is sentinel.protrusive_position


def test_set_distance_moves_to_exact_position(
    controller: ProtrusionController,
    assembly: MagicMock,
    protrusion: MagicMock,
) -> None:
    position = controller.set_distance(
        4.0
    )

    assert controller.distance_mm == pytest.approx(
        4.0
    )

    assert not controller.is_centered

    protrusion.position_at.assert_called_once_with(
        assembly=assembly,
        distance=4.0,
    )

    assert position is sentinel.protrusive_position


def test_advance_moves_one_step(
    controller: ProtrusionController,
) -> None:
    position = controller.advance()

    assert controller.distance_mm == pytest.approx(
        2.0
    )

    assert position is sentinel.protrusive_position


def test_advance_stops_at_limit(
    controller: ProtrusionController,
) -> None:
    for _ in range(20):
        controller.advance()

    assert controller.distance_mm == pytest.approx(
        10.0
    )

    assert controller.is_at_limit


def test_retreat_moves_one_step_toward_center(
    controller: ProtrusionController,
) -> None:
    controller.set_distance(
        6.0
    )

    position = controller.retreat()

    assert controller.distance_mm == pytest.approx(
        4.0
    )

    assert position is sentinel.protrusive_position


def test_retreat_stops_at_center(
    controller: ProtrusionController,
) -> None:
    controller.set_distance(
        2.0
    )

    for _ in range(20):
        controller.retreat()

    assert controller.distance_mm == pytest.approx(
        0.0
    )

    assert controller.is_centered


def test_reset_returns_to_center(
    controller: ProtrusionController,
) -> None:
    controller.set_distance(
        8.0
    )

    position = controller.reset()

    assert controller.distance_mm == pytest.approx(
        0.0
    )

    assert controller.is_centered
    assert position is sentinel.protrusive_position


@pytest.mark.parametrize(
    "distance_mm",
    [
        -1.0,
        11.0,
        float("nan"),
        float("inf"),
    ],
)
def test_set_distance_rejects_invalid_values(
    controller: ProtrusionController,
    distance_mm: float,
) -> None:
    with pytest.raises(ValueError):
        controller.set_distance(
            distance_mm
        )


@pytest.mark.parametrize(
    (
        "maximum_distance_mm",
        "step_mm",
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
    protrusion: MagicMock,
    maximum_distance_mm: float,
    step_mm: float,
) -> None:
    with pytest.raises(ValueError):
        ProtrusionController(
            assembly=assembly,
            protrusion=protrusion,
            maximum_distance_mm=(
                maximum_distance_mm
            ),
            step_mm=step_mm,
        )


def test_controller_rejects_limit_beyond_guides(
    assembly: MagicMock,
    protrusion: MagicMock,
) -> None:
    with pytest.raises(
        ValueError,
        match="common guide limit",
    ):
        ProtrusionController(
            assembly=assembly,
            protrusion=protrusion,
            maximum_distance_mm=18.0,
            step_mm=1.0,
        )