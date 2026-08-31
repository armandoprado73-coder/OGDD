from unittest.mock import MagicMock, sentinel

import pytest

from ogdd.articulator.combined_controller import CombinedController
from ogdd.articulator.combined_movement import (
    MandibularCombinedPosition,
)
from ogdd.articulator.functional_calibration_controller import (
    FunctionalCalibrationController,
)
from ogdd.articulator.functional_limits import (
    FunctionalLimitKind,
    FunctionalLimits,
)
from ogdd.articulator.lateral_excursion import LateralSide
from ogdd.articulator.occlusal_closure import (
    MandibularOcclusalPosition,
)
from ogdd.articulator.occlusal_closure_controller import (
    OcclusalClosureController,
)


def make_combined_position(
    opening: float,
    lateral: float,
    protrusion: float,
) -> MagicMock:
    position = MagicMock(spec=MandibularCombinedPosition)
    position.opening_angle_degrees = opening
    position.lateral_angle_degrees = lateral
    position.protrusion_distance_mm = protrusion

    if lateral > 0.0:
        position.working_side = LateralSide.RIGHT
    elif lateral < 0.0:
        position.working_side = LateralSide.LEFT
    else:
        position.working_side = None

    return position


def make_occlusal_position(
    combined: MagicMock,
    adjustment: float,
) -> MagicMock:
    position = MagicMock(spec=MandibularOcclusalPosition)
    position.base_opening_angle_degrees = (
        combined.opening_angle_degrees
    )
    position.adjustment_angle_degrees = adjustment
    position.total_opening_angle_degrees = (
        combined.opening_angle_degrees + adjustment
    )
    position.lateral_angle_degrees = (
        combined.lateral_angle_degrees
    )
    position.protrusion_distance_mm = (
        combined.protrusion_distance_mm
    )
    position.working_side = combined.working_side

    return position


@pytest.fixture
def combined() -> MagicMock:
    controller = MagicMock(spec=CombinedController)
    controller.opening_angle_degrees = 0.0
    controller.lateral_angle_degrees = 0.0
    controller.protrusion_distance_mm = 0.0
    controller.opening_step_degrees = 1.0
    controller.lateral_step_degrees = 1.0
    controller.protrusion_step_mm = 1.0
    controller.maximum_opening_angle_degrees = 30.0
    controller.maximum_current_protrusion_distance_mm = 17.0
    controller.maximum_right_lateral_angle_degrees = 8.0
    controller.maximum_left_lateral_angle_degrees = 8.0
    controller.position = make_combined_position(0.0, 0.0, 0.0)

    def set_position(
        opening_angle_degrees: float,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> MagicMock:
        controller.opening_angle_degrees = opening_angle_degrees
        controller.lateral_angle_degrees = lateral_angle_degrees
        controller.protrusion_distance_mm = protrusion_distance_mm
        controller.position = make_combined_position(
            opening_angle_degrees,
            lateral_angle_degrees,
            protrusion_distance_mm,
        )
        return controller.position

    controller.set_position.side_effect = set_position

    return controller


@pytest.fixture
def closure() -> MagicMock:
    controller = MagicMock(spec=OcclusalClosureController)
    controller.adjustment_angle_degrees = 0.0

    def set_base_position(base: MagicMock) -> MagicMock:
        controller.base_position = base
        controller.adjustment_angle_degrees = 0.0
        controller.position = make_occlusal_position(base, 0.0)
        return controller.position

    def set_adjustment(angle: float) -> MagicMock:
        controller.adjustment_angle_degrees = angle
        controller.position = make_occlusal_position(
            controller.base_position,
            angle,
        )
        return controller.position

    controller.set_base_position.side_effect = set_base_position
    controller.set_adjustment.side_effect = set_adjustment
    controller.open.side_effect = lambda: set_adjustment(
        controller.adjustment_angle_degrees + 0.1
    )
    controller.close.side_effect = lambda: set_adjustment(
        controller.adjustment_angle_degrees - 0.1
    )
    controller.reset.side_effect = lambda: set_adjustment(0.0)

    return controller


@pytest.fixture
def controller(
    combined: MagicMock,
    closure: MagicMock,
) -> FunctionalCalibrationController:
    controller = FunctionalCalibrationController(
        combined=combined,
        closure=closure,
    )
    combined.reset_mock()
    closure.reset_mock()

    return controller


def save_protrusive_at(
    controller: FunctionalCalibrationController,
    distance: float,
    adjustment: float = -0.3,
) -> None:
    controller.set_position(2.0, 0.0, distance)
    controller.set_adjustment(adjustment)
    controller.save_protrusive_limit()


def save_right_at(
    controller: FunctionalCalibrationController,
    angle: float,
    adjustment: float = -0.2,
) -> None:
    controller.set_position(1.0, angle, 0.0)
    controller.set_adjustment(adjustment)
    controller.save_right_canine_limit()


def save_left_at(
    controller: FunctionalCalibrationController,
    angle: float,
    adjustment: float = -0.4,
) -> None:
    controller.set_position(1.0, -angle, 0.0)
    controller.set_adjustment(adjustment)
    controller.save_left_canine_limit()


def test_initialization_synchronizes_base(
    combined: MagicMock,
    closure: MagicMock,
) -> None:
    FunctionalCalibrationController(
        combined=combined,
        closure=closure,
    )

    closure.set_base_position.assert_called_once_with(
        combined.position
    )


def test_controller_starts_without_functional_limits(
    controller: FunctionalCalibrationController,
) -> None:
    assert len(controller.limits) == 0
    assert not controller.limits.is_complete
    assert controller.maximum_protrusion_distance_mm == pytest.approx(17.0)
    assert (
        controller.maximum_right_lateral_angle_degrees
        == pytest.approx(8.0)
    )
    assert (
        controller.maximum_left_lateral_angle_degrees
        == pytest.approx(8.0)
    )


def test_accepts_existing_functional_limits(
    combined: MagicMock,
    closure: MagicMock,
) -> None:
    limits = FunctionalLimits()
    controller = FunctionalCalibrationController(
        combined=combined,
        closure=closure,
        limits=limits,
    )

    assert controller.limits is limits


def test_position_and_components_delegate_to_controllers(
    controller: FunctionalCalibrationController,
    closure: MagicMock,
) -> None:
    assert controller.opening_angle_degrees == pytest.approx(0.0)
    assert controller.lateral_angle_degrees == pytest.approx(0.0)
    assert controller.protrusion_distance_mm == pytest.approx(0.0)
    assert controller.adjustment_angle_degrees == pytest.approx(0.0)
    assert controller.position is closure.position


def test_changed_combined_position_resets_adjustment(
    controller: FunctionalCalibrationController,
    closure: MagicMock,
) -> None:
    controller.set_adjustment(-0.4)
    closure.set_base_position.reset_mock()

    controller.set_position(2.0, 3.0, 4.0)

    assert controller.adjustment_angle_degrees == pytest.approx(0.0)
    closure.set_base_position.assert_called_once()


def test_unchanged_combined_position_preserves_adjustment(
    controller: FunctionalCalibrationController,
    closure: MagicMock,
) -> None:
    controller.set_adjustment(-0.4)
    closure.set_base_position.reset_mock()

    position = controller.set_position(0.0, 0.0, 0.0)

    closure.set_base_position.assert_not_called()
    assert controller.adjustment_angle_degrees == pytest.approx(-0.4)
    assert position is closure.position


def test_combined_set_is_atomic_when_underlying_rejects(
    controller: FunctionalCalibrationController,
    combined: MagicMock,
    closure: MagicMock,
) -> None:
    controller.set_adjustment(-0.3)
    combined.set_position.side_effect = ValueError("mechanical limit")

    with pytest.raises(ValueError, match="mechanical"):
        controller.set_position(10.0, 3.0, 4.0)

    assert controller.adjustment_angle_degrees == pytest.approx(-0.3)
    closure.set_base_position.assert_not_called()


def test_open_and_close_mandible_use_combined_step(
    controller: FunctionalCalibrationController,
) -> None:
    controller.open_mandible()
    controller.open_mandible()
    controller.close_mandible()

    assert controller.opening_angle_degrees == pytest.approx(1.0)


def test_mandibular_opening_stops_at_mechanical_limits(
    controller: FunctionalCalibrationController,
) -> None:
    for _ in range(40):
        controller.open_mandible()

    assert controller.opening_angle_degrees == pytest.approx(30.0)

    for _ in range(40):
        controller.close_mandible()

    assert controller.opening_angle_degrees == pytest.approx(0.0)


def test_adjust_open_and_close_delegate_fine_steps(
    controller: FunctionalCalibrationController,
) -> None:
    controller.adjust_close()
    controller.adjust_close()
    controller.adjust_open()

    assert controller.adjustment_angle_degrees == pytest.approx(-0.1)


def test_exact_adjustment_and_reset_delegate(
    controller: FunctionalCalibrationController,
) -> None:
    controller.set_adjustment(-1.3)
    assert controller.adjustment_angle_degrees == pytest.approx(-1.3)

    controller.reset_adjustment()
    assert controller.adjustment_angle_degrees == pytest.approx(0.0)


def test_save_protrusive_uses_current_adjusted_position(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4, -0.3)

    saved = controller.limits.protrusive
    assert saved is not None
    assert saved.protrusion_distance_mm == pytest.approx(6.4)
    assert saved.adjustment_angle_degrees == pytest.approx(-0.3)


def test_save_right_canine_uses_current_adjusted_position(
    controller: FunctionalCalibrationController,
) -> None:
    save_right_at(controller, 3.2, -0.2)

    saved = controller.limits.right_canine
    assert saved is not None
    assert saved.lateral_angle_degrees == pytest.approx(3.2)
    assert saved.adjustment_angle_degrees == pytest.approx(-0.2)


def test_save_left_canine_uses_current_adjusted_position(
    controller: FunctionalCalibrationController,
) -> None:
    save_left_at(controller, 3.6, -0.4)

    saved = controller.limits.left_canine
    assert saved is not None
    assert saved.lateral_angle_degrees == pytest.approx(-3.6)
    assert saved.adjustment_angle_degrees == pytest.approx(-0.4)


def test_saved_protrusion_reduces_effective_limit(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4)
    controller.reset_movement()

    assert controller.maximum_protrusion_distance_mm == pytest.approx(6.4)

    for _ in range(20):
        controller.advance()

    assert controller.protrusion_distance_mm == pytest.approx(6.4)


def test_saved_right_canine_reduces_effective_limit(
    controller: FunctionalCalibrationController,
) -> None:
    save_right_at(controller, 3.2)
    controller.reset_movement()

    for _ in range(20):
        controller.move_right()

    assert controller.lateral_angle_degrees == pytest.approx(3.2)


def test_saved_left_canine_reduces_effective_limit(
    controller: FunctionalCalibrationController,
) -> None:
    save_left_at(controller, 3.6)
    controller.reset_movement()

    for _ in range(20):
        controller.move_left()

    assert controller.lateral_angle_degrees == pytest.approx(-3.6)


def test_mechanical_limit_remains_authoritative(
    controller: FunctionalCalibrationController,
    combined: MagicMock,
) -> None:
    save_protrusive_at(controller, 6.4)
    combined.maximum_current_protrusion_distance_mm = 5.0

    assert controller.maximum_protrusion_distance_mm == pytest.approx(5.0)


def test_direct_setters_are_clamped_by_saved_limits(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4)
    controller.clear_limit(
        FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
    )
    save_right_at(controller, 3.2)
    save_left_at(controller, 3.6)
    controller.reset_movement()

    controller.set_lateral(7.0)
    assert controller.lateral_angle_degrees == pytest.approx(3.2)

    controller.set_lateral(-7.0)
    assert controller.lateral_angle_degrees == pytest.approx(-3.6)


def test_protrusion_direct_setter_is_clamped(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4)
    controller.reset_movement()

    controller.set_protrusion(12.0)

    assert controller.protrusion_distance_mm == pytest.approx(6.4)


def test_direct_combined_set_uses_candidate_mechanical_state(
    controller: FunctionalCalibrationController,
    combined: MagicMock,
) -> None:
    combined.protrusion_distance_mm = 10.0
    combined.maximum_right_lateral_angle_degrees = 2.0

    controller.set_position(0.0, 5.0, 0.0)

    assert controller.lateral_angle_degrees == pytest.approx(5.0)


def test_command_at_limit_preserves_occlusal_adjustment(
    controller: FunctionalCalibrationController,
    closure: MagicMock,
) -> None:
    save_protrusive_at(controller, 6.4)
    controller.set_adjustment(-0.3)
    closure.set_base_position.reset_mock()

    position = controller.advance()

    closure.set_base_position.assert_not_called()
    assert controller.adjustment_angle_degrees == pytest.approx(-0.3)
    assert position is closure.position


def test_moving_away_from_limit_resets_adjustment(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4)
    controller.set_adjustment(-0.3)

    controller.retreat()

    assert controller.protrusion_distance_mm == pytest.approx(5.4)
    assert controller.adjustment_angle_degrees == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kind",
    [
        FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE,
        FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP,
        FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP,
    ],
)
def test_go_to_limit_reproduces_saved_position(
    controller: FunctionalCalibrationController,
    kind: FunctionalLimitKind,
) -> None:
    if kind is FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE:
        save_protrusive_at(controller, 6.4, -0.3)
    elif kind is FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP:
        save_right_at(controller, 3.2, -0.2)
    else:
        save_left_at(controller, 3.6, -0.4)

    saved = controller.limits.get(kind)
    controller.reset_movement()

    position = controller.go_to_limit(kind)

    assert controller.opening_angle_degrees == pytest.approx(
        saved.base_opening_angle_degrees
    )
    assert controller.lateral_angle_degrees == pytest.approx(
        saved.lateral_angle_degrees
    )
    assert controller.protrusion_distance_mm == pytest.approx(
        saved.protrusion_distance_mm
    )
    assert controller.adjustment_angle_degrees == pytest.approx(
        saved.adjustment_angle_degrees
    )
    assert position is controller.position


def test_go_to_unsaved_limit_is_rejected_without_change(
    controller: FunctionalCalibrationController,
) -> None:
    controller.set_position(2.0, 0.0, 3.0)
    controller.set_adjustment(-0.2)

    with pytest.raises(ValueError, match="not saved"):
        controller.go_to_limit(
            FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
        )

    assert controller.protrusion_distance_mm == pytest.approx(3.0)
    assert controller.adjustment_angle_degrees == pytest.approx(-0.2)


def test_go_to_limit_reproduces_snapshot_without_cross_clamping(
    controller: FunctionalCalibrationController,
) -> None:
    controller.set_position(1.0, 3.2, 0.8)
    controller.set_adjustment(-0.2)
    controller.save_right_canine_limit()

    controller.reset_movement()
    save_protrusive_at(controller, 0.5, -0.1)
    controller.reset_movement()

    controller.go_to_limit(
        FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP
    )

    assert controller.lateral_angle_degrees == pytest.approx(3.2)
    assert controller.protrusion_distance_mm == pytest.approx(0.8)
    assert controller.adjustment_angle_degrees == pytest.approx(-0.2)


def test_clear_limit_allows_recalibration_beyond_old_endpoint(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4)
    controller.reset_movement()

    removed = controller.clear_limit(
        FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
    )
    controller.set_protrusion(8.0)

    assert removed is not None
    assert controller.protrusion_distance_mm == pytest.approx(8.0)


def test_clear_limits_removes_every_functional_stop(
    controller: FunctionalCalibrationController,
) -> None:
    save_protrusive_at(controller, 6.4)
    save_right_at(controller, 3.2)
    save_left_at(controller, 3.6)

    controller.clear_limits()

    assert len(controller.limits) == 0
    assert controller.maximum_protrusion_distance_mm == pytest.approx(17.0)


def test_reset_movement_returns_to_center_and_resets_adjustment(
    controller: FunctionalCalibrationController,
) -> None:
    controller.set_position(3.0, 2.0, 4.0)
    controller.set_adjustment(-0.2)

    controller.reset_movement()

    assert controller.opening_angle_degrees == pytest.approx(0.0)
    assert controller.lateral_angle_degrees == pytest.approx(0.0)
    assert controller.protrusion_distance_mm == pytest.approx(0.0)
    assert controller.adjustment_angle_degrees == pytest.approx(0.0)


def test_invalid_direct_value_reaches_combined_validation(
    controller: FunctionalCalibrationController,
    combined: MagicMock,
) -> None:
    combined.set_position.side_effect = ValueError("invalid value")

    with pytest.raises(ValueError, match="invalid value"):
        controller.set_protrusion(float("nan"))


def test_failed_save_does_not_create_limit(
    controller: FunctionalCalibrationController,
) -> None:
    controller.set_position(0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="forward"):
        controller.save_protrusive_limit()

    assert controller.limits.protrusive is None
