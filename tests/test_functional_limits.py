from dataclasses import FrozenInstanceError
import math
from unittest.mock import MagicMock

import pytest

from ogdd.articulator.functional_limits import (
    FunctionalLimit,
    FunctionalLimitKind,
    FunctionalLimits,
)
from ogdd.articulator.lateral_excursion import LateralSide
from ogdd.articulator.occlusal_closure import (
    MandibularOcclusalPosition,
)


def make_position(
    *,
    base_opening: float = 2.0,
    adjustment: float = -0.3,
    total_opening: float = 1.7,
    lateral: float = 0.0,
    protrusion: float = 6.4,
    working_side: LateralSide | None = None,
) -> MagicMock:
    position = MagicMock(spec=MandibularOcclusalPosition)
    position.base_opening_angle_degrees = base_opening
    position.adjustment_angle_degrees = adjustment
    position.total_opening_angle_degrees = total_opening
    position.lateral_angle_degrees = lateral
    position.protrusion_distance_mm = protrusion
    position.working_side = working_side

    return position


@pytest.fixture
def limits() -> FunctionalLimits:
    return FunctionalLimits()


def test_limits_start_empty(limits: FunctionalLimits) -> None:
    assert len(limits) == 0
    assert limits.values == ()
    assert limits.protrusive is None
    assert limits.right_canine is None
    assert limits.left_canine is None
    assert not limits.is_complete


def test_save_protrusive_captures_all_components(
    limits: FunctionalLimits,
) -> None:
    saved = limits.save_protrusive(make_position())

    assert saved.kind is FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
    assert saved.base_opening_angle_degrees == pytest.approx(2.0)
    assert saved.adjustment_angle_degrees == pytest.approx(-0.3)
    assert saved.total_opening_angle_degrees == pytest.approx(1.7)
    assert saved.lateral_angle_degrees == pytest.approx(0.0)
    assert saved.protrusion_distance_mm == pytest.approx(6.4)
    assert saved.working_side is None
    assert limits.protrusive is saved


def test_save_right_canine_captures_combined_position(
    limits: FunctionalLimits,
) -> None:
    position = make_position(
        lateral=3.2,
        protrusion=0.8,
        working_side=LateralSide.RIGHT,
    )

    saved = limits.save_right_canine(position)

    assert saved.kind is FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP
    assert saved.lateral_angle_degrees == pytest.approx(3.2)
    assert saved.protrusion_distance_mm == pytest.approx(0.8)
    assert saved.working_side is LateralSide.RIGHT
    assert limits.right_canine is saved


def test_save_left_canine_captures_combined_position(
    limits: FunctionalLimits,
) -> None:
    position = make_position(
        lateral=-3.6,
        protrusion=0.6,
        working_side=LateralSide.LEFT,
    )

    saved = limits.save_left_canine(position)

    assert saved.kind is FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP
    assert saved.lateral_angle_degrees == pytest.approx(-3.6)
    assert saved.protrusion_distance_mm == pytest.approx(0.6)
    assert saved.working_side is LateralSide.LEFT
    assert limits.left_canine is saved


def test_generic_save_accepts_enum_and_string(
    limits: FunctionalLimits,
) -> None:
    saved = limits.save(
        "protrusive_edge_to_edge",
        make_position(),
    )

    assert saved.kind is FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE


def test_saving_again_replaces_prior_calibration(
    limits: FunctionalLimits,
) -> None:
    first = limits.save_protrusive(make_position(protrusion=5.2))
    corrected = limits.save_protrusive(make_position(protrusion=6.1))

    assert len(limits) == 1
    assert limits.protrusive is corrected
    assert limits.protrusive is not first
    assert limits.protrusive.protrusion_distance_mm == pytest.approx(6.1)


def test_limit_is_an_immutable_snapshot(
    limits: FunctionalLimits,
) -> None:
    position = make_position()
    saved = limits.save_protrusive(position)

    position.protrusion_distance_mm = 9.9

    assert saved.protrusion_distance_mm == pytest.approx(6.4)

    with pytest.raises(FrozenInstanceError):
        saved.protrusion_distance_mm = 7.0


def test_snapshot_does_not_retain_geometry(
    limits: FunctionalLimits,
) -> None:
    saved = limits.save_protrusive(make_position())

    assert not hasattr(saved, "mesh")
    assert not hasattr(saved, "balkwill")
    assert not hasattr(saved, "bonwill")
    assert not hasattr(saved, "hinge_axis")


def test_get_returns_saved_limit(limits: FunctionalLimits) -> None:
    saved = limits.save_protrusive(make_position())

    assert limits.get(
        FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
    ) is saved
    assert limits.get("protrusive_edge_to_edge") is saved


def test_get_returns_none_for_unsaved_limit(
    limits: FunctionalLimits,
) -> None:
    assert limits.get(
        FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP
    ) is None


def test_is_complete_requires_all_three_limits(
    limits: FunctionalLimits,
) -> None:
    limits.save_protrusive(make_position())
    limits.save_right_canine(
        make_position(
            lateral=3.0,
            working_side=LateralSide.RIGHT,
        )
    )

    assert not limits.is_complete

    limits.save_left_canine(
        make_position(
            lateral=-3.0,
            working_side=LateralSide.LEFT,
        )
    )

    assert limits.is_complete
    assert len(limits) == 3


def test_values_use_stable_clinical_order(
    limits: FunctionalLimits,
) -> None:
    left = limits.save_left_canine(
        make_position(
            lateral=-3.0,
            working_side=LateralSide.LEFT,
        )
    )
    right = limits.save_right_canine(
        make_position(
            lateral=3.0,
            working_side=LateralSide.RIGHT,
        )
    )
    protrusive = limits.save_protrusive(make_position())

    assert limits.values == (protrusive, right, left)


def test_clear_removes_only_selected_limit(
    limits: FunctionalLimits,
) -> None:
    protrusive = limits.save_protrusive(make_position())
    limits.save_right_canine(
        make_position(
            lateral=3.0,
            working_side=LateralSide.RIGHT,
        )
    )

    removed = limits.clear(
        FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
    )

    assert removed is protrusive
    assert limits.protrusive is None
    assert limits.right_canine is not None
    assert len(limits) == 1


def test_clear_unsaved_limit_returns_none(
    limits: FunctionalLimits,
) -> None:
    assert limits.clear(
        FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
    ) is None


def test_clear_all_removes_every_limit(
    limits: FunctionalLimits,
) -> None:
    limits.save_protrusive(make_position())
    limits.save_right_canine(
        make_position(
            lateral=3.0,
            working_side=LateralSide.RIGHT,
        )
    )

    limits.clear_all()

    assert len(limits) == 0
    assert limits.values == ()


@pytest.mark.parametrize("operation", ["save", "get", "clear"])
def test_rejects_unknown_limit_kind(
    limits: FunctionalLimits,
    operation: str,
) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        if operation == "save":
            limits.save("unknown", make_position())
        elif operation == "get":
            limits.get("unknown")
        else:
            limits.clear("unknown")


@pytest.mark.parametrize(
    "lateral, working_side",
    [
        (1.0, LateralSide.RIGHT),
        (-1.0, LateralSide.LEFT),
        (0.0, LateralSide.RIGHT),
    ],
)
def test_protrusive_rejects_lateral_components(
    limits: FunctionalLimits,
    lateral: float,
    working_side: LateralSide | None,
) -> None:
    with pytest.raises(ValueError, match="lateral"):
        limits.save_protrusive(
            make_position(
                lateral=lateral,
                working_side=working_side,
            )
        )


@pytest.mark.parametrize("protrusion", [0.0, -0.1])
def test_protrusive_requires_forward_movement(
    limits: FunctionalLimits,
    protrusion: float,
) -> None:
    with pytest.raises(ValueError, match="forward"):
        limits.save_protrusive(
            make_position(protrusion=protrusion)
        )


@pytest.mark.parametrize(
    "lateral, working_side",
    [
        (0.0, None),
        (-3.0, LateralSide.LEFT),
        (3.0, LateralSide.LEFT),
    ],
)
def test_right_canine_requires_right_excursion(
    limits: FunctionalLimits,
    lateral: float,
    working_side: LateralSide | None,
) -> None:
    with pytest.raises(ValueError, match="right excursion"):
        limits.save_right_canine(
            make_position(
                lateral=lateral,
                working_side=working_side,
            )
        )


@pytest.mark.parametrize(
    "lateral, working_side",
    [
        (0.0, None),
        (3.0, LateralSide.RIGHT),
        (-3.0, LateralSide.RIGHT),
    ],
)
def test_left_canine_requires_left_excursion(
    limits: FunctionalLimits,
    lateral: float,
    working_side: LateralSide | None,
) -> None:
    with pytest.raises(ValueError, match="left excursion"):
        limits.save_left_canine(
            make_position(
                lateral=lateral,
                working_side=working_side,
            )
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "base_opening_angle_degrees",
        "adjustment_angle_degrees",
        "total_opening_angle_degrees",
        "lateral_angle_degrees",
        "protrusion_distance_mm",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_components(
    limits: FunctionalLimits,
    field_name: str,
    value: float,
) -> None:
    position = make_position()
    setattr(position, field_name, value)

    with pytest.raises(ValueError, match="finite"):
        limits.save_protrusive(position)

    assert len(limits) == 0


def test_rejects_inconsistent_opening_components(
    limits: FunctionalLimits,
) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        limits.save_protrusive(
            make_position(total_opening=1.8)
        )


def test_failed_replacement_preserves_prior_limit(
    limits: FunctionalLimits,
) -> None:
    original = limits.save_protrusive(make_position(protrusion=6.4))

    with pytest.raises(ValueError):
        limits.save_protrusive(make_position(protrusion=0.0))

    assert limits.protrusive is original


def test_signed_zero_is_normalized(limits: FunctionalLimits) -> None:
    saved = limits.save_protrusive(
        make_position(lateral=-0.0)
    )

    assert saved.lateral_angle_degrees == 0.0
    assert math.copysign(1.0, saved.lateral_angle_degrees) == 1.0
