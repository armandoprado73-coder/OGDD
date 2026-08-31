"""
OGDD - Operator-Calibrated Functional Limits

Stores mandibular positions that the operator has visually
confirmed as functional movement endpoints.
"""

from dataclasses import dataclass, field
from enum import Enum
import math

from ogdd.articulator.lateral_excursion import LateralSide
from ogdd.articulator.occlusal_closure import (
    MandibularOcclusalPosition,
)


class FunctionalLimitKind(str, Enum):
    """
    Operator-confirmed functional relationship.
    """

    PROTRUSIVE_EDGE_TO_EDGE = "protrusive_edge_to_edge"
    RIGHT_CANINE_CUSP_TO_CUSP = "right_canine_cusp_to_cusp"
    LEFT_CANINE_CUSP_TO_CUSP = "left_canine_cusp_to_cusp"


@dataclass(frozen=True)
class FunctionalLimit:
    """
    Immutable numeric snapshot of a confirmed position.

    Geometry is intentionally excluded. These movement
    components are sufficient to reproduce the position
    from the original mandibular assembly.
    """

    kind: FunctionalLimitKind

    base_opening_angle_degrees: float

    adjustment_angle_degrees: float

    total_opening_angle_degrees: float

    lateral_angle_degrees: float

    protrusion_distance_mm: float

    working_side: LateralSide | None

    @staticmethod
    def _finite(value: float, name: str) -> float:
        """
        Return a finite numeric component.
        """

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"The functional {name} must be finite."
            )

        if math.isclose(value, 0.0, abs_tol=1e-12):
            return 0.0

        return value

    @classmethod
    def from_position(
        cls,
        kind: FunctionalLimitKind | str,
        position: MandibularOcclusalPosition,
    ) -> "FunctionalLimit":
        """
        Capture and validate one operator-confirmed endpoint.
        """

        try:
            kind = FunctionalLimitKind(kind)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Unknown functional limit kind."
            ) from error

        base_opening = cls._finite(
            position.base_opening_angle_degrees,
            "base opening angle",
        )
        adjustment = cls._finite(
            position.adjustment_angle_degrees,
            "adjustment angle",
        )
        total_opening = cls._finite(
            position.total_opening_angle_degrees,
            "total opening angle",
        )
        lateral = cls._finite(
            position.lateral_angle_degrees,
            "lateral angle",
        )
        protrusion = cls._finite(
            position.protrusion_distance_mm,
            "protrusion distance",
        )

        if not math.isclose(
            total_opening,
            base_opening + adjustment,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "The functional opening components are inconsistent."
            )

        working_side = position.working_side

        cls._validate_relationship(
            kind=kind,
            lateral_angle_degrees=lateral,
            protrusion_distance_mm=protrusion,
            working_side=working_side,
        )

        return cls(
            kind=kind,
            base_opening_angle_degrees=base_opening,
            adjustment_angle_degrees=adjustment,
            total_opening_angle_degrees=total_opening,
            lateral_angle_degrees=lateral,
            protrusion_distance_mm=protrusion,
            working_side=working_side,
        )

    @staticmethod
    def _validate_relationship(
        kind: FunctionalLimitKind,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
        working_side: LateralSide | None,
    ) -> None:
        """
        Require movement components that match the label.
        """

        if kind is FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE:
            if lateral_angle_degrees != 0.0 or working_side is not None:
                raise ValueError(
                    "A protrusive limit cannot contain lateral movement."
                )

            if protrusion_distance_mm <= 0.0:
                raise ValueError(
                    "A protrusive limit requires forward movement."
                )

            return

        if kind is FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP:
            if (
                lateral_angle_degrees <= 0.0
                or working_side is not LateralSide.RIGHT
            ):
                raise ValueError(
                    "A right canine limit requires a right excursion."
                )

            return

        if (
            lateral_angle_degrees >= 0.0
            or working_side is not LateralSide.LEFT
        ):
            raise ValueError(
                "A left canine limit requires a left excursion."
            )


@dataclass
class FunctionalLimits:
    """
    Mutable set of endpoints confirmed by the operator.

    Saving the same relationship again replaces its prior
    value, allowing the operator to correct a calibration.
    """

    _limits: dict[FunctionalLimitKind, FunctionalLimit] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def save(
        self,
        kind: FunctionalLimitKind | str,
        position: MandibularOcclusalPosition,
    ) -> FunctionalLimit:
        """
        Save or replace one confirmed endpoint.
        """

        limit = FunctionalLimit.from_position(
            kind=kind,
            position=position,
        )
        self._limits[limit.kind] = limit

        return limit

    def save_protrusive(
        self,
        position: MandibularOcclusalPosition,
    ) -> FunctionalLimit:
        """
        Save the operator-confirmed edge-to-edge position.
        """

        return self.save(
            FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE,
            position,
        )

    def save_right_canine(
        self,
        position: MandibularOcclusalPosition,
    ) -> FunctionalLimit:
        """
        Save the right canine cusp-to-cusp position.
        """

        return self.save(
            FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP,
            position,
        )

    def save_left_canine(
        self,
        position: MandibularOcclusalPosition,
    ) -> FunctionalLimit:
        """
        Save the left canine cusp-to-cusp position.
        """

        return self.save(
            FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP,
            position,
        )

    def get(
        self,
        kind: FunctionalLimitKind | str,
    ) -> FunctionalLimit | None:
        """
        Return a saved endpoint, if available.
        """

        try:
            kind = FunctionalLimitKind(kind)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Unknown functional limit kind."
            ) from error

        return self._limits.get(kind)

    @property
    def protrusive(self) -> FunctionalLimit | None:
        """
        Saved protrusive edge-to-edge endpoint.
        """

        return self.get(
            FunctionalLimitKind.PROTRUSIVE_EDGE_TO_EDGE
        )

    @property
    def right_canine(self) -> FunctionalLimit | None:
        """
        Saved right canine cusp-to-cusp endpoint.
        """

        return self.get(
            FunctionalLimitKind.RIGHT_CANINE_CUSP_TO_CUSP
        )

    @property
    def left_canine(self) -> FunctionalLimit | None:
        """
        Saved left canine cusp-to-cusp endpoint.
        """

        return self.get(
            FunctionalLimitKind.LEFT_CANINE_CUSP_TO_CUSP
        )

    @property
    def is_complete(self) -> bool:
        """
        Whether all three functional endpoints are saved.
        """

        return len(self._limits) == len(FunctionalLimitKind)

    @property
    def values(self) -> tuple[FunctionalLimit, ...]:
        """
        Saved endpoints in stable clinical order.
        """

        return tuple(
            self._limits[kind]
            for kind in FunctionalLimitKind
            if kind in self._limits
        )

    def clear(
        self,
        kind: FunctionalLimitKind | str,
    ) -> FunctionalLimit | None:
        """
        Remove and return one saved endpoint.
        """

        try:
            kind = FunctionalLimitKind(kind)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Unknown functional limit kind."
            ) from error

        return self._limits.pop(kind, None)

    def clear_all(self) -> None:
        """
        Remove every saved functional endpoint.
        """

        self._limits.clear()

    def __len__(self) -> int:
        """
        Number of currently saved endpoints.
        """

        return len(self._limits)
