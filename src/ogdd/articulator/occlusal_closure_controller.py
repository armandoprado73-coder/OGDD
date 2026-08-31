"""
OGDD - Occlusal Closure Controller

Controls manual opening and closure adjustments around
the mobile hinge axis in operator-selected steps.
"""

from dataclasses import dataclass, field
import math

from ogdd.articulator.combined_movement import (
    MandibularCombinedPosition,
)
from ogdd.articulator.occlusal_closure import (
    MandibularOcclusalPosition,
    OcclusalClosure,
)


@dataclass
class OcclusalClosureController:
    """
    Stateful manual control of occlusal adjustment.

    Positive values add opening and negative values add
    closure. No clinical endpoint is imposed: the operator
    observes the dental relationship and decides where to
    save a functional position.

    Changing the combined base position resets adjustment
    to zero so a value is never carried silently into a
    different mandibular relationship.
    """

    closure: OcclusalClosure

    base_position: MandibularCombinedPosition

    step_degrees: float = 0.1

    _adjustment_angle_degrees: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate the operator adjustment step.
        """

        self.step_degrees = float(
            self.step_degrees
        )

        if (
            not math.isfinite(self.step_degrees)
            or self.step_degrees <= 0.0
        ):
            raise ValueError(
                "Occlusal adjustment step must be "
                "positive and finite."
            )

    @staticmethod
    def _normalized_angle(
        angle_degrees: float,
    ) -> float:
        """
        Normalize a finite angle for stable manual steps.
        """

        angle_degrees = float(
            angle_degrees
        )

        if not math.isfinite(angle_degrees):
            raise ValueError(
                "Occlusal adjustment angle must be finite."
            )

        angle_degrees = round(
            angle_degrees,
            12,
        )

        if math.isclose(
            angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return 0.0

        return angle_degrees

    @property
    def adjustment_angle_degrees(self) -> float:
        """
        Current signed operator adjustment.
        """

        return self._adjustment_angle_degrees

    @property
    def total_opening_angle_degrees(self) -> float:
        """
        Base opening plus the manual adjustment.
        """

        return (
            self.base_position.opening_angle_degrees
            + self._adjustment_angle_degrees
        )

    @property
    def is_unadjusted(self) -> bool:
        """
        Whether adjustment is at the base position.
        """

        return math.isclose(
            self._adjustment_angle_degrees,
            0.0,
            abs_tol=1e-12,
        )

    @property
    def is_opened(self) -> bool:
        """
        Whether manual adjustment adds opening.
        """

        return self._adjustment_angle_degrees > 0.0

    @property
    def is_closed(self) -> bool:
        """
        Whether manual adjustment adds closure.
        """

        return self._adjustment_angle_degrees < 0.0

    @property
    def position(self) -> MandibularOcclusalPosition:
        """
        Adjusted geometry at the current signed angle.
        """

        return self.closure.position_at(
            position=self.base_position,
            adjustment_angle_degrees=(
                self._adjustment_angle_degrees
            ),
        )

    def set_adjustment(
        self,
        angle_degrees: float,
    ) -> MandibularOcclusalPosition:
        """
        Set an exact operator-selected adjustment.
        """

        angle_degrees = self._normalized_angle(
            angle_degrees
        )

        self._adjustment_angle_degrees = (
            angle_degrees
        )

        return self.position

    def open(self) -> MandibularOcclusalPosition:
        """
        Open one configured manual step.
        """

        return self.set_adjustment(
            self._adjustment_angle_degrees
            + self.step_degrees
        )

    def close(self) -> MandibularOcclusalPosition:
        """
        Close one configured manual step.
        """

        return self.set_adjustment(
            self._adjustment_angle_degrees
            - self.step_degrees
        )

    def reset(self) -> MandibularOcclusalPosition:
        """
        Return to the unadjusted combined position.
        """

        return self.set_adjustment(
            0.0
        )

    def set_base_position(
        self,
        position: MandibularCombinedPosition,
    ) -> MandibularOcclusalPosition:
        """
        Select a new base and reset manual adjustment.
        """

        self.base_position = position
        self._adjustment_angle_degrees = 0.0

        return self.position
