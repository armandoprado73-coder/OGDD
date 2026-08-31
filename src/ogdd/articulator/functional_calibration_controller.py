"""
OGDD - Functional Calibration Controller

Coordinates combined movement, manual occlusal closure
and operator-confirmed functional movement limits.
"""

from dataclasses import dataclass, field
import math
from typing import Callable

from ogdd.articulator.combined_controller import CombinedController
from ogdd.articulator.combined_movement import (
    MandibularCombinedPosition,
)
from ogdd.articulator.functional_limits import (
    FunctionalLimit,
    FunctionalLimitKind,
    FunctionalLimits,
)
from ogdd.articulator.occlusal_closure import (
    MandibularOcclusalPosition,
)
from ogdd.articulator.occlusal_closure_controller import (
    OcclusalClosureController,
)


@dataclass
class FunctionalCalibrationController:
    """
    Operate and calibrate functional mandibular endpoints.

    Mechanical condylar-guide limits remain authoritative.
    Saved functional limits can only reduce available travel.
    They never enlarge the movement permitted by the model.

    A genuine combined-position change resets the manual
    occlusal adjustment. A command already stopped at a
    limit leaves that adjustment untouched.
    """

    combined: CombinedController

    closure: OcclusalClosureController

    limits: FunctionalLimits = field(
        default_factory=FunctionalLimits,
    )

    def __post_init__(self) -> None:
        """
        Synchronize closure with the current combined base.
        """

        self.closure.set_base_position(
            self.combined.position
        )

    @property
    def opening_angle_degrees(self) -> float:
        """
        Current base opening angle.
        """

        return self.combined.opening_angle_degrees

    @property
    def lateral_angle_degrees(self) -> float:
        """
        Current signed lateral angle.
        """

        return self.combined.lateral_angle_degrees

    @property
    def protrusion_distance_mm(self) -> float:
        """
        Current protrusive distance.
        """

        return self.combined.protrusion_distance_mm

    @property
    def adjustment_angle_degrees(self) -> float:
        """
        Current operator-controlled occlusal adjustment.
        """

        return self.closure.adjustment_angle_degrees

    @property
    def position(self) -> MandibularOcclusalPosition:
        """
        Current complete operator-adjusted position.
        """

        return self.closure.position

    def _state(self) -> tuple[float, float, float]:
        """
        Return the current combined numeric state.
        """

        return (
            self.combined.opening_angle_degrees,
            self.combined.lateral_angle_degrees,
            self.combined.protrusion_distance_mm,
        )

    def _apply_combined_change(
        self,
        change: Callable[[], MandibularCombinedPosition],
    ) -> MandibularOcclusalPosition:
        """
        Apply movement and synchronize only when it changed.
        """

        previous_state = self._state()
        base_position = change()

        if self._state() == previous_state:
            return self.position

        return self.closure.set_base_position(
            base_position
        )

    @property
    def maximum_protrusion_distance_mm(self) -> float:
        """
        Current mechanical limit reduced by calibration.
        """

        maximum = (
            self.combined
            .maximum_current_protrusion_distance_mm
        )
        saved = self.limits.protrusive

        if saved is not None:
            maximum = min(
                maximum,
                saved.protrusion_distance_mm,
            )

        return maximum

    @property
    def maximum_right_lateral_angle_degrees(self) -> float:
        """
        Current right mechanical limit reduced by calibration.
        """

        maximum = (
            self.combined
            .maximum_right_lateral_angle_degrees
        )
        saved = self.limits.right_canine

        if saved is not None:
            maximum = min(
                maximum,
                saved.lateral_angle_degrees,
            )

        return maximum

    @property
    def maximum_left_lateral_angle_degrees(self) -> float:
        """
        Current left magnitude reduced by calibration.
        """

        maximum = (
            self.combined
            .maximum_left_lateral_angle_degrees
        )
        saved = self.limits.left_canine

        if saved is not None:
            maximum = min(
                maximum,
                -saved.lateral_angle_degrees,
            )

        return maximum

    def set_position(
        self,
        opening_angle_degrees: float,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> MandibularOcclusalPosition:
        """
        Set a combined position within saved endpoints.

        Saved limits clamp otherwise valid positive travel.
        The combined controller still validates mechanical
        and shared-guide constraints atomically.
        """

        lateral_angle_degrees = float(
            lateral_angle_degrees
        )
        protrusion_distance_mm = float(
            protrusion_distance_mm
        )

        if math.isfinite(lateral_angle_degrees):
            if lateral_angle_degrees > 0.0:
                saved = self.limits.right_canine

                if saved is not None:
                    lateral_angle_degrees = min(
                        lateral_angle_degrees,
                        saved.lateral_angle_degrees,
                    )
            elif lateral_angle_degrees < 0.0:
                saved = self.limits.left_canine

                if saved is not None:
                    lateral_angle_degrees = max(
                        lateral_angle_degrees,
                        saved.lateral_angle_degrees,
                    )

        if (
            math.isfinite(protrusion_distance_mm)
            and protrusion_distance_mm > 0.0
        ):
            saved = self.limits.protrusive

            if saved is not None:
                protrusion_distance_mm = min(
                    protrusion_distance_mm,
                    saved.protrusion_distance_mm,
                )

        return self._apply_combined_change(
            lambda: self.combined.set_position(
                opening_angle_degrees=(
                    opening_angle_degrees
                ),
                lateral_angle_degrees=(
                    lateral_angle_degrees
                ),
                protrusion_distance_mm=(
                    protrusion_distance_mm
                ),
            )
        )

    def set_opening(
        self,
        angle_degrees: float,
    ) -> MandibularOcclusalPosition:
        """
        Set base opening while preserving other movement.
        """

        return self.set_position(
            opening_angle_degrees=angle_degrees,
            lateral_angle_degrees=(
                self.lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                self.protrusion_distance_mm
            ),
        )

    def set_lateral(
        self,
        angle_degrees: float,
    ) -> MandibularOcclusalPosition:
        """
        Set lateral excursion within a saved endpoint.
        """

        return self.set_position(
            opening_angle_degrees=(
                self.opening_angle_degrees
            ),
            lateral_angle_degrees=angle_degrees,
            protrusion_distance_mm=(
                self.protrusion_distance_mm
            ),
        )

    def set_protrusion(
        self,
        distance_mm: float,
    ) -> MandibularOcclusalPosition:
        """
        Set protrusion within a saved endpoint.
        """

        return self.set_position(
            opening_angle_degrees=(
                self.opening_angle_degrees
            ),
            lateral_angle_degrees=(
                self.lateral_angle_degrees
            ),
            protrusion_distance_mm=distance_mm,
        )

    def open_mandible(self) -> MandibularOcclusalPosition:
        """
        Increase base opening by its configured step.
        """

        return self.set_opening(
            min(
                self.opening_angle_degrees
                + self.combined.opening_step_degrees,
                self.combined.maximum_opening_angle_degrees,
            )
        )

    def close_mandible(self) -> MandibularOcclusalPosition:
        """
        Decrease base opening by its configured step.
        """

        return self.set_opening(
            max(
                self.opening_angle_degrees
                - self.combined.opening_step_degrees,
                0.0,
            )
        )

    def move_right(self) -> MandibularOcclusalPosition:
        """
        Move one step toward the effective right endpoint.
        """

        return self.set_lateral(
            min(
                self.lateral_angle_degrees
                + self.combined.lateral_step_degrees,
                self.maximum_right_lateral_angle_degrees,
            )
        )

    def move_left(self) -> MandibularOcclusalPosition:
        """
        Move one step toward the effective left endpoint.
        """

        return self.set_lateral(
            max(
                self.lateral_angle_degrees
                - self.combined.lateral_step_degrees,
                -self.maximum_left_lateral_angle_degrees,
            )
        )

    def advance(self) -> MandibularOcclusalPosition:
        """
        Advance one step toward the effective endpoint.
        """

        return self.set_protrusion(
            min(
                self.protrusion_distance_mm
                + self.combined.protrusion_step_mm,
                self.maximum_protrusion_distance_mm,
            )
        )

    def retreat(self) -> MandibularOcclusalPosition:
        """
        Retreat one configured step toward centric relation.
        """

        return self.set_protrusion(
            max(
                self.protrusion_distance_mm
                - self.combined.protrusion_step_mm,
                0.0,
            )
        )

    def reset_movement(self) -> MandibularOcclusalPosition:
        """
        Return all combined movement components to zero.
        """

        return self.set_position(
            opening_angle_degrees=0.0,
            lateral_angle_degrees=0.0,
            protrusion_distance_mm=0.0,
        )

    def adjust_open(self) -> MandibularOcclusalPosition:
        """
        Add one fine opening adjustment step.
        """

        return self.closure.open()

    def adjust_close(self) -> MandibularOcclusalPosition:
        """
        Add one fine closing adjustment step.
        """

        return self.closure.close()

    def set_adjustment(
        self,
        angle_degrees: float,
    ) -> MandibularOcclusalPosition:
        """
        Set the exact fine occlusal adjustment.
        """

        return self.closure.set_adjustment(
            angle_degrees
        )

    def reset_adjustment(self) -> MandibularOcclusalPosition:
        """
        Remove the fine occlusal adjustment only.
        """

        return self.closure.reset()

    def save_protrusive_limit(self) -> FunctionalLimit:
        """
        Save the current confirmed edge-to-edge position.
        """

        return self.limits.save_protrusive(
            self.position
        )

    def save_right_canine_limit(self) -> FunctionalLimit:
        """
        Save the current confirmed right canine endpoint.
        """

        return self.limits.save_right_canine(
            self.position
        )

    def save_left_canine_limit(self) -> FunctionalLimit:
        """
        Save the current confirmed left canine endpoint.
        """

        return self.limits.save_left_canine(
            self.position
        )

    def go_to_limit(
        self,
        kind: FunctionalLimitKind | str,
    ) -> MandibularOcclusalPosition:
        """
        Reproduce one saved operator-confirmed endpoint.
        """

        saved = self.limits.get(kind)

        if saved is None:
            raise ValueError(
                "The requested functional limit is not saved."
            )

        self._apply_combined_change(
            lambda: self.combined.set_position(
                opening_angle_degrees=(
                    saved.base_opening_angle_degrees
                ),
                lateral_angle_degrees=(
                    saved.lateral_angle_degrees
                ),
                protrusion_distance_mm=(
                    saved.protrusion_distance_mm
                ),
            )
        )

        return self.closure.set_adjustment(
            saved.adjustment_angle_degrees
        )

    def clear_limit(
        self,
        kind: FunctionalLimitKind | str,
    ) -> FunctionalLimit | None:
        """
        Remove one endpoint before optional recalibration.
        """

        return self.limits.clear(kind)

    def clear_limits(self) -> None:
        """
        Remove every operator-confirmed endpoint.
        """

        self.limits.clear_all()
