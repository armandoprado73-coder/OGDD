"""
OGDD - Combined Movement Controller

Stateful control of opening, lateral excursion and
protrusion while protecting the condylar guide limits.
"""

from dataclasses import dataclass, field
import math

from ogdd.articulator.combined_movement import (
    CombinedMovement,
    MandibularCombinedPosition,
)
from ogdd.articulator.guided_lateral_excursion import (
    GuidedLateralExcursion,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)


@dataclass
class CombinedController:
    """
    Control all current mandibular movements together.

    Negative lateral angles represent left excursion.
    Positive lateral angles represent right excursion.
    Opening and protrusion use non-negative values.

    Lateral excursion and protrusion share condylar guide
    travel. Their available limits are therefore dynamic.
    """

    movement: CombinedMovement

    maximum_opening_angle_degrees: float

    maximum_lateral_angle_degrees: float

    maximum_protrusion_distance_mm: float

    opening_step_degrees: float = 1.0

    lateral_step_degrees: float = 1.0

    protrusion_step_mm: float = 1.0

    _opening_angle_degrees: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    _lateral_angle_degrees: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    _protrusion_distance_mm: float = field(
        default=0.0,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate controller limits and step sizes.
        """

        self.maximum_opening_angle_degrees = float(
            self.maximum_opening_angle_degrees
        )
        self.maximum_lateral_angle_degrees = float(
            self.maximum_lateral_angle_degrees
        )
        self.maximum_protrusion_distance_mm = float(
            self.maximum_protrusion_distance_mm
        )

        self.opening_step_degrees = float(
            self.opening_step_degrees
        )
        self.lateral_step_degrees = float(
            self.lateral_step_degrees
        )
        self.protrusion_step_mm = float(
            self.protrusion_step_mm
        )

        self._validate_positive_finite(
            value=self.maximum_opening_angle_degrees,
            name="Maximum opening angle",
        )
        self._validate_positive_finite(
            value=self.maximum_lateral_angle_degrees,
            name="Maximum lateral angle",
        )
        self._validate_positive_finite(
            value=self.maximum_protrusion_distance_mm,
            name="Maximum protrusion distance",
        )
        self._validate_positive_finite(
            value=self.opening_step_degrees,
            name="Opening step",
        )
        self._validate_positive_finite(
            value=self.lateral_step_degrees,
            name="Lateral step",
        )
        self._validate_positive_finite(
            value=self.protrusion_step_mm,
            name="Protrusion step",
        )

        if (
            self.maximum_lateral_angle_degrees
            > self.movement
            .right_excursion
            .maximum_angle_degrees
            or self.maximum_lateral_angle_degrees
            > self.movement
            .left_excursion
            .maximum_angle_degrees
        ):
            raise ValueError(
                "Maximum lateral angle cannot exceed "
                "either condylar guide limit."
            )

        if (
            self.maximum_protrusion_distance_mm
            > self.movement
            .protrusion
            .maximum_translation
        ):
            raise ValueError(
                "Maximum protrusion distance cannot "
                "exceed the common guide limit."
            )

        if (
            self.opening_step_degrees
            > self.maximum_opening_angle_degrees
        ):
            raise ValueError(
                "Opening step cannot exceed its maximum."
            )

        if (
            self.lateral_step_degrees
            > self.maximum_lateral_angle_degrees
        ):
            raise ValueError(
                "Lateral step cannot exceed its maximum."
            )

        if (
            self.protrusion_step_mm
            > self.maximum_protrusion_distance_mm
        ):
            raise ValueError(
                "Protrusion step cannot exceed its maximum."
            )

    @staticmethod
    def _validate_positive_finite(
        value: float,
        name: str,
    ) -> None:
        """
        Require a positive finite configuration value.
        """

        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be positive and finite."
            )

    @staticmethod
    def _normalized_zero(value: float) -> float:
        """
        Normalize numerical values close to zero.
        """

        if math.isclose(
            value,
            0.0,
            abs_tol=1e-12,
        ):
            return 0.0

        return value

    @property
    def opening_angle_degrees(self) -> float:
        """
        Current opening angle.
        """

        return self._opening_angle_degrees

    @property
    def lateral_angle_degrees(self) -> float:
        """
        Current signed lateral angle.
        """

        return self._lateral_angle_degrees

    @property
    def protrusion_distance_mm(self) -> float:
        """
        Current protrusive distance.
        """

        return self._protrusion_distance_mm

    @property
    def working_side(self) -> LateralSide | None:
        """
        Current working side, or None at lateral center.
        """

        if math.isclose(
            self._lateral_angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return None

        if self._lateral_angle_degrees > 0.0:
            return LateralSide.RIGHT

        return LateralSide.LEFT

    @property
    def is_centered(self) -> bool:
        """
        Whether all three movements are at zero.
        """

        return (
            math.isclose(
                self._opening_angle_degrees,
                0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                self._lateral_angle_degrees,
                0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                self._protrusion_distance_mm,
                0.0,
                abs_tol=1e-12,
            )
        )

    @property
    def position(self) -> MandibularCombinedPosition:
        """
        Complete position at the current controller state.
        """

        return self.movement.position_at(
            opening_angle_degrees=(
                self._opening_angle_degrees
            ),
            lateral_angle_degrees=(
                self._lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                self._protrusion_distance_mm
            ),
        )

    def _excursion_for_side(
        self,
        side: LateralSide,
    ) -> GuidedLateralExcursion:
        """
        Return the configured excursion for one side.
        """

        if side is LateralSide.RIGHT:
            return self.movement.right_excursion

        if side is LateralSide.LEFT:
            return self.movement.left_excursion

        raise ValueError(
            "Side must be a LateralSide."
        )

    def maximum_lateral_angle_for(
        self,
        side: LateralSide,
        protrusion_distance_mm: float | None = None,
    ) -> float:
        """
        Available lateral angle at a protrusive distance.
        """

        excursion = self._excursion_for_side(side)

        if protrusion_distance_mm is None:
            protrusion_distance_mm = (
                self._protrusion_distance_mm
            )

        protrusion_distance_mm = float(
            protrusion_distance_mm
        )

        if (
            not math.isfinite(protrusion_distance_mm)
            or not 0.0
            <= protrusion_distance_mm
            <= self.maximum_protrusion_distance_mm
        ):
            raise ValueError(
                "Protrusion distance must remain "
                "within the configured limits."
            )

        remaining_guide_travel = max(
            excursion
            .balancing_guide
            .maximum_translation
            - protrusion_distance_mm,
            0.0,
        )

        ratio = min(
            remaining_guide_travel
            / excursion.hinge_axis.length,
            1.0,
        )

        guide_angle = math.degrees(
            math.asin(ratio)
        )

        return min(
            self.maximum_lateral_angle_degrees,
            excursion.maximum_angle_degrees,
            guide_angle,
        )

    @property
    def maximum_right_lateral_angle_degrees(
        self,
    ) -> float:
        """
        Current dynamic right lateral limit.
        """

        return self.maximum_lateral_angle_for(
            LateralSide.RIGHT
        )

    @property
    def maximum_left_lateral_angle_degrees(
        self,
    ) -> float:
        """
        Current dynamic left lateral magnitude limit.
        """

        return self.maximum_lateral_angle_for(
            LateralSide.LEFT
        )

    def maximum_protrusion_at(
        self,
        lateral_angle_degrees: float,
    ) -> float:
        """
        Available protrusion at a signed lateral angle.
        """

        lateral_angle_degrees = float(
            lateral_angle_degrees
        )

        if not math.isfinite(lateral_angle_degrees):
            raise ValueError(
                "Lateral angle must be finite."
            )

        if math.isclose(
            lateral_angle_degrees,
            0.0,
            abs_tol=1e-12,
        ):
            return self.maximum_protrusion_distance_mm

        if lateral_angle_degrees > 0.0:
            side = LateralSide.RIGHT
            angle_magnitude = lateral_angle_degrees
        else:
            side = LateralSide.LEFT
            angle_magnitude = -lateral_angle_degrees

        excursion = self._excursion_for_side(side)

        if (
            angle_magnitude
            > self.maximum_lateral_angle_degrees
            or angle_magnitude
            > excursion.maximum_angle_degrees
        ):
            raise ValueError(
                "Lateral angle must remain within "
                "the configured limits."
            )

        remaining_guide_travel = max(
            excursion
            .balancing_guide
            .maximum_translation
            - excursion.guide_distance_at(
                angle_magnitude
            ),
            0.0,
        )

        return min(
            self.maximum_protrusion_distance_mm,
            remaining_guide_travel,
        )

    @property
    def maximum_current_protrusion_distance_mm(
        self,
    ) -> float:
        """
        Dynamic protrusive limit at current lateral angle.
        """

        return self.maximum_protrusion_at(
            self._lateral_angle_degrees
        )

    def _validated_state(
        self,
        opening_angle_degrees: float,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> tuple[float, float, float]:
        """
        Validate a complete candidate state atomically.
        """

        opening_angle_degrees = float(
            opening_angle_degrees
        )
        lateral_angle_degrees = float(
            lateral_angle_degrees
        )
        protrusion_distance_mm = float(
            protrusion_distance_mm
        )

        if (
            not math.isfinite(opening_angle_degrees)
            or not 0.0
            <= opening_angle_degrees
            <= self.maximum_opening_angle_degrees
        ):
            raise ValueError(
                "Opening angle must remain within "
                "the configured limits."
            )

        if (
            not math.isfinite(lateral_angle_degrees)
            or not -self.maximum_lateral_angle_degrees
            <= lateral_angle_degrees
            <= self.maximum_lateral_angle_degrees
        ):
            raise ValueError(
                "Lateral angle must remain within "
                "the configured limits."
            )

        if (
            not math.isfinite(protrusion_distance_mm)
            or not 0.0
            <= protrusion_distance_mm
            <= self.maximum_protrusion_distance_mm
        ):
            raise ValueError(
                "Protrusion distance must remain "
                "within the configured limits."
            )

        maximum_lateral_angle = (
            self.maximum_right_lateral_angle_degrees
        )

        if lateral_angle_degrees < 0.0:
            maximum_lateral_angle = (
                self.maximum_lateral_angle_for(
                    LateralSide.LEFT,
                    protrusion_distance_mm,
                )
            )
        elif lateral_angle_degrees > 0.0:
            maximum_lateral_angle = (
                self.maximum_lateral_angle_for(
                    LateralSide.RIGHT,
                    protrusion_distance_mm,
                )
            )

        if (
            abs(lateral_angle_degrees)
            > maximum_lateral_angle
            + 1e-12
        ):
            raise ValueError(
                "The combined movement exceeds the "
                "balancing condylar guide limit."
            )

        maximum_protrusion = self.maximum_protrusion_at(
            lateral_angle_degrees
        )

        if (
            protrusion_distance_mm
            > maximum_protrusion
            + 1e-12
        ):
            raise ValueError(
                "The combined movement exceeds the "
                "balancing condylar guide limit."
            )

        return (
            self._normalized_zero(
                opening_angle_degrees
            ),
            self._normalized_zero(
                lateral_angle_degrees
            ),
            self._normalized_zero(
                protrusion_distance_mm
            ),
        )

    def set_position(
        self,
        opening_angle_degrees: float,
        lateral_angle_degrees: float,
        protrusion_distance_mm: float,
    ) -> MandibularCombinedPosition:
        """
        Set all movement components atomically.
        """

        (
            opening_angle_degrees,
            lateral_angle_degrees,
            protrusion_distance_mm,
        ) = self._validated_state(
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

        self._opening_angle_degrees = (
            opening_angle_degrees
        )
        self._lateral_angle_degrees = (
            lateral_angle_degrees
        )
        self._protrusion_distance_mm = (
            protrusion_distance_mm
        )

        return self.position

    def set_opening(
        self,
        angle_degrees: float,
    ) -> MandibularCombinedPosition:
        """
        Set opening while preserving other components.
        """

        return self.set_position(
            opening_angle_degrees=angle_degrees,
            lateral_angle_degrees=(
                self._lateral_angle_degrees
            ),
            protrusion_distance_mm=(
                self._protrusion_distance_mm
            ),
        )

    def set_lateral(
        self,
        angle_degrees: float,
    ) -> MandibularCombinedPosition:
        """
        Set lateral angle while preserving other movement.
        """

        return self.set_position(
            opening_angle_degrees=(
                self._opening_angle_degrees
            ),
            lateral_angle_degrees=angle_degrees,
            protrusion_distance_mm=(
                self._protrusion_distance_mm
            ),
        )

    def set_protrusion(
        self,
        distance_mm: float,
    ) -> MandibularCombinedPosition:
        """
        Set protrusion while preserving other movement.
        """

        return self.set_position(
            opening_angle_degrees=(
                self._opening_angle_degrees
            ),
            lateral_angle_degrees=(
                self._lateral_angle_degrees
            ),
            protrusion_distance_mm=distance_mm,
        )

    def open(self) -> MandibularCombinedPosition:
        """
        Increase opening by one configured step.
        """

        return self.set_opening(
            min(
                self._opening_angle_degrees
                + self.opening_step_degrees,
                self.maximum_opening_angle_degrees,
            )
        )

    def close(self) -> MandibularCombinedPosition:
        """
        Decrease opening by one configured step.
        """

        return self.set_opening(
            max(
                self._opening_angle_degrees
                - self.opening_step_degrees,
                0.0,
            )
        )

    def move_right(self) -> MandibularCombinedPosition:
        """
        Move one step toward right lateral excursion.
        """

        return self.set_lateral(
            min(
                self._lateral_angle_degrees
                + self.lateral_step_degrees,
                self.maximum_right_lateral_angle_degrees,
            )
        )

    def move_left(self) -> MandibularCombinedPosition:
        """
        Move one step toward left lateral excursion.
        """

        return self.set_lateral(
            max(
                self._lateral_angle_degrees
                - self.lateral_step_degrees,
                -self.maximum_left_lateral_angle_degrees,
            )
        )

    def advance(self) -> MandibularCombinedPosition:
        """
        Increase protrusion by one available step.
        """

        return self.set_protrusion(
            min(
                self._protrusion_distance_mm
                + self.protrusion_step_mm,
                self.maximum_current_protrusion_distance_mm,
            )
        )

    def retreat(self) -> MandibularCombinedPosition:
        """
        Decrease protrusion by one configured step.
        """

        return self.set_protrusion(
            max(
                self._protrusion_distance_mm
                - self.protrusion_step_mm,
                0.0,
            )
        )

    def reset(self) -> MandibularCombinedPosition:
        """
        Return opening, lateral and protrusion to zero.
        """

        return self.set_position(
            opening_angle_degrees=0.0,
            lateral_angle_degrees=0.0,
            protrusion_distance_mm=0.0,
        )
