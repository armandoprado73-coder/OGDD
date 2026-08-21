"""
OGDD - Articulator Configuration

Configuration parameters for a virtual dental articulator.
"""

from dataclasses import dataclass
from enum import Enum


class IntercondylarPreset(Enum):
    """
    Standard intercondylar width presets in millimeters.
    """

    S = 95.0
    M = 110.0
    L = 140.0


@dataclass(frozen=True)
class ArticulatorConfiguration:
    """
    Mechanical configuration of a virtual articulator.
    """

    intercondylar_width: float = IntercondylarPreset.M.value
    balkwill_angle_degrees: float = 25.0
    
    def __post_init__(self) -> None:
        
        if self.intercondylar_width <= 0:
            raise ValueError(
                "Intercondylar width must be positive."
            )
        if not 0.0 < self.balkwill_angle_degrees < 90.0:
            raise ValueError(
                "Balkwill angle must be between 0 and 90 degrees."
            )    

    @property
    def bonwill_side_length(self) -> float:
        """
        Side length of the virtual Bonwill triangle.

        OGDD uses the selected articulator size
        to scale the complete Bonwill triangle.
        """

        return self.intercondylar_width



    @classmethod
    def from_preset(
        cls,
        preset: IntercondylarPreset,
    ) -> "ArticulatorConfiguration":
        """
        Creates an articulator configuration
        from a standard intercondylar preset.
        """

        return cls(
            intercondylar_width=preset.value
        )