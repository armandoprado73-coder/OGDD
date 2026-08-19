"""
OGDD - Balkwill Triangle

Anatomical reference representing the Balkwill triangle.

Defined by two posterior occlusal landmarks and
the dental midline.
"""
from dataclasses import dataclass

from ogdd.anatomy.landmark import Landmark

from ogdd.geometry.triangle import Triangle

from ogdd.geometry.coordinate_system import CoordinateSystem

@dataclass(frozen=True)
class BalkwillTriangle:
    """
    Anatomical reference representing the Balkwill triangle.

    Defined by two posterior occlusal landmarks and
    the dental midline.
    """
   
    left_posterior: Landmark
    right_posterior: Landmark
    dental_midline: Landmark

    @property
    def triangle(self) -> Triangle:
        """
        Returns the geometric triangle corresponding to the
        Balkwill anatomical reference.
        """
        return Triangle(
            a=self.right_posterior.point,
            b=self.left_posterior.point,
            c=self.dental_midline.point,
        )

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """
        Returns the anatomical coordinate system defined
        by the Balkwill landmarks.

        OGDD anatomical convention:

        Origin = dental midline
        +X = patient's right
        +Y = anterior
        +Z = superior
        """

        return CoordinateSystem.from_dental_landmarks(
            right_molar=self.right_posterior.point,
            left_molar=self.left_posterior.point,
            dental_midline=self.dental_midline.point,
        )

    @property
    def right_side(self) -> float:
        """
        Distance between the right posterior landmark
        and the dental midline.
        """

        return self.triangle.side_ca


    @property
    def left_side(self) -> float:
        """
        Distance between the left posterior landmark
        and the dental midline.
        """

        return self.triangle.side_bc


    @property
    def intermolar_width(self) -> float:
        """
        Distance between both posterior landmarks.
        """

        return self.triangle.side_ab

    @property
    def symmetry_difference(self) -> float:
        """
        Difference between the left and right sides.
        """

        return abs(
            self.right_side - self.left_side
        )