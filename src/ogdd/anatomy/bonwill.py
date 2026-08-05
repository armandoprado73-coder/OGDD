"""
OGDD - Bonwill Triangle

Anatomical reference representing the Bonwill triangle.

Defined by two condylar landmarks and
the dental midline.
"""

from dataclasses import dataclass

from ogdd.anatomy.landmark import Landmark

from ogdd.geometry.triangle import Triangle


@dataclass(frozen=True)
class BonwillTriangle:
    """
    Anatomical reference representing the Bonwill triangle.

    Defined by two condylar landmarks and
    the dental midline.
    """

    left_condyle: Landmark
    right_condyle: Landmark
    dental_midline: Landmark

    @property
    def triangle(self) -> Triangle:
        """
        Returns the geometric triangle corresponding to the
        Bonwill anatomical reference.
        """

        return Triangle(
            a=self.right_condyle.point,
            b=self.left_condyle.point,
            c=self.dental_midline.point,
        )

    @property
    def right_side(self) -> float:
        """
        Distance between the right condyle
        and the dental midline.
        """

        return self.triangle.side_ca

    @property
    def left_side(self) -> float:
        """
        Distance between the left condyle
        and the dental midline.
        """

        return self.triangle.side_bc

    @property
    def condylar_width(self) -> float:
        """
        Distance between both condyles.
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