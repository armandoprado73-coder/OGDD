"""
OGDD - Lateral Excursion

Mandibular lateral movement around the working condyle.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.geometry.transform import Transform
from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.mesh import Mesh
from ogdd.anatomy.mandibular_assembly import MandibularAssembly

class LateralSide(Enum):
    """
    Working side of a mandibular lateral excursion.
    """

    RIGHT = "right"
    LEFT = "left"

@dataclass(frozen=True)
class MandibularLateralPosition:
    """
    Complete mandibular position during a lateral excursion.
    """

    mesh: Mesh

    balkwill: BalkwillTriangle

    bonwill: BonwillTriangle

    hinge_axis: HingeAxis

    working_side: LateralSide

    angle_degrees: float

@dataclass(frozen=True)
class LateralExcursion:
    """
    Mandibular lateral excursion around the working condyle.

    The working condyle remains fixed and acts as the
    center of rotation. The balancing condyle translates
    along an arc while preserving the intercondylar
    distance.

    The excursion angle represents mandibular rotation,
    not the Bennett angle.
    """

    hinge_axis: HingeAxis

    superior_direction: np.ndarray

    working_side: LateralSide

    def __post_init__(self) -> None:
        """
        Validate and normalize the superior direction.
        """

        superior_direction = np.asarray(
            self.superior_direction,
            dtype=float,
        )

        if superior_direction.shape != (3,):
            raise ValueError(
                "The superior direction must be a 3D vector."
            )

        direction_length = np.linalg.norm(
            superior_direction
        )

        if np.isclose(direction_length, 0.0):
            raise ValueError(
                "The superior direction cannot have zero length."
            )

        if not isinstance(
            self.working_side,
            LateralSide,
        ):
            raise ValueError(
                "The working side must be a LateralSide."
            )

        object.__setattr__(
            self,
            "superior_direction",
            superior_direction / direction_length,
        )

    @property
    def working_condyle(self) -> Landmark:
        """
        Condyle that remains fixed and rotates.
        """

        if self.working_side is LateralSide.RIGHT:
            return self.hinge_axis.right_condyle

        return self.hinge_axis.left_condyle

    @property
    def balancing_condyle(self) -> Landmark:
        """
        Condyle that translates during lateral movement.
        """

        if self.working_side is LateralSide.RIGHT:
            return self.hinge_axis.left_condyle

        return self.hinge_axis.right_condyle

    def transform_at(
        self,
        angle_degrees: float,
    ) -> Transform:
        """
        Build the rigid transformation for an excursion.

        Positive input angles move the balancing condyle
        toward the anterior region on either working side.
        """

        angle_degrees = float(
            angle_degrees
        )

        if angle_degrees < 0.0:
            raise ValueError(
                "The excursion angle cannot be negative."
            )

        if self.working_side is LateralSide.RIGHT:
            signed_angle = -angle_degrees
        else:
            signed_angle = angle_degrees

        return Transform.rotation_about_axis(
            origin=self.working_condyle.point,
            axis=self.superior_direction,
            angle_degrees=signed_angle,
        )

    def rotate_points(
        self,
        points: np.ndarray,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Rotate 3D points around the working condyle.
        """

        return self.transform_at(
            angle_degrees=angle_degrees,
        ).apply(points)

    def working_condyle_at(
        self,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Return the fixed working-condyle center.
        """

        result = self.rotate_points(
            points=np.array(
                [
                    self.working_condyle.point,
                ]
            ),
            angle_degrees=angle_degrees,
        )

        return result[0]

    def balancing_condyle_at(
        self,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Return the translated balancing-condyle center.
        """

        result = self.rotate_points(
            points=np.array(
                [
                    self.balancing_condyle.point,
                ]
            ),
            angle_degrees=angle_degrees,
        )

        return result[0]
    def rotate_landmark(
        self,
        landmark: Landmark,
        angle_degrees: float,
    ) -> Landmark:
        """
        Return a rotated copy of an anatomical landmark.
        """

        rotated_point = self.rotate_points(
            points=np.array(
                [
                    landmark.point,
                ]
            ),
            angle_degrees=angle_degrees,
        )[0]

        return Landmark(
            name=landmark.name,
            point=rotated_point,
            reference_used=landmark.reference_used,
            confidence=landmark.confidence,
            created_by=landmark.created_by,
        )

    def rotate_balkwill(
        self,
        balkwill: BalkwillTriangle,
        angle_degrees: float,
    ) -> BalkwillTriangle:
        """
        Rotate the complete Balkwill triangle.
        """

        return BalkwillTriangle(
            left_posterior=self.rotate_landmark(
                landmark=balkwill.left_posterior,
                angle_degrees=angle_degrees,
            ),
            right_posterior=self.rotate_landmark(
                landmark=balkwill.right_posterior,
                angle_degrees=angle_degrees,
            ),
            dental_midline=self.rotate_landmark(
                landmark=balkwill.dental_midline,
                angle_degrees=angle_degrees,
            ),
        )

    def rotate_bonwill(
        self,
        bonwill: BonwillTriangle,
        angle_degrees: float,
    ) -> BonwillTriangle:
        """
        Rotate the complete Bonwill triangle.

        The working condyle remains fixed while the
        balancing condyle translates.
        """

        return BonwillTriangle(
            left_condyle=self.rotate_landmark(
                landmark=bonwill.left_condyle,
                angle_degrees=angle_degrees,
            ),
            right_condyle=self.rotate_landmark(
                landmark=bonwill.right_condyle,
                angle_degrees=angle_degrees,
            ),
            dental_midline=self.rotate_landmark(
                landmark=bonwill.dental_midline,
                angle_degrees=angle_degrees,
            ),
        )

    def rotate_mesh(
        self,
        mesh: Mesh,
        angle_degrees: float,
    ) -> Mesh:
        """
        Return a rigidly rotated copy of a mesh.
        """

        transform = self.transform_at(
            angle_degrees=angle_degrees
        )

        rotated_vertices = transform.apply(
            mesh.vertices
        )

        rotated_normals = None

        if mesh.normals is not None:
            rotated_normals = (
                transform.apply_vectors(
                    mesh.normals
                )
            )

        return Mesh(
            vertices=rotated_vertices,
            faces=mesh.faces.copy(),
            normals=rotated_normals,
            attributes={
                name: values.copy()
                for name, values
                in mesh.attributes.items()
            },
            metadata=mesh.metadata.copy(),
        )
    def position_at(
        self,
        assembly: MandibularAssembly,
        angle_degrees: float,
    ) -> MandibularLateralPosition:
        """
        Calculate a complete mandibular lateral position.

        Every position is calculated from the original
        assembly to avoid accumulated rotation errors.
        """

        if not np.allclose(
            assembly.hinge_axis.left_condyle.point,
            self.hinge_axis.left_condyle.point,
        ):
            raise ValueError(
                "The assembly and excursion must share "
                "the same left condyle."
            )

        if not np.allclose(
            assembly.hinge_axis.right_condyle.point,
            self.hinge_axis.right_condyle.point,
        ):
            raise ValueError(
                "The assembly and excursion must share "
                "the same right condyle."
            )

        angle_degrees = float(
            angle_degrees
        )

        rotated_mesh = self.rotate_mesh(
            mesh=assembly.mesh,
            angle_degrees=angle_degrees,
        )

        rotated_balkwill = self.rotate_balkwill(
            balkwill=assembly.balkwill,
            angle_degrees=angle_degrees,
        )

        rotated_bonwill = self.rotate_bonwill(
            bonwill=assembly.bonwill,
            angle_degrees=angle_degrees,
        )

        rotated_hinge_axis = HingeAxis(
            left_condyle=rotated_bonwill.left_condyle,
            right_condyle=rotated_bonwill.right_condyle,
        )

        return MandibularLateralPosition(
            mesh=rotated_mesh,
            balkwill=rotated_balkwill,
            bonwill=rotated_bonwill,
            hinge_axis=rotated_hinge_axis,
            working_side=self.working_side,
            angle_degrees=angle_degrees,
        )