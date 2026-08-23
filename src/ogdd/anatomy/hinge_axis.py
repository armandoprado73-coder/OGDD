"""
OGDD - Hinge Axis

Anatomical axis defined by the right and left condyles.
"""

from dataclasses import dataclass

import numpy as np

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.mesh import Mesh
from ogdd.anatomy.landmark import Landmark


@dataclass(frozen=True)
class HingeAxis:
    """
    Anatomical hinge axis defined by two condylar landmarks.

    The axis direction points from the left condyle
    toward the right condyle, following the OGDD
    anatomical convention where +X points to the
    patient's right.
    """

    left_condyle: Landmark
    right_condyle: Landmark

    def __post_init__(self) -> None:
        """
        Validates that the condyles define a valid axis.
        """

        if np.allclose(
            self.left_condyle.point,
            self.right_condyle.point,
        ):
            raise ValueError(
                "The condyles must define two different points."
            )

    @property
    def vector(self) -> np.ndarray:
        """
        Vector from the left condyle to the right condyle.
        """

        return (
            self.right_condyle.point
            - self.left_condyle.point
        )

    @property
    def length(self) -> float:
        """
        Distance between both condyles.
        """

        return float(np.linalg.norm(self.vector))

    @property
    def direction(self) -> np.ndarray:
        """
        Unit vector pointing from the left condyle
        toward the right condyle.
        """

        return self.vector / self.length

    @property
    def midpoint(self) -> np.ndarray:
        """
        Midpoint between both condyles.
        """

        return (
            self.left_condyle.point
            + self.right_condyle.point
        ) / 2.0

    def rotate_point(
        self,
        point: np.ndarray,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Rotates a 3D point around the hinge axis.

        Positive angles represent mandibular opening.
        Following the OGDD anatomical convention,
        the anterior region moves toward -Z.

        Parameters
        ----------
        point : np.ndarray
            Point to rotate in 3D space.

        angle_degrees : float
            Opening angle expressed in degrees.

        Returns
        -------
        np.ndarray
            Rotated point.
        """

        point = np.asarray(point, dtype=float)

        if point.shape != (3,):
            raise ValueError(
                "The point must be a 3D vector."
            )

        angle_radians = np.deg2rad(
            -float(angle_degrees)
        )

        axis = self.direction
        relative_point = point - self.midpoint

        cosine = np.cos(angle_radians)
        sine = np.sin(angle_radians)

        rotated_relative_point = (
            relative_point * cosine
            + np.cross(axis, relative_point) * sine
            + axis
            * np.dot(axis, relative_point)
            * (1.0 - cosine)
        )

        return (
            self.midpoint
            + rotated_relative_point
        )

    def rotate_points(
        self,
        points: np.ndarray,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Rotates multiple 3D points around the hinge axis.

        All points receive the same rigid rotation.
        Positive angles represent mandibular opening.

        Parameters
        ----------
        points : np.ndarray
            Array of 3D points with shape (n, 3).

        angle_degrees : float
            Opening angle expressed in degrees.

        Returns
        -------
        np.ndarray
            Rotated points with shape (n, 3).
        """

        points = np.asarray(points, dtype=float)

        if (
            points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValueError(
                "The points must have shape (n, 3)."
            )

        angle_radians = np.deg2rad(
            -float(angle_degrees)
        )

        axis = self.direction
        relative_points = (
            points - self.midpoint
        )

        cosine = np.cos(angle_radians)
        sine = np.sin(angle_radians)

        parallel_component = (
            np.outer(
                relative_points @ axis,
                axis,
            )
        )

        perpendicular_component = (
            relative_points - parallel_component
        )

        cross_component = np.cross(
            axis,
            relative_points,
        )

        rotated_relative_points = (
            parallel_component
            + perpendicular_component * cosine
            + cross_component * sine
        )

        return (
            self.midpoint
            + rotated_relative_points
        )

    def rotate_vectors(
        self,
        vectors: np.ndarray,
        angle_degrees: float,
    ) -> np.ndarray:
        """
        Rotates multiple 3D direction vectors.

        Unlike points, direction vectors are rotated
        without translation around the axis midpoint.
        """

        vectors = np.asarray(vectors, dtype=float)

        if (
            vectors.ndim != 2
            or vectors.shape[1] != 3
        ):
            raise ValueError(
                "The vectors must have shape (n, 3)."
            )

        angle_radians = np.deg2rad(
            -float(angle_degrees)
        )

        axis = self.direction

        cosine = np.cos(angle_radians)
        sine = np.sin(angle_radians)

        parallel_component = np.outer(
            vectors @ axis,
            axis,
        )

        perpendicular_component = (
            vectors - parallel_component
        )

        cross_component = np.cross(
            axis,
            vectors,
        )

        return (
            parallel_component
            + perpendicular_component * cosine
            + cross_component * sine
        )

    def rotate_landmark(
        self,
        landmark: Landmark,
        angle_degrees: float,
    ) -> Landmark:
        """
        Returns a rotated copy of an anatomical landmark.

        The original landmark remains unchanged.
        All landmark metadata is preserved.
        """

        rotated_point = self.rotate_point(
            point=landmark.point,
            angle_degrees=angle_degrees,
        )

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
        Returns a rotated copy of a Balkwill triangle.

        The three anatomical landmarks receive the
        same rigid rotation around the hinge axis.
        The original triangle remains unchanged.
        """

        rotated_left_posterior = (
            self.rotate_landmark(
                landmark=balkwill.left_posterior,
                angle_degrees=angle_degrees,
            )
        )

        rotated_right_posterior = (
            self.rotate_landmark(
                landmark=balkwill.right_posterior,
                angle_degrees=angle_degrees,
            )
        )

        rotated_dental_midline = (
            self.rotate_landmark(
                landmark=balkwill.dental_midline,
                angle_degrees=angle_degrees,
            )
        )

        return BalkwillTriangle(
            left_posterior=rotated_left_posterior,
            right_posterior=rotated_right_posterior,
            dental_midline=rotated_dental_midline,
        )

    def rotate_bonwill(
        self,
        bonwill: BonwillTriangle,
        angle_degrees: float,
    ) -> BonwillTriangle:
        """
        Returns a rotated copy of a Bonwill triangle.

        Both condyles remain fixed because they lie
        on the hinge axis. The dental midline rotates
        around the intercondylar axis.

        The original triangle remains unchanged.
        """

        rotated_left_condyle = (
            self.rotate_landmark(
                landmark=bonwill.left_condyle,
                angle_degrees=angle_degrees,
            )
        )

        rotated_right_condyle = (
            self.rotate_landmark(
                landmark=bonwill.right_condyle,
                angle_degrees=angle_degrees,
            )
        )

        rotated_dental_midline = (
            self.rotate_landmark(
                landmark=bonwill.dental_midline,
                angle_degrees=angle_degrees,
            )
        )

        return BonwillTriangle(
            left_condyle=rotated_left_condyle,
            right_condyle=rotated_right_condyle,
            dental_midline=rotated_dental_midline,
        )

    def rotate_mesh(
        self,
        mesh: Mesh,
        angle_degrees: float,
    ) -> Mesh:
        """
        Returns a rotated copy of a mesh.

        Vertices rotate around the hinge axis.
        Face indices remain unchanged.
        Normals rotate as direction vectors.
        Attributes and metadata are preserved.

        The original mesh remains unchanged.
        """

        rotated_vertices = self.rotate_points(
            points=mesh.vertices,
            angle_degrees=angle_degrees,
        )

        rotated_normals = None

        if mesh.normals is not None:
            rotated_normals = self.rotate_vectors(
                vectors=mesh.normals,
                angle_degrees=angle_degrees,
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