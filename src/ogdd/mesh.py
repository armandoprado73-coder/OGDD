"""
OGDD Mesh Core Module

Defines the fundamental 3D mesh representation used by OGDD.

A mesh is represented by:

    vertices -> Nx3 numpy array
    faces    -> Mx3 numpy array

The class intentionally contains no algorithms.
Computational operations belong to specialized modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    """
    Fundamental 3D mesh container.

    Parameters
    ----------
    vertices:
        Array containing vertex coordinates.
        Shape must be (N,3).

    faces:
        Array containing triangular faces.
        Shape must be (M,3).

    Attributes
    ----------
    normals:
        Optional face or vertex normals.

    attributes:
        Dictionary for computed data.

    metadata:
        Additional information about the mesh.
    """

    vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=float)
    )

    faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int32)
    )

    normals: np.ndarray | None = None

    attributes: dict[str, np.ndarray] = field(
        default_factory=dict
    )

    metadata: dict = field(
        default_factory=dict
    )


    def __post_init__(self) -> None:
        """
        Validate mesh data after initialization.
        """

        self.vertices = np.asarray(
            self.vertices,
            dtype=float
        )

        self.faces = np.asarray(
            self.faces,
            dtype=np.int32
        )

        self._validate()


    def _validate(self) -> None:
        """
        Validate mesh array dimensions.
        """

        if self.vertices.size > 0:

            if self.vertices.ndim != 2:
                raise ValueError(
                    "Vertices must be a 2D array."
                )

            if self.vertices.shape[1] != 3:
                raise ValueError(
                    "Vertices must have shape (N,3)."
                )


        if self.faces.size > 0:

            if self.faces.ndim != 2:
                raise ValueError(
                    "Faces must be a 2D array."
                )

            if self.faces.shape[1] != 3:
                raise ValueError(
                    "Only triangular faces are supported."
                )


    @property
    def vertex_count(self) -> int:
        """
        Number of vertices.
        """

        return len(self.vertices)


    @property
    def face_count(self) -> int:
        """
        Number of triangular faces.
        """

        return len(self.faces)


    def add_attribute(
        self,
        name: str,
        values: np.ndarray
    ) -> None:
        """
        Add computed data associated with vertices.

        Examples
        --------

        mesh.add_attribute(
            "curvature",
            curvature_values
        )

        mesh.add_attribute(
            "probability",
            occlusal_probability
        )
        """

        values = np.asarray(values)

        if len(values) != self.vertex_count:
            raise ValueError(
                "Attribute size must match vertex count."
            )

        self.attributes[name] = values


    def __repr__(self) -> str:
        """
        Developer representation.
        """

        return (
            f"Mesh("
            f"vertices={self.vertex_count}, "
            f"faces={self.face_count}"
            f")"
        )
