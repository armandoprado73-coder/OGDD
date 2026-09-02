"""
OGDD - Occlusal Record

Represents a rigid interarch record used to compare
centric relation and maximum intercuspation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ogdd.geometry.transform import Transform
from ogdd.mesh import Mesh


def _validated_vertex_indices(
    values,
    name: str,
    vertex_count: int,
) -> np.ndarray:
    """
    Validate one registration region.
    """

    indices = np.asarray(values)

    if indices.ndim != 1:
        raise ValueError(
            f"{name} vertex indices must be one-dimensional."
        )

    if len(indices) == 0:
        raise ValueError(
            f"{name} registration region cannot be empty."
        )

    if not np.issubdtype(
        indices.dtype,
        np.integer,
    ):
        raise ValueError(
            f"{name} vertex indices must be integers."
        )

    indices = indices.astype(
        np.int64,
        copy=True,
    )

    if (
        np.any(indices < 0)
        or np.any(indices >= vertex_count)
    ):
        raise ValueError(
            f"{name} vertex indices are outside the mesh."
        )

    if len(np.unique(indices)) != len(indices):
        raise ValueError(
            f"{name} vertex indices cannot contain duplicates."
        )

    indices.setflags(
        write=False
    )

    return indices


def _validate_rigid_transform(
    transform: Transform,
) -> None:
    """
    Ensure that a transform contains only rotation
    and translation.
    """

    if not isinstance(
        transform,
        Transform,
    ):
        raise TypeError(
            "Position must be defined by a Transform."
        )

    matrix = transform.matrix

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Rigid transformation must contain finite values."
        )

    if not np.allclose(
        matrix[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),
    ):
        raise ValueError(
            "Rigid transformation must use homogeneous coordinates."
        )

    rotation = matrix[:3, :3]

    is_orthonormal = np.allclose(
        rotation.T @ rotation,
        np.eye(3),
    )

    has_positive_orientation = np.isclose(
        np.linalg.det(rotation),
        1.0,
    )

    if (
        not is_orthonormal
        or not has_positive_orientation
    ):
        raise ValueError(
            "Occlusal records accept only rigid "
            "rotation and translation."
        )


def _positioned_mesh(
    mesh: Mesh,
    transform: Transform,
) -> Mesh:
    """
    Create a transformed mesh without changing
    the original mesh.
    """

    transformed_normals = None

    if mesh.normals is not None:
        transformed_normals = (
            transform.apply_vectors(
                mesh.normals
            )
        )

    return Mesh(
        vertices=transform.apply(
            mesh.vertices
        ),
        faces=mesh.faces.copy(),
        normals=transformed_normals,
        attributes={
            name: values.copy()
            for name, values
            in mesh.attributes.items()
        },
        metadata=deepcopy(
            mesh.metadata
        ),
    )


@dataclass(frozen=True)
class OcclusalRecordPosition:
    """
    Occlusal record evaluated at one rigid position.
    """

    mesh: Mesh

    maxillary_vertex_indices: np.ndarray

    mandibular_vertex_indices: np.ndarray

    transform: Transform

    @property
    def maxillary_points(
        self,
    ) -> np.ndarray:
        """
        Positioned maxillary registration points.
        """

        return self.mesh.vertices[
            self.maxillary_vertex_indices
        ].copy()

    @property
    def mandibular_points(
        self,
    ) -> np.ndarray:
        """
        Positioned mandibular registration points.
        """

        return self.mesh.vertices[
            self.mandibular_vertex_indices
        ].copy()


@dataclass(frozen=True)
class OcclusalRecord:
    """
    Combined rigid record of one occlusal relation.

    The mesh remains one object. Vertex regions identify
    the maxillary and mandibular surfaces used during
    registration.
    """

    mesh: Mesh

    maxillary_vertex_indices: np.ndarray

    mandibular_vertex_indices: np.ndarray

    def __post_init__(
        self,
    ) -> None:
        """
        Validate the combined mesh and both regions.
        """

        if self.mesh.vertex_count == 0:
            raise ValueError(
                "Occlusal record mesh cannot be empty."
            )

        maxillary_indices = (
            _validated_vertex_indices(
                values=self.maxillary_vertex_indices,
                name="Maxillary",
                vertex_count=self.mesh.vertex_count,
            )
        )

        mandibular_indices = (
            _validated_vertex_indices(
                values=self.mandibular_vertex_indices,
                name="Mandibular",
                vertex_count=self.mesh.vertex_count,
            )
        )

        if np.intersect1d(
            maxillary_indices,
            mandibular_indices,
        ).size > 0:
            raise ValueError(
                "Maxillary and mandibular registration "
                "regions cannot overlap."
            )

        object.__setattr__(
            self,
            "maxillary_vertex_indices",
            maxillary_indices,
        )

        object.__setattr__(
            self,
            "mandibular_vertex_indices",
            mandibular_indices,
        )

    @property
    def maxillary_points(
        self,
    ) -> np.ndarray:
        """
        Original maxillary registration points.
        """

        return self.mesh.vertices[
            self.maxillary_vertex_indices
        ].copy()

    @property
    def mandibular_points(
        self,
    ) -> np.ndarray:
        """
        Original mandibular registration points.
        """

        return self.mesh.vertices[
            self.mandibular_vertex_indices
        ].copy()

    def position_at(
        self,
        transform: Transform,
    ) -> OcclusalRecordPosition:
        """
        Evaluate the entire record at one rigid position.

        Both registration regions receive exactly the
        same transformation. The original mesh remains
        unchanged.
        """

        _validate_rigid_transform(
            transform
        )

        positioned_mesh = _positioned_mesh(
            mesh=self.mesh,
            transform=transform,
        )

        return OcclusalRecordPosition(
            mesh=positioned_mesh,
            maxillary_vertex_indices=(
                self.maxillary_vertex_indices.copy()
            ),
            mandibular_vertex_indices=(
                self.mandibular_vertex_indices.copy()
            ),
            transform=Transform(
                transform.matrix.copy()
            ),
        )