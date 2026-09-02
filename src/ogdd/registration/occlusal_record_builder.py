"""
OGDD - Occlusal Record Builder

Builds a rigid occlusal record from a combined mesh
using one maxillary and one mandibular seed vertex.
"""

from __future__ import annotations

import numpy as np

from ogdd.mesh import Mesh
from ogdd.registration.occlusal_record import (
    OcclusalRecord,
)


def _validated_seed_vertex(
    value,
    name: str,
    vertex_count: int,
) -> int:
    """
    Validate one operator-selected seed vertex.
    """

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise TypeError(
            f"{name} seed vertex must be an integer."
        )

    value = int(value)

    if (
        value < 0
        or value >= vertex_count
    ):
        raise ValueError(
            f"{name} seed vertex is outside the mesh."
        )

    return value


class OcclusalRecordBuilder:
    """
    Build maxillary and mandibular registration regions
    from a single combined mesh.

    The operator supplies one seed vertex on each arch.
    Every vertex topologically connected to that seed
    becomes part of its registration region.
    """

    @staticmethod
    def from_seed_vertices(
        mesh: Mesh,
        maxillary_seed_vertex: int,
        mandibular_seed_vertex: int,
    ) -> OcclusalRecord:
        """
        Build an occlusal record from two seed vertices.
        """

        if not isinstance(
            mesh,
            Mesh,
        ):
            raise TypeError(
                "Combined record must be an OGDD Mesh."
            )

        if mesh.vertex_count == 0:
            raise ValueError(
                "Combined record mesh cannot be empty."
            )

        if mesh.face_count == 0:
            raise ValueError(
                "Combined record mesh must contain faces."
            )

        if (
            np.any(mesh.faces < 0)
            or np.any(
                mesh.faces
                >= mesh.vertex_count
            )
        ):
            raise ValueError(
                "Mesh faces reference vertices "
                "outside the mesh."
            )

        maxillary_seed_vertex = (
            _validated_seed_vertex(
                value=maxillary_seed_vertex,
                name="Maxillary",
                vertex_count=mesh.vertex_count,
            )
        )

        mandibular_seed_vertex = (
            _validated_seed_vertex(
                value=mandibular_seed_vertex,
                name="Mandibular",
                vertex_count=mesh.vertex_count,
            )
        )

        component_labels = (
            OcclusalRecordBuilder
            ._connected_component_labels(
                mesh
            )
        )

        maxillary_component = (
            component_labels[
                maxillary_seed_vertex
            ]
        )

        mandibular_component = (
            component_labels[
                mandibular_seed_vertex
            ]
        )

        if (
            maxillary_component
            == mandibular_component
        ):
            raise ValueError(
                "Maxillary and mandibular seeds "
                "must belong to different connected "
                "components."
            )

        maxillary_indices = np.flatnonzero(
            component_labels
            == maxillary_component
        )

        mandibular_indices = np.flatnonzero(
            component_labels
            == mandibular_component
        )

        return OcclusalRecord(
            mesh=mesh,
            maxillary_vertex_indices=(
                maxillary_indices
            ),
            mandibular_vertex_indices=(
                mandibular_indices
            ),
        )

    @staticmethod
    def _connected_component_labels(
        mesh: Mesh,
    ) -> np.ndarray:
        """
        Return one connected-component label per vertex.

        A union-find structure joins the three vertices
        of every triangular face. Geometrically close
        vertices are never joined unless a face connects
        them.
        """

        parent = np.arange(
            mesh.vertex_count,
            dtype=np.int64,
        )

        rank = np.zeros(
            mesh.vertex_count,
            dtype=np.uint8,
        )

        def find(
            vertex: int,
        ) -> int:
            """
            Find a component root with path compression.
            """

            while parent[vertex] != vertex:
                parent[vertex] = parent[
                    parent[vertex]
                ]

                vertex = int(
                    parent[vertex]
                )

            return vertex

        def union(
            first: int,
            second: int,
        ) -> None:
            """
            Join two vertex components by rank.
            """

            first_root = find(
                first
            )

            second_root = find(
                second
            )

            if first_root == second_root:
                return

            if (
                rank[first_root]
                < rank[second_root]
            ):
                parent[first_root] = (
                    second_root
                )

            elif (
                rank[first_root]
                > rank[second_root]
            ):
                parent[second_root] = (
                    first_root
                )

            else:
                parent[second_root] = (
                    first_root
                )

                rank[first_root] += 1

        for face in mesh.faces:
            first = int(
                face[0]
            )

            second = int(
                face[1]
            )

            third = int(
                face[2]
            )

            union(
                first,
                second,
            )

            union(
                first,
                third,
            )

        labels = np.empty(
            mesh.vertex_count,
            dtype=np.int64,
        )

        for vertex in range(
            mesh.vertex_count
        ):
            labels[vertex] = find(
                vertex
            )

        return labels