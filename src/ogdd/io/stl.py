"""
OGDD STL Reader

Supports ASCII and Binary STL files.

Converts STL triangle data into
the OGDD Mesh representation.
"""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from ..mesh import Mesh


class STLReader:
    """
    Reader for STL geometry files.

    Supports:

    - ASCII STL
    - Binary STL
    """

    @staticmethod
    def read(
        filename: str | Path
    ) -> Mesh:
        """
        Read STL file and return OGDD Mesh.

        Automatically detects ASCII or Binary STL.
        """

        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(filename)

        if STLReader._is_binary(filename):
            return STLReader._read_binary(filename)

        return STLReader._read_ascii(filename)

    @staticmethod
    def _is_binary(filename: Path) -> bool:
        """
        Detect whether an STL file is binary.

        Binary STL format:

        - 80 bytes header
        - 4 bytes triangle count
        - 50 bytes per triangle
        """

        file_size = filename.stat().st_size

        if file_size < 84:
            return False

        with open(filename, "rb") as file:

            header = file.read(80)

            triangle_count_bytes = file.read(4)

        triangle_count = struct.unpack(
            "<I",
            triangle_count_bytes
        )[0]

        expected_size = (
            84
            + triangle_count * 50
        )

        if file_size == expected_size:
            return True

        try:

            header_text = header.decode(
                "ascii",
                errors="ignore"
            ).strip().lower()

            if header_text.startswith("solid"):
                return False

        except Exception:
            pass

        return True

    @staticmethod
    def _read_ascii(filename: Path) -> Mesh:
        """
        Read ASCII STL file.
        """

        vertices = []

        faces = []

        vertex_map = {}

        current_face = []

        with open(
            filename,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:

            for line in file:

                line = line.strip()

                if line.startswith("vertex"):

                    parts = line.split()

                    vertex = tuple(
                        float(x)
                        for x in parts[1:4]
                    )

                    if vertex not in vertex_map:

                        vertex_map[vertex] = len(
                            vertices
                        )

                        vertices.append(
                            vertex
                        )

                    current_face.append(
                        vertex_map[vertex]
                    )

                    if len(current_face) == 3:

                        faces.append(
                            current_face
                        )

                        current_face = []

        return Mesh(
            vertices=np.asarray(
                vertices,
                dtype=float
            ),
            faces=np.asarray(
                faces,
                dtype=np.int32
            )
        )

    @staticmethod
    def _read_binary(filename: Path) -> Mesh:
        """
        Read Binary STL file.
        """

        vertices = []

        faces = []

        vertex_map = {}

        with open(
            filename,
            "rb"
        ) as file:

            file.seek(80)

            triangle_count = struct.unpack(
                "<I",
                file.read(4)
            )[0]

            for _ in range(triangle_count):

                data = file.read(50)

                if len(data) != 50:
                    raise ValueError(
                        "Invalid Binary STL file."
                    )

                triangle = struct.unpack(
                    "<12fH",
                    data
                )

                triangle_vertices = [
                    tuple(triangle[3:6]),
                    tuple(triangle[6:9]),
                    tuple(triangle[9:12]),
                ]

                face = []

                for vertex in triangle_vertices:

                    if vertex not in vertex_map:

                        vertex_map[vertex] = len(
                            vertices
                        )

                        vertices.append(
                            vertex
                        )

                    face.append(
                        vertex_map[vertex]
                    )

                faces.append(face)

        return Mesh(
            vertices=np.asarray(
                vertices,
                dtype=float
            ),
            faces=np.asarray(
                faces,
                dtype=np.int32
            )
        )