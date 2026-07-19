"""
OGDD STL Reader

Initial STL ASCII reader.

Converts STL triangle data into
the OGDD Mesh representation.
"""

from __future__ import annotations


from pathlib import Path

import numpy as np

from ..mesh import Mesh



class STLReader:
    """
    Reader for STL geometry files.

    Current support:

    - ASCII STL

    Future:

    - Binary STL
    """


    @staticmethod
    def read(
        filename: str | Path
    ) -> Mesh:
        """
        Read STL file and return OGDD Mesh.
        """

        filename = Path(filename)


        if not filename.exists():

            raise FileNotFoundError(
                filename
            )


        vertices = []

        faces = []

        vertex_map = {}


        with open(
            filename,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as file:


            for line in file:

                line = line.strip()


                if line.startswith(
                    "vertex"
                ):

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


                    current_index = vertex_map[
                        vertex
                    ]


                    if (
                        len(faces) == 0
                        or len(faces[-1]) == 3
                    ):

                        faces.append([])


                    faces[-1].append(
                        current_index
                    )


        faces = [
            face
            for face in faces
            if len(face) == 3
        ]


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
