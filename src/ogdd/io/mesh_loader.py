from pathlib import Path
from abc import ABC, abstractmethod

from ..geometry.mesh import Mesh


class MeshLoader(ABC):
    """
    Abstract base class for loading meshes.

    Every mesh loader in OGDD must implement this interface,
    regardless of the file format or data source.
    """

    @abstractmethod
    def load(self, path: Path) -> Mesh:
        """
        Load a mesh from the given path.

        Parameters
        ----------
        path : Path
            Path to the mesh file.

        Returns
        -------
        Mesh
            The loaded mesh.
        """
        raise NotImplementedError
