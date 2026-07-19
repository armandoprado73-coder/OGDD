from pathlib import Path
from abc import ABC, abstractmethod

from ..geometry.mesh import Mesh


class MeshLoader(ABC):
    """Base class for all mesh loaders."""

    @abstractmethod
    def load(self, path: Path) -> Mesh:
        """Load a mesh from a file."""
        raise NotImplementedError
