"""
OGDD - Open Geometry for Digital Dentistry

Core package initialization.

This module defines the public interface of the OGDD SDK.
"""

from __future__ import annotations


# ---------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------

__title__ = "OGDD"
__description__ = (
    "Open Geometry for Digital Dentistry - "
    "Computational geometry tools for digital dentistry."
)

__version__ = "0.1.0-alpha"

__author__ = "Armando Prado"

__license__ = "MIT"


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
#
# Future public classes will be imported here.
#
# Example:
#
# from .mesh import Mesh
#
# The goal is to keep the external API simple:
#
#     import ogdd
#     mesh = ogdd.Mesh()
#
# ---------------------------------------------------------------------


__all__ = [
    "__title__",
    "__version__",
    "__author__",
]
