"""
OGDD Core Results

Standard result container for
all analysis algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any



@dataclass
class AnalysisResult:
    """
    Generic OGDD analysis output.
    """

    value: Any


    algorithm: str


    version: str = "0.1"


    confidence: Any = None


    metadata: dict = field(
        default_factory=dict
    )



    def add_metadata(
        self,
        key: str,
        value: Any
    ):
        """
        Add information to result.
        """

        self.metadata[key] = value
