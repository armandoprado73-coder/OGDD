"""
OGDD Occlusal Plane Estimator v0.2 Tests

Tests automatic alignment integration.
"""

import numpy as np

from ogdd.algorithms.occlusal import (
    OcclusalPlaneEstimator
)

from ogdd.geometry.transform import (
    Transform
)



def create_surface():
    """
    Creates a simple dental-like surface.
    """

    return np.array(
        [
            [0,0,10],
            [10,0,10],
            [0,10,10],
            [10,10,10],
            [5,5,10],

            [2,2,2],
            [8,8,3],
            [4,7,1],
        ],
        dtype=float
    )



def test_occlusal_estimation_basic():

    points = create_surface()


    estimator = OcclusalPlaneEstimator(
        top_percentage=40
    )


    plane = estimator.compute(
        points
    )


    assert np.isclose(
        np.linalg.norm(
            plane.normal
        ),
        1.0
    )



def test_occlusal_estimation_after_translation():

    points = create_surface()


    translation = Transform.translation(
        [100,50,-30]
    )


    moved_points = translation.apply(
        points
    )


    estimator = OcclusalPlaneEstimator(
        top_percentage=40
    )


    plane = estimator.compute(
        moved_points
    )


    assert np.isclose(
        np.linalg.norm(
            plane.normal
        ),
        1.0
    )



def test_occlusal_estimation_rotated_model():

    points = create_surface()


    rotation = np.array(
        [
            [0,-1,0,0],
            [1,0,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ],
        dtype=float
    )


    transform = Transform(
        rotation
    )


    rotated_points = transform.apply(
        points
    )


    estimator = OcclusalPlaneEstimator(
        top_percentage=40
    )


    plane = estimator.compute(
        rotated_points
    )


    assert np.isclose(
        np.linalg.norm(
            plane.normal
        ),
        1.0
    )
