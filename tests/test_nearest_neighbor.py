"""
Tests for spatial nearest-neighbor search.
"""

import numpy as np
import pytest

from ogdd.registration.nearest_neighbor import (
    NearestNeighborSearch,
)


def target_points() -> np.ndarray:
    """
    Build a small fixed spatial reference.
    """

    return np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 5.0],
        ]
    )


def source_points() -> np.ndarray:
    """
    Build points close to known targets.
    """

    return np.array(
        [
            [0.1, 0.0, 0.0],
            [4.8, 0.2, 0.0],
            [0.0, 4.7, 0.4],
        ]
    )


def test_query_finds_expected_target_indices():
    search = NearestNeighborSearch(
        target_points()
    )

    result = search.query(
        source_points()
    )

    np.testing.assert_array_equal(
        result.target_indices,
        np.array(
            [0, 1, 2]
        ),
    )

    np.testing.assert_array_equal(
        result.source_indices,
        np.array(
            [0, 1, 2]
        ),
    )


def test_query_returns_corresponding_target_points():
    targets = target_points()

    search = NearestNeighborSearch(
        targets
    )

    result = search.query(
        source_points()
    )

    np.testing.assert_allclose(
        result.corresponding_target_points,
        targets[
            [0, 1, 2]
        ],
    )


def test_query_reports_correct_distances():
    search = NearestNeighborSearch(
        target_points()
    )

    result = search.query(
        source_points()
    )

    expected = np.array(
        [
            0.1,
            np.sqrt(0.08),
            0.5,
        ]
    )

    np.testing.assert_allclose(
        result.distances,
        expected,
    )


def test_result_reports_match_count_and_fraction():
    search = NearestNeighborSearch(
        target_points()
    )

    result = search.query(
        source_points()
    )

    assert result.match_count == 3
    assert result.match_fraction == pytest.approx(
        1.0
    )


def test_maximum_distance_filters_far_matches():
    search = NearestNeighborSearch(
        target_points()
    )

    result = search.query(
        source_points(),
        maximum_distance=0.15,
    )

    np.testing.assert_array_equal(
        result.source_indices,
        np.array(
            [0]
        ),
    )

    np.testing.assert_array_equal(
        result.target_indices,
        np.array(
            [0]
        ),
    )

    assert result.match_count == 1

    assert result.match_fraction == pytest.approx(
        1.0 / 3.0
    )


def test_query_can_return_no_matches():
    search = NearestNeighborSearch(
        target_points()
    )

    result = search.query(
        np.array(
            [
                [100.0, 100.0, 100.0],
                [200.0, 200.0, 200.0],
            ]
        ),
        maximum_distance=1.0,
    )

    assert result.match_count == 0
    assert result.match_fraction == 0.0

    assert result.source_indices.shape == (0,)
    assert result.target_indices.shape == (0,)
    assert result.distances.shape == (0,)

    assert (
        result.corresponding_target_points.shape
        == (0, 3)
    )


def test_search_copies_fixed_target_points():
    targets = target_points()

    expected = targets.copy()

    search = NearestNeighborSearch(
        targets
    )

    targets[:] = 1000.0

    np.testing.assert_allclose(
        search.target_points,
        expected,
    )


def test_target_points_property_returns_copy():
    search = NearestNeighborSearch(
        target_points()
    )

    returned = search.target_points

    returned[:] = -500.0

    np.testing.assert_allclose(
        search.target_points,
        target_points(),
    )


def test_query_does_not_change_source_points():
    source = source_points()

    original = source.copy()

    search = NearestNeighborSearch(
        target_points()
    )

    search.query(
        source
    )

    np.testing.assert_array_equal(
        source,
        original,
    )


def test_result_arrays_are_read_only():
    search = NearestNeighborSearch(
        target_points()
    )

    result = search.query(
        source_points()
    )

    with pytest.raises(
        ValueError,
    ):
        result.distances[0] = 100.0

    with pytest.raises(
        ValueError,
    ):
        result.target_indices[0] = 4


@pytest.mark.parametrize(
    "invalid_points",
    [
        np.array(
            [0.0, 1.0, 2.0]
        ),
        np.zeros(
            (4, 2)
        ),
    ],
)
def test_invalid_target_shape_is_rejected(
    invalid_points,
):
    with pytest.raises(
        ValueError,
        match=r"shape \(N,3\)",
    ):
        NearestNeighborSearch(
            invalid_points
        )


def test_empty_target_set_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        NearestNeighborSearch(
            np.empty(
                (0, 3)
            )
        )


def test_nonfinite_target_points_are_rejected():
    targets = target_points()

    targets[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        NearestNeighborSearch(
            targets
        )


def test_invalid_source_shape_is_rejected():
    search = NearestNeighborSearch(
        target_points()
    )

    with pytest.raises(
        ValueError,
        match=r"shape \(N,3\)",
    ):
        search.query(
            np.zeros(
                (3, 2)
            )
        )


def test_empty_source_set_is_rejected():
    search = NearestNeighborSearch(
        target_points()
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        search.query(
            np.empty(
                (0, 3)
            )
        )


def test_nonfinite_source_points_are_rejected():
    search = NearestNeighborSearch(
        target_points()
    )

    source = source_points()
    source[0, 1] = np.inf

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        search.query(
            source
        )


@pytest.mark.parametrize(
    "maximum_distance",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_numeric_maximum_distance_is_rejected(
    maximum_distance,
):
    search = NearestNeighborSearch(
        target_points()
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        search.query(
            source_points(),
            maximum_distance=maximum_distance,
        )


@pytest.mark.parametrize(
    "maximum_distance",
    [
        "near",
        True,
    ],
)
def test_nonnumeric_maximum_distance_is_rejected(
    maximum_distance,
):
    search = NearestNeighborSearch(
        target_points()
    )

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        search.query(
            source_points(),
            maximum_distance=maximum_distance,
        )


def test_large_exact_point_set_is_queried():
    generator = np.random.default_rng(
        42
    )

    targets = generator.normal(
        size=(10000, 3)
    )

    search = NearestNeighborSearch(
        targets
    )

    result = search.query(
        targets
    )

    assert result.match_count == 10000

    np.testing.assert_array_equal(
        result.target_indices,
        np.arange(
            10000
        ),
    )

    assert np.max(
        result.distances
    ) == pytest.approx(
        0.0,
        abs=1e-12,
    )