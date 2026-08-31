import math

import numpy as np
import pytest

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.combined_movement import (
    MandibularCombinedPosition,
)
from ogdd.articulator.lateral_excursion import (
    LateralSide,
)
from ogdd.articulator.occlusal_closure import (
    MandibularOcclusalPosition,
    OcclusalClosure,
)
from ogdd.mesh import Mesh


@pytest.fixture
def mobile_hinge_axis() -> HingeAxis:
    return HingeAxis(
        left_condyle=Landmark(
            name="LEFT_CONDYLE",
            point=np.array([-55.0, 5.0, -5.0]),
            reference_used="mobile condylar center",
        ),
        right_condyle=Landmark(
            name="RIGHT_CONDYLE",
            point=np.array([55.0, 5.0, -5.0]),
            reference_used="mobile condylar center",
        ),
    )


@pytest.fixture
def combined_position(
    mobile_hinge_axis: HingeAxis,
) -> MandibularCombinedPosition:
    dental_midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 105.0, -5.0]),
        reference_used="mandibular dental midline",
    )

    balkwill = BalkwillTriangle(
        left_posterior=Landmark(
            name="LEFT_SECOND_MOLAR",
            point=np.array([-50.0, 25.0, -5.0]),
            reference_used="distobuccal cusp",
        ),
        right_posterior=Landmark(
            name="RIGHT_SECOND_MOLAR",
            point=np.array([50.0, 25.0, -5.0]),
            reference_used="distobuccal cusp",
        ),
        dental_midline=dental_midline,
    )

    bonwill = BonwillTriangle(
        left_condyle=mobile_hinge_axis.left_condyle,
        right_condyle=mobile_hinge_axis.right_condyle,
        dental_midline=dental_midline,
    )

    mesh = Mesh(
        vertices=np.array([
            [-50.0, 25.0, -5.0],
            [50.0, 25.0, -5.0],
            [0.0, 105.0, -5.0],
            [0.0, 65.0, -25.0],
        ]),
        faces=np.array([
            [0, 1, 2],
            [0, 2, 3],
        ]),
        normals=np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]),
        attributes={
            "test_value": np.array([
                1.0,
                2.0,
                3.0,
                4.0,
            ]),
        },
        metadata={
            "source": "combined synthetic position",
        },
    )

    return MandibularCombinedPosition(
        mesh=mesh,
        balkwill=balkwill,
        bonwill=bonwill,
        hinge_axis=mobile_hinge_axis,
        opening_angle_degrees=0.0,
        lateral_angle_degrees=3.0,
        protrusion_distance_mm=5.0,
        working_side=LateralSide.RIGHT,
    )


@pytest.fixture
def closure() -> OcclusalClosure:
    return OcclusalClosure()


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        points[:, None, :]
        - points[None, :, :],
        axis=2,
    )


def test_zero_returns_original_geometry(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=0.0,
    )

    assert np.allclose(
        result.mesh.vertices,
        combined_position.mesh.vertices,
    )


def test_positive_adjustment_opens_mandible(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=0.1,
    )

    assert (
        result.balkwill.dental_midline.point[2]
        < combined_position.balkwill.dental_midline.point[2]
    )


def test_negative_adjustment_closes_mandible(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-0.1,
    )

    assert (
        result.balkwill.dental_midline.point[2]
        > combined_position.balkwill.dental_midline.point[2]
    )


def test_tenth_degree_steps_produce_distinct_positions(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    closed_01 = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-0.1,
    )
    closed_02 = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-0.2,
    )

    assert not np.allclose(
        closed_01.mesh.vertices,
        closed_02.mesh.vertices,
    )


@pytest.mark.parametrize(
    "adjustment_angle_degrees",
    [
        -2.0,
        -0.1,
        0.0,
        0.1,
        2.0,
    ],
)
def test_condyles_remain_fixed(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
    adjustment_angle_degrees: float,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=(
            adjustment_angle_degrees
        ),
    )

    assert np.allclose(
        result.bonwill.left_condyle.point,
        combined_position.bonwill.left_condyle.point,
    )
    assert np.allclose(
        result.bonwill.right_condyle.point,
        combined_position.bonwill.right_condyle.point,
    )


def test_mobile_hinge_axis_is_preserved(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert result.hinge_axis.length == pytest.approx(
        combined_position.hinge_axis.length
    )
    assert np.allclose(
        result.hinge_axis.midpoint,
        combined_position.hinge_axis.midpoint,
    )


def test_adjustment_is_rigid(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert np.allclose(
        pairwise_distances(result.mesh.vertices),
        pairwise_distances(
            combined_position.mesh.vertices
        ),
    )


def test_balkwill_and_bonwill_dimensions_are_preserved(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert result.balkwill.intermolar_width == (
        pytest.approx(
            combined_position.balkwill.intermolar_width
        )
    )
    assert result.bonwill.condylar_width == (
        pytest.approx(
            combined_position.bonwill.condylar_width
        )
    )


def test_all_structures_adjust_together(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert np.allclose(
        result.mesh.vertices[2],
        result.balkwill.dental_midline.point,
    )
    assert np.allclose(
        result.balkwill.dental_midline.point,
        result.bonwill.dental_midline.point,
    )


def test_mesh_data_and_normal_lengths_are_preserved(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert np.array_equal(
        result.mesh.faces,
        combined_position.mesh.faces,
    )
    assert np.array_equal(
        result.mesh.attributes["test_value"],
        combined_position.mesh.attributes["test_value"],
    )
    assert result.mesh.metadata == (
        combined_position.mesh.metadata
    )
    assert np.allclose(
        np.linalg.norm(result.mesh.normals, axis=1),
        np.linalg.norm(
            combined_position.mesh.normals,
            axis=1,
        ),
    )


def test_position_records_operator_adjustment(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert isinstance(
        result,
        MandibularOcclusalPosition,
    )
    assert result.base_opening_angle_degrees == (
        pytest.approx(0.0)
    )
    assert result.adjustment_angle_degrees == (
        pytest.approx(-1.3)
    )
    assert result.total_opening_angle_degrees == (
        pytest.approx(-1.3)
    )


def test_position_preserves_combined_components(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert result.lateral_angle_degrees == (
        pytest.approx(3.0)
    )
    assert result.protrusion_distance_mm == (
        pytest.approx(5.0)
    )
    assert result.working_side is LateralSide.RIGHT


def test_base_opening_and_adjustment_are_added(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    opened_base = MandibularCombinedPosition(
        mesh=combined_position.mesh,
        balkwill=combined_position.balkwill,
        bonwill=combined_position.bonwill,
        hinge_axis=combined_position.hinge_axis,
        opening_angle_degrees=5.0,
        lateral_angle_degrees=(
            combined_position.lateral_angle_degrees
        ),
        protrusion_distance_mm=(
            combined_position.protrusion_distance_mm
        ),
        working_side=combined_position.working_side,
    )

    result = closure.position_at(
        position=opened_base,
        adjustment_angle_degrees=-1.3,
    )

    assert result.total_opening_angle_degrees == (
        pytest.approx(3.7)
    )


def test_positions_do_not_accumulate_adjustment(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-0.1,
    )

    result = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    direct = closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert np.allclose(
        result.mesh.vertices,
        direct.mesh.vertices,
    )


def test_adjustment_does_not_modify_combined_position(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
) -> None:
    original_vertices = (
        combined_position.mesh.vertices.copy()
    )

    closure.position_at(
        position=combined_position,
        adjustment_angle_degrees=-1.3,
    )

    assert np.allclose(
        combined_position.mesh.vertices,
        original_vertices,
    )


@pytest.mark.parametrize(
    "adjustment_angle_degrees",
    [
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_rejects_nonfinite_adjustment(
    closure: OcclusalClosure,
    combined_position: MandibularCombinedPosition,
    adjustment_angle_degrees: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        closure.position_at(
            position=combined_position,
            adjustment_angle_degrees=(
                adjustment_angle_degrees
            ),
        )
