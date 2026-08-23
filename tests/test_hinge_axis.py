import numpy as np
import pytest

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.geometry.triangle import Triangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.mesh import Mesh

@pytest.fixture
def left_condyle() -> Landmark:
    return Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-55.0, 0.0, 0.0]),
        reference_used="condylar center",
    )


@pytest.fixture
def right_condyle() -> Landmark:
    return Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([55.0, 0.0, 0.0]),
        reference_used="condylar center",
    )


@pytest.fixture
def hinge_axis(
    left_condyle: Landmark,
    right_condyle: Landmark,
) -> HingeAxis:
    return HingeAxis(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
    )

@pytest.fixture
def balkwill_triangle() -> BalkwillTriangle:
    left_posterior = Landmark(
        name="LEFT_SECOND_MOLAR",
        point=np.array([-50.0, 20.0, 0.0]),
        reference_used="distobuccal cusp",
    )

    right_posterior = Landmark(
        name="RIGHT_SECOND_MOLAR",
        point=np.array([50.0, 20.0, 0.0]),
        reference_used="distobuccal cusp",
    )

    dental_midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
    )

    return BalkwillTriangle(
        left_posterior=left_posterior,
        right_posterior=right_posterior,
        dental_midline=dental_midline,
    )

@pytest.fixture
def bonwill_triangle(
    left_condyle: Landmark,
    right_condyle: Landmark,
) -> BonwillTriangle:
    dental_midline = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
    )

    return BonwillTriangle(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
        dental_midline=dental_midline,
    )

def test_hinge_axis_creation(
    hinge_axis: HingeAxis,
    left_condyle: Landmark,
    right_condyle: Landmark,
) -> None:
    assert hinge_axis.left_condyle is left_condyle
    assert hinge_axis.right_condyle is right_condyle


def test_hinge_axis_vector(
    hinge_axis: HingeAxis,
) -> None:
    assert np.allclose(
        hinge_axis.vector,
        np.array([110.0, 0.0, 0.0]),
    )


def test_hinge_axis_length(
    hinge_axis: HingeAxis,
) -> None:
    assert hinge_axis.length == pytest.approx(110.0)


def test_hinge_axis_direction(
    hinge_axis: HingeAxis,
) -> None:
    assert np.allclose(
        hinge_axis.direction,
        np.array([1.0, 0.0, 0.0]),
    )


def test_hinge_axis_midpoint(
    hinge_axis: HingeAxis,
) -> None:
    assert np.allclose(
        hinge_axis.midpoint,
        np.array([0.0, 0.0, 0.0]),
    )


def test_hinge_axis_rejects_coincident_condyles() -> None:
    point = np.array([10.0, 20.0, 30.0])

    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=point,
        reference_used="condylar center",
    )

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=point.copy(),
        reference_used="condylar center",
    )

    with pytest.raises(
        ValueError,
        match="two different points",
    ):
        HingeAxis(
            left_condyle=left_condyle,
            right_condyle=right_condyle,
        )

def test_rotate_point_zero_degrees(
    hinge_axis: HingeAxis,
) -> None:
    point = np.array([0.0, 100.0, 0.0])

    rotated = hinge_axis.rotate_point(
        point=point,
        angle_degrees=0.0,
    )

    assert np.allclose(rotated, point)


def test_rotate_point_during_opening(
    hinge_axis: HingeAxis,
) -> None:
    point = np.array([0.0, 100.0, 0.0])

    rotated = hinge_axis.rotate_point(
        point=point,
        angle_degrees=30.0,
    )

    expected = np.array([
        0.0,
        86.60254038,
        -50.0,
    ])

    assert np.allclose(rotated, expected)


def test_right_condyle_remains_fixed(
    hinge_axis: HingeAxis,
    right_condyle: Landmark,
) -> None:
    rotated = hinge_axis.rotate_point(
        point=right_condyle.point,
        angle_degrees=30.0,
    )

    assert np.allclose(
        rotated,
        right_condyle.point,
    )


def test_left_condyle_remains_fixed(
    hinge_axis: HingeAxis,
    left_condyle: Landmark,
) -> None:
    rotated = hinge_axis.rotate_point(
        point=left_condyle.point,
        angle_degrees=30.0,
    )

    assert np.allclose(
        rotated,
        left_condyle.point,
    )


def test_rotation_preserves_distance_to_midpoint(
    hinge_axis: HingeAxis,
) -> None:
    point = np.array([20.0, 80.0, -10.0])

    rotated = hinge_axis.rotate_point(
        point=point,
        angle_degrees=25.0,
    )

    original_distance = np.linalg.norm(
        point - hinge_axis.midpoint
    )

    rotated_distance = np.linalg.norm(
        rotated - hinge_axis.midpoint
    )

    assert rotated_distance == pytest.approx(
        original_distance
    )


def test_rotate_point_rejects_invalid_shape(
    hinge_axis: HingeAxis,
) -> None:
    point = np.array([10.0, 20.0])

    with pytest.raises(
        ValueError,
        match="3D vector",
    ):
        hinge_axis.rotate_point(
            point=point,
            angle_degrees=30.0,
        )

def test_rotate_multiple_points(
    hinge_axis: HingeAxis,
) -> None:
    points = np.array([
        [0.0, 100.0, 0.0],
        [20.0, 100.0, 0.0],
        [-20.0, 100.0, 0.0],
    ])

    rotated = hinge_axis.rotate_points(
        points=points,
        angle_degrees=30.0,
    )

    expected = np.array([
        [0.0, 86.60254038, -50.0],
        [20.0, 86.60254038, -50.0],
        [-20.0, 86.60254038, -50.0],
    ])

    assert np.allclose(rotated, expected)


def test_rotate_points_zero_degrees(
    hinge_axis: HingeAxis,
) -> None:
    points = np.array([
        [10.0, 20.0, 30.0],
        [-15.0, 40.0, -5.0],
        [25.0, -10.0, 8.0],
    ])

    rotated = hinge_axis.rotate_points(
        points=points,
        angle_degrees=0.0,
    )

    assert np.allclose(rotated, points)


def test_rotate_points_preserves_balkwill_triangle(
    hinge_axis: HingeAxis,
) -> None:
    points = np.array([
        [50.0, 20.0, 0.0],
        [-50.0, 20.0, 0.0],
        [0.0, 100.0, 0.0],
    ])

    original_triangle = Triangle(
        a=points[0],
        b=points[1],
        c=points[2],
    )

    rotated_points = hinge_axis.rotate_points(
        points=points,
        angle_degrees=30.0,
    )

    rotated_triangle = Triangle(
        a=rotated_points[0],
        b=rotated_points[1],
        c=rotated_points[2],
    )

    assert rotated_triangle.side_ab == pytest.approx(
        original_triangle.side_ab
    )

    assert rotated_triangle.side_bc == pytest.approx(
        original_triangle.side_bc
    )

    assert rotated_triangle.side_ca == pytest.approx(
        original_triangle.side_ca
    )


def test_rotate_points_keeps_condyles_fixed(
    hinge_axis: HingeAxis,
    left_condyle: Landmark,
    right_condyle: Landmark,
) -> None:
    condyles = np.array([
        left_condyle.point,
        right_condyle.point,
    ])

    rotated = hinge_axis.rotate_points(
        points=condyles,
        angle_degrees=30.0,
    )

    assert np.allclose(rotated, condyles)


def test_rotate_points_rejects_single_point(
    hinge_axis: HingeAxis,
) -> None:
    point = np.array([10.0, 20.0, 30.0])

    with pytest.raises(
        ValueError,
        match=r"shape \(n, 3\)",
    ):
        hinge_axis.rotate_points(
            points=point,
            angle_degrees=30.0,
        )

def test_rotate_landmark(
    hinge_axis: HingeAxis,
) -> None:
    landmark = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
    )

    rotated = hinge_axis.rotate_landmark(
        landmark=landmark,
        angle_degrees=30.0,
    )

    expected_point = np.array([
        0.0,
        86.60254038,
        -50.0,
    ])

    assert np.allclose(
        rotated.point,
        expected_point,
    )


def test_rotate_landmark_preserves_metadata(
    hinge_axis: HingeAxis,
) -> None:
    landmark = Landmark(
        name="DENTAL_MIDLINE",
        point=np.array([0.0, 100.0, 0.0]),
        reference_used="mandibular dental midline",
        confidence=0.95,
        created_by="operator",
    )

    rotated = hinge_axis.rotate_landmark(
        landmark=landmark,
        angle_degrees=30.0,
    )

    assert rotated.name == landmark.name

    assert (
        rotated.reference_used
        == landmark.reference_used
    )

    assert rotated.confidence == landmark.confidence

    assert rotated.created_by == landmark.created_by


def test_rotate_landmark_does_not_modify_original(
    hinge_axis: HingeAxis,
) -> None:
    original_point = np.array([
        0.0,
        100.0,
        0.0,
    ])

    landmark = Landmark(
        name="DENTAL_MIDLINE",
        point=original_point.copy(),
        reference_used="mandibular dental midline",
    )

    rotated = hinge_axis.rotate_landmark(
        landmark=landmark,
        angle_degrees=30.0,
    )

    assert np.allclose(
        landmark.point,
        original_point,
    )

    assert rotated is not landmark

    assert not np.allclose(
        rotated.point,
        landmark.point,
    )

def test_rotate_balkwill_landmarks(
    hinge_axis: HingeAxis,
    balkwill_triangle: BalkwillTriangle,
) -> None:
    rotated = hinge_axis.rotate_balkwill(
        balkwill=balkwill_triangle,
        angle_degrees=30.0,
    )

    expected_left = np.array([
        -50.0,
        17.32050808,
        -10.0,
    ])

    expected_right = np.array([
        50.0,
        17.32050808,
        -10.0,
    ])

    expected_midline = np.array([
        0.0,
        86.60254038,
        -50.0,
    ])

    assert np.allclose(
        rotated.left_posterior.point,
        expected_left,
    )

    assert np.allclose(
        rotated.right_posterior.point,
        expected_right,
    )

    assert np.allclose(
        rotated.dental_midline.point,
        expected_midline,
    )


def test_rotate_balkwill_preserves_measurements(
    hinge_axis: HingeAxis,
    balkwill_triangle: BalkwillTriangle,
) -> None:
    rotated = hinge_axis.rotate_balkwill(
        balkwill=balkwill_triangle,
        angle_degrees=30.0,
    )

    assert rotated.right_side == pytest.approx(
        balkwill_triangle.right_side
    )

    assert rotated.left_side == pytest.approx(
        balkwill_triangle.left_side
    )

    assert rotated.intermolar_width == pytest.approx(
        balkwill_triangle.intermolar_width
    )

    assert (
        rotated.symmetry_difference
        == pytest.approx(
            balkwill_triangle.symmetry_difference
        )
    )


def test_rotate_balkwill_does_not_modify_original(
    hinge_axis: HingeAxis,
    balkwill_triangle: BalkwillTriangle,
) -> None:
    original_left = (
        balkwill_triangle.left_posterior.point.copy()
    )

    original_right = (
        balkwill_triangle.right_posterior.point.copy()
    )

    original_midline = (
        balkwill_triangle.dental_midline.point.copy()
    )

    rotated = hinge_axis.rotate_balkwill(
        balkwill=balkwill_triangle,
        angle_degrees=30.0,
    )

    assert np.allclose(
        balkwill_triangle.left_posterior.point,
        original_left,
    )

    assert np.allclose(
        balkwill_triangle.right_posterior.point,
        original_right,
    )

    assert np.allclose(
        balkwill_triangle.dental_midline.point,
        original_midline,
    )

    assert rotated is not balkwill_triangle
def test_rotate_bonwill_landmarks(
    hinge_axis: HingeAxis,
    bonwill_triangle: BonwillTriangle,
) -> None:
    rotated = hinge_axis.rotate_bonwill(
        bonwill=bonwill_triangle,
        angle_degrees=30.0,
    )

    expected_midline = np.array([
        0.0,
        86.60254038,
        -50.0,
    ])

    assert np.allclose(
        rotated.left_condyle.point,
        bonwill_triangle.left_condyle.point,
    )

    assert np.allclose(
        rotated.right_condyle.point,
        bonwill_triangle.right_condyle.point,
    )

    assert np.allclose(
        rotated.dental_midline.point,
        expected_midline,
    )


def test_rotate_bonwill_preserves_measurements(
    hinge_axis: HingeAxis,
    bonwill_triangle: BonwillTriangle,
) -> None:
    rotated = hinge_axis.rotate_bonwill(
        bonwill=bonwill_triangle,
        angle_degrees=30.0,
    )

    assert rotated.right_side == pytest.approx(
        bonwill_triangle.right_side
    )

    assert rotated.left_side == pytest.approx(
        bonwill_triangle.left_side
    )

    assert rotated.condylar_width == pytest.approx(
        bonwill_triangle.condylar_width
    )

    assert (
        rotated.symmetry_difference
        == pytest.approx(
            bonwill_triangle.symmetry_difference
        )
    )


def test_rotate_bonwill_does_not_modify_original(
    hinge_axis: HingeAxis,
    bonwill_triangle: BonwillTriangle,
) -> None:
    original_left_condyle = (
        bonwill_triangle.left_condyle.point.copy()
    )

    original_right_condyle = (
        bonwill_triangle.right_condyle.point.copy()
    )

    original_midline = (
        bonwill_triangle.dental_midline.point.copy()
    )

    rotated = hinge_axis.rotate_bonwill(
        bonwill=bonwill_triangle,
        angle_degrees=30.0,
    )

    assert np.allclose(
        bonwill_triangle.left_condyle.point,
        original_left_condyle,
    )

    assert np.allclose(
        bonwill_triangle.right_condyle.point,
        original_right_condyle,
    )

    assert np.allclose(
        bonwill_triangle.dental_midline.point,
        original_midline,
    )

    assert rotated is not bonwill_triangle

def test_balkwill_and_bonwill_share_rotated_midline(
    hinge_axis: HingeAxis,
    balkwill_triangle: BalkwillTriangle,
    bonwill_triangle: BonwillTriangle,
) -> None:
    rotated_balkwill = hinge_axis.rotate_balkwill(
        balkwill=balkwill_triangle,
        angle_degrees=30.0,
    )

    rotated_bonwill = hinge_axis.rotate_bonwill(
        bonwill=bonwill_triangle,
        angle_degrees=30.0,
    )

    assert np.allclose(
        rotated_balkwill.dental_midline.point,
        rotated_bonwill.dental_midline.point,
    )
@pytest.fixture
def mandibular_mesh() -> Mesh:
    return Mesh(
        vertices=np.array([
            [-50.0, 20.0, 0.0],
            [50.0, 20.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 60.0, -20.0],
        ]),
        faces=np.array([
            [0, 1, 2],
            [0, 2, 3],
            [1, 3, 2],
            [0, 3, 1],
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
            "model": "mandibular",
        },
    )
def test_rotate_mesh_vertices(
    hinge_axis: HingeAxis,
    mandibular_mesh: Mesh,
) -> None:
    rotated = hinge_axis.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=30.0,
    )

    expected_midline = np.array([
        0.0,
        86.60254038,
        -50.0,
    ])

    assert np.allclose(
        rotated.vertices[2],
        expected_midline,
    )


def test_rotate_mesh_preserves_structure(
    hinge_axis: HingeAxis,
    mandibular_mesh: Mesh,
) -> None:
    rotated = hinge_axis.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=30.0,
    )

    assert (
        rotated.vertex_count
        == mandibular_mesh.vertex_count
    )

    assert (
        rotated.face_count
        == mandibular_mesh.face_count
    )

    assert np.array_equal(
        rotated.faces,
        mandibular_mesh.faces,
    )


def test_rotate_mesh_preserves_distances(
    hinge_axis: HingeAxis,
    mandibular_mesh: Mesh,
) -> None:
    rotated = hinge_axis.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=30.0,
    )

    original_distance = np.linalg.norm(
        mandibular_mesh.vertices[2]
        - mandibular_mesh.vertices[0]
    )

    rotated_distance = np.linalg.norm(
        rotated.vertices[2]
        - rotated.vertices[0]
    )

    assert rotated_distance == pytest.approx(
        original_distance
    )


def test_rotate_mesh_rotates_normals(
    hinge_axis: HingeAxis,
    mandibular_mesh: Mesh,
) -> None:
    rotated = hinge_axis.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=30.0,
    )

    expected_normal = np.array([
        0.0,
        0.5,
        0.86602540,
    ])

    assert rotated.normals is not None

    assert np.allclose(
        rotated.normals[0],
        expected_normal,
    )


def test_rotate_mesh_preserves_data(
    hinge_axis: HingeAxis,
    mandibular_mesh: Mesh,
) -> None:
    rotated = hinge_axis.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=30.0,
    )

    assert np.array_equal(
        rotated.attributes["test_value"],
        mandibular_mesh.attributes["test_value"],
    )

    assert (
        rotated.metadata
        == mandibular_mesh.metadata
    )


def test_rotate_mesh_does_not_modify_original(
    hinge_axis: HingeAxis,
    mandibular_mesh: Mesh,
) -> None:
    original_vertices = (
        mandibular_mesh.vertices.copy()
    )

    original_faces = (
        mandibular_mesh.faces.copy()
    )

    rotated = hinge_axis.rotate_mesh(
        mesh=mandibular_mesh,
        angle_degrees=30.0,
    )

    assert np.allclose(
        mandibular_mesh.vertices,
        original_vertices,
    )

    assert np.array_equal(
        mandibular_mesh.faces,
        original_faces,
    )

    assert rotated is not mandibular_mesh