import numpy as np
import pytest

from ogdd.articulator.condylar_guide import (
    CondylarGuide,
)
from ogdd.geometry.coordinate_system import (
    CoordinateSystem,
)


@pytest.fixture
def coordinate_system() -> CoordinateSystem:
    return CoordinateSystem.identity()


@pytest.fixture
def guide(
    coordinate_system: CoordinateSystem,
) -> CondylarGuide:
    return CondylarGuide(
        condyle_center=np.array([
            55.0,
            0.0,
            0.0,
        ]),
        coordinate_system=coordinate_system,
    )


def test_condylar_guide_creation(
    guide: CondylarGuide,
) -> None:
    assert np.allclose(
        guide.condyle_center,
        np.array([55.0, 0.0, 0.0]),
    )

    assert guide.angle_degrees == 45.0
    assert guide.length == 20.0
    assert guide.width == 20.0


def test_default_condyle_radius(
    guide: CondylarGuide,
) -> None:
    assert guide.condyle_radius == 3.0


def test_trajectory_direction_at_45_degrees(
    guide: CondylarGuide,
) -> None:
    expected = np.array([
        0.0,
        np.sqrt(0.5),
        -np.sqrt(0.5),
    ])

    assert np.allclose(
        guide.trajectory_direction,
        expected,
    )


def test_surface_normal_at_45_degrees(
    guide: CondylarGuide,
) -> None:
    expected = np.array([
        0.0,
        np.sqrt(0.5),
        np.sqrt(0.5),
    ])

    assert np.allclose(
        guide.surface_normal,
        expected,
    )


def test_trajectory_and_surface_are_orthogonal(
    guide: CondylarGuide,
) -> None:
    assert np.dot(
        guide.trajectory_direction,
        guide.surface_normal,
    ) == pytest.approx(0.0)


def test_main_surface_is_tangent_to_condyle(
    guide: CondylarGuide,
) -> None:
    distance = np.linalg.norm(
        guide.guide_contact_point
        - guide.condyle_center
    )

    assert distance == pytest.approx(3.0)


def test_posterior_stop_is_tangent_to_condyle(
    guide: CondylarGuide,
) -> None:
    distance = np.linalg.norm(
        guide.posterior_stop_contact_point
        - guide.condyle_center
    )

    assert distance == pytest.approx(3.0)


def test_center_at_zero_is_centric_position(
    guide: CondylarGuide,
) -> None:
    assert np.allclose(
        guide.center_at(0.0),
        guide.condyle_center,
    )


def test_center_translates_along_guide(
    guide: CondylarGuide,
) -> None:
    expected = np.array([
        55.0,
        10.0 * np.sqrt(0.5),
        -10.0 * np.sqrt(0.5),
    ])

    assert np.allclose(
        guide.center_at(10.0),
        expected,
    )


def test_contact_follows_condyle_translation(
    guide: CondylarGuide,
) -> None:
    contact_difference = (
        guide.guide_contact_at(10.0)
        - guide.guide_contact_point
    )

    expected = (
        10.0
        * guide.trajectory_direction
    )

    assert np.allclose(
        contact_difference,
        expected,
    )


def test_rejects_translation_beyond_guide(
    guide: CondylarGuide,
) -> None:
    with pytest.raises(
        ValueError,
        match="guide length",
    ):
        guide.center_at(21.0)


def test_zero_degree_guide_is_horizontal(
    coordinate_system: CoordinateSystem,
) -> None:
    guide = CondylarGuide(
        condyle_center=np.zeros(3),
        coordinate_system=coordinate_system,
        angle_degrees=0.0,
    )

    assert np.allclose(
        guide.trajectory_direction,
        np.array([0.0, 1.0, 0.0]),
    )

    assert np.allclose(
        guide.surface_normal,
        np.array([0.0, 0.0, 1.0]),
    )


def test_rejects_invalid_condyle_center(
    coordinate_system: CoordinateSystem,
) -> None:
    with pytest.raises(
        ValueError,
        match="3D point",
    ):
        CondylarGuide(
            condyle_center=np.array([0.0, 0.0]),
            coordinate_system=coordinate_system,
        )


def test_rejects_invalid_guidance_angle(
    coordinate_system: CoordinateSystem,
) -> None:
    with pytest.raises(
        ValueError,
        match="guidance angle",
    ):
        CondylarGuide(
            condyle_center=np.zeros(3),
            coordinate_system=coordinate_system,
            angle_degrees=90.0,
        )
def test_maximum_translation_is_17_mm(
    guide: CondylarGuide,
) -> None:
    assert guide.maximum_translation == 17.0


def test_posterior_stop_uses_condyle_diameter(
    guide: CondylarGuide,
) -> None:
    assert guide.posterior_stop_length == 6.0


def test_l_corner_connects_both_contacts(
    guide: CondylarGuide,
) -> None:
    expected_main_contact = (
        guide.l_corner
        + guide.condyle_radius
        * guide.trajectory_direction
    )

    expected_stop_contact = (
        guide.l_corner
        - guide.condyle_radius
        * guide.surface_normal
    )

    assert np.allclose(
        expected_main_contact,
        guide.guide_contact_point,
    )

    assert np.allclose(
        expected_stop_contact,
        guide.posterior_stop_contact_point,
    )


def test_main_surface_has_configured_dimensions(
    guide: CondylarGuide,
) -> None:
    vertices = guide.main_surface_vertices

    measured_width = np.linalg.norm(
        vertices[1] - vertices[0]
    )

    measured_length = np.linalg.norm(
        vertices[3] - vertices[0]
    )

    assert measured_width == pytest.approx(20.0)
    assert measured_length == pytest.approx(20.0)


def test_main_surface_is_on_tangent_plane(
    guide: CondylarGuide,
) -> None:
    relative_vertices = (
        guide.main_surface_vertices
        - guide.guide_contact_point
    )

    distances = (
        relative_vertices
        @ guide.surface_normal
    )

    assert np.allclose(
        distances,
        np.zeros(4),
    )


def test_maximum_translation_reaches_surface_end(
    guide: CondylarGuide,
) -> None:
    vertices = guide.main_surface_vertices

    surface_end_center = (
        vertices[2] + vertices[3]
    ) / 2.0

    final_contact = guide.guide_contact_at(
        guide.maximum_translation
    )

    assert np.allclose(
        final_contact,
        surface_end_center,
    )


def test_posterior_stop_has_6_by_20_dimensions(
    guide: CondylarGuide,
) -> None:
    vertices = guide.posterior_stop_vertices

    measured_width = np.linalg.norm(
        vertices[1] - vertices[0]
    )

    measured_length = np.linalg.norm(
        vertices[3] - vertices[0]
    )

    assert measured_width == pytest.approx(20.0)
    assert measured_length == pytest.approx(6.0)


def test_posterior_stop_is_on_tangent_plane(
    guide: CondylarGuide,
) -> None:
    relative_vertices = (
        guide.posterior_stop_vertices
        - guide.posterior_stop_contact_point
    )

    distances = (
        relative_vertices
        @ guide.posterior_stop_normal
    )

    assert np.allclose(
        distances,
        np.zeros(4),
    )