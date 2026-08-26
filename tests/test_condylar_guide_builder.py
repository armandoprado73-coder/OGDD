import numpy as np
import pytest

from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.articulator.condylar_guide_builder import (
    CondylarGuideBuilder,
    CondylarGuidePair,
)
from ogdd.articulator.configuration import (
    ArticulatorConfiguration,
)
from ogdd.geometry.coordinate_system import (
    CoordinateSystem,
)


@pytest.fixture
def coordinate_system() -> CoordinateSystem:
    return CoordinateSystem.identity()


@pytest.fixture
def hinge_axis() -> HingeAxis:
    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-55.0, 0.0, 0.0]),
        reference_used="Virtual articulator",
    )

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([55.0, 0.0, 0.0]),
        reference_used="Virtual articulator",
    )

    return HingeAxis(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
    )


@pytest.fixture
def configuration() -> ArticulatorConfiguration:
    return ArticulatorConfiguration(
        right_condylar_guidance_degrees=35.0,
        left_condylar_guidance_degrees=42.0,
        condyle_diameter=6.0,
        condylar_guide_length=20.0,
        condylar_guide_width=20.0,
    )


@pytest.fixture
def guide_pair(
    hinge_axis: HingeAxis,
    coordinate_system: CoordinateSystem,
    configuration: ArticulatorConfiguration,
) -> CondylarGuidePair:
    return CondylarGuideBuilder.build(
        hinge_axis=hinge_axis,
        coordinate_system=coordinate_system,
        configuration=configuration,
    )


def test_builder_creates_named_guide_pair(
    guide_pair: CondylarGuidePair,
) -> None:
    assert isinstance(
        guide_pair,
        CondylarGuidePair,
    )


def test_right_guide_uses_right_condyle(
    guide_pair: CondylarGuidePair,
) -> None:
    assert np.allclose(
        guide_pair.right_guide.condyle_center,
        np.array([55.0, 0.0, 0.0]),
    )


def test_left_guide_uses_left_condyle(
    guide_pair: CondylarGuidePair,
) -> None:
    assert np.allclose(
        guide_pair.left_guide.condyle_center,
        np.array([-55.0, 0.0, 0.0]),
    )


def test_guidance_angles_are_independent(
    guide_pair: CondylarGuidePair,
) -> None:
    assert (
        guide_pair.right_guide.angle_degrees
        == 35.0
    )

    assert (
        guide_pair.left_guide.angle_degrees
        == 42.0
    )


def test_guides_use_configured_condyle_size(
    guide_pair: CondylarGuidePair,
) -> None:
    assert (
        guide_pair.right_guide.condyle_diameter
        == 6.0
    )

    assert (
        guide_pair.left_guide.condyle_radius
        == 3.0
    )


def test_guides_use_configured_dimensions(
    guide_pair: CondylarGuidePair,
) -> None:
    assert guide_pair.right_guide.length == 20.0
    assert guide_pair.right_guide.width == 20.0
    assert guide_pair.left_guide.length == 20.0
    assert guide_pair.left_guide.width == 20.0


def test_guides_have_different_trajectories(
    guide_pair: CondylarGuidePair,
) -> None:
    assert not np.allclose(
        guide_pair.right_guide.trajectory_direction,
        guide_pair.left_guide.trajectory_direction,
    )