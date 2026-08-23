import numpy as np
import pytest

from ogdd.anatomy.balkwill import BalkwillTriangle
from ogdd.anatomy.bonwill import BonwillTriangle
from ogdd.anatomy.hinge_axis import HingeAxis
from ogdd.anatomy.landmark import Landmark
from ogdd.anatomy.mandibular_assembly import (
    MandibularAssembly,
    MandibularPosition,
)
from ogdd.mesh import Mesh


@pytest.fixture
def mandibular_assembly() -> MandibularAssembly:
    left_condyle = Landmark(
        name="LEFT_CONDYLE",
        point=np.array([-55.0, 0.0, 0.0]),
        reference_used="condylar center",
    )

    right_condyle = Landmark(
        name="RIGHT_CONDYLE",
        point=np.array([55.0, 0.0, 0.0]),
        reference_used="condylar center",
    )

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

    mesh = Mesh(
        vertices=np.array([
            [-50.0, 20.0, 0.0],
            [50.0, 20.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 60.0, -20.0],
        ]),
        faces=np.array([
            [0, 1, 2],
            [0, 2, 3],
        ]),
    )

    balkwill = BalkwillTriangle(
        left_posterior=left_posterior,
        right_posterior=right_posterior,
        dental_midline=dental_midline,
    )

    bonwill = BonwillTriangle(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
        dental_midline=dental_midline,
    )

    hinge_axis = HingeAxis(
        left_condyle=left_condyle,
        right_condyle=right_condyle,
    )

    return MandibularAssembly(
        mesh=mesh,
        balkwill=balkwill,
        bonwill=bonwill,
        hinge_axis=hinge_axis,
    )


def test_mandibular_assembly_creation(
    mandibular_assembly: MandibularAssembly,
) -> None:
    assert mandibular_assembly.mesh.vertex_count == 4

    assert (
        mandibular_assembly.hinge_axis.length
        == pytest.approx(110.0)
    )


def test_position_at_returns_mandibular_position(
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = mandibular_assembly.position_at(
        angle_degrees=30.0,
    )

    assert isinstance(
        position,
        MandibularPosition,
    )

    assert position.angle_degrees == pytest.approx(
        30.0
    )


def test_position_at_moves_all_structures_together(
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = mandibular_assembly.position_at(
        angle_degrees=30.0,
    )

    mesh_midline = position.mesh.vertices[2]

    balkwill_midline = (
        position.balkwill.dental_midline.point
    )

    bonwill_midline = (
        position.bonwill.dental_midline.point
    )

    assert np.allclose(
        mesh_midline,
        balkwill_midline,
    )

    assert np.allclose(
        balkwill_midline,
        bonwill_midline,
    )


def test_position_at_keeps_hinge_axis_fixed(
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = mandibular_assembly.position_at(
        angle_degrees=30.0,
    )

    assert (
        position.hinge_axis
        is mandibular_assembly.hinge_axis
    )

    assert np.allclose(
        position.bonwill.left_condyle.point,
        mandibular_assembly.hinge_axis.left_condyle.point,
    )

    assert np.allclose(
        position.bonwill.right_condyle.point,
        mandibular_assembly.hinge_axis.right_condyle.point,
    )


def test_position_at_zero_returns_closed_geometry(
    mandibular_assembly: MandibularAssembly,
) -> None:
    position = mandibular_assembly.position_at(
        angle_degrees=0.0,
    )

    assert np.allclose(
        position.mesh.vertices,
        mandibular_assembly.mesh.vertices,
    )

    assert np.allclose(
        position.balkwill.dental_midline.point,
        mandibular_assembly.balkwill.dental_midline.point,
    )

    assert np.allclose(
        position.bonwill.dental_midline.point,
        mandibular_assembly.bonwill.dental_midline.point,
    )


def test_positions_do_not_accumulate_rotation(
    mandibular_assembly: MandibularAssembly,
) -> None:
    position_10 = mandibular_assembly.position_at(
        angle_degrees=10.0,
    )

    position_30 = mandibular_assembly.position_at(
        angle_degrees=30.0,
    )

    direct_position_30 = (
        mandibular_assembly.position_at(
            angle_degrees=30.0,
        )
    )

    assert not np.allclose(
        position_10.mesh.vertices,
        position_30.mesh.vertices,
    )

    assert np.allclose(
        position_30.mesh.vertices,
        direct_position_30.mesh.vertices,
    )


def test_negative_opening_angle_is_rejected(
    mandibular_assembly: MandibularAssembly,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        mandibular_assembly.position_at(
            angle_degrees=-1.0,
        )