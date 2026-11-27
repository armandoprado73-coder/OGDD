"""MCD results agree across the UI and exports, including reopened data."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from ogdd.app.main_window import MainWindow
from ogdd.app.rc_mic_diagnosis_panel import RcMicDiagnosisPanel
from ogdd.app.results_export import ResultsExporter
from ogdd.geometry.coordinate_system import CoordinateSystem
from ogdd.registration.centric_relation_registration import CondylarDisplacementResult


def test_saved_anatomical_geometry_is_converted_once_in_every_result_view():
    frame = CoordinateSystem(
        origin=[10, 20, 30], x_axis=[0, 1, 0],
        y_axis=[-1, 0, 0], z_axis=[0, 0, 1],
    )
    anatomical = [np.array([0.9136, 1.3235, 0.4433]),
                  np.array([0.9406, -0.9251, -0.4932])]
    rc_local = [np.array([55, 80, 40]), np.array([-55, 80, 40])]
    displacement = CondylarDisplacementResult(
        right_rc_point=frame.to_world(rc_local[0]),
        right_mic_point=frame.to_world(rc_local[0] + anatomical[0]),
        left_rc_point=frame.to_world(rc_local[1]),
        left_mic_point=frame.to_world(rc_local[1] + anatomical[1]),
    )
    window = SimpleNamespace(
        _articulator_configuration=None,
        _functional_calibration_controller=None,
        _rc_mic_registration=SimpleNamespace(
            converged=True,
            maxillary_registration=SimpleNamespace(root_mean_square_error=0.0),
            mandibular_registration=SimpleNamespace(root_mean_square_error=0.014305),
        ),
        _condylar_displacement=displacement,
        _coordinate_system=frame,
        _study_display_name=lambda: "Montaje 2",
        _study_filenames=lambda: {},
    )
    snapshot = MainWindow._results_snapshot(window)
    rows = {(s, m): v for s, m, v, _ in ResultsExporter.csv_rows(snapshot)}
    for side, source, expected in zip(
        ("right", "left"), anatomical,
        ([1.3235, 0.9136, -0.4433], [-0.9251, 0.9406, 0.4932]),
    ):
        stored = snapshot.rc_mic[f"{side}_vector_mm"]
        np.testing.assert_allclose(stored, source)
        np.testing.assert_allclose(
            [rows[("rc_mic", f"{side}_{axis}")] for axis in "xyz"], expected,
        )
        panel = RcMicDiagnosisPanel._format_vector(np.array(stored))
        tree = MainWindow._format_diagnostic_vector(np.array(stored))
        report = ResultsExporter._vector_text(stored, np.linalg.norm(source))
        for axis, value in zip("XYZ", expected):
            text = f"{axis} {value:+.4f}"
            assert text in panel and text in tree and text in report
        assert rows[("rc_mic", f"{side}_distance")] == pytest.approx(
            np.linalg.norm(source)
        )
    assert MainWindow._results_snapshot(window).rc_mic == snapshot.rc_mic
