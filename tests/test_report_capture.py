"""Clinical RC report capture and restoration regression tests."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from ogdd.app.main_window import MainWindow
from ogdd.app.scene_view import SceneView


@pytest.mark.parametrize("capture_fails", [False, True])
def test_top_capture_restores_maxillary_opacity(capture_fails):
    prop = MagicMock()
    prop.GetOpacity.return_value = 0.75
    actor = MagicMock()
    actor.GetProperty.return_value = prop
    scene = SimpleNamespace(
        _layer_actors={"maxillary_rc": [actor]},
        set_anatomical_view=MagicMock(),
        save_screenshot=MagicMock(),
    )
    def capture(path):
        prop.SetOpacity.assert_called_once_with(0.2)
        if capture_fails:
            raise OSError("capture failed")
    scene.save_screenshot.side_effect = capture
    if capture_fails:
        with pytest.raises(OSError, match="capture failed"):
            SceneView.save_superior_rc_screenshot(scene, "top.png")
    else:
        SceneView.save_superior_rc_screenshot(scene, "top.png")
    scene.set_anatomical_view.assert_called_once_with("top")
    assert prop.SetOpacity.call_args.args == (0.75,)


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("capture_fails", [False, True])
def test_report_captures_confirmed_rc_and_restores_state(
    tmp_path, confirmed, capture_fails,
):
    controller = MagicMock()
    controller.centric_relation_confirmed = confirmed
    controller.opening_angle_degrees = 1.2
    controller.lateral_angle_degrees = 0.0
    controller.protrusion_distance_mm = 2.0
    controller.adjustment_angle_degrees = -0.1
    controller.limits.get.return_value = None
    controller.reset_adjustment.return_value = "confirmed RC position"
    window = SimpleNamespace(
        _functional_calibration_controller=controller,
        _coordinate_system=object(),
        _rc_mic_registration=None,
        _restoring_study=False,
        _study_dirty=True,
        layer_actions={"maxillary_rc": MagicMock(), "mandibular_rc": MagicMock()},
        scene=MagicMock(),
        rc_mic_diagnosis_panel=MagicMock(),
        _show_functional_position=MagicMock(),
        _set_report_layers=MagicMock(),
        _update_window_title=MagicMock(),
        _refresh_results_panel=MagicMock(),
    )
    def capture(path):
        window._show_functional_position.assert_called_once_with(
            "confirmed RC position"
        )
        window._set_report_layers.assert_called_with(
            {"maxillary_rc", "mandibular_rc"}
        )
        if capture_fails:
            raise OSError("capture failed")
    window.scene.save_superior_rc_screenshot.side_effect = capture
    if confirmed and capture_fails:
        with pytest.raises(OSError, match="capture failed"):
            MainWindow._capture_report_visuals(window, tmp_path)
    else:
        captures = MainWindow._capture_report_visuals(window, tmp_path)
        assert captures["centric_top"] == (
            tmp_path / "centric_top.png" if confirmed else None
        )
    controller.set_position.assert_called_once_with(
        opening_angle_degrees=1.2,
        lateral_angle_degrees=0.0,
        protrusion_distance_mm=2.0,
    )
    controller.set_adjustment.assert_called_once_with(-0.1)
    window.scene.restore_camera_state.assert_called_once_with(
        window.scene.camera_state.return_value
    )
    assert window._study_dirty is True
    assert window._restoring_study is False
