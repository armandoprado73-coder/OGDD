"""Welcome overlay lifecycle without an OpenGL context."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

from PySide6.QtWidgets import QApplication, QFrame
from ogdd.app import scene_view


def test_logo_hides_on_load_and_returns_for_a_new_study(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class Interactor(QFrame):
        def __init__(self, parent):
            super().__init__(parent)
            self.interactor = self
            self.set_background = MagicMock()
            self.remove_actor = MagicMock()

    monkeypatch.setattr(scene_view, "QtInteractor", Interactor)
    scene = scene_view.SceneView()
    scene.reset_camera = MagicMock()
    scene.resize(900, 650)
    scene.show()
    app.processEvents()
    welcome = scene._welcome_panel
    assert welcome.isVisible()
    assert not welcome.logo.pixmap().isNull()
    assert welcome.geometry() == scene.plotter.rect()
    scene.finish_study_load()
    assert welcome.isHidden()
    scene.show_welcome()
    scene.resize(600, 440)
    app.processEvents()
    assert welcome.isVisible()
    assert welcome.geometry() == scene.plotter.rect()
    assert welcome.logo.pixmap().height() <= int(scene.plotter.height() * 0.45)
    scene.close()
