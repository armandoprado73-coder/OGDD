"""Contract tests for reproducible OGDD installers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

from ogdd.app.application import headless_startup_check_enabled


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_installer_sources_are_present() -> None:
    expected = (
        "src/ogdd/app/assets/ogdd-logo.png",
        "installer/ogdd.spec",
        "installer/windows/ogdd.ico",
        "installer/windows/ogdd.iss",
        "installer/linux/ogdd.desktop",
        "installer/linux/installer-header.sh",
        "tools/build_installer.py",
        ".github/workflows/build-installers.yml",
    )

    for relative_path in expected:
        assert (PROJECT_ROOT / relative_path).is_file()


def test_application_entry_point_is_available_to_installer() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject:
        configuration = tomllib.load(pyproject)

    assert configuration["project"]["scripts"]["ogdd"] == (
        "ogdd.app.application:main"
    )
    assert "build" in configuration["project"]["optional-dependencies"]
    assert configuration["tool"]["setuptools"]["package-data"]["ogdd.app"] == [
        "assets/*.png"
    ]


def test_linux_desktop_entry_has_build_time_executable_placeholder() -> None:
    desktop_entry = (
        PROJECT_ROOT / "installer" / "linux" / "ogdd.desktop"
    ).read_text(encoding="utf-8")

    assert "Exec=@OGDD_EXECUTABLE@" in desktop_entry
    assert "Icon=@OGDD_ICON@" in desktop_entry
    assert "Terminal=false" in desktop_entry


def test_windows_installer_is_per_user_and_includes_uninstaller() -> None:
    inno_setup = (
        PROJECT_ROOT / "installer" / "windows" / "ogdd.iss"
    ).read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in inno_setup
    assert "DefaultDirName={localappdata}\\Programs\\OGDD" in inno_setup
    assert "UninstallDisplayIcon={app}\\OGDD.exe" in inno_setup
    assert "SetupIconFile=ogdd.ico" in inno_setup


def test_application_and_installers_use_the_ogdd_logo() -> None:
    application = (
        PROJECT_ROOT / "src" / "ogdd" / "app" / "application.py"
    ).read_text(encoding="utf-8")
    pyinstaller = (PROJECT_ROOT / "installer" / "ogdd.spec").read_text(
        encoding="utf-8"
    )
    linux_installer = (
        PROJECT_ROOT / "installer" / "linux" / "installer-header.sh"
    ).read_text(encoding="utf-8")

    assert "application.setWindowIcon" in application
    assert '"ogdd-logo.png"' in pyinstaller
    assert '"ogdd.ico"' in pyinstaller
    assert "@OGDD_ICON@" in linux_installer


def test_workflow_preserves_windows_installer_and_startup_diagnostics() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "build-installers.yml"
    ).read_text(encoding="utf-8")
    entry_point = (
        PROJECT_ROOT / "src" / "ogdd" / "app" / "__main__.py"
    ).read_text(encoding="utf-8")

    assert "if: ${{ !cancelled() }}" in workflow
    assert "Upload Windows startup diagnostics" in workflow
    assert "ogdd-startup-error.log" in workflow
    assert "ogdd-startup-error.log" in entry_point


def test_headless_startup_check_requires_check_argument_and_environment() -> None:
    enabled = {"OGDD_HEADLESS_STARTUP_CHECK": "1"}

    assert not headless_startup_check_enabled(
        check_startup=False,
        environment=enabled,
    )
    assert not headless_startup_check_enabled(
        check_startup=True,
        environment={},
    )
    assert headless_startup_check_enabled(
        check_startup=True,
        environment=enabled,
    )


def test_workflow_checks_installed_application_lifecycle() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "build-installers.yml"
    ).read_text(encoding="utf-8")

    assert "OGDD_HEADLESS_STARTUP_CHECK" in workflow
    assert "Test Windows installer lifecycle" in workflow
    assert '"/VERYSILENT"' in workflow
    assert '"unins000.exe"' in workflow
    assert "Test Linux installer lifecycle" in workflow
    assert 'installed_icon="${XDG_DATA_HOME}/ogdd/OGDD/ogdd-logo.png"' in workflow
    assert "uninstall-ogdd.sh" in workflow


def test_checksum_uses_installer_filename(tmp_path: Path) -> None:
    module_path = PROJECT_ROOT / "tools" / "build_installer.py"
    spec = importlib.util.spec_from_file_location("ogdd_installer_builder", module_path)
    assert spec is not None
    assert spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    installer = tmp_path / "OGDD-test.run"
    installer.write_bytes(b"Fangorn")

    checksum = builder.write_checksum(installer)

    digest, filename = checksum.read_text(encoding="ascii").split()
    assert len(digest) == 64
    assert filename == installer.name
