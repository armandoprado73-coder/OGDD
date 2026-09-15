"""Contract tests for reproducible OGDD installers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_installer_sources_are_present() -> None:
    expected = (
        "installer/ogdd.spec",
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


def test_linux_desktop_entry_has_build_time_executable_placeholder() -> None:
    desktop_entry = (
        PROJECT_ROOT / "installer" / "linux" / "ogdd.desktop"
    ).read_text(encoding="utf-8")

    assert "Exec=@OGDD_EXECUTABLE@" in desktop_entry
    assert "Terminal=false" in desktop_entry


def test_windows_installer_is_per_user_and_includes_uninstaller() -> None:
    inno_setup = (
        PROJECT_ROOT / "installer" / "windows" / "ogdd.iss"
    ).read_text(encoding="utf-8")

    assert "PrivilegesRequired=lowest" in inno_setup
    assert "DefaultDirName={localappdata}\\Programs\\OGDD" in inno_setup
    assert "UninstallDisplayIcon={app}\\OGDD.exe" in inno_setup


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
