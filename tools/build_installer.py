"""Build the native OGDD installer for the current operating system."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
PYINSTALLER_DIR = DIST_DIR / "pyinstaller"
INSTALLER_DIR = DIST_DIR / "installers"
SPEC_FILE = PROJECT_ROOT / "installer" / "ogdd.spec"


def project_version() -> str:
    """Read the package version from the single authoritative source."""

    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject:
        return tomllib.load(pyproject)["project"]["version"]


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    """Run one build command and stop immediately on failure."""

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def build_frozen_application() -> Path:
    """Create the PyInstaller one-directory application."""

    run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(PYINSTALLER_DIR),
            "--workpath",
            str(PROJECT_ROOT / "build" / "pyinstaller"),
            str(SPEC_FILE),
        ]
    )
    application = PYINSTALLER_DIR / "OGDD"
    if not application.is_dir():
        raise RuntimeError("PyInstaller did not create the OGDD application.")
    return application


def find_inno_compiler() -> Path:
    """Locate the Inno Setup command-line compiler on Windows."""

    candidates: list[str | Path | None] = [
        shutil.which("ISCC.exe"),
        shutil.which("iscc"),
    ]
    installation_roots = (
        (os.environ.get("LOCALAPPDATA"), "Programs"),
        (os.environ.get("ProgramFiles"),),
        (os.environ.get("ProgramFiles(x86)"),),
    )
    candidates.extend(
        Path(root).joinpath(*subdirectories, "Inno Setup 6", "ISCC.exe")
        for root, *subdirectories in installation_roots
        if root
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError(
        "Inno Setup 6 was not found. Install it and run this command again."
    )


def build_windows_installer(version: str) -> Path:
    """Compile the per-user Windows setup executable."""

    environment = os.environ.copy()
    environment["OGDD_VERSION"] = version
    run(
        [
            str(find_inno_compiler()),
            str(PROJECT_ROOT / "installer" / "windows" / "ogdd.iss"),
        ],
        environment=environment,
    )
    return INSTALLER_DIR / f"OGDD-{version}-Windows-x64-Setup.exe"


def build_linux_installer(application: Path, version: str) -> Path:
    """Create a self-extracting, per-user Linux installer."""

    desktop_file = PROJECT_ROOT / "installer" / "linux" / "ogdd.desktop"
    shutil.copy2(desktop_file, application / desktop_file.name)

    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    archive = DIST_DIR / f"OGDD-{version}-Linux-x86_64.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(application, arcname="OGDD")

    output = INSTALLER_DIR / f"OGDD-{version}-Linux-x86_64.run"
    header = PROJECT_ROOT / "installer" / "linux" / "installer-header.sh"
    with output.open("wb") as installer, header.open("rb") as script:
        shutil.copyfileobj(script, installer)
        with archive.open("rb") as payload:
            shutil.copyfileobj(payload, installer)
    output.chmod(0o755)
    archive.unlink()
    return output


def write_checksum(installer: Path) -> Path:
    """Write a SHA-256 checksum beside a completed installer."""

    digest = hashlib.sha256()
    with installer.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    checksum = installer.with_name(f"{installer.name}.sha256")
    checksum.write_text(
        f"{digest.hexdigest()}  {installer.name}\n",
        encoding="ascii",
    )
    return checksum


def main() -> int:
    """Build the application and installer for this host platform."""

    version = project_version()
    application = build_frozen_application()
    if sys.platform == "win32":
        installer = build_windows_installer(version)
    elif sys.platform.startswith("linux"):
        installer = build_linux_installer(application, version)
    else:
        raise RuntimeError(f"Unsupported installer platform: {sys.platform}")

    if not installer.is_file() or installer.stat().st_size == 0:
        raise RuntimeError("The OGDD installer was not created correctly.")
    write_checksum(installer)
    print(installer.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
