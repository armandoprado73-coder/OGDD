# OGDD
# Open Geometry for Digital Dentistry

![OGDD Logo](docs/assets/ogdd_logo_placeholder.png)

## Overview

OGDD (Open Geometry for Digital Dentistry) is an open-source computational geometry library designed specifically for digital dentistry applications.

The main objective of OGDD is to provide a transparent, reproducible, and extensible framework for geometric analysis of dental 3D models.

The first clinical application target is the automatic detection and calculation of an occlusal reference plane from 3D dental meshes.

---

# Vision

Digital dentistry depends increasingly on 3D data generated from:

- Intraoral scanners
- Laboratory scanners
- CBCT systems
- CAD/CAM workflows

However, many geometric operations used in dentistry remain dependent on proprietary solutions and closed algorithms.

OGDD aims to provide an open computational geometry foundation for dental research and software development.

---

# Main Goals

OGDD is designed to provide:

- Open geometric algorithms for dental applications.
- Reproducible computational methods.
- Independent validation of results.
- Extensible architecture.
- Compatibility with different 3D environments.

---

# Architecture

OGDD is designed as a modular Python SDK.

The architecture separates the mathematical core from visualization platforms.

             Blender
                |
          OGDD Add-on
                |
    -------------------------
          OGDD SDK
    -------------------------
                |
    Computational Geometry
                |
             NumPy

   
The core library does not depend on Blender.

Blender is considered only one possible visualization environment.

---

# Current Development Status

## Version

     
## Development Phase

Foundation stage.

Current objectives:

- Project infrastructure.
- Core data structures.
- Mesh representation.
- Basic geometry utilities.

---

# Planned Features

## Geometry Engine

- Mesh processing.
- Surface analysis.
- Normal calculation.
- Curvature estimation.
- Spatial transformations.

## Dental Analysis

- Occlusal surface detection.
- Anatomical landmark detection.
- Cusp identification.
- Occlusal reference plane calculation.
- Confidence estimation.

## Future Development

- Curve of Spee analysis.
- Curve of Wilson analysis.
- Dental symmetry analysis.
- Digital articulation tools.
- Research-oriented measurements.

---

# Installation

## Native installers

Release builds do not require Python. Download the installer for your platform
from the GitHub release or from the artifacts produced by the **Build
installers** workflow.

### Windows 10/11 (64-bit)

Run `OGDD-<version>-Windows-x64-Setup.exe`. OGDD is installed for the current
user, appears in the Start menu and includes an uninstaller. Administrator
access is not required.

### Linux (64-bit)

Make the downloaded installer executable and run it:

```bash
chmod +x OGDD-<version>-Linux-x86_64.run
./OGDD-<version>-Linux-x86_64.run
```

OGDD is installed for the current user and appears in the desktop application
menu. It can be removed with:

```bash
~/.local/share/ogdd/uninstall-ogdd.sh
```

## Development installation

```bash
git clone https://github.com/armandoprado73-coder/OGDD.git
cd OGDD
python -m pip install -e ".[gui]"
ogdd
```

## Building an installer

Install the packaging dependencies and build on the target operating system:

```bash
python -m pip install ".[gui,build]"
python tools/build_installer.py
```

Windows builds also require Inno Setup 6. Generated installers are written to
`dist/installers/` with a SHA-256 checksum. The GitHub Actions workflow builds
and checks both systems independently whenever it is started manually or a
`v*` tag is pushed.
