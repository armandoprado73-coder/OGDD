# Changelog

All notable changes to this project will be documented in this file.

The format is based on:

https://keepachangelog.com/

This project follows Semantic Versioning.

---

# [0.1.0-alpha] - 2026-07-18

## Added

### Project Foundation

- Initial creation of OGDD (Open Geometry for Digital Dentistry).
- Established open-source project structure.
- Added MIT License.
- Added initial project documentation.

### Architecture

- Defined OGDD as an independent computational geometry SDK.
- Established separation between:
  - OGDD Core
  - Blender Integration
  - Future visualization environments.

### Development Environment

- Added modern Python packaging using `pyproject.toml`.
- Defined Python >= 3.11 requirement.
- Added NumPy as the first numerical dependency.

### Design Decisions

- Adopted Blender 4.5 LTS as the initial visualization platform.
- Decided that Blender will not be a dependency of the computational core.
- Established NumPy-based vectorized geometry representation.
- Chosen MIT License for maximum openness and collaboration.

---

# [Unreleased]

## Planned

### Core Geometry

- Mesh data structure.
- Vertex and face representation.
- Bounding box calculations.
- Basic geometric transformations.

### Input / Output

- STL reader.
- STL writer.

### Testing

- Initial unit testing framework.
- Basic geometry validation tests.

---

# Versioning Strategy

OGDD follows semantic versioning:
