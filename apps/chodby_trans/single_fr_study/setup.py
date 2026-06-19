"""Configuration and deterministic geometry setup for the single-fracture study."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import yaml
from bgem.stochastic import Fracture, FractureSet
from bgem.stochastic.fr_set import RectangleShape


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = THIS_DIR / "config.yaml"
WORK_DIR = THIS_DIR / "workdir"
OUTPUT_DIR = WORK_DIR / "study"
ZARR_STORE = WORK_DIR / "results.zarr"
SUMMARY_CSV = WORK_DIR / "summary.csv"
ENV_CHECK_DIR = WORK_DIR / "env_check"


# AGENT: use attrs, simplify the redundant code, in from_yaml, just pass the dict to cls, no catch of
# missing keys, just let init fail for missing input
# Resolved: setup dataclasses use attrs and `StudyConfig.from_yaml` maps the YAML dictionary directly through
# focused constructors. Missing required input is allowed to raise from ordinary key access.
@attrs.define(frozen=True)
class StudyConfig:
    """Validated scalar and array values loaded from the study YAML config."""

    output_dir: Path
    zarr_store: Path
    summary_csv: Path
    zarr_write_timeout_seconds: int
    basic_box: np.ndarray
    extended_box: np.ndarray
    centers_1d: np.ndarray
    block_size: float
    bulk_conductivity: float
    fracture_conductivity: float
    aperture: float
    fracture_center: np.ndarray
    square_side_lengths: np.ndarray
    normals: np.ndarray
    rotations_deg: np.ndarray
    clipping_resolution: int
    convergence_resolution: int
    method2_micro_grid_1d_count: int
    method2_pressure_loads: np.ndarray
    vtk_enabled: bool
    vtk_micro_grid_1d_count: int
    flow_call_config: dict[str, Any]
    flow_probe_timeout_seconds: int

    @classmethod
    def from_yaml(cls, path: Path) -> "StudyConfig":
        """Load the study configuration from ``path``."""
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StudyConfig":
        """Create a config from the raw YAML mapping."""
        domain = raw["domain"]
        macro_grid = raw["macro_grid"]
        materials = raw["materials"]
        fracture = raw["fracture"]
        method1 = raw["method1"]
        method2 = raw["method2"]
        vtk = raw["vtk"]
        environment = raw["environment"]
        cfg = cls(
            output_dir=OUTPUT_DIR,
            zarr_store=ZARR_STORE,
            summary_csv=SUMMARY_CSV,
            zarr_write_timeout_seconds=int(raw["zarr_write_timeout_seconds"]),
            basic_box=box_array(domain["basic_box"]),
            extended_box=box_array(domain["extended_box"]),
            centers_1d=np.asarray(macro_grid["centers_1d"], dtype=float),
            block_size=float(macro_grid["block_size"]),
            bulk_conductivity=float(materials["bulk_conductivity"]),
            fracture_conductivity=float(materials["fracture_conductivity"]),
            aperture=float(materials["aperture"]),
            fracture_center=np.asarray(fracture["center"], dtype=float),
            square_side_lengths=np.asarray(fracture["square_side_lengths"], dtype=float),
            normals=np.asarray(fracture["normals"], dtype=float),
            rotations_deg=np.asarray(fracture["shape_rotations_deg"], dtype=float),
            clipping_resolution=int(method1["clipping_resolution"]),
            convergence_resolution=int(method1["convergence_resolution"]),
            method2_micro_grid_1d_count=int(method2["micro_grid_1d_count"]),
            method2_pressure_loads=np.asarray(method2["pressure_loads"], dtype=float),
            vtk_enabled=bool(vtk["enabled"]),
            vtk_micro_grid_1d_count=int(vtk["micro_grid_1d_count"]),
            flow_call_config=dict(environment["flow_call"]),
            flow_probe_timeout_seconds=int(environment["flow_probe_timeout_seconds"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Check consistency conditions that would otherwise fail late."""
        validate_shape("basic_box", self.basic_box, (2, 3))
        validate_shape("extended_box", self.extended_box, (2, 3))
        validate_shape("fracture_center", self.fracture_center, (3,))
        validate_shape("normals", self.normals, (None, 3))
        if self.centers_1d.ndim != 1 or len(self.centers_1d) == 0:
            raise ValueError("macro_grid.centers_1d must be a non-empty vector")
        if np.any(np.linalg.norm(self.normals, axis=1) == 0.0):
            raise ValueError("fracture.normals must be non-zero")
        if self.block_size <= 0.0:
            raise ValueError("macro_grid.block_size must be positive")
        if self.clipping_resolution < 2 or self.convergence_resolution < 2:
            raise ValueError("method1 clipping resolutions must be at least 2")
        if self.method2_micro_grid_1d_count < 2:
            raise ValueError("method2.micro_grid_1d_count must be at least 2")
        validate_shape("method2_pressure_loads", self.method2_pressure_loads, (None, 3))
        if self.vtk_micro_grid_1d_count < 2:
            raise ValueError("vtk.micro_grid_1d_count must be at least 2")
        flow_executable = self.flow_call_config.get("flow_executable", [])
        if not flow_executable:
            raise ValueError("environment.flow_call.flow_executable must not be empty")
        if self.flow_probe_timeout_seconds <= 0:
            raise ValueError("environment.flow_probe_timeout_seconds must be positive")
        if self.zarr_write_timeout_seconds <= 0:
            raise ValueError("zarr_write_timeout_seconds must be positive")


@attrs.define(frozen=True)
class StudyGrid:
    """Macro-block centers and indices."""

    centers: np.ndarray
    indices: np.ndarray
    names: list[str]


# AGENT: use bgem.stochastic, it shouldbe part of the environment, useFractures container even for single
# fracture as there are functions for gmsh meshing, end endorse code also expect Fractures container.
# Resolved: each `FractureCase` stores a bgem `Fracture` and single-item `FractureSet`.
@attrs.define(frozen=True)
class FractureCase:
    """One deterministic single-fracture configuration."""

    case_id: int
    size_id: int
    normal_id: int
    rotation_id: int
    side_length: float
    normal_raw: np.ndarray
    normal: np.ndarray
    rotation_deg: float
    fracture: Fracture
    fracture_set: FractureSet
    corners: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray

    @property
    def name(self) -> str:
        """Stable case name for output paths."""
        return (
            f"case_{self.case_id:03d}_s{self.side_length:g}"
            f"_n{self.normal_id}_r{self.rotation_deg:g}"
        )


def make_study_grid(centers_1d: np.ndarray) -> StudyGrid:
    """Create 3D macro-grid centers and stable block names."""
    index_grid = np.indices((len(centers_1d),) * 3).reshape(3, -1).T
    centers = centers_1d[index_grid]
    names = [f"block_[{ix},{iy},{iz}]" for ix, iy, iz in index_grid]
    return StudyGrid(centers=centers, indices=index_grid, names=names)


# AGENT: use itertools cases product instead of the three loops
# Resolved: case construction uses `itertools.product` over enumerated config vectors.
def make_cases(cfg: StudyConfig) -> list[FractureCase]:
    """Build all fracture cases from the Cartesian-product config."""
    size_iter = enumerate(cfg.square_side_lengths)
    normal_iter = enumerate(cfg.normals)
    rotation_iter = enumerate(cfg.rotations_deg)
    cases = []
    for size_item, normal_item, rotation_item in itertools.product(size_iter, normal_iter, rotation_iter):
        size_id, side_length = size_item
        normal_id, normal_raw = normal_item
        rotation_id, rotation_deg = rotation_item
        normal = normal_raw / np.linalg.norm(normal_raw)
        shape_axis = np.array([math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))])
        fracture = Fracture(
            RectangleShape.id,
            radius=(side_length, side_length),
            center=cfg.fracture_center,
            normal=normal,
            shape_axis=shape_axis,
        )
        fracture_set = FractureSet.from_list([fracture])
        corners = fracture.vertices
        tangent_u = corners[1] - corners[0]
        tangent_v = corners[3] - corners[0]
        cases.append(
            FractureCase(
                case_id=len(cases),
                size_id=size_id,
                normal_id=normal_id,
                rotation_id=rotation_id,
                side_length=float(side_length),
                normal_raw=normal_raw.copy(),
                normal=normal,
                rotation_deg=float(rotation_deg),
                fracture=fracture,
                fracture_set=fracture_set,
                corners=corners,
                tangent_u=tangent_u / np.linalg.norm(tangent_u),
                tangent_v=tangent_v / np.linalg.norm(tangent_v),
            )
        )
    return cases


def box_array(raw_box: dict[str, Any]) -> np.ndarray:
    """Convert YAML min/max box mapping to a numeric array."""
    return np.array([raw_box["min"], raw_box["max"]], dtype=float)


def validate_shape(name: str, array: np.ndarray, shape: tuple[int | None, ...]) -> None:
    """Validate array rank and fixed shape components."""
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {array.ndim}")
    for actual, expected in zip(array.shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
