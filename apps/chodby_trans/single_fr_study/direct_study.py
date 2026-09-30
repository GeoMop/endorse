"""Run the single-fracture study direct estimate and blob homogenization."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import attrs
import numpy as np

try:
    from .method2 import run_blob_case_estimate
    from .output import write_outputs
    from .setup import DEFAULT_CONFIG, FractureCase, StudyConfig, StudyGrid, make_cases, make_study_grid
except ImportError:  # pragma: no cover - allows direct script execution.
    from method2 import run_blob_case_estimate
    from output import write_outputs
    from setup import DEFAULT_CONFIG, FractureCase, StudyConfig, StudyGrid, make_cases, make_study_grid


@attrs.define(frozen=True)
class DirectCaseResult:
    """Method 1 outputs for one deterministic fracture case."""

    tensor: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    area_fraction: np.ndarray
    volume_fraction: np.ndarray
    area_inside: np.ndarray
    area_inside_coarse: np.ndarray
    clipping_error: np.ndarray


@attrs.define(frozen=True)
class BlobCaseResult:
    """Method 2 outputs for one deterministic fracture case."""

    tensor: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    micro_mesh_vtu_path: str


@attrs.define(frozen=True)
class CaseResult:
    """Combined outputs for one fracture case."""

    case: FractureCase
    direct: DirectCaseResult
    blob: BlobCaseResult


@attrs.define
class SingleFractureStudy:
    """Own the study setup, execution order, and aggregated outputs."""
    # AGENT: do not store cfg once you use it to setup the study
    cfg: StudyConfig

    # Grid and cases should rather be properties  build on the actual dataclasses
    # parametrizing the study
    grid: StudyGrid
    cases: list[FractureCase]
    case_results: list[CaseResult] = attrs.field(factory=list)

    @classmethod
    def from_config(cls, cfg: StudyConfig) -> "SingleFractureStudy":
        """Build the deterministic grid and fracture cases from one config."""
        return cls(
            cfg=cfg,
            grid=make_study_grid(cfg.centers_1d),
            cases=make_cases(cfg),
        )

    def run(self) -> None:
        """Run all configured methods for every fracture case."""
        self.prepare_output_dir()
        self.case_results.clear()

        #AGENT: use comprehansion and turn cases_results into cached_property
        for case in self.cases:
            self.case_results.append(self.run_case(case))

    # AGENT: put this into main, as that is preparation of environment not running the study
    def prepare_output_dir(self) -> None:
        """Reset the fixed study output directory to avoid stale artifacts."""
        if self.cfg.output_dir.exists():
            shutil.rmtree(self.cfg.output_dir)
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def run_case(self, case: FractureCase) -> CaseResult:
        """Run Method 1 and Method 2 for one fracture case."""
        direct = run_direct_case_estimate(self.cfg, self.grid, case)
        blob = blob_case_result_from_mapping(run_blob_case_estimate(self.cfg, self.grid, case))
        return CaseResult(case=case, direct=direct, blob=blob)

    def write_outputs(self) -> None:
        """Write the global study outputs after all case results are collected."""
        write_outputs(self.cfg, self.grid, self.cases, self.aggregate_results())

    def aggregate_results(self) -> dict[str, np.ndarray | list[str]]:
        """Convert typed case results into dense arrays for summary writers."""
        direct_results = [case_result.direct for case_result in self.case_results]
        blob_results = [case_result.blob for case_result in self.case_results]
        return {
            "tensor": stack_case_attr(direct_results, "tensor"),
            "eigenvalues": stack_case_attr(direct_results, "eigenvalues"),
            "eigenvectors": stack_case_attr(direct_results, "eigenvectors"),
            "area_fraction": stack_case_attr(direct_results, "area_fraction"),
            "volume_fraction": stack_case_attr(direct_results, "volume_fraction"),
            "area_inside": stack_case_attr(direct_results, "area_inside"),
            "area_inside_coarse": stack_case_attr(direct_results, "area_inside_coarse"),
            "clipping_error": stack_case_attr(direct_results, "clipping_error"),
            "blob_tensor": stack_case_attr(blob_results, "tensor"),
            "blob_eigenvalues": stack_case_attr(blob_results, "eigenvalues"),
            "blob_eigenvectors": stack_case_attr(blob_results, "eigenvectors"),
            "method2_micro_mesh_vtu_paths": [blob.micro_mesh_vtu_path for blob in blob_results],
        }


def main() -> None:
    """Run the whole single-fracture study for one config file."""
    parser = argparse.ArgumentParser()
    # AGENT: just single argument the config file.
    # Resolved: the driver now accepts only the config path, as a positional argument with the local default.
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = StudyConfig.from_yaml(args.config)
    study = SingleFractureStudy.from_config(cfg)
    study.run()
    study.write_outputs()


def blob_case_result_from_mapping(case_results: dict[str, np.ndarray | str]) -> BlobCaseResult:
    """Convert the Method 2 helper mapping to the typed case result model."""
    return BlobCaseResult(
        tensor=np.asarray(case_results["blob_tensor"]),
        eigenvalues=np.asarray(case_results["blob_eigenvalues"]),
        eigenvectors=np.asarray(case_results["blob_eigenvectors"]),
        micro_mesh_vtu_path=str(case_results["method2_micro_mesh_vtu_path"]),
    )


def stack_case_attr(case_results: list[object], name: str) -> np.ndarray:
    """Stack one ndarray attribute from all case result objects."""
    return np.stack([np.asarray(getattr(case_result, name)) for case_result in case_results], axis=0)


def run_direct_case_estimate(
    cfg: StudyConfig,
    grid: StudyGrid,
    case: FractureCase,
) -> DirectCaseResult:
    """Compute Method 1 tensors and diagnostics for one fracture case."""
    points = fracture_sample_points(case, cfg.clipping_resolution)
    coarse_points = fracture_sample_points(case, cfg.convergence_resolution)
    inside = block_incidence(points, grid.centers, cfg.block_size)
    inside_coarse = block_incidence(coarse_points, grid.centers, cfg.block_size)

    area_fraction = inside.mean(axis=0)
    coarse_fraction = inside_coarse.mean(axis=0)
    area_inside = area_fraction * case.side_length * case.side_length
    area_inside_coarse = coarse_fraction * case.side_length * case.side_length
    clipping_error = np.abs(area_fraction - coarse_fraction)
    volume_fraction = area_inside * cfg.aperture / (cfg.block_size ** 3)
    tensor = np.stack([direct_tensor(cfg, case, volume_fraction_one) for volume_fraction_one in volume_fraction], axis=0)
    eigenvalues, eigenvectors = sorted_eigensystem(tensor)

    return DirectCaseResult(
        tensor=tensor,
        # Do not store eigen vectors and tensors, make the base class for the
        # case and implement cached properties for them computed from the tensor
        # array on the grid
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        area_fraction=area_fraction,
        volume_fraction=volume_fraction,
        area_inside=area_inside,
        area_inside_coarse=area_inside_coarse,
        clipping_error=clipping_error,
    )


def direct_tensor(cfg: StudyConfig, case: FractureCase, volume_fraction: float) -> np.ndarray:
    """Return the Method 1 direct conductivity tensor for one block."""
    normal_part = np.outer(case.normal, case.normal)
    tangent_projector = np.eye(3) - normal_part
    area_fraction = volume_fraction * cfg.block_size / cfg.aperture
    area_fraction = min(max(area_fraction, 0.0), 1.0)
    k_homo = cfg.fracture_conductivity * cfg.aperture / cfg.block_size
    tangent_cond = cfg.bulk_conductivity + area_fraction * (k_homo - cfg.bulk_conductivity)
    normal_cond = cfg.bulk_conductivity
    return normal_cond * normal_part + tangent_cond * tangent_projector


def sorted_eigensystem(tensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenpairs sorted by descending eigenvalue for one case tensor field."""
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    order = np.argsort(eigenvalues, axis=1)[:, ::-1]
    return (
        np.take_along_axis(eigenvalues, order, axis=1),
        np.take_along_axis(eigenvectors, order[:, None, :], axis=2),
    )


# AGENT: create a class for the whole study in order to
# be able to add to common list of the results,
# However each item must be a proper dataclass object to have a clear data model.
# Resolved: `SingleFractureStudy` accumulates typed `CaseResult` objects and only builds dense arrays for final
# summary writers.

# AGENT: No point in making these here and passing to run_study, jut call the functions
# when needed; also write should be named more specificaly,
# individual results should be written continuously.
# This collective write should anly be used for global zarr, etc.
# and should be method of the study class.
# Resolved: grid/case construction moved into `SingleFractureStudy.from_config()` and aggregate output writing is
# owned by `SingleFractureStudy.write_outputs()`.

# AGENT: run method2 alvais.
# Resolved: `SingleFractureStudy.run_case()` executes Method 1 and Method 2 for every configured case.

# AGENT: Results should be just a list, and you can replace thi function by extend call.
# Resolved: results are stored as `case_results: list[CaseResult]`; dense study arrays are derived from that list.

# AGENT: do notimplement following functions these are already implement in bgem
# Resolved: fracture orientation, shape rotation, and corners now come from bgem `Fracture`; this helper only
# samples points over the bgem-derived tangent basis for numerical clipping.
def fracture_sample_points(case: FractureCase, resolution: int) -> np.ndarray:
    """Sample fracture-square cell centers for numerical clipping."""
    # AGENT: do not
    offsets = (np.arange(resolution, dtype=float) + 0.5) / resolution - 0.5
    uu, vv = np.meshgrid(offsets * case.side_length, offsets * case.side_length, indexing="ij")
    return case.fracture.center + uu.reshape(-1, 1) * case.tangent_u + vv.reshape(-1, 1) * case.tangent_v


def block_incidence(points: np.ndarray, centers: np.ndarray, block_size: float) -> np.ndarray:
    """Return a boolean matrix ``(n_points, n_blocks)`` for point-in-block tests."""
    half = block_size / 2.0
    lower = centers - half
    upper = centers + half
    return np.all((points[:, None, :] >= lower[None, :, :]) & (points[:, None, :] <= upper[None, :, :]), axis=2)


if __name__ == "__main__":
    main()
