"""Run Method 1 direct estimates for the single-fracture study."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

try:
    from .method2 import run_blob_case_estimate
    from .output import write_outputs
    from .setup import DEFAULT_CONFIG, FractureCase, StudyConfig, StudyGrid, make_cases, make_study_grid
except ImportError:  # pragma: no cover - allows direct script execution.
    from method2 import run_blob_case_estimate
    from output import write_outputs
    from setup import DEFAULT_CONFIG, FractureCase, StudyConfig, StudyGrid, make_cases, make_study_grid


def main() -> None:
    """Run Method 1 direct estimates for all configured fracture cases."""
    # AGENT:  just single argument the config file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-zarr", action="store_true")
    parser.add_argument("--include-method2", action="store_true")
    parser.add_argument("--max-method2-cases", type=int, default=None)
    args = parser.parse_args()

    cfg = StudyConfig.from_yaml(args.config)
    if args.clean and cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # AGENT: No point in making these here and passing to run_study, jut call the functions
    # when needed; also write should be named more specificaly,
    # individual results should be written continuously.
    # This collective write should anly be used for global zarr, etc.
    # and should be method of the study class.
    grid = make_study_grid(cfg.centers_1d)
    cases = make_cases(cfg)
    results = run_study(cfg, grid, cases, include_method2=args.include_method2, max_method2_cases=args.max_method2_cases)
    write_outputs(cfg, grid, cases, results, write_zarr=not args.skip_zarr)

# AGENT: create a class for the whole study in order to
# be able to add to common list of the results,
# However each item must be a proper dataclass object to have a clear data model.
def run_study(
    cfg: StudyConfig,
    grid: StudyGrid,
    cases: list[FractureCase],
    include_method2: bool = False,
    max_method2_cases: int | None = None,
) -> dict[str, object]:
    """Run all enabled study methods for each fracture case."""
    n_cases = len(cases)
    n_blocks = len(grid.centers)
    results = initialize_results(n_cases, n_blocks, include_method2)
    method2_case_limit = n_cases if max_method2_cases is None else min(max_method2_cases, n_cases)

    for case in cases:
        case_id = case.case_id
        direct_results = run_direct_case_estimate(cfg, grid, case)
        store_case_results(results, direct_results, case_id)
        # AGENT: run method2 alvais.
        if include_method2 and case_id < method2_case_limit:
            blob_results = run_blob_case_estimate(cfg, grid, case)
            store_case_results(results, blob_results, case_id)

    return results


def initialize_results(n_cases: int, n_blocks: int, include_method2: bool) -> dict[str, object]:
    """Allocate result arrays for all enabled study methods."""
    results: dict[str, object] = {
        "tensor": np.empty((n_cases, n_blocks, 3, 3), dtype=float),
        "eigenvalues": np.empty((n_cases, n_blocks, 3), dtype=float),
        "eigenvectors": np.empty((n_cases, n_blocks, 3, 3), dtype=float),
        "area_fraction": np.empty((n_cases, n_blocks), dtype=float),
        "volume_fraction": np.empty((n_cases, n_blocks), dtype=float),
        "area_inside": np.empty((n_cases, n_blocks), dtype=float),
        "area_inside_coarse": np.empty((n_cases, n_blocks), dtype=float),
        "clipping_error": np.empty((n_cases, n_blocks), dtype=float),
    }
    if include_method2:
        results.update(
            {
                "blob_tensor": np.full((n_cases, n_blocks, 3, 3), np.nan, dtype=float),
                "blob_eigenvalues": np.full((n_cases, n_blocks, 3), np.nan, dtype=float),
                "blob_eigenvectors": np.full((n_cases, n_blocks, 3, 3), np.nan, dtype=float),
                "method2_micro_mesh_vtu_paths": [None] * n_cases,
            }
        )
    return results

# AGENT: Results should be just a list, and you can replace thi function by extend call.
def store_case_results(results: dict[str, object], case_results: dict[str, object], case_id: int) -> None:
    """Store one case result mapping into the full study result arrays."""
    for name, value in case_results.items():
        if name == "method2_micro_mesh_vtu_path":
            results["method2_micro_mesh_vtu_paths"][case_id] = value
            continue
        if name not in results:
            continue
        results[name][case_id] = value


def run_direct_case_estimate(
    cfg: StudyConfig,
    grid: StudyGrid,
    case: FractureCase,
) -> dict[str, np.ndarray]:
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
    tensor = np.stack([direct_tensor(cfg, case, vf) for vf in volume_fraction], axis=0)
    eigenvalues, eigenvectors = sorted_eigensystem(tensor)

    return {
        "tensor": tensor,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "area_fraction": area_fraction,
        "volume_fraction": volume_fraction,
        "area_inside": area_inside,
        "area_inside_coarse": area_inside_coarse,
        "clipping_error": clipping_error,
    }


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


# AGENT: do notimplement following functions these are already implement in bgem
# Resolved: fracture orientation, shape rotation, and corners now come from bgem `Fracture`; this helper only
# samples points over the bgem-derived tangent basis for numerical clipping.
def fracture_sample_points(case: FractureCase, resolution: int) -> np.ndarray:
    """Sample fracture-square cell centers for numerical clipping."""
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
