"""Run Method 1 direct estimates for the single-fracture study."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

try:
    from .output import write_outputs
    from .setup import DEFAULT_CONFIG, FractureCase, StudyConfig, StudyGrid, make_cases, make_study_grid
except ImportError:  # pragma: no cover - allows direct script execution.
    from output import write_outputs
    from setup import DEFAULT_CONFIG, FractureCase, StudyConfig, StudyGrid, make_cases, make_study_grid


def main() -> None:
    """Run Method 1 direct estimates for all configured fracture cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-zarr", action="store_true")
    args = parser.parse_args()

    cfg = StudyConfig.from_yaml(args.config)
    if args.clean and cfg.output_dir.exists():
        shutil.rmtree(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    grid = make_study_grid(cfg.centers_1d)
    cases = make_cases(cfg)
    results = run_direct_estimates(cfg, grid, cases)
    write_outputs(cfg, grid, cases, results, write_zarr=not args.skip_zarr)


def run_direct_estimates(
    cfg: StudyConfig, grid: StudyGrid, cases: list[FractureCase]
) -> dict[str, np.ndarray]:
    """Compute Method 1 tensors and diagnostics for all cases and blocks."""
    n_cases = len(cases)
    n_blocks = len(grid.centers)
    tensors = np.empty((n_cases, n_blocks, 3, 3), dtype=float)
    eigenvalues = np.empty((n_cases, n_blocks, 3), dtype=float)
    eigenvectors = np.empty((n_cases, n_blocks, 3, 3), dtype=float)
    area_fraction = np.empty((n_cases, n_blocks), dtype=float)
    volume_fraction = np.empty((n_cases, n_blocks), dtype=float)
    area_inside = np.empty((n_cases, n_blocks), dtype=float)
    area_inside_coarse = np.empty((n_cases, n_blocks), dtype=float)
    clipping_error = np.empty((n_cases, n_blocks), dtype=float)

    for i_case, case in enumerate(cases):
        points = fracture_sample_points(case, cfg.clipping_resolution)
        coarse_points = fracture_sample_points(case, cfg.convergence_resolution)
        inside = block_incidence(points, grid.centers, cfg.block_size)
        inside_coarse = block_incidence(coarse_points, grid.centers, cfg.block_size)

        area_fraction[i_case, :] = inside.mean(axis=0)
        coarse_fraction = inside_coarse.mean(axis=0)
        area_inside[i_case, :] = area_fraction[i_case, :] * case.side_length * case.side_length
        area_inside_coarse[i_case, :] = coarse_fraction * case.side_length * case.side_length
        clipping_error[i_case, :] = np.abs(area_fraction[i_case, :] - coarse_fraction)
        volume_fraction[i_case, :] = area_inside[i_case, :] * cfg.aperture / (cfg.block_size ** 3)

        for i_block in range(n_blocks):
            tensors[i_case, i_block] = direct_tensor(cfg, case, volume_fraction[i_case, i_block])
            evals, evecs = np.linalg.eigh(tensors[i_case, i_block])
            order = np.argsort(evals)[::-1]
            eigenvalues[i_case, i_block] = evals[order]
            eigenvectors[i_case, i_block] = evecs[:, order]

    return {
        "tensor": tensors,
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
