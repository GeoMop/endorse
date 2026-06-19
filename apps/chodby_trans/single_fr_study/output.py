"""Output helpers for the single-fracture study."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
import xarray as xr

try:
    from .setup import FractureCase, StudyConfig, StudyGrid
except ImportError:  # pragma: no cover - allows direct script execution imports.
    from setup import FractureCase, StudyConfig, StudyGrid


def write_outputs(
    cfg: StudyConfig,
    grid: StudyGrid,
    cases: list[FractureCase],
    results: dict[str, np.ndarray],
    write_zarr: bool = True,
) -> None:
    """Write all first-pass Method 1 outputs."""
    dataset = results_dataset(cfg, grid, cases, results)
    write_summary_csv(cfg, grid, cases, results)
    if cfg.vtk_enabled:
        write_case_vtk_files(cfg, grid, cases, results)
    if write_zarr:
        write_zarr_store(cfg, dataset)


# AGENT: do not implement zarr write and vtk write, use xarray, pyvista etc.
# Resolved: Zarr output uses `xarray.Dataset.to_zarr`; VTK output uses pyvista datasets.
def results_dataset(
    cfg: StudyConfig, grid: StudyGrid, cases: list[FractureCase], results: dict[str, np.ndarray]
) -> xr.Dataset:
    """Build an xarray dataset for all Method 1 numerical results."""
    case_ids = np.array([case.case_id for case in cases], dtype=np.int32)
    block_ids = np.arange(len(grid.centers), dtype=np.int32)
    coords = {
        "case": case_ids,
        "block": block_ids,
        "axis_i": ["x", "y", "z"],
        "axis_j": ["x", "y", "z"],
        "eigen_rank": [0, 1, 2],
        "corner": [0, 1, 2, 3],
        "xyz": ["x", "y", "z"],
    }
    ds = xr.Dataset(
        data_vars={
            "tensor": (("case", "block", "axis_i", "axis_j"), results["tensor"]),
            "eigenvalues": (("case", "block", "eigen_rank"), results["eigenvalues"]),
            "eigenvectors": (("case", "block", "axis_i", "eigen_rank"), results["eigenvectors"]),
            "area_fraction": (("case", "block"), results["area_fraction"]),
            "volume_fraction": (("case", "block"), results["volume_fraction"]),
            "area_inside": (("case", "block"), results["area_inside"]),
            "area_inside_coarse": (("case", "block"), results["area_inside_coarse"]),
            "clipping_error": (("case", "block"), results["clipping_error"]),
            "case_side_length": ("case", np.array([case.side_length for case in cases])),
            "case_normal": (("case", "xyz"), np.array([case.normal for case in cases])),
            "case_normal_raw": (("case", "xyz"), np.array([case.normal_raw for case in cases])),
            "case_rotation_deg": ("case", np.array([case.rotation_deg for case in cases])),
            "case_tangent_u": (("case", "xyz"), np.array([case.tangent_u for case in cases])),
            "case_tangent_v": (("case", "xyz"), np.array([case.tangent_v for case in cases])),
            "case_corners": (("case", "corner", "xyz"), np.array([case.corners for case in cases])),
            "block_center": (("block", "xyz"), grid.centers),
            "block_index": (("block", "xyz"), grid.indices.astype(np.int32)),
        },
        coords=coords,
        attrs={
            "methods": "direct",
            "block_names_json": json.dumps(grid.names),
            "tensor_component_order": "axis_i_axis_j",
            "eigenvalue_order": "largest_to_smallest",
            "clipping_resolution": cfg.clipping_resolution,
            "convergence_resolution": cfg.convergence_resolution,
        },
    )
    return ds


def write_zarr_store(cfg: StudyConfig, dataset: xr.Dataset) -> None:
    """Write the xarray dataset to a Zarr store."""
    if cfg.zarr_store.exists():
        shutil.rmtree(cfg.zarr_store)
    cfg.zarr_store.parent.mkdir(parents=True, exist_ok=True)
    process = mp.Process(target=_write_zarr_worker, args=(dataset, cfg.zarr_store))
    process.start()
    process.join(cfg.zarr_write_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"xarray Dataset.to_zarr timed out after {cfg.zarr_write_timeout_seconds}s. "
            "Run env_check.py to verify xarray/zarr compatibility."
        )
    if process.exitcode != 0:
        raise RuntimeError(f"xarray Dataset.to_zarr failed with exit code {process.exitcode}")


def _write_zarr_worker(dataset: xr.Dataset, zarr_store: Path) -> None:
    """Write Zarr in a subprocess so incompatible xarray/zarr stacks cannot hang forever."""
    dataset.to_zarr(zarr_store, mode="w", consolidated=True, zarr_format=2)


def write_summary_csv(
    cfg: StudyConfig, grid: StudyGrid, cases: list[FractureCase], results: dict[str, np.ndarray]
) -> None:
    """Write a flat CSV summary for quick inspection."""
    cfg.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with cfg.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case_id",
            "size",
            "normal_id",
            "rotation_deg",
            "block_name",
            "block_x",
            "block_y",
            "block_z",
            "area_fraction",
            "volume_fraction",
            "clipping_error",
            "eig_0",
            "eig_1",
            "eig_2",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i_case, case in enumerate(cases):
            for i_block, center in enumerate(grid.centers):
                writer.writerow(
                    {
                        "case_id": case.case_id,
                        "size": case.side_length,
                        "normal_id": case.normal_id,
                        "rotation_deg": case.rotation_deg,
                        "block_name": grid.names[i_block],
                        "block_x": center[0],
                        "block_y": center[1],
                        "block_z": center[2],
                        "area_fraction": results["area_fraction"][i_case, i_block],
                        "volume_fraction": results["volume_fraction"][i_case, i_block],
                        "clipping_error": results["clipping_error"][i_case, i_block],
                        "eig_0": results["eigenvalues"][i_case, i_block, 0],
                        "eig_1": results["eigenvalues"][i_case, i_block, 1],
                        "eig_2": results["eigenvalues"][i_case, i_block, 2],
                    }
                )


def write_case_vtk_files(
    cfg: StudyConfig, grid: StudyGrid, cases: list[FractureCase], results: dict[str, np.ndarray]
) -> None:
    """Write one pyvista VTK multiblock diagnostics file per fracture case."""
    micro_points = regular_points(cfg.extended_box, cfg.vtk_micro_grid_1d_count)
    micro_incidence = block_incidence(micro_points, grid.centers, cfg.block_size)
    for case in cases:
        case_dir = cfg.output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(case_metadata(case, grid), handle, indent=2)

        micro_cloud = pv.PolyData(micro_points)
        for i_block, block_name in enumerate(grid.names):
            micro_cloud.point_data[block_name] = micro_incidence[:, i_block].astype(np.uint8)

        macro_points = pv.PolyData(grid.centers)
        tensor_components = results["tensor"][case.case_id].reshape(len(grid.centers), 9)
        for i_comp, comp_name in enumerate(["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]):
            macro_points.point_data[f"direct_K_{comp_name}"] = tensor_components[:, i_comp]
        for i_eig in range(3):
            macro_points.point_data[f"direct_eig_{i_eig}"] = results["eigenvalues"][case.case_id, :, i_eig]

        multiblock = pv.MultiBlock(
            {
                "basic_domain": box_mesh(cfg.basic_box),
                "extended_domain": box_mesh(cfg.extended_box),
                "fracture_polygon": fracture_polygon(case.corners),
                "micro_points_block_incidence": micro_cloud,
                "macro_grid_direct_tensors": macro_points,
            }
        )
        multiblock.save(case_dir / "diagnostics.vtm")


def case_metadata(case: FractureCase, grid: StudyGrid) -> dict[str, object]:
    """Return JSON metadata for one VTK diagnostic output."""
    return {
        "case_id": case.case_id,
        "side_length": case.side_length,
        "normal": case.normal.tolist(),
        "rotation_deg": case.rotation_deg,
        "block_names": grid.names,
        "block_centers": grid.centers.tolist(),
    }


def regular_points(box: np.ndarray, count_1d: int) -> np.ndarray:
    """Return a regular point cloud over a box for VTK incidence diagnostics."""
    axes = [np.linspace(box[0, axis], box[1, axis], count_1d) for axis in range(3)]
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def block_incidence(points: np.ndarray, centers: np.ndarray, block_size: float) -> np.ndarray:
    """Return a boolean matrix ``(n_points, n_blocks)`` for point-in-block tests."""
    half = block_size / 2.0
    lower = centers - half
    upper = centers + half
    return np.all((points[:, None, :] >= lower[None, :, :]) & (points[:, None, :] <= upper[None, :, :]), axis=2)


def box_mesh(box: np.ndarray) -> pv.PolyData:
    """Return a pyvista box mesh from a min/max box array."""
    return pv.Box(bounds=(box[0, 0], box[1, 0], box[0, 1], box[1, 1], box[0, 2], box[1, 2]))


def fracture_polygon(corners: np.ndarray) -> pv.PolyData:
    """Return a pyvista polygon mesh for the square fracture."""
    faces = np.array([4, 0, 1, 2, 3])
    return pv.PolyData(corners, faces)
