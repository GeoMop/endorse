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
from vtk import VTK_VERTEX

try:
    from .setup import FractureCase, StudyConfig, StudyGrid
except ImportError:  # pragma: no cover - allows direct script execution imports.
    from setup import FractureCase, StudyConfig, StudyGrid


def write_outputs(
    cfg: StudyConfig,
    grid: StudyGrid,
    cases: list[FractureCase],
    results: dict[str, object],
    write_zarr: bool = True,
) -> None:
    """Write all aggregated study outputs."""
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
            **optional_data_vars(results),
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


def optional_data_vars(results: dict[str, object]) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Return optional xarray variables for methods beyond Method 1."""
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    if "blob_tensor" in results:
        data_vars["blob_tensor"] = (("case", "block", "axis_i", "axis_j"), results["blob_tensor"])
        data_vars["blob_eigenvalues"] = (("case", "block", "eigen_rank"), results["blob_eigenvalues"])
        data_vars["blob_eigenvectors"] = (("case", "block", "axis_i", "eigen_rank"), results["blob_eigenvectors"])
    return data_vars


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
    cfg: StudyConfig, grid: StudyGrid, cases: list[FractureCase], results: dict[str, object]
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
            "blob_eig_0",
            "blob_eig_1",
            "blob_eig_2",
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
                        "blob_eig_0": optional_scalar(results, "blob_eigenvalues", i_case, i_block, 0),
                        "blob_eig_1": optional_scalar(results, "blob_eigenvalues", i_case, i_block, 1),
                        "blob_eig_2": optional_scalar(results, "blob_eigenvalues", i_case, i_block, 2),
                    }
                )


def optional_scalar(
    results: dict[str, object],
    name: str,
    i_case: int,
    i_block: int,
    i_value: int,
) -> float | str:
    """Return an optional scalar result for the CSV writer."""
    if name not in results:
        return ""
    return float(results[name][i_case, i_block, i_value])


def write_case_vtk_files(
    cfg: StudyConfig, grid: StudyGrid, cases: list[FractureCase], results: dict[str, object]
) -> None:
    """Write one macro-grid VTU diagnostics file per fracture case."""
    for case in cases:
        case_dir = cfg.output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(case_metadata(case, grid), handle, indent=2)

        macro_grid = macro_tensor_grid(grid, case, results)
        macro_grid.save(case_dir / "diagnostics.vtu")


def macro_tensor_grid(grid: StudyGrid, case: FractureCase, results: dict[str, object]) -> pv.UnstructuredGrid:
    """Return a single-vertex-cell VTU grid carrying macro tensor diagnostics."""
    vertices = np.column_stack(
        [np.ones(len(grid.centers), dtype=np.int64), np.arange(len(grid.centers), dtype=np.int64)]
    ).ravel()
    cell_types = np.full(len(grid.centers), VTK_VERTEX, dtype=np.uint8)
    macro_grid = pv.UnstructuredGrid(vertices, cell_types, grid.centers)
    macro_grid.point_data["direct_tensor"] = results["tensor"][case.case_id].reshape(len(grid.centers), 9)
    macro_grid.point_data["direct_eigenvalues"] = results["eigenvalues"][case.case_id]
    macro_grid.point_data["block_index"] = grid.indices.astype(np.int32)
    macro_grid.field_data["case_id"] = np.array([case.case_id], dtype=np.int32)
    macro_grid.field_data["side_length"] = np.array([case.side_length], dtype=float)
    macro_grid.field_data["rotation_deg"] = np.array([case.rotation_deg], dtype=float)
    macro_grid.field_data["normal"] = case.normal.reshape(1, 3)
    if "blob_tensor" in results:
        macro_grid.point_data["blob_tensor"] = results["blob_tensor"][case.case_id].reshape(len(grid.centers), 9)
        macro_grid.point_data["blob_eigenvalues"] = results["blob_eigenvalues"][case.case_id]
    return macro_grid


def method2_micro_mesh(case: FractureCase, results: dict[str, object]) -> pv.DataSet | None:
    """Read the Method 2 micro mesh for a case when it is available."""
    paths = results.get("method2_micro_mesh_vtu_paths")
    if paths is None:
        return None
    path = paths[case.case_id]
    if path is None:
        return None
    return pv.read(path)


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
