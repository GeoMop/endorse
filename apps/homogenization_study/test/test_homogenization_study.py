"""Contracts for the independent homogenized-conductivity study."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import subprocess

import numpy as np
import pytest
import xarray as xr

from endorse import common
from homogenization_study import ConductivityFieldResult
from homogenization_study import job
from homogenization_study.model import TENSOR_COMPONENTS, XYZ_COMPONENTS
from homogenization_study.problem_simulation import FIELD_GROUP, ensure_field_storage, write_field_result
from homogenization_study.sensitivity_sampling import submit_pbs
from homogenization_study.zarr_to_fuse import convert_to_zarr_fuse, normalize_source_dataset


SCHEMA_PATH = Path(__file__).parents[1] / "input_data" / "data_schema.yaml"


def test_conductivity_field_result_matches_storage_axes() -> None:
    """Accept one finite six-component tensor and geometry row per macro element."""
    result = ConductivityFieldResult(
        conductivity=np.arange(18, dtype=float).reshape(3, 6),
        el_barycenter=np.arange(9, dtype=float).reshape(3, 3),
        gmsh_id=np.asarray([11, 17, 23], dtype=np.int64),
    )

    assert result.n_elements == 3
    np.testing.assert_array_equal(TENSOR_COMPONENTS, [1, 2, 3, 4, 5, 6])
    assert XYZ_COMPONENTS == ("x", "y", "z")


@pytest.mark.parametrize(
    ("conductivity", "el_barycenter", "gmsh_id", "message"),
    [
        (np.zeros((2, 9)), np.zeros((2, 3)), np.asarray([1, 2]), "conductivity"),
        (np.zeros((2, 6)), np.zeros((3, 3)), np.asarray([1, 2]), "el_barycenter"),
        (np.zeros((2, 6)), np.zeros((2, 3)), np.asarray([1, 1]), "unique"),
        (np.zeros((2, 6)), np.zeros((2, 3)), np.asarray([1.0, 2.0]), "integer"),
    ],
)
def test_conductivity_field_result_rejects_inconsistent_arrays(
    conductivity: np.ndarray,
    el_barycenter: np.ndarray,
    gmsh_id: np.ndarray,
    message: str,
) -> None:
    """Reject arrays that would make the sample and element axes ambiguous."""
    with pytest.raises(ValueError, match=message):
        ConductivityFieldResult(conductivity, el_barycenter, gmsh_id)


def test_study_artifact_paths_are_self_contained(tmp_path: Path) -> None:
    """Resolve all persistent artifacts below the independent study workdir."""
    job.set_workdir(tmp_path)

    assert job.input.study_cfg_path == tmp_path / "input_data" / "homogenization_study.yaml"
    assert job.input.data_schema_yaml == tmp_path / "input_data" / "data_schema.yaml"
    assert job.output.zarr_store_path == tmp_path / "homogenization_sampling"
    assert job.output.mlmc_hdf_path == tmp_path / "homogenization_mlmc.hdf"
    assert job.output.homogenization_mesh_dir == tmp_path / "homogenization_mesh"
    assert job.output.pbs_script == tmp_path / "homogenization_sampling.pbs"


def test_pbs_submission_runs_sampling_through_dask_cluster(tmp_path: Path) -> None:
    """Start, use, and stop the multi-node Dask runner in a generated PBS job."""
    job.set_workdir(tmp_path)
    machine_cfg = {
        "pbs": {
            "n_nodes": 2,
            "n_cores": 4,
            "mem": "16gb",
            "scratch": "20gb",
            "walltime": "01:00:00",
            "queue": "default",
            "pbs_name": "homogenization",
        }
    }

    script_path = submit_pbs(common.dotdict.create(machine_cfg), cmd="meta", submit=False)
    script = script_path.read_text(encoding="utf-8")

    assert 'bash "$DASK_RUNNER" "$OUTPUT_DIR" start' in script
    assert 'bash "$DASK_RUNNER" "$OUTPUT_DIR" run meta' in script
    assert 'bash "$DASK_RUNNER" "$OUTPUT_DIR" stop' in script
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_schema_storage_and_zarr_fuse_conversion(tmp_path: Path) -> None:
    """Use the shared tensor schema for local storage and the independent Fuse export."""
    source_store = tmp_path / "source.zarr"
    destination_store = tmp_path / "fused.zarr"
    barycenters = np.arange(15, dtype=float).reshape(5, 3)
    gmsh_ids = np.arange(101, 106, dtype=np.int64)
    ensure_field_storage(
        source_store,
        schema_path=SCHEMA_PATH,
        n_samples=2,
        n_elements=5,
        parameter_names=["alpha", "beta"],
        el_barycenter=barycenters,
        gmsh_id=gmsh_ids,
    )
    conductivity = np.arange(30, dtype=float).reshape(5, 6)
    write_field_result(
        source_store,
        sample_id=1,
        conductivity=conductivity,
        parameter_values=[0.25, 0.75],
        return_code=0,
        eval_time=12.5,
    )

    source = xr.open_zarr(source_store, group=FIELD_GROUP, consolidated=False)
    assert source["conductivity"].dims == ("sample_id", "el_id", "tensor_comp")
    assert source["conductivity"].encoding["chunks"] == (1, 5, 6)
    assert source["parameter"].dims == ("sample_id", "param_name")
    np.testing.assert_array_equal(source["gmsh_id"], gmsh_ids)
    np.testing.assert_allclose(source["conductivity"].isel(sample_id=1), conductivity)

    fused = convert_to_zarr_fuse(
        source_store,
        SCHEMA_PATH,
        store_url=str(destination_store),
        overwrite=True,
    )
    assert set(fused.data_vars) == set(source.data_vars)
    np.testing.assert_allclose(fused["conductivity"], source["conductivity"])

    legacy = source.rename({"parameter": "param_val", "param_name": "parameter"})
    normalized = normalize_source_dataset(legacy)
    assert normalized["parameter"].dims == ("sample_id", "param_name")


@pytest.mark.skip(reason="integration test requires a configured Flow123d run")
def test_deterministic_homogenization_problem(tmp_path: Path) -> None:
    """Build both meshes and return a valid field from exactly three load cases."""
    create_study_meshes = getattr(
        import_module("homogenization_study.create_mesh"),
        "create_study_meshes",
    )
    run_homogenization_problem = getattr(
        import_module("homogenization_study.problem"),
        "run_homogenization_problem",
    )

    cfg = {
        "homogenization": {
            "pressure_loads": np.eye(3).tolist(),
            "tensor_components": 6,
        }
    }
    meshes = create_study_meshes(cfg, tmp_path / "meshes")
    result = run_homogenization_problem(cfg, tmp_path / "sample", meshes)

    assert meshes.micro_mesh.exists()
    assert meshes.homogenization_mesh.exists()
    assert result.n_elements > 0
