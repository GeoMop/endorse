"""Contracts for the independent homogenized-conductivity study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from homogenization_study import ConductivityFieldResult
from homogenization_study import job
from homogenization_study.create_mesh import create_study_meshes
from homogenization_study.model import TENSOR_COMPONENTS, XYZ_COMPONENTS
from homogenization_study.problem import run_homogenization_problem


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
    assert job.output.zarr_store_path == tmp_path / "homogenization_sampling"
    assert job.output.mlmc_hdf_path == tmp_path / "homogenization_mlmc.hdf"
    assert job.output.homogenization_mesh_dir == tmp_path / "homogenization_mesh"
    assert job.output.pbs_script == tmp_path / "homogenization_sampling.pbs"


@pytest.mark.skip(reason="USER: deterministic study meshes and three-load homogenization are pending")
def test_deterministic_homogenization_problem(tmp_path: Path) -> None:
    """Build both meshes and return a valid field from exactly three load cases."""
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
