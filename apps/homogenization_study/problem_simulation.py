"""MLMC forward simulation and persistent field storage for Stage B."""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr
import yaml
from dask.distributed import Lock, get_client

from endorse import common
from endorse.common import dotdict
from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sim.simulation import Simulation

import homogenization_study.job as job
from homogenization_study.exception_wrapper import ReturnCode, WrapperException
from homogenization_study.create_mesh import create_micro_mesh, run_in_subprocess, update_dfn_params
from homogenization_study.model import ConductivityFieldResult, StudyMeshFiles
from homogenization_study.ot_sa import SensitivityAnalysis
from homogenization_study.problem import run_homogenization_problem


FIELD_GROUP = "tensors"


@run_in_subprocess
def run_sample_homogenization(
    cfg: dotdict,
    workdir: Path,
    meshes: StudyMeshFiles,
    fractures: Sequence[Any],
) -> ConductivityFieldResult:
    """Load meshes and run the three pressure loads on a process main thread."""
    return run_homogenization_problem(cfg, workdir, meshes, fractures=fractures)


def expand_sample_parameters(cfg: dotdict, parameters: Sequence[float]) -> np.ndarray:
    """Expand grouped unit-cube values to the full configured parameter vector."""
    analysis = SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    values = np.asarray(parameters, dtype=float)
    if values.size == len(analysis.parameters):
        return values
    if values.size != len(analysis.groups):
        raise ValueError(
            f"Expected {len(analysis.groups)} grouped or {len(analysis.parameters)} full parameters, "
            f"got {values.size}."
        )
    group_to_column = {group: index for index, group in enumerate(analysis.groups)}
    return np.asarray([
        parameter.map_from_group(np.asarray([values[group_to_column[parameter.group]]]))[0]
        for parameter in analysis.parameters.values()
    ], dtype=float)


def apply_sample_parameters(cfg: dotdict, parameters: Sequence[float]) -> tuple[dotdict, dict[str, float]]:
    """Apply a sampled vector to configuration paths and return its full values."""
    analysis = SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    full_values = expand_sample_parameters(cfg, parameters)
    parameter_dict = analysis.param_vec_to_dict(full_values)
    patch: dict[str, float] = {}
    for name, parameter_cfg in cfg.ot_sensitivity.parameters.items():
        for path in parameter_cfg.get("path", []):
            patch[str(path)] = parameter_dict[name]
    return common.apply_variant(cfg, patch), parameter_dict


def full_parameter_names(cfg: dotdict) -> list[str]:
    """Return parameter names in the canonical storage order."""
    return list(SensitivityAnalysis.from_cfg(cfg.ot_sensitivity).parameters.keys())


def full_parameter_values(cfg: dotdict, values: dict[str, float]) -> np.ndarray:
    """Convert a parameter dictionary to the canonical storage vector."""
    return np.asarray([values[name] for name in full_parameter_names(cfg)], dtype=float)


def parse_sample_workspace(path: str | Path) -> tuple[int, int]:
    """Extract MLMC level and sample ids from ``L00_S0000001`` directories."""
    for candidate in (Path(path), *Path(path).parents):
        parts = candidate.name.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("L") and parts[1].startswith("S"):
            return int(parts[0][1:]), int(parts[1][1:])
    raise ValueError(f"No MLMC sample workspace found in {path!s}.")


def _sample_seed(sample_id: str) -> int:
    """Return a stable integer seed independent of Python hash randomization."""
    return int(np.frombuffer(hashlib.md5(sample_id.encode("ascii")).digest(), dtype="uint32")[0])


def _sample_parameters(cfg: dotdict, sample_id: str) -> np.ndarray:
    """Generate reproducible grouped unit-cube parameters for one sample id."""
    analysis = SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    return np.random.default_rng(_sample_seed(sample_id)).random(len(analysis.groups))


def ensure_field_storage(
    store_path: str | Path,
    *,
    schema_path: str | Path,
    n_samples: int,
    n_elements: int,
    parameter_names: Sequence[str],
    el_barycenter: np.ndarray,
    gmsh_id: np.ndarray,
) -> None:
    """Create or validate the schema-defined, level-free scientific Zarr store."""
    path = Path(store_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(schema_path).open("r", encoding="utf-8") as schema_file:
        root_schema = yaml.safe_load(schema_file)
    field_schema = root_schema[FIELD_GROUP]
    variable_schema = field_schema["VARS"]
    coordinate_schema = field_schema["COORDS"]
    barycenter = np.asarray(el_barycenter, dtype=float)
    element_ids = np.asarray(gmsh_id, dtype=np.int64)
    names = np.asarray(parameter_names, dtype=str)
    if barycenter.shape != (n_elements, 3) or element_ids.shape != (n_elements,):
        raise ValueError("Homogenization geometry does not match the declared element count.")
    if path.exists():
        ds = xr.open_zarr(path, group=FIELD_GROUP, consolidated=False)
        if ds.sizes.get("el_id") != n_elements or not np.array_equal(ds.gmsh_id.values, element_ids):
            raise ValueError("Existing homogenization storage has incompatible element identity.")
        if not np.allclose(ds.el_barycenter.values, barycenter):
            raise ValueError("Existing homogenization storage has incompatible element geometry.")
        if ds.sizes.get("sample_id") != n_samples or ds.sizes.get("param_name") != len(names):
            raise ValueError("Existing homogenization storage has incompatible sample or parameter dimensions.")
        return

    def metadata(name: str, definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        definition = definitions[name]
        return {key: definition[key] for key in ("unit", "description") if key in definition}

    ds = xr.Dataset(
        data_vars={
            "conductivity": (("sample_id", "el_id", "tensor_comp"),
                             np.zeros((n_samples, n_elements, 6), dtype=float),
                             metadata("conductivity", variable_schema)),
            "el_barycenter": (("el_id", "xyz"), barycenter,
                              metadata("el_barycenter", variable_schema)),
            "gmsh_id": (("el_id",), element_ids, metadata("gmsh_id", variable_schema)),
            "parameter": (("sample_id", "param_name"), np.full((n_samples, len(names)), np.nan),
                          metadata("parameter", variable_schema)),
            "return_code": (("sample_id",), np.full(n_samples, ReturnCode.NONE, dtype=np.int32),
                            metadata("return_code", variable_schema)),
            "eval_time": (("sample_id",), np.full(n_samples, -1.0, dtype=float),
                          metadata("eval_time", variable_schema)),
        },
        coords={
            "sample_id": ("sample_id", np.arange(n_samples, dtype=np.int64),
                          metadata("sample_id", coordinate_schema)),
            "el_id": ("el_id", np.arange(n_elements, dtype=np.int64),
                      metadata("el_id", coordinate_schema)),
            "param_name": ("param_name", names, metadata("param_name", coordinate_schema)),
            "tensor_comp": ("tensor_comp", np.arange(1, 7, dtype=np.int8),
                            metadata("tensor_comp", coordinate_schema)),
            "xyz": ("xyz", np.asarray(["x", "y", "z"], dtype=str),
                    metadata("xyz", coordinate_schema)),
        },
        attrs=dict(field_schema.get("ATTRS", {})),
    )
    sample_chunk = min(int(coordinate_schema["sample_id"]["chunk_size"]), n_samples)
    element_chunk = min(int(coordinate_schema["el_id"]["chunk_size"]), n_elements)
    parameter_chunk = min(int(coordinate_schema["param_name"]["chunk_size"]), len(names))
    component_chunk = min(int(coordinate_schema["tensor_comp"]["chunk_size"]), 6)
    xyz_chunk = min(int(coordinate_schema["xyz"]["chunk_size"]), 3)
    encoding = {
        "conductivity": {"chunks": (sample_chunk, element_chunk, component_chunk)},
        "el_barycenter": {"chunks": (element_chunk, xyz_chunk)},
        "gmsh_id": {"chunks": (element_chunk,)},
        "parameter": {"chunks": (sample_chunk, parameter_chunk)},
        "return_code": {"chunks": (sample_chunk,)},
        "eval_time": {"chunks": (sample_chunk,)},
        "sample_id": {"chunks": (sample_chunk,)},
        "el_id": {"chunks": (element_chunk,)},
        "param_name": {"chunks": (parameter_chunk,)},
        "tensor_comp": {"chunks": (component_chunk,)},
        "xyz": {"chunks": (xyz_chunk,)},
    }
    ds.to_zarr(path, group=FIELD_GROUP, mode="w", consolidated=False, encoding=encoding)


def write_field_result(
    store_path: str | Path,
    *,
    sample_id: int,
    conductivity: np.ndarray | None,
    parameter_values: Sequence[float],
    return_code: int,
    eval_time: float,
) -> None:
    """Write one sample atomically into the level-free Zarr field store."""
    path = Path(store_path)
    ds = xr.open_zarr(path, group=FIELD_GROUP, consolidated=False)
    if not 0 <= int(sample_id) < ds.sizes["sample_id"]:
        raise IndexError(f"sample_id {sample_id} is outside the storage capacity.")
    tensor = np.zeros((ds.sizes["el_id"], 6), dtype=float)
    if conductivity is not None:
        tensor = np.asarray(conductivity, dtype=float)
        if tensor.shape != (ds.sizes["el_id"], 6):
            raise ValueError(f"Expected conductivity shape {(ds.sizes['el_id'], 6)}, got {tensor.shape}.")
    values = np.asarray(parameter_values, dtype=float)
    if values.shape != (ds.sizes["param_name"],):
        raise ValueError(f"Expected parameter vector shape {(ds.sizes['param_name'],)}, got {values.shape}.")

    try:
        get_client()
        lock = Lock(f"homogenization-field-sample-{int(sample_id)}")
        lock.acquire()
    except ValueError:
        lock = None
    try:
        region = {"sample_id": slice(int(sample_id), int(sample_id) + 1)}
        update = xr.Dataset({
            "conductivity": (("sample_id", "el_id", "tensor_comp"), tensor[None, ...]),
            "parameter": (("sample_id", "param_name"), values[None, ...]),
            "return_code": (("sample_id",), np.asarray([return_code], dtype=np.int32)),
            "eval_time": (("sample_id",), np.asarray([eval_time], dtype=float)),
        })
        update.to_zarr(path, group=FIELD_GROUP, mode="r+", region=region, consolidated=False)
    finally:
        if lock is not None:
            lock.release()


class HomogenizationLevelSimulation(LevelSimulation):
    """Level metadata that keeps deterministic parameter planning on the driver."""

    def prepare_samples(self, sample_ids: list[str]):
        cfg = dotdict.create(self.config_dict["root_cfg"])
        return [(sample_id, *_sample_parameters(cfg, sample_id)) for sample_id in sample_ids]


class HomogenizationSimulation(Simulation):
    """Single-level MLMC adapter for homogenized conductivity fields."""

    def __init__(self, cfg: dotdict, workdir: Path, meshes: Any, n_elements: int, store_path: Path):
        self.cfg = cfg
        self.workdir = Path(workdir)
        self.meshes = meshes
        self.n_elements = int(n_elements)
        self.store_path = Path(store_path)

    def level_instance(self, fine_level_params: list[float], coarse_level_params: list[float]) -> LevelSimulation:
        """Create the sole level; the coarse result is intentionally zero."""
        return HomogenizationLevelSimulation(
            config_dict={
                "root_cfg": dotdict.serialize(self.cfg),
                "meshes": {
                    "micro_mesh": str(self.meshes.micro_mesh),
                    "homogenization_mesh": str(self.meshes.homogenization_mesh),
                },
                "store_path": str(self.store_path),
                "workdir": str(self.workdir),
                "input_dir": str(job.input.dir_path),
                "n_elements": self.n_elements,
            },
            need_sample_workspace=True,
            task_size=float(self.cfg.mlmc.levels[0].task_size),
        )

    def result_format(self) -> list[QuantitySpec]:
        """Describe one conductivity tensor field with no level dimension."""
        return [QuantitySpec("conductivity", "m^2", (self.n_elements, 6), [0.0], ["homogenization"])]

    @staticmethod
    def calculate(config_dict: dict[str, Any], sample_input: Sequence[float]):
        """Run one sample and persist both success and failure metadata."""
        started = time.perf_counter()
        sample_level, sample_number = parse_sample_workspace(Path.cwd())
        job.set_workdir(Path(config_dict["workdir"]), Path(config_dict["input_dir"]))
        cfg = dotdict.create(copy.deepcopy(config_dict["root_cfg"]))
        cfg, parameter_dict = apply_sample_parameters(cfg, sample_input)
        parameter_values = full_parameter_values(cfg, parameter_dict)
        update_dfn_params(cfg.mesh.fractures, parameter_dict, Path(config_dict["input_dir"]))
        store_path = Path(config_dict["store_path"])
        try:
            fixed_meshes = StudyMeshFiles(**config_dict["meshes"])
            micro_path, fractures = create_micro_mesh(cfg, Path.cwd())
            meshes = StudyMeshFiles(micro_mesh=micro_path, homogenization_mesh=fixed_meshes.homogenization_mesh)
            result = run_sample_homogenization(cfg, Path.cwd(), meshes, fractures)
            fine = np.asarray(result.conductivity, dtype=float)
            code = ReturnCode.OK
        except Exception as error:
            logging.exception("Homogenization sample L%02d/S%07d failed.", sample_level, sample_number)
            fine = None
            code = int(error.code) if isinstance(error, WrapperException) else ReturnCode.HOMOGENIZATION_ERROR
        elapsed = time.perf_counter() - started
        write_field_result(store_path, sample_id=sample_number, conductivity=fine,
                           parameter_values=parameter_values, return_code=code, eval_time=elapsed)
        if fine is None:
            raise RuntimeError(f"Homogenization sample failed with return code {code}.")
        return fine, np.zeros_like(fine)


class RandomSimulation(HomogenizationSimulation):
    """Deterministic fast simulation for validating MLMC, restart, and storage paths."""

    @staticmethod
    def calculate(config_dict: dict[str, Any], sample_input: Sequence[float]):
        """Generate a reproducible tensor field without meshing or Flow123d."""
        started = time.perf_counter()
        _sample_level, sample_number = parse_sample_workspace(Path.cwd())
        cfg = dotdict.create(copy.deepcopy(config_dict["root_cfg"]))
        _cfg, parameter_dict = apply_sample_parameters(cfg, sample_input)
        parameter_values = full_parameter_values(cfg, parameter_dict)
        n_elements = int(config_dict["n_elements"])
        base = np.arange(1, n_elements * 6 + 1, dtype=float).reshape(n_elements, 6)
        fine = base * (1.0 + 0.01 * float(np.sum(parameter_values)))
        elapsed = time.perf_counter() - started
        write_field_result(
            config_dict["store_path"],
            sample_id=sample_number,
            conductivity=fine,
            parameter_values=parameter_values,
            return_code=ReturnCode.OK,
            eval_time=elapsed,
        )
        return fine, np.zeros_like(fine)
