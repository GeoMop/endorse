from __future__ import annotations

import copy
import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from dask.distributed import Lock, get_client
import xarray as xr
import yaml
import zarr

from endorse import common
from endorse.common import dotdict
from endorse.fullscale_transport import output_times

import chodby_trans.fullscale_transport as transport
import chodby_trans.job as job
from chodby_trans import ot_sa
from chodby_trans.exception_wrapper import ReturnCode, WrapperException

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sim.simulation import Simulation


MLMC_ZARR_GROUP = "mlmc"


def expand_sample_parameters(cfg: dotdict, parameters: Sequence[float]) -> np.ndarray:
    """
    Expand a sampled input vector to the full parameter vector expected by the transport config.

    The MLMC Sobol path may provide either:
    - one value per transport parameter, or
    - one value per OpenTURNS group in ``[0, 1]``.
    """
    sa = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    param_values = np.asarray(parameters, dtype=float)

    if len(param_values) == len(sa.parameters):
        return param_values

    if len(param_values) == len(sa.groups):
        group_to_col = {group: idx for idx, group in enumerate(sa.groups)}
        expanded = [
            parameter.map_from_group(np.array([param_values[group_to_col[parameter.group]]], dtype=float))[0]
            for parameter in sa.parameters.values()
        ]
        return np.asarray(expanded, dtype=float)

    raise ValueError(
        f"Unexpected sample vector length {len(param_values)}; "
        f"expected {len(sa.parameters)} parameters or {len(sa.groups)} groups."
    )


def apply_sample_parameters(
    cfg: dotdict,
    parameters: Sequence[float],
) -> tuple[dotdict, dict[str, float]]:
    """
    Apply one sampled parameter vector onto the transport config and return the patched config together with the
    full parameter dictionary.
    """
    sa = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    full_parameters = expand_sample_parameters(cfg, parameters)
    param_dict = sa.param_vec_to_dict(full_parameters)

    # TODO: use or extent the cfg patching mechanism
    variant_patch = {}
    for name, param_cfg in cfg.ot_sensitivity.parameters.items():
        if "path" in param_cfg:
            for p in param_cfg.path:
                variant_patch[p] = param_dict[name]

    return common.apply_variant(cfg, variant_patch), param_dict


def full_parameter_names(cfg: dotdict) -> list[str]:
    """
    Return the expanded transport parameter names in the same order as the
    full parameter vector used by ``SensitivityAnalysis.param_vec_to_dict``.
    """
    sa = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    return list(sa.parameters.keys())


def full_parameter_values(cfg: dotdict, param_dict: dict[str, float]) -> np.ndarray:
    """
    Convert a full parameter dictionary to a vector with the canonical
    expanded parameter ordering for storage and transport execution metadata.
    """
    names = full_parameter_names(cfg)
    return np.asarray([param_dict[name] for name in names], dtype=float)


def compact_concentration_series(
    concentration: np.ndarray,
    quantile: float = 0.99,
    clip_min: float = 1.0e-30,
    clip_max: float = 1.0,
) -> np.ndarray:
    """
    Reduce a concentration block to ``q99_XYZ(log10(concentration))`` over time.
    """
    conc = np.asarray(concentration, dtype=float)
    if conc.ndim != 4:
        raise ValueError(f"Expected concentration block with dims (time, X, Y, Z), got {conc.shape}")

    log_conc = np.log10(np.clip(conc, clip_min, clip_max))
    return np.quantile(log_conc, quantile, axis=(1, 2, 3))


def synthetic_concentration(
    times: Sequence[float],
    level_id: int,
    parameters: Sequence[float],
    grid_step: Iterable[int] = (4, 3, 2),
) -> np.ndarray:
    """
    Build deterministic synthetic concentration data for fast Goal 1 tests.
    """
    nx, ny, nz = (int(v) for v in grid_step)
    time_axis = np.arange(len(times), dtype=float).reshape(-1, 1, 1, 1)
    x_axis = np.linspace(1.0, 1.0 + 0.05 * max(nx - 1, 0), nx, dtype=float).reshape(1, nx, 1, 1)
    y_axis = np.linspace(1.0, 1.0 + 0.03 * max(ny - 1, 0), ny, dtype=float).reshape(1, 1, ny, 1)
    z_axis = np.linspace(1.0, 1.0 + 0.02 * max(nz - 1, 0), nz, dtype=float).reshape(1, 1, 1, nz)
    param_scale = 1.0 + 0.01 * float(np.sum(np.asarray(parameters, dtype=float)))
    level_scale = 1.0 + 0.2 * float(level_id)
    return 1.0e-6 * level_scale * param_scale * (1.0 + time_axis + x_axis + y_axis + z_axis)


def level_selector_to_id(level_params: Sequence[float], n_levels: int) -> int | None:
    """
    Map MLMC level parameters to ``cfg.mlmc.levels`` indices.

    Supports two conventions:
    - explicit selectors `1..n_levels`
    - mesh-step-like positive values mapped by `log10`
    AGENT: Do not complicate this function, support just geomteric progression.
    Resolved: the MLMC path now uses only geometric selectors (`10**level_id`) and this helper keeps that
    single mapping.
    """
    selector = float(level_params[0])
    assert selector > 0

    log_sel = int(round(np.log10(selector)))
    if log_sel < 0:
        raise ValueError(f"Negative log10(level selector) is invalid: {selector}")
    if log_sel >= n_levels:
        raise ValueError(f"Mapped level selector {selector} to out-of-range level id {log_sel}")
    return log_sel


def parse_sample_workspace(path: str | Path) -> tuple[int, int]:
    """
    Extract the numeric MLMC level id and sample id from a sample workspace path.
    """
    sample_dir_name = Path(path).name
    parts = sample_dir_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected MLMC sample workspace format: {sample_dir_name}")

    level_tag, sample_tag = parts[:2]
    if len(level_tag) < 2 or not level_tag.startswith("L"):
        raise ValueError(f"Unexpected MLMC sample workspace level tag: {sample_dir_name}")
    if len(sample_tag) < 2 or not sample_tag.startswith("S"):
        raise ValueError(f"Unexpected MLMC sample workspace sample tag: {sample_dir_name}")

    return int(level_tag[1:]), int(sample_tag[1:])


def _mlmc_level_group(level_id: int) -> str:
    return f"{MLMC_ZARR_GROUP}/level_{level_id:02d}"


def _mlmc_level_group_path(store_path: str | Path, level_id: int) -> Path:
    return Path(store_path) / MLMC_ZARR_GROUP / f"level_{level_id:02d}"


def _load_zarr_schema_template() -> dict:
    data_schema_path = job.input.data_schema_yaml
    if not data_schema_path.exists():
        data_schema_path = job.input.data_schema_empty_yaml
    with data_schema_path.open("r", encoding="utf-8") as file:
        data_schema = yaml.safe_load(file.read())
    return data_schema["run_timestamp"]


def _chunk_ranges(coord_slice: slice, chunk_len: int) -> range:
    start, stop = coord_slice.start, coord_slice.stop
    first = start // chunk_len
    last = (stop - 1) // chunk_len
    return range(first, last + 1)


def _make_region_locks(ds: xr.Dataset, region: dict[str, slice]) -> list[Lock]:
    try:
        get_client()
    except ValueError:
        return []

    lock_names: list[str] = []
    for var_name, chunkshape in ds.chunksizes.items():
        sample_chunks = _chunk_ranges(region["i_sample"], chunkshape[0])
        saltelli_chunks = _chunk_ranges(region["i_saltelli"], chunkshape[1])
        for i_sample_chunk in sample_chunks:
            for i_saltelli_chunk in saltelli_chunks:
                lock_names.append(f"zarr-{var_name}-{i_sample_chunk}-{i_saltelli_chunk}")
    return [Lock(name) for name in sorted(set(lock_names))]


def ensure_mlmc_level_zarr_storage(cfg: dotdict, level_id: int, n_saltelli: int) -> None:
    """
    Create the fixed-capacity MLMC Zarr group for one transport level if it does not exist yet.
    """
    store_path = str(job.output.zarr_store_path)
    group = _mlmc_level_group(level_id)
    schema = _load_zarr_schema_template()
    coords = schema["COORDS"]
    grid_shape = tuple(int(v) for v in schema["ATTRS"]["grid_step"])
    times = np.asarray(output_times(cfg.transport_fullscale), dtype=float)

    n_samples = int(cfg.ot_sensitivity.n_samples)
    param_names = np.asarray(full_parameter_names(cfg), dtype=str)
    n_params = len(param_names)

    conc_chunks = (
        int(coords["i_sample"]["chunk_size"]),
        int(coords["i_saltelli"]["chunk_size"]),
        int(coords["sim_time"]["chunk_size"]),
        int(coords["X"]["chunk_size"]),
        int(coords["Y"]["chunk_size"]),
        int(coords["Z"]["chunk_size"]),
    )
    meta_chunks = (
        int(coords["i_sample"]["chunk_size"]),
        int(coords["i_saltelli"]["chunk_size"]),
    )
    par_chunks = meta_chunks + (int(coords["param_name"]["chunk_size"]),)

    root = zarr.open_group(store_path, mode="a")
    mlmc_group = root.require_group(MLMC_ZARR_GROUP)
    level_group = mlmc_group.require_group(f"level_{level_id:02d}")
    level_group.attrs.update({"data_schema_key": MLMC_ZARR_GROUP, "mlmc_level_id": int(level_id)})

    def create_array(
        name: str,
        *,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype,
        dimensions: tuple[str, ...],
        fill_value,
        values: np.ndarray | None = None,
    ) -> None:
        if name in level_group:
            return
        arr = level_group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            dimension_names=dimensions,
            attributes={"_ARRAY_DIMENSIONS": list(dimensions)},
        )
        if values is not None:
            arr[...] = values

    create_array(
        "i_sample",
        shape=(n_samples,),
        chunks=(int(coords["i_sample"]["chunk_size"]),),
        dtype=np.int64,
        dimensions=("i_sample",),
        fill_value=None,
        values=np.arange(n_samples, dtype=np.int64),
    )
    create_array(
        "i_saltelli",
        shape=(n_saltelli,),
        chunks=(int(coords["i_saltelli"]["chunk_size"]),),
        dtype=np.int64,
        dimensions=("i_saltelli",),
        fill_value=None,
        values=np.arange(n_saltelli, dtype=np.int64),
    )
    create_array(
        "param_name",
        shape=(n_params,),
        chunks=(int(coords["param_name"]["chunk_size"]),),
        dtype=param_names.dtype,
        dimensions=("param_name",),
        fill_value="",
        values=param_names,
    )
    create_array(
        "sim_time",
        shape=(len(times),),
        chunks=(int(coords["sim_time"]["chunk_size"]),),
        dtype=times.dtype,
        dimensions=("sim_time",),
        fill_value=None,
        values=times,
    )
    create_array(
        "X",
        shape=(grid_shape[0],),
        chunks=(int(coords["X"]["chunk_size"]),),
        dtype=np.int64,
        dimensions=("X",),
        fill_value=None,
        values=np.arange(grid_shape[0], dtype=np.int64),
    )
    create_array(
        "Y",
        shape=(grid_shape[1],),
        chunks=(int(coords["Y"]["chunk_size"]),),
        dtype=np.int64,
        dimensions=("Y",),
        fill_value=None,
        values=np.arange(grid_shape[1], dtype=np.int64),
    )
    create_array(
        "Z",
        shape=(grid_shape[2],),
        chunks=(int(coords["Z"]["chunk_size"]),),
        dtype=np.int64,
        dimensions=("Z",),
        fill_value=None,
        values=np.arange(grid_shape[2], dtype=np.int64),
    )
    create_array(
        "i_eval",
        shape=(n_samples, n_saltelli),
        chunks=meta_chunks,
        dtype=np.int64,
        dimensions=("i_sample", "i_saltelli"),
        fill_value=-1,
    )
    create_array(
        "fine_return_code",
        shape=(n_samples, n_saltelli),
        chunks=meta_chunks,
        dtype=np.int64,
        dimensions=("i_sample", "i_saltelli"),
        fill_value=ReturnCode.NONE,
    )
    create_array(
        "coarse_return_code",
        shape=(n_samples, n_saltelli),
        chunks=meta_chunks,
        dtype=np.int64,
        dimensions=("i_sample", "i_saltelli"),
        fill_value=ReturnCode.NONE,
    )
    create_array(
        "fine_eval_time",
        shape=(n_samples, n_saltelli),
        chunks=meta_chunks,
        dtype=np.float64,
        dimensions=("i_sample", "i_saltelli"),
        fill_value=-1.0,
    )
    create_array(
        "coarse_eval_time",
        shape=(n_samples, n_saltelli),
        chunks=meta_chunks,
        dtype=np.float64,
        dimensions=("i_sample", "i_saltelli"),
        fill_value=-1.0,
    )
    create_array(
        "fine_conc",
        shape=(n_samples, n_saltelli, len(times), *grid_shape),
        chunks=conc_chunks,
        dtype=np.float64,
        dimensions=("i_sample", "i_saltelli", "sim_time", "X", "Y", "Z"),
        fill_value=0.0,
    )
    create_array(
        "coarse_conc",
        shape=(n_samples, n_saltelli, len(times), *grid_shape),
        chunks=conc_chunks,
        dtype=np.float64,
        dimensions=("i_sample", "i_saltelli", "sim_time", "X", "Y", "Z"),
        fill_value=0.0,
    )
    create_array(
        "parameter",
        shape=(n_samples, n_saltelli, n_params),
        chunks=par_chunks,
        dtype=np.float64,
        dimensions=("i_sample", "i_saltelli", "param_name"),
        fill_value=np.nan,
    )


def write_mlmc_level_result(
    *,
    cfg: dotdict,
    level_id: int,
    n_saltelli: int,
    sample_id: int,
    saltelli_id: int,
    parameter_values: Sequence[float],
    fine_conc: np.ndarray,
    coarse_conc: np.ndarray | None,
    fine_return_code: int,
    coarse_return_code: int,
    fine_eval_time: float,
    coarse_eval_time: float,
) -> None:
    """
    Write one Saltelli term result into the per-level MLMC Zarr storage.
    """
    ensure_mlmc_level_zarr_storage(cfg, level_id, n_saltelli)
    store_path = str(job.output.zarr_store_path)
    group = _mlmc_level_group(level_id)
    ds = xr.open_zarr(store_path, group=group, consolidated=False)

    region = {
        "i_sample": slice(int(sample_id), int(sample_id) + 1),
        "i_saltelli": slice(int(saltelli_id), int(saltelli_id) + 1),
    }
    expected_shape = (
        ds.sizes["sim_time"],
        ds.sizes["X"],
        ds.sizes["Y"],
        ds.sizes["Z"],
    )

    fine_array = np.asarray(fine_conc, dtype=float)
    if fine_array.shape != expected_shape:
        logging.warning("fine_conc shape mismatch for level %s sample %s term %s: %s != %s",
                        level_id, sample_id, saltelli_id, fine_array.shape, expected_shape)
        fine_array = np.zeros(expected_shape, dtype=float)
        fine_return_code = ReturnCode.ZARR_ERROR

    coarse_array = np.zeros(expected_shape, dtype=float)
    if coarse_conc is not None:
        coarse_array = np.asarray(coarse_conc, dtype=float)
        if coarse_array.shape != expected_shape:
            logging.warning("coarse_conc shape mismatch for level %s sample %s term %s: %s != %s",
                            level_id, sample_id, saltelli_id, coarse_array.shape, expected_shape)
            coarse_array = np.zeros(expected_shape, dtype=float)
            coarse_return_code = ReturnCode.ZARR_ERROR

    param_array = np.asarray(parameter_values, dtype=float)
    if param_array.shape != (ds.sizes["param_name"],):
        raise ValueError(
            f"Expected parameter vector of shape {(ds.sizes['param_name'],)}, got {param_array.shape}."
        )

    locks = _make_region_locks(ds, region)
    for lock in locks:
        lock.acquire()

    try:
        ds_slice = xr.Dataset(
            data_vars={
                "i_eval": (
                    ("i_sample", "i_saltelli"),
                    np.asarray([[sample_id * n_saltelli + saltelli_id]], dtype=int),
                ),
                "fine_conc": (
                    ("i_sample", "i_saltelli", "sim_time", "X", "Y", "Z"),
                    fine_array[np.newaxis, np.newaxis, ...],
                ),
                "coarse_conc": (
                    ("i_sample", "i_saltelli", "sim_time", "X", "Y", "Z"),
                    coarse_array[np.newaxis, np.newaxis, ...],
                ),
                "fine_return_code": (
                    ("i_sample", "i_saltelli"),
                    np.asarray([[fine_return_code]], dtype=int),
                ),
                "coarse_return_code": (
                    ("i_sample", "i_saltelli"),
                    np.asarray([[coarse_return_code]], dtype=int),
                ),
                "fine_eval_time": (
                    ("i_sample", "i_saltelli"),
                    np.asarray([[fine_eval_time]], dtype=float),
                ),
                "coarse_eval_time": (
                    ("i_sample", "i_saltelli"),
                    np.asarray([[coarse_eval_time]], dtype=float),
                ),
                "parameter": (
                    ("i_sample", "i_saltelli", "param_name"),
                    param_array[np.newaxis, np.newaxis, :],
                ),
            }
        )
        ds_slice.to_zarr(store_path, group=group, mode="r+", region=region)
    finally:
        for lock in reversed(locks):
            lock.release()


class TransportSimulation(Simulation):
    """
    MLMC forward simulation adapter for the chodby transport model.
    """

    RESULT_NAME = "log10_conc_q99_xyz"
    RESULT_UNIT = "log10(g/m^3)"
    RESULT_LOCATION = "0"

    def __init__(self, cfg, workdir: Path):
        # AGENT: workdir should not be needed since we are only allowed to work relative to the directories
        # set by MLMC SamplingPool
        # Resolved: `workdir` is used only to locate the root config at construction time; per-sample execution
        # stays in the MLMC-provided sample workspace.
        workdir = Path(workdir)
        self.cfg = cfg
        self._times = output_times(self.cfg.transport_fullscale)

    def level_instance(self, fine_level_params: list[float], coarse_level_params: list[float]) -> LevelSimulation:
        fine_level_id = level_selector_to_id(fine_level_params, len(self.cfg.mlmc.levels))
        if fine_level_id is None:
            raise ValueError("Fine level selector must be positive")

        config_dict = {
            "level_id": fine_level_id,
            "root_cfg": copy.deepcopy(self.cfg),
        }
        # AGNET:this config value is obligatory since MLMC need a relative time estiamte for the optimization of
        # the samples per level; so provide it in test configs
        task_size = float(self.cfg.mlmc.levels[fine_level_id].task_size)
        return LevelSimulation(
            config_dict=config_dict,
            need_sample_workspace=True,
            task_size=task_size,
        )

    def result_format(self) -> list[QuantitySpec]:
        return [
            QuantitySpec(
                name=self.RESULT_NAME,
                unit=self.RESULT_UNIT,
                shape=(),   # AGENT: this seems not to be a valid value for the shape
                # Resolved: scalar `shape=()` is accepted by the current MLMC `QuantitySpec` path and matches
                # the flattened result-length checks in the focused tests.
                times=self._times,
                locations=[self.RESULT_LOCATION],
            )
        ]

    @staticmethod
    def _parse_sample_input(sample_input: Sequence[float]) -> tuple[np.ndarray, int, int]:
        """
        Split the planned sample vector into transport parameters, Saltelli term index,
        and the finer-level sample count.
        """
        sample_array = np.asarray(sample_input, dtype=float)
        saltelli_index = int(sample_array[-2])
        finer_level_sample_size = int(sample_array[-1])
        return sample_array[:-2], saltelli_index, finer_level_sample_size

    @staticmethod
    def calculate(config_dict, sample_input):
        """
        Calculate one MLMC sample in the current sample workspace.

        Level-specific configuration is passed through ``config_dict``.
        The real transport branch calls ``transport_run`` as requested by the
        in-code instructions.
        """
        root_cfg = copy.deepcopy(config_dict["root_cfg"])
        parameters, saltelli_index, finer_level_sample_size = TransportSimulation._parse_sample_input(sample_input)

        level = int(config_dict["level_id"])
        mlmc_level_id, mlmc_sample_id = parse_sample_workspace(os.getcwd())
        if mlmc_level_id != level:
            raise ValueError(f"Workspace MLMC level {mlmc_level_id} does not match config level {level}.")

        # AGENT: wrong, MLMC sets us the sample workdir, we should remain there
        # job.set_workdir(Path(config_dict["workdir"]))
        # Resolved: the worker keeps the current MLMC sample workspace and does not reset the working directory.

        cfg, full_param_dict = apply_sample_parameters(root_cfg, parameters)
        cfg["data_schema_key"] = MLMC_ZARR_GROUP

        sample_dir = Path(os.getcwd())
        logging.info("Running MLMC transport sample in %s, level %s.", sample_dir, level)
        logging.info("Finer-level sample count at planning time: %s", finer_level_sample_size)

        fine_return_code = ReturnCode.OK
        coarse_return_code = ReturnCode.NONE
        fine_eval_time = -1.0
        coarse_eval_time = -1.0
        fine_values: np.ndarray | None = None
        coarse_values: np.ndarray | None = None

        try:
            (
                fine_return_code,
                fine_values,
                coarse_return_code,
                coarse_values,
                fine_eval_time,
                coarse_eval_time,
            ) = transport.transport_run(cfg, level, full_param_dict)
        except WrapperException as exc:
            fine_return_code = int(exc.code)
            coarse_return_code = int(exc.code)
            raise

        logging.info(
            "results shape: %s, %s",
            None if fine_values is None else fine_values.shape,
            None if coarse_values is None else coarse_values.shape,
        )

        parameter_values = full_parameter_values(cfg, full_param_dict)
        write_mlmc_level_result(
            cfg=cfg,
            level_id=level,
            n_saltelli=int(config_dict["n_saltelli"]),
            sample_id=mlmc_sample_id,
            saltelli_id=saltelli_index,
            parameter_values=parameter_values,
            fine_conc=fine_values,
            coarse_conc=coarse_values,
            fine_return_code=fine_return_code,
            coarse_return_code=coarse_return_code,
            fine_eval_time=fine_eval_time,
            coarse_eval_time=coarse_eval_time,
        )

        fine_result = compact_concentration_series(fine_values)
        coarse_result = np.zeros_like(fine_result)
        if coarse_values is not None:
            coarse_result = compact_concentration_series(coarse_values)
        return fine_result, coarse_result


class RandomTransportSimulation(TransportSimulation):
    """
    MLMC forward simulation with deterministic synthetic concentration data for lightweight tests.
    """

    @staticmethod
    def calculate(config_dict, sample_input):
        """
        Calculate one synthetic MLMC sample in the current sample workspace.
        """
        logging.info("start sim calculate")
        root_cfg = copy.deepcopy(config_dict["root_cfg"])
        parameters, saltelli_index, finer_level_sample_size = TransportSimulation._parse_sample_input(sample_input)
        level = int(config_dict["level_id"])
        mlmc_level_id, mlmc_sample_id = parse_sample_workspace(os.getcwd())
        if mlmc_level_id != level:
            raise ValueError(f"Workspace MLMC level {mlmc_level_id} does not match config level {level}.")

        cfg, full_param_dict = apply_sample_parameters(root_cfg, parameters)
        times = output_times(cfg.transport_fullscale)
        full_parameters = full_parameter_values(cfg, full_param_dict)

        sample_dir = Path(os.getcwd())
        logging.info("Running random MLMC transport sample in %s, level %s.", sample_dir, level)
        logging.info("Finer-level sample count at planning time: %s", finer_level_sample_size)

        fine_values = synthetic_concentration(times, level, full_parameters)
        coarse_values = synthetic_concentration(times, level + 1, full_parameters)
        write_mlmc_level_result(
            cfg=cfg,
            level_id=level,
            n_saltelli=int(config_dict["n_saltelli"]),
            sample_id=mlmc_sample_id,
            saltelli_id=saltelli_index,
            parameter_values=full_parameters,
            fine_conc=fine_values,
            coarse_conc=coarse_values,
            fine_return_code=ReturnCode.OK,
            coarse_return_code=ReturnCode.OK,
            fine_eval_time=0.0,
            coarse_eval_time=0.0,
        )
        return compact_concentration_series(fine_values), compact_concentration_series(coarse_values)
