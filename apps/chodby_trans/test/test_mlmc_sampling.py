from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

import cloudpickle
import numpy as np
import pytest
from dask.distributed import Client
import xarray as xr

from endorse import common

from chodby_trans import fullscale_transport, ot_sa, job, sensitivity_sampling
from chodby_trans.sensitivity_sampling import (
    TransportSaltelliSimulation,
    make_transport_simulation,
    make_group_matrix_generator,
    mlmc_level_parameters,
    validate_mlmc_scheduler_preload,
    validate_mlmc_worker_preloads,
    validate_result_grid_size,
)
from chodby_trans.mlmc_worker import (
    TransportLevelSimulation,
    calculate_transport_saltelli,
    return_result_format,
    scheduler_preload_status,
    transport_preload_status,
)
from chodby_trans.transport_simulation import (
    RandomTransportSimulation,
    TransportSimulation,
    failure_return_code,
)
from chodby_trans.exception_wrapper import (
    Flow123dException,
    MeshException,
    ReturnCode,
)
from mlmc.sampling_pool import SamplingPool
from mlmc.level_simulation import LevelSimulation
script_dir = Path(__file__).absolute().parent


def test_transport_preload_status_is_lightweight(monkeypatch):
    monkeypatch.delitem(
        sensitivity_sampling.sys.modules,
        "chodby_trans.dask_worker_preload",
        raising=False,
    )

    status = transport_preload_status()

    assert status["completed"] is False
    assert status["seconds"] is None
    assert status["peak_rss_mib"] is None


def test_scheduler_preload_status_is_lightweight(monkeypatch):
    monkeypatch.delitem(
        sensitivity_sampling.sys.modules,
        "chodby_trans.dask_scheduler_preload",
        raising=False,
    )

    status = scheduler_preload_status()

    assert status["completed"] is False
    assert status["seconds"] is None
    assert status["peak_rss_mib"] is None


def test_validate_mlmc_scheduler_preload():
    with pytest.raises(RuntimeError, match="Dask scheduler"):
        validate_mlmc_scheduler_preload({"completed": False})

    validate_mlmc_scheduler_preload(
        {
            "completed": True,
            "pid": 9,
            "seconds": 1.5,
            "peak_rss_mib": 256.0,
        }
    )


def test_validate_mlmc_worker_preloads():
    with pytest.raises(RuntimeError, match="No Dask workers"):
        validate_mlmc_worker_preloads({})

    states = {
        "tcp://worker-0": {
            "completed": True,
            "pid": 10,
            "seconds": 2.5,
            "peak_rss_mib": 512.0,
        },
    }
    validate_mlmc_worker_preloads(states)

    with pytest.raises(RuntimeError, match="tcp://worker-1"):
        validate_mlmc_worker_preloads(
            {
                **states,
                "tcp://worker-1": {
                    "completed": False,
                    "pid": 11,
                    "seconds": None,
                    "peak_rss_mib": None,
                },
            }
        )


def test_validate_result_grid_size():
    data_schema = {"ATTRS": {"grid_step": [20, 20, 2]}}

    assert validate_result_grid_size(
        common.dotdict.create({"grid_size": [20, 20, 2]}),
        data_schema,
    ) == (20, 20, 2)

    with pytest.raises(ValueError, match="does not match"):
        validate_result_grid_size(
            common.dotdict.create({"grid_size": [20, 10, 2]}),
            data_schema,
        )


def test_process_results_uses_configured_grid_size(monkeypatch):
    expected_values = np.ones((2, 3), dtype=float)
    observed = {}

    def get_indicator(cfg, flow_output, grid_size):
        observed["grid_size"] = grid_size
        return object(), expected_values

    monkeypatch.setattr(fullscale_transport, "get_indicator", get_indicator)
    cfg = common.dotdict.create({"grid_size": [20, 20, 2]})

    assert fullscale_transport.process_results(cfg, object()) is expected_values
    assert observed["grid_size"] == (20, 20, 2)


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (MeshException("mesh failed"), ReturnCode.BGEM_GMSH_ERROR),
        (Flow123dException("flow failed"), ReturnCode.FLOW123_ERROR),
        (RuntimeError("unexpected failure"), ReturnCode.UNKNOWN_ERROR),
    ],
)
def test_failure_return_code(error, expected_code):
    assert failure_return_code(error) == expected_code


def test_transport_failure_is_written_to_mlmc_zarr(smart_tmp_path: Path, monkeypatch):
    workdir = make_full_input_workdir(smart_tmp_path, "transport_failure_zarr")
    input_dir = workdir / "input_data"
    job.set_workdir(workdir, input_dir)

    cfg = common.load_config(job.input.transport_cfg_path)
    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    simulation = TransportSimulation(cfg, workdir)
    level_sim = simulation.make_level_simulation([1.0], [0], level_id=0)
    sample_input = np.concatenate(
        (
            np.full(len(sa_obj.groups), 0.5, dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )
    )

    def fail_transport(*_args, **_kwargs):
        raise MeshException("synthetic mesh failure")

    monkeypatch.setattr(
        sensitivity_sampling.transport_simulation.transport,
        "transport_run",
        fail_transport,
    )
    sample_dir = workdir / "pool" / "L00_S0000000"
    with common.workdir(str(sample_dir), clean=True):
        with pytest.raises(MeshException, match="synthetic mesh failure"):
            level_sim._calculate(level_sim.config_dict, sample_input)

    ds = xr.open_zarr(
        str(job.output.zarr_store_path),
        group="mlmc/level_00",
        consolidated=False,
    )
    term = {"i_sample": 0, "i_saltelli": 0}
    fine_code = ds["fine_return_code"].isel(**term).to_numpy().item()
    coarse_code = ds["coarse_return_code"].isel(**term).to_numpy().item()
    fine_time = ds["fine_eval_time"].isel(**term).to_numpy().item()
    coarse_time = ds["coarse_eval_time"].isel(**term).to_numpy().item()
    assert fine_code == ReturnCode.BGEM_GMSH_ERROR
    assert coarse_code == ReturnCode.BGEM_GMSH_ERROR
    assert fine_time == -1.0
    assert coarse_time == -1.0
    assert np.count_nonzero(ds["fine_conc"].isel(**term).to_numpy()) == 0
    assert np.count_nonzero(ds["coarse_conc"].isel(**term).to_numpy()) == 0
    assert np.all(np.isfinite(ds["parameter"].isel(**term).to_numpy()))


def make_mlmc_workdir(
    tmp_path: Path,
    source_name: str = "trans_mesh_config_mlmc_goal1.yaml",
    test_random_data: bool = True,
    source_root: Path | None = None,
) -> Path:
    """
    Build a minimal MLMC workdir for synchronous sample execution tests.

    The default fixture keeps the synthetic Goal 1 path fast. Some tests use
    ``transport_mlmc.yaml`` as the source fixture, but switch ``mlmc.sim_class``
    to ``RandomTransportSimulation`` so execution remains lightweight.
    """
    workdir = tmp_path / "mlmc_goal23"
    input_dir = workdir / "input_data"
    input_dir.mkdir(parents=True)

    if source_root is None:
        source_root = Path(__file__).parent / "input_data"
    source_path = source_root / source_name
    target_path = input_dir / "trans_mesh_config.yaml"
    shutil.copyfile(source_path, target_path)
    if test_random_data:
        content = target_path.read_text(encoding="utf-8")
        if "test_random_data:" in content:
            content = content.replace("test_random_data: False", "test_random_data: True", 1)
        else:
            content = "test_random_data: True\n" + content
        if "sim_class:" in content:
            content = content.replace("sim_class: TransportSimulation", "sim_class: RandomTransportSimulation", 1)
        else:
            content = content.replace("mlmc:\n", "mlmc:\n  sim_class: RandomTransportSimulation\n", 1)
        target_path.write_text(content, encoding="utf-8")
    return workdir


def make_full_input_workdir(tmp_path: Path, name: str) -> Path:
    workdir = tmp_path / name
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    source_input_dir = Path(__file__).parent.parent / "input_data"
    shutil.copytree(source_input_dir, workdir / "input_data")
    return workdir

@pytest.mark.skip
def test_goal3_prepare_samples_prefixes_finer_count(tmp_path: Path):
    workdir = make_mlmc_workdir(tmp_path)
    cfg = common.config.load_config(str(workdir / "input_data" / "trans_mesh_config.yaml"))
    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    simulation = TransportSaltelliSimulation(
        cfg_levels=cfg.mlmc.levels,
        forward_simulation=RandomTransportSimulation(cfg, workdir),
        matrix_generator=make_group_matrix_generator(sa_obj),
        n_parameters=len(sa_obj.groups),
        finer_samples_collected=lambda _sample_ids: 7,
    )

    coarse_level = simulation.level_instance([1.0], [0])
    fine_level = simulation.level_instance([10.0], [1.0])

    coarse_input = coarse_level.prepare_samples(["L00_S0000000"])[0][1]
    fine_input = fine_level.prepare_samples(["L01_S0000000"])[0][1]

    assert coarse_input.shape[1] == len(sa_obj.groups) + 1
    assert fine_input.shape[1] == len(sa_obj.groups) + 1
    assert np.all(coarse_input[:, 0] == 7)
    assert np.all(fine_input[:, 0] == 0)


@pytest.mark.skip
def test_goal3_grouped_sample_runs_transport_simulation(tmp_path: Path):
    workdir = make_mlmc_workdir(tmp_path)
    cfg = common.config.load_config(str(workdir / "input_data" / "trans_mesh_config.yaml"))
    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    simulation = TransportSaltelliSimulation(
        cfg_levels=cfg.mlmc.levels,
        forward_simulation=RandomTransportSimulation(cfg, workdir),
        matrix_generator=make_group_matrix_generator(sa_obj),
        n_parameters=len(sa_obj.groups),
        finer_samples_collected=lambda _sample_ids: 3,
    )

    assert mlmc_level_parameters(cfg) == [[1.0], [10.0]]

    level_sim = simulation.level_instance([10.0], [1.0])
    sample_input = level_sim.prepare_samples(["L01_S0000000"])[0]
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    sample_id, result, err_msg, _running_time = SamplingPool.calculate_sample(
        sample_input,
        level_sim,
        work_dir=str(pool_dir),
    )

    fine, coarse = result
    expected_len = simulation.schema.n_terms * len(RandomTransportSimulation(cfg, workdir).result_format()[0].times)

    assert sample_id == "L01_S0000000"
    assert err_msg == ""
    assert fine.shape == (expected_len,)
    assert coarse.shape == (expected_len,)
    assert np.all(np.isfinite(fine))
    assert np.all(np.isfinite(coarse))

@pytest.mark.skip
def test_transport_mlmc_random(smart_tmp_path: Path):
    """
    Document the synchronous forward-simulation path against the real MLMC transport config.

    This does not exercise the heavy transport solve. It uses
    ``input_data/transport_mlmc.yaml`` as the source fixture, switches
    ``mlmc.sim_class`` to ``RandomTransportSimulation``, and runs the MLMC
    driver through ``run_mlmc_sampling``.
    """
    workdir = smart_tmp_path / "transport_mlmc"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    #input_dir = Path(__file__).parent.parent / "input_data"
    job.set_workdir(workdir)
    shutil.copytree(script_dir.parent / job.input.dir_path.name, job.input.dir_path, dirs_exist_ok=True)

    # Patch config to use the random simulation class.
    cfg = common.load_config(job.input.transport_cfg_path)
    cfg_random = common.apply_variant(cfg, {"mlmc/sim_class": "RandomTransportSimulation"})
    common.dump_config(cfg_random, job.input.transport_cfg_path)
    reloaded_cfg = common.load_config(job.input.transport_cfg_path)

    assert reloaded_cfg.mlmc.sim_class == "RandomTransportSimulation"

    sensitivity_sampling.resolve_subcmd("mlmc", workdir, None, copy_flag=False)

#@pytest.mark.skip
def test_transport_simulation(smart_tmp_path: Path):
    """
    Exercise `TransportSimulation` directly for the fine-level setup from `transport_mlmc.yaml`.

    The test keeps the runtime lightweight with `RandomTransportSimulation`, but it bypasses the
    MLMC driver and runs the fine-level `LevelSimulation` through `SamplingPool.calculate_sample`.
    """
    workdir = smart_tmp_path / "transport_simulation_fine"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    source_input_dir = Path(__file__).parent.parent / "input_data"
    input_dir = workdir / "input_data"
    if not input_dir.exists():
        shutil.copytree(source_input_dir, input_dir)
    else:
        print(f"Using the existing input dir: {input_dir}")

    job.set_workdir(workdir, input_dir)

    cfg = common.load_config(job.input.transport_cfg_path)
    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)

    data_schema_key, data_schema = sensitivity_sampling.initialize_data_schema()
    with common.workdir(str(workdir), clean=False):
      sensitivity_sampling.prepare_common_homogenization_mesh(cfg)

    #cfg = common.apply_variant(cfg, {"mlmc/sim_class": "RandomTransportSimulation"})
    simulation = make_transport_simulation(cfg)

    fine_level_sim = simulation.make_level_simulation([10.0], [1.0], level_id=1)

    sample_input = np.concatenate(
        (
            np.full(len(sa_obj.groups), 0.5, dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )
    )
    pool_dir = workdir / "pool"
    pool_dir.mkdir(exist_ok=True)

    sample_id, result, err_msg, _running_time = SamplingPool.calculate_sample(
        ("L01_S0000000", *sample_input.tolist()),
        fine_level_sim,
        work_dir=str(pool_dir),
    )

    fine_result, coarse_result = result
    expected_len = len(simulation.result_format()[0].times)

    assert sample_id == "L01_S0000000"
    assert err_msg == ""
    assert fine_level_sim.config_dict["level_id"] == 1
    assert fine_result.shape == (expected_len,)
    assert coarse_result.shape == (expected_len,)
    assert np.all(np.isfinite(fine_result))
    assert np.all(np.isfinite(coarse_result))


def test_transport_saltelli_simulation_writes_mlmc_zarr(smart_tmp_path: Path):
    workdir = make_full_input_workdir(smart_tmp_path, "transport_saltelli_zarr")
    input_dir = workdir / "input_data"
    job.set_workdir(workdir, input_dir)

    cfg = common.load_config(job.input.transport_cfg_path)
    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    simulation = TransportSaltelliSimulation(
        cfg_levels=cfg.mlmc.levels,
        forward_simulation=RandomTransportSimulation(cfg, workdir),
        matrix_generator=make_group_matrix_generator(sa_obj),
        n_parameters=len(sa_obj.groups),
        finer_samples_collected=lambda _sample_ids: 0,
    )

    level_sim = simulation.make_level_simulation([1.0], [0], level_id=0)
    sample_input = level_sim.prepare_samples(["L00_S0000000"])[0]
    worker_level_sim = cloudpickle.loads(cloudpickle.dumps(level_sim))
    assert worker_level_sim._calculate is calculate_transport_saltelli
    assert "prepare_samples" not in worker_level_sim.__dict__

    sample_id = sample_input[0]
    with common.workdir(str(workdir / "pool" / sample_id), clean=True):
        result = level_sim._calculate(level_sim.config_dict, sample_input[1:])

    assert sample_id == "L00_S0000000"
    assert result[0].shape[0] == simulation.schema.n_terms * len(simulation.result_format()[0].times)

    ds = xr.open_zarr(
        str(job.output.zarr_store_path),
        group="mlmc/level_00",
        consolidated=False,
    )
    assert ds["fine_return_code"].isel(i_sample=0, i_saltelli=0).to_numpy().item() == 0
    assert ds["coarse_return_code"].isel(i_sample=0, i_saltelli=0).to_numpy().item() == 0
    assert ds["fine_eval_time"].isel(i_sample=0, i_saltelli=0).to_numpy().item() == 0.0
    assert ds["coarse_eval_time"].isel(i_sample=0, i_saltelli=0).to_numpy().item() == 0.0
    assert np.any(ds["fine_conc"].isel(i_sample=0).to_numpy() != 0.0)
    assert np.any(ds["coarse_conc"].isel(i_sample=0).to_numpy() != 0.0)
    assert np.all(np.isfinite(ds["parameter"].isel(i_sample=0, i_saltelli=0).to_numpy()))


class _SerializationBomb:
    """Fail the test if driver-only planning state reaches cloudpickle."""

    def __reduce__(self):
        raise AssertionError("Driver-only Saltelli planning state was serialized")


def test_transport_saltelli_worker_serialization_omits_planning_state():
    planning_state = _SerializationBomb()

    def prepare_samples(_sample_ids):
        assert planning_state is not None
        return []

    level_sim = TransportLevelSimulation(config_dict={})
    level_sim._calculate = calculate_transport_saltelli
    level_sim._result_format = partial(return_result_format, [])
    level_sim.prepare_samples = prepare_samples

    assert level_sim.prepare_samples(["L00_S0000000"]) == []
    restored = cloudpickle.loads(cloudpickle.dumps(level_sim))

    assert "prepare_samples" not in restored.__dict__
    assert restored._calculate is calculate_transport_saltelli
    assert restored._calculate.__module__ == "chodby_trans.mlmc_worker"
    assert restored._result_format() == []
