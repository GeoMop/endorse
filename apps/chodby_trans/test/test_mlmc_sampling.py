from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from dask.distributed import Client

from endorse import common

from chodby_trans import ot_sa, job, sensitivity_sampling
from chodby_trans.sensitivity_sampling import (
    TransportSaltelliSimulation,
    make_transport_simulation,
    make_group_matrix_generator,
    mlmc_level_parameters,
)
from chodby_trans.transport_simulation import RandomTransportSimulation
from mlmc.sampling_pool import SamplingPool
script_dir = Path(__file__).absolute().parent


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

    coarse_level = simulation.level_instance([10.0], [0])
    fine_level = simulation.level_instance([1.0], [10.0])

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

    assert mlmc_level_parameters(cfg) == [[10.0], [1.0]]

    level_sim = simulation.level_instance([1.0], [10.0])
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

    fine_level_sim = simulation.make_level_simulation([1.0], [10.0], level_id=1)

    sample_input = np.concatenate(
        (
            np.full(len(sa_obj.groups), 0.5, dtype=float),
            np.array([0.0], dtype=float),
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
    assert fine_level_sim.config_dict["level_id"] == 0
    assert fine_result.shape == (expected_len,)
    assert coarse_result.shape == (expected_len,)
    assert np.all(np.isfinite(fine_result))
    assert np.all(np.isfinite(coarse_result))
