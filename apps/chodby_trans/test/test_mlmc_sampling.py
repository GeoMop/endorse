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
    make_group_matrix_generator,
    mlmc_level_parameters,
)
from chodby_trans.transport_simulation import TransportSimulation
from mlmc.sampling_pool import SamplingPool


def make_mlmc_workdir(
    tmp_path: Path,
    source_name: str = "trans_mesh_config_mlmc_goal1.yaml",
    test_random_data: bool = True,
    source_root: Path | None = None,
) -> Path:
    """
    Build a minimal MLMC workdir for synchronous sample execution tests.

    The default fixture keeps the synthetic Goal 1 path fast.  Some tests use
    ``transport_mlmc.yaml`` as the source fixture, but flip ``test_random_data``
    on so the forward simulation remains synchronous and lightweight.
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
        target_path.write_text(content, encoding="utf-8")
    return workdir

@pytest.mark.skip
def test_goal3_prepare_samples_prefixes_finer_count(tmp_path: Path):
    workdir = make_mlmc_workdir(tmp_path)
    cfg = common.config.load_config(str(workdir / "input_data" / "trans_mesh_config.yaml"))
    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    simulation = TransportSaltelliSimulation(
        cfg_levels=cfg.mlmc.levels,
        forward_simulation=TransportSimulation(workdir),
        matrix_generator=make_group_matrix_generator(sa_obj),
        n_parameters=len(sa_obj.groups),
        finer_finished_count=lambda _level_id: 7,
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
        forward_simulation=TransportSimulation(workdir),
        matrix_generator=make_group_matrix_generator(sa_obj),
        n_parameters=len(sa_obj.groups),
        finer_finished_count=lambda _level_id: 3,
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
    expected_len = simulation.schema.n_terms * len(TransportSimulation(workdir).result_format()[0].times)

    assert sample_id == "L01_S0000000"
    assert err_msg == ""
    assert fine.shape == (expected_len,)
    assert coarse.shape == (expected_len,)
    assert np.all(np.isfinite(fine))
    assert np.all(np.isfinite(coarse))


def test_transport_mlmc_random(smart_tmp_path: Path):
    """
    Document the synchronous forward-simulation path against the real MLMC transport config.

    This does not exercise the heavy transport solve.  It uses
    ``input_data/transport_mlmc.yaml`` as the source fixture, forces
    ``test_random_data=True``, and runs the MLMC driver through
    ``run_mlmc_sampling``.
    """
    workdir = smart_tmp_path / "transport_mlmc"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(__file__).parent.parent / "input_data"
    job.set_workdir(workdir, input_dir)

    # Patch config to use random data
    cfg = common.load_config(job.input.transport_cfg_path)
    cfg_random = common.apply_variant(cfg, {"test_random_data": True})
    cfg_random_path = job.input.dir_path / "__transport_mlmc_random.yaml"
    common.dump_config(cfg_random, cfg_random_path)

    sensitivity_sampling.resolve_subcmd("mlmc", workdir, None)
