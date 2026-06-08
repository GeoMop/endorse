from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from mlmc.sampling_pool import SamplingPool

from chodby_trans.transport_simulation import (
    TransportSimulation,
    compact_concentration_series,
)


def make_goal1_workdir(tmp_path: Path) -> Path:
    workdir = tmp_path / "mlmc_goal1"
    input_dir = workdir / "input_data"
    input_dir.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parent / "input_data" / "trans_mesh_config_mlmc_goal1.yaml",
        input_dir / "trans_mesh_config.yaml",
    )
    return workdir


def test_compact_concentration_series():
    conc = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2) + 1.0e-6
    reduced = compact_concentration_series(conc, quantile=0.5)
    assert reduced.shape == (2,)
    assert np.all(np.isfinite(reduced))


def test_transport_simulation_result_format_uses_time_axis(tmp_path: Path):
    workdir = make_goal1_workdir(tmp_path)
    sim = TransportSimulation(workdir)

    result_format = sim.result_format()

    assert len(result_format) == 1
    assert result_format[0].shape == ()
    assert result_format[0].times == [0, 5, 10, 10]


def test_transport_simulation_runs_one_pair_with_sample_workspace(tmp_path: Path):
    workdir = make_goal1_workdir(tmp_path)
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    sim = TransportSimulation(workdir)
    level_sim = sim.make_level_simulation([2], [1], level_id=1)
    sample_input = ("L01_S0000000", np.array([3.0, 0.2, -0.5], dtype=float))
    old_cwd = Path.cwd()
    try:
        sample_id, result, err_msg, _running_time = SamplingPool.calculate_sample(
            sample_input,
            level_sim,
            work_dir=str(pool_dir),
        )
    finally:
        os.chdir(old_cwd)

    assert sample_id == "L01_S0000000"
    assert err_msg == ""

    fine, coarse = result
    assert fine.shape == (4,)
    assert coarse.shape == (4,)
    assert np.all(np.isfinite(fine))
    assert np.all(np.isfinite(coarse))
    assert not np.allclose(fine, coarse)
