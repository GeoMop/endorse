"""Lightweight, importable callables used at the MLMC Dask boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from attrs import evolve
from mlmc.level_simulation import LevelSimulation, SampleInput
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.sampling_pool import SamplingPool
from mlmc.sampling_pool_dask import SamplingPoolDask

from homogenization_study import job


def _calculate_on_worker(
    sample_input: SampleInput, level_sim: LevelSimulation, debug: bool,
) -> tuple[str, Any, str, float]:
    """Run MLMC and apply its file retention policy on the node that owns the workspace."""
    output = str(job.scratch.mlmc_output_dir)
    previous_cwd = Path.cwd()
    try:
        sample_id, result, error, elapsed = SamplingPool.calculate_sample(sample_input, level_sim, output)
    finally:
        os.chdir(previous_cwd)

    if error:
        SamplingPool.move_failed_rm(sample_id, level_sim, output, SamplingPool.FAILED_DIR)
    elif not debug:
        SamplingPool.move_successful_rm(sample_id, level_sim, output, SamplingPool.SEVERAL_SUCCESSFUL_DIR)
    return sample_id, result, error, elapsed


class WorkerScratchSamplingPool(SamplingPoolDask):
    """Use each PBS worker's scratch for both sample execution and file retention."""

    def schedule_sample(self, sample_input: SampleInput, level_sim: LevelSimulation) -> None:
        """Submit the MLMC task without passing the driver's scratch path."""
        sample_id, *_ = sample_input
        if sample_id in self._sample_to_future:
            return
        future = self._client.submit(
            _calculate_on_worker, sample_input, level_sim, self._debug,
            key=self._future_key(sample_id), pure=True, **self._submit_kwargs,
        )
        self._distributed.fire_and_forget(future)
        self._sample_to_future[sample_id] = (future, level_sim)
        self._n_running += 1

    def _process_result(
        self, sample_id: str, result: Any, err_msg: str, running_time: float, level_sim: LevelSimulation,
    ) -> None:
        # Reuse MLMC accounting without repeating filesystem operations on the driver.
        super()._process_result(
            sample_id, result, err_msg, running_time, evolve(level_sim, need_sample_workspace=False),
        )


class HomogenizationLevelSimulation(LevelSimulation):
    """Level metadata with sample planning performed only by the driver."""

    def prepare_samples(self, sample_ids: list[str]) -> list[tuple[Any, ...]]:
        """Create deterministic parameter vectors before samples reach Dask."""
        from endorse.common import dotdict
        from homogenization_study.problem_simulation import _sample_parameters

        cfg = dotdict.create(self.config_dict["root_cfg"])
        return [(sample_id, *_sample_parameters(cfg, sample_id)) for sample_id in sample_ids]


def return_result_format(result_format: list[QuantitySpec]) -> list[QuantitySpec]:
    """Return precomputed quantity metadata without retaining the driver simulation."""
    return result_format


def calculate_homogenization_sample(
    config_dict: dict[str, Any],
    sample_input: Sequence[float],
) -> tuple[Any, Any]:
    """Load and execute the physical homogenization model on a Dask worker."""
    from homogenization_study.problem_simulation import run_homogenization_sample

    return run_homogenization_sample(config_dict, sample_input)


def calculate_random_sample(
    config_dict: dict[str, Any],
    sample_input: Sequence[float],
) -> tuple[Any, Any]:
    """Load and execute the deterministic infrastructure model on a Dask worker."""
    from homogenization_study.problem_simulation import run_random_sample

    return run_random_sample(config_dict, sample_input)
