"""Lightweight, importable callables used at the MLMC Dask boundary."""

from __future__ import annotations

from typing import Any, Sequence

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec


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
