"""
Lightweight, importable callables used by MLMC Dask workers.

Keep module imports small so deserializing a scheduled ``LevelSimulation`` does
not execute the full sensitivity-analysis driver in the Dask scheduler.
"""

import logging
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from mlmc.level_simulation import LevelSimulation
from mlmc.quantity.quantity_spec import QuantitySpec


class TransportLevelSimulation(LevelSimulation):
    """
    Level configuration whose driver-only sample planner is not serialized.
    """

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("prepare_samples", None)
        return state


def return_result_format(result_format: list[QuantitySpec]) -> list[QuantitySpec]:
    """
    Return a precomputed MLMC result format without capturing the simulation.
    """
    return result_format


def transport_preload_status() -> dict[str, Any]:
    """
    Return preload state without importing the heavy transport stack.
    """
    module = sys.modules.get("chodby_trans.dask_worker_preload")
    if module is None:
        return {
            "completed": False,
            "pid": os.getpid(),
            "seconds": None,
            "peak_rss_mib": None,
        }

    return {
        "completed": bool(getattr(module, "PRELOAD_COMPLETED", False)),
        "pid": int(getattr(module, "PRELOAD_PID", os.getpid())),
        "seconds": float(getattr(module, "PRELOAD_SECONDS", -1.0)),
        "peak_rss_mib": float(getattr(module, "PRELOAD_PEAK_RSS_MIB", -1.0)),
    }


def _parse_sample_workspace(path: str | Path) -> tuple[int, int]:
    """
    Extract numeric MLMC level and sample ids without importing transport code.
    """
    sample_dir_name = Path(path).name
    parts = sample_dir_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected MLMC sample workspace format: {sample_dir_name}")

    level_tag, sample_tag = parts[:2]
    if not level_tag.startswith("L") or not sample_tag.startswith("S"):
        raise ValueError(f"Unexpected MLMC sample workspace format: {sample_dir_name}")
    return int(level_tag[1:]), int(sample_tag[1:])


def calculate_transport_saltelli(
    config_dict: dict[str, Any],
    sample_input: Sequence[float],
) -> tuple[Any, Any]:
    """
    Evaluate every Saltelli term through the configured forward simulation.

    Heavy chodby transport modules and NumPy are imported only after Dask has
    deserialized the task and started executing it on a worker.
    """
    if sample_input is None:
        raise ValueError("Missing planned Saltelli sample input")

    n_terms = int(config_dict["n_saltelli_terms"])
    n_parameters = int(config_dict["n_parameters"])
    own_size = n_terms * n_parameters
    mlmc_level_id, mlmc_sample_id = _parse_sample_workspace(os.getcwd())
    sample_dir_name = Path.cwd().name
    logging.info(
        "Evaluating Saltelli MLMC sample %s (sample_no=%s) on MLMC level %s with %s terms.",
        sample_dir_name,
        mlmc_sample_id,
        mlmc_level_id,
        n_terms,
    )

    import numpy as np
    import chodby_trans.transport_simulation as transport_simulation

    sample_array = np.asarray(sample_input, dtype=float)
    sample_matrix = sample_array[:own_size].reshape(n_terms, n_parameters)
    forward_params = sample_array[own_size:]
    simulation_class = getattr(
        transport_simulation,
        config_dict["forward_simulation_class"],
    )

    fine_results = []
    coarse_results = []
    for i_saltelli, input_vector in enumerate(sample_matrix):
        fine_result, coarse_result = simulation_class.calculate(
            config_dict["forward_config"],
            (*input_vector, i_saltelli, *forward_params),
        )
        fine_results.append(np.asarray(fine_result).flatten())
        coarse_results.append(np.asarray(coarse_result).flatten())

    return np.asarray(fine_results).flatten(), np.asarray(coarse_results).flatten()
