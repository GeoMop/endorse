from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

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

    variant_patch = {}
    for name, param_cfg in cfg.ot_sensitivity.parameters.items():
        if "path" in param_cfg:
            variant_patch[param_cfg.path] = param_dict[name]

    return common.apply_variant(cfg, variant_patch), param_dict


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


class TransportSimulation(Simulation):
    """
    MLMC forward simulation adapter for the chodby transport model.
    """

    RESULT_NAME = "log10_conc_q99_xyz"
    RESULT_UNIT = "log10(g/m^3)"
    RESULT_LOCATION = "0"

    def __init__(self, workdir: Path | str, transport_config_path: Path | None = None):
        # AGENT: workdir should not be needed since we are only allowed to work relative to the directories
        # set by MLMC SamplingPool
        # Resolved: `workdir` is used only to locate the root config at construction time; per-sample execution
        # stays in the MLMC-provided sample workspace.
        workdir = Path(workdir)

        transport_config_path = (
            Path(transport_config_path)
            if transport_config_path is not None
            else workdir / "input_data" / "trans_mesh_config.yaml"
        )
        self.cfg = common.config.load_config(str(transport_config_path))
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
    def calculate(config_dict, sample_input):
        """
        Calculate one MLMC sample in the current sample workspace.

        Level-specific configuration is passed through ``config_dict``.
        The real transport branch calls ``transport_run`` as requested by the
        in-code instructions. The coarse output remains a placeholder until the
        transport-side pair result is exposed directly.
        """
        root_cfg = copy.deepcopy(config_dict["root_cfg"])
        sample_input = np.asarray(sample_input, dtype=float)
        finer_level_sample_size = int(sample_input[0])
        parameters = sample_input[1:]

        # AGENT: this is not allowed
        # else:
        #
        #     finer_level_sample_size = 0
        #     parameters = sample_input
        # Resolved: the MLMC/Saltelli driver always prepends the planning-time finer-level sample count, so the
        # worker keeps one explicit input layout only.

        level = int(config_dict["level_id"])

        # AGENT: wrong, MLMC sets us the sample workdir, we should remain there
        # job.set_workdir(Path(config_dict["workdir"]))
        # Resolved: the worker keeps the current MLMC sample workspace and does not reset the working directory.

        cfg, full_param_dict = apply_sample_parameters(root_cfg, parameters)
        cfg["data_schema_key"] = "run_timestamp"

        sample_dir = Path(os.getcwd())
        logging.info("Running MLMC transport sample in %s, level %s.", sample_dir, level)
        logging.info("Finer-level sample count at planning time: %s", finer_level_sample_size)

        if cfg.test_random_data:
            times = output_times(cfg.transport_fullscale)
            full_parameters = np.asarray(list(full_param_dict.values()), dtype=float)
            fine_values = synthetic_concentration(times, level, full_parameters)
            coarse_values = synthetic_concentration(times, level + 1, full_parameters)
            fine_rc = ReturnCode.OK
        else:
            fine_values, coarse_values = transport.transport_run(cfg, level, full_param_dict)
            fine_rc = ReturnCode.OK


        if fine_rc != ReturnCode.OK:
            raise WrapperException(
                f"Transport evaluation failed with return code {fine_rc}",
                code=fine_rc,
            )

        fine_result = compact_concentration_series(fine_values)
        coarse_result = np.zeros_like(fine_result)
        if coarse_values is not None:
            coarse_result = compact_concentration_series(coarse_values)
        return fine_result, coarse_result
