from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from endorse import common

import chodby_trans.job as job
from chodby_trans import mlmc_var_analysis
from chodby_trans.mlmc_analysis import run_mlmc_analysis

from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.quantity.sobol import SaltelliSchema
from mlmc.sample_storage_hdf import SampleStorageHDF


def _sample_values(schema: SaltelliSchema, n_outputs: int, i_sample: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one fine/coarse pair whose difference has much smaller variance than the coarse side.
    """
    terms = np.arange(schema.n_terms, dtype=float).reshape(schema.n_terms, 1)
    outputs = np.arange(n_outputs, dtype=float).reshape(1, n_outputs)
    sample_shift = float(i_sample)
    coarse = 10.0 + terms + outputs + sample_shift
    fine = coarse + 0.05 * sample_shift
    return fine.flatten(), coarse.flatten()


def _write_test_hdf(path: Path, schema: SaltelliSchema, n_outputs: int = 2) -> None:
    result_format = [
        QuantitySpec(
            name="value",
            unit="1",
            shape=(schema.n_terms,),
            times=[0.0, 1.0],
            locations=["0"],
        )
    ]
    storage = SampleStorageHDF(str(path))
    storage.save_global_data(level_parameters=[[1.0], [10.0]], result_format=result_format)

    successful = {0: [], 1: []}
    for i_sample in range(4):
        fine, coarse = _sample_values(schema, n_outputs, i_sample)
        successful[0].append((f"L00_S{i_sample:07d}", (fine, coarse)))
        successful[1].append((f"L01_S{i_sample:07d}", (fine, coarse)))
    storage.save_samples(successful, {0: [], 1: []})


def _write_paired_test_hdf(
    path: Path,
    n_outputs: int = 2,
    collected_levels: tuple[int, ...] = (0, 1),
) -> None:
    result_format = [
        QuantitySpec(
            name="value",
            unit="1",
            shape=(1,),
            times=[0.0, 1.0],
            locations=["0"],
        )
    ]
    storage = SampleStorageHDF(str(path))
    storage.save_global_data(level_parameters=[[1.0], [10.0]], result_format=result_format)

    successful = {level_id: [] for level_id in collected_levels}
    for i_sample in range(4):
        outputs = np.arange(n_outputs, dtype=float)
        coarse = 10.0 + outputs + float(i_sample)
        fine = coarse + 0.05 * float(i_sample)
        for level_id in collected_levels:
            successful[level_id].append((f"L{level_id:02d}_S{i_sample:07d}", (fine, coarse)))
    storage.save_samples(successful, {0: [], 1: []})


def _paired_metadata(collected_levels: tuple[int, ...] = (0, 1), n_samples: int = 4) -> pd.DataFrame:
    rows = []
    for level_id in collected_levels:
        for sample_id in range(n_samples):
            rows.append(
                {
                    "level_id": level_id,
                    "sample_id": sample_id,
                    "fine_return_code": 0,
                    "coarse_return_code": 0,
                    "fine_eval_time": 2.0,
                    "coarse_eval_time": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_run_mlmc_analysis_writes_variance_diagnostics(tmp_path: Path):
    """
    Check that the MLMC analysis subcommand reads HDF sample pairs and writes variance diagnostics.
    """
    workdir = tmp_path / "workdir"
    input_dir = workdir / "input_data"
    input_dir.mkdir(parents=True)
    cfg_path = Path(__file__).parent / "input_data" / "trans_mesh_config_mlmc_goal1.yaml"
    cfg = common.config.load_config(str(cfg_path))
    job.set_workdir(workdir, input_dir)

    schema = SaltelliSchema.make(n_parameters=2)
    _write_test_hdf(job.output.mlmc_hdf_path, schema)

    run_mlmc_analysis(cfg)

    analysis_dir = job.output.plots / "mlmc_analysis"
    csv_path = analysis_dir / "mlmc_variance_diagnostics.csv"
    assert csv_path.exists()
    assert (analysis_dir / "mlmc_variance_mean.pdf").exists()
    assert (analysis_dir / "mlmc_variance_denominator.pdf").exists()

    diagnostics = pd.read_csv(csv_path)
    paired_mean = diagnostics[
        (diagnostics["level_id"] == 1)
        & (diagnostics["quantity"] == "mean")
        & diagnostics["coarse_variance"].notna()
    ]
    assert not paired_mean.empty
    assert np.all(paired_mean["diff_variance"] < paired_mean["coarse_variance"])


def test_run_mlmc_analysis_writes_paired_diagnostics(tmp_path: Path, monkeypatch):
    """
    Check that paired MLMC HDF samples write diagnostics without a Saltelli axis.
    """
    workdir = tmp_path / "workdir"
    input_dir = workdir / "input_data"
    input_dir.mkdir(parents=True)
    cfg_path = Path(__file__).parent / "input_data" / "trans_mesh_config_mlmc_goal1.yaml"
    cfg = common.config.load_config(str(cfg_path))
    cfg.mlmc["sample_mode"] = "paired"
    job.set_workdir(workdir, input_dir)

    _write_paired_test_hdf(job.output.mlmc_hdf_path)
    monkeypatch.setattr(
        mlmc_var_analysis,
        "read_mlmc_paired_zarr_metadata",
        lambda: _paired_metadata(),
    )

    run_mlmc_analysis(cfg)

    analysis_dir = job.output.plots / "mlmc_analysis"
    csv_path = analysis_dir / "mlmc_paired_diagnostics.csv"
    assert csv_path.exists()
    assert (analysis_dir / "mlmc_paired_zarr_metadata.csv").exists()
    assert (analysis_dir / "value_level_01_fine_coarse_mlmc_diagnostics.pdf").exists()
    assert (analysis_dir / "subfigs" / "value_level_01_fine_coarse_variances.pdf").exists()
    assert (analysis_dir / "subfigs" / "value_level_01_fine_coarse_variance_reduction.pdf").exists()
    assert (analysis_dir / "subfigs" / "value_level_01_fine_coarse_correlation.pdf").exists()
    assert (analysis_dir / "subfigs" / "value_level_01_fine_coarse_difference.pdf").exists()

    diagnostics = pd.read_csv(csv_path)
    assert set(["correlation", "bias", "diff_variance"]).issubset(diagnostics.columns)
    paired_level = diagnostics[diagnostics["level_id"] == 1]
    assert not paired_level.empty
    assert np.all(paired_level["diff_variance"] < paired_level["coarse_variance"])


def test_run_mlmc_analysis_skips_empty_paired_level(tmp_path: Path, monkeypatch):
    """
    Check that analysis tolerates staged HDF files where coarse level has no collected values yet.
    """
    workdir = tmp_path / "workdir"
    input_dir = workdir / "input_data"
    input_dir.mkdir(parents=True)
    cfg_path = Path(__file__).parent / "input_data" / "trans_mesh_config_mlmc_goal1.yaml"
    cfg = common.config.load_config(str(cfg_path))
    cfg.mlmc["sample_mode"] = "paired"
    job.set_workdir(workdir, input_dir)

    _write_paired_test_hdf(job.output.mlmc_hdf_path, collected_levels=(1,))
    monkeypatch.setattr(
        mlmc_var_analysis,
        "read_mlmc_paired_zarr_metadata",
        lambda: _paired_metadata(collected_levels=(1,)),
    )

    run_mlmc_analysis(cfg)

    csv_path = job.output.plots / "mlmc_analysis" / "mlmc_paired_diagnostics.csv"
    diagnostics = pd.read_csv(csv_path)
    assert set(diagnostics["level_id"]) == {1}
    assert not diagnostics.empty
