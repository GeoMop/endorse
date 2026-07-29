from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numcodecs import Zstd

import chodby_trans.job as job
from chodby_trans import transport_simulation

from mlmc.sample_storage_hdf import SampleStorageHDF


def _variance(values: np.ndarray) -> np.ndarray:
    """
    Return sample variance over the last axis, using NaN for undersampled data.
    """
    if values.shape[-1] < 2:
        return np.full(values.shape[:-1], np.nan)
    return np.var(values, axis=-1, ddof=1)


def _paired_output_labels(result_spec) -> list[str]:
    """
    Build labels for paired-mode result specs without a leading Saltelli axis.
    """
    tail_shape = tuple(result_spec.shape)
    tail_size = int(np.prod(tail_shape, dtype=int)) if tail_shape else 1
    labels = []
    i_output = 0
    for time in result_spec.times:
        for location in result_spec.locations:
            for i_tail in range(tail_size):
                tail_label = "" if tail_size == 1 else f", component={i_tail}"
                labels.append(f"i={i_output}, t={time}, loc={location}{tail_label}")
                i_output += 1
    return labels


def _level_collected_ids(level_group) -> list[str]:
    """
    Read collected sample ids for one MLMC HDF level group.
    """
    with h5py.File(level_group.file_name, "r") as hdf_file:
        group = hdf_file[level_group.level_group_path]
        if "collected_ids" not in group:
            return []
        return [sample["sample_id"].decode() for sample in group["collected_ids"][()]]


def _collected_level_pairs(storage: SampleStorageHDF) -> Iterable[tuple[int, np.ndarray, list[str]]]:
    """
    Yield collected sample-pair arrays and ids for levels that have ``collected_values``.
    """
    for level_id in storage.get_level_ids():
        level_chunks = []
        try:
            chunk_specs = list(storage.chunks(level_id=int(level_id)))
        except AttributeError:
            logging.info("Skipping MLMC level %s without collected_values in HDF storage.", level_id)
            continue

        for chunk_spec in chunk_specs:
            chunk = storage.sample_pairs_level(chunk_spec)
            if chunk is not None and len(chunk) > 0:
                level_chunks.append(chunk)

        if level_chunks:
            level_group = storage._level_groups[int(level_id)]
            yield int(level_id), np.concatenate(level_chunks, axis=1), _level_collected_ids(level_group)


def _split_paired_level_blocks(storage: SampleStorageHDF) -> Iterable[tuple]:
    """
    Yield paired-mode HDF result blocks as ``(spec, level_id, values, labels, sample_ids)``.
    """
    result_format = storage.load_result_format()
    for level_id, sample_pairs, sample_ids in _collected_level_pairs(storage):
        offset = 0
        for result_spec in result_format:
            output_labels = _paired_output_labels(result_spec)
            block_size = len(output_labels)
            block = sample_pairs[offset:offset + block_size]
            if block.shape[0] != block_size:
                raise ValueError(
                    f"Stored paired result for {result_spec.name!r} has too few values: "
                    f"expected {block_size}, got {block.shape[0]}."
                )
            values = block.reshape(block_size, block.shape[1], block.shape[2])
            yield result_spec, level_id, values, output_labels, sample_ids
            offset += block_size


def _correlation(fine_values: np.ndarray, coarse_values: np.ndarray) -> np.ndarray:
    """
    Compute per-output fine/coarse sample correlation with NaN for undersampled data.
    """
    corr = np.full(fine_values.shape[0], np.nan, dtype=float)
    for i_output in range(fine_values.shape[0]):
        fine = fine_values[i_output]
        coarse = coarse_values[i_output]
        mask = np.isfinite(fine) & np.isfinite(coarse)
        if np.count_nonzero(mask) < 2:
            continue
        if np.nanstd(fine[mask]) == 0.0 or np.nanstd(coarse[mask]) == 0.0:
            continue
        corr[i_output] = np.corrcoef(fine[mask], coarse[mask])[0, 1]
    return corr


def _bootstrap_variance_iqr(
    values: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-output variance uncertainty by bootstrap resampling.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    lower = np.full(values.shape[1], np.nan, dtype=float)
    upper = np.full(values.shape[1], np.nan, dtype=float)

    for i_output in range(values.shape[1]):
        output_values = values[:, i_output]
        output_values = output_values[np.isfinite(output_values)]
        if output_values.size < 2:
            continue
        sample_idx = rng.integers(0, output_values.size, size=(n_bootstrap, output_values.size))
        boot_vars = np.var(output_values[sample_idx], axis=1, ddof=1)
        lower[i_output], upper[i_output] = np.nanquantile(boot_vars, [0.25, 0.75])

    return lower, upper


def _time_axis(times: list[float]) -> tuple[np.ndarray, bool]:
    """
    Match the sequential diagnostic plot time axis convention.
    """
    t_raw = np.asarray(times, dtype=float) / 1000.0
    t_pos = t_raw[t_raw > 0]
    if t_pos.size == 0:
        return np.arange(len(times), dtype=float), False
    return np.where(t_raw > 0, t_raw, float(t_pos.min()) * 0.999), True


def _sample_number(sample_id: str) -> int:
    """
    Extract the numeric sample index from an MLMC sample id.
    """
    sample_tag = sample_id.split("_", 1)[1]
    if not sample_tag.startswith("S"):
        raise ValueError(f"Unexpected MLMC sample id format: {sample_id}")
    return int(sample_tag[1:])


def _paired_level_metadata(metadata: pd.DataFrame, level_id: int, sample_ids: list[str]) -> pd.DataFrame:
    """
    Select Zarr metadata rows matching collected HDF sample ids.
    """
    sample_numbers = [_sample_number(sample_id) for sample_id in sample_ids]
    selected = metadata[
        (metadata["level_id"] == level_id)
        & metadata["sample_id"].isin(sample_numbers)
    ].copy()
    selected["_order"] = pd.Categorical(selected["sample_id"], categories=sample_numbers, ordered=True)
    selected = selected.sort_values("_order").drop(columns=["_order"])
    if len(selected) != len(sample_ids):
        raise ValueError(
            f"Expected {len(sample_ids)} Zarr metadata rows for level {level_id}, got {len(selected)}."
        )
    return selected


def mlmc_paired_diagnostics(storage: SampleStorageHDF) -> pd.DataFrame:
    """
    Build variance, correlation, and bias diagnostics for paired MLMC samples.
    """
    rows: list[dict] = []
    for result_spec, level_id, values, output_labels, _sample_ids in _split_paired_level_blocks(storage):
        fine_values = values[:, :, 0]
        if values.shape[-1] > 1:
            coarse_values = values[:, :, 1]
        else:
            coarse_values = np.full_like(fine_values, np.nan)
        diff_values = fine_values - coarse_values

        fine_var = _variance(fine_values)
        coarse_var = _variance(coarse_values)
        diff_var = _variance(diff_values)
        corr = _correlation(fine_values, coarse_values)
        if values.shape[-1] > 1:
            bias = np.nanmean(diff_values, axis=1)
        else:
            bias = np.full(fine_values.shape[0], np.nan, dtype=float)

        for i_output, output_label in enumerate(output_labels):
            coarse = coarse_var[i_output]
            diff = diff_var[i_output]
            rows.append(
                {
                    "result": result_spec.name,
                    "level_id": level_id,
                    "n_samples": values.shape[1],
                    "output_index": i_output,
                    "output_label": output_label,
                    "fine_variance": fine_var[i_output],
                    "coarse_variance": coarse,
                    "diff_variance": diff,
                    "diff_to_coarse": diff / coarse if coarse > 0 else np.nan,
                    "correlation": corr[i_output],
                    "bias": bias[i_output],
                }
            )

    if not rows:
        raise ValueError("No collected paired MLMC samples found in the HDF storage.")
    return pd.DataFrame(rows)


def read_mlmc_paired_zarr_metadata() -> pd.DataFrame:
    """
    Read paired-mode return codes, parameters, and timings from MLMC Zarr groups.
    """
    store_path = job.output.zarr_store_path / transport_simulation.MLMC_ZARR_GROUP
    if not store_path.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for group_path in sorted(path for path in store_path.iterdir() if path.is_dir()):
        group_name = group_path.name
        if not group_name.startswith("level_"):
            continue
        level_id = int(group_name.split("_", 1)[1])
        param_names = [str(name) for name in _read_zarr_v3_array(group_path / "param_name")]
        i_sample = _read_zarr_v3_array(group_path / "i_sample")
        fine_rc = _read_zarr_v3_array(group_path / "fine_return_code")[:, 0]
        coarse_rc = _read_zarr_v3_array(group_path / "coarse_return_code")[:, 0]
        fine_time = _read_zarr_v3_array(group_path / "fine_eval_time")[:, 0]
        coarse_time = _read_zarr_v3_array(group_path / "coarse_eval_time")[:, 0]
        parameters = _read_zarr_v3_array(group_path / "parameter")[:, 0, :]
        for row_id, sample_id in enumerate(i_sample):
            row = {
                "level_id": level_id,
                "sample_id": int(sample_id),
                "fine_return_code": int(fine_rc[row_id]),
                "coarse_return_code": int(coarse_rc[row_id]),
                "fine_eval_time": float(fine_time[row_id]),
                "coarse_eval_time": float(coarse_time[row_id]),
            }
            row.update(
                {
                    f"param:{param_name}": float(parameters[row_id, i_param])
                    for i_param, param_name in enumerate(param_names)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _zarr_dtype(data_type) -> np.dtype:
    """
    Convert the limited Zarr v3 data-type metadata used by MLMC stores to NumPy dtype.
    """
    if data_type == "float64":
        return np.dtype("<f8")
    if data_type == "int64":
        return np.dtype("<i8")
    if data_type == "int32":
        return np.dtype("<i4")
    if isinstance(data_type, dict) and data_type.get("name") == "fixed_length_utf32":
        length_bytes = int(data_type["configuration"]["length_bytes"])
        return np.dtype(f"<U{length_bytes // 4}")
    raise TypeError(f"Unsupported Zarr metadata dtype: {data_type!r}")


def _zarr_fill_value(fill_value, dtype: np.dtype):
    if fill_value == "NaN":
        return np.nan
    if fill_value is None:
        return 0
    return fill_value


def _read_zarr_v3_array(array_path: Path) -> np.ndarray:
    """
    Read a local Zarr v3 array without using zarr.open_array.

    The current zarr synchronous API hangs on some local metadata stores in this
    project. The MLMC analysis only needs small metadata arrays, all written as
    bytes plus zstd chunks, so direct decoding is simpler and deterministic.
    """
    metadata = json.loads((array_path / "zarr.json").read_text(encoding="utf-8"))
    shape = tuple(int(size) for size in metadata["shape"])
    chunk_shape = tuple(int(size) for size in metadata["chunk_grid"]["configuration"]["chunk_shape"])
    dtype = _zarr_dtype(metadata["data_type"])
    fill_value = _zarr_fill_value(metadata.get("fill_value"), dtype)
    result = np.full(shape, fill_value, dtype=dtype)
    chunk_root = array_path / "c"
    if not chunk_root.exists():
        return result

    n_chunks = tuple((size + chunk - 1) // chunk for size, chunk in zip(shape, chunk_shape))
    decoder = Zstd(level=0, checksum=False)
    for chunk_index in product(*(range(n_chunk) for n_chunk in n_chunks)):
        chunk_path = chunk_root.joinpath(*(str(idx) for idx in chunk_index))
        if not chunk_path.exists():
            continue
        decoded = decoder.decode(chunk_path.read_bytes())
        chunk = np.frombuffer(decoded, dtype=dtype).reshape(chunk_shape)
        slices = tuple(
            slice(idx * chunk_len, min((idx + 1) * chunk_len, size))
            for idx, chunk_len, size in zip(chunk_index, chunk_shape, shape)
        )
        local_slices = tuple(slice(0, sl.stop - sl.start) for sl in slices)
        result[slices] = chunk[local_slices]
    return result


def _plot_mlmc_paired_summary(
    *,
    result_name: str,
    level_id: int,
    fine_values: np.ndarray,
    coarse_values: np.ndarray,
    times: list[float],
    fine_times: np.ndarray,
    coarse_times: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """
    Write sequential-style fine/coarse MLMC diagnostic plots for one paired result block.
    """
    diff_values = fine_values - coarse_values
    n_samples = fine_values.shape[0]
    cost_reduction = np.nanmean(fine_times + coarse_times) / np.nanmean(coarse_times)

    fine_var = np.nanvar(fine_values, axis=0, ddof=1)
    coarse_var = np.nanvar(coarse_values, axis=0, ddof=1)
    diff_var = np.nanvar(diff_values, axis=0, ddof=1)
    fine_var_q25, fine_var_q75 = _bootstrap_variance_iqr(fine_values, seed=12345)
    coarse_var_q25, coarse_var_q75 = _bootstrap_variance_iqr(coarse_values, seed=12346)
    diff_var_q25, diff_var_q75 = _bootstrap_variance_iqr(diff_values, seed=12347)
    ratio = np.divide(coarse_var, fine_var, out=np.full_like(fine_var, np.nan), where=fine_var > 0)
    reduction = np.divide(coarse_var, diff_var, out=np.full_like(coarse_var, np.nan), where=diff_var > 0)
    bias = np.nanmean(diff_values, axis=0)
    diff_q25, diff_q75 = np.nanquantile(diff_values, [0.25, 0.75], axis=0)
    corr = _correlation(fine_values.T, coarse_values.T)
    t, use_log_x = _time_axis(times)

    output_dir.mkdir(parents=True, exist_ok=True)

    def configure_x_axis(ax) -> None:
        if use_log_x:
            ax.set_xscale("log")
            ax.set_xlabel("Time from 50y pulse (ky)")
        else:
            ax.set_xlabel("Output time index")

    def plot_variances(ax) -> None:
        for lower, upper, color in [
            (coarse_var_q25, coarse_var_q75, "tab:orange"),
            (diff_var_q25, diff_var_q75, "tab:green"),
        ]:
            lower_plot = np.where(lower > 0, lower, np.nan)
            upper_plot = np.where(upper > 0, upper, np.nan)
            ax.fill_between(t, lower_plot, upper_plot, color=color, alpha=0.16, linewidth=0.0)

        ax.plot(t, coarse_var, label="Var(coarse)", color="tab:orange")
        ax.plot(t, diff_var, label="Var(fine - coarse)", color="tab:green")
        ax.set_yscale("log")
        ax.set_ylabel("Variance")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    def plot_variance_reduction(ax) -> None:
        ax.plot(t, reduction, label="Var(coarse) / Var(fine - coarse)", color="tab:purple")
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax.set_ylabel("Variance reduction")
        legend = ax.legend(loc="best")
        ax.grid(alpha=0.25)

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_bbox_axes = legend_bbox.transformed(ax.transAxes.inverted())
        ax.text(
            legend_bbox_axes.x0 + 0.1,
            legend_bbox_axes.y0 - 0.1,
            f"Cost reduction = {cost_reduction:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.5",
                "alpha": 0.9,
            },
        )

    def plot_correlation(ax) -> None:
        ax.plot(t, corr, label="Corr(fine, coarse)", color="tab:red")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Correlation")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    def plot_difference(ax) -> None:
        ax.fill_between(t, diff_q25, diff_q75, color="tab:green", alpha=0.2, label="IQR(fine - coarse)")
        ax.plot(t, bias, label="Mean(fine - coarse)", color="tab:green")
        ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
        ax.set_ylabel("Difference")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)
    plot_variances(axes[0])
    plot_variance_reduction(axes[1])
    plot_correlation(axes[2])
    plot_difference(axes[3])
    configure_x_axis(axes[-1])
    fig.suptitle(f"Fine/coarse MLMC diagnostics, n={n_samples}")
    fig.tight_layout()

    prefix = "" if result_name == "log10_conc_q99_xyz" else f"{result_name}_level_{level_id:02d}_"
    out_path = output_dir / f"{prefix}fine_coarse_mlmc_diagnostics.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    subfigs_dir = output_dir / "subfigs"
    subfigs_dir.mkdir(parents=True, exist_ok=True)
    written_paths = [out_path]
    individual_plots = [
        ("fine_coarse_variances.pdf", "Fine/coarse variances", plot_variances),
        ("fine_coarse_variance_reduction.pdf", "Fine/coarse variance reduction", plot_variance_reduction),
        ("fine_coarse_correlation.pdf", "Fine/coarse correlation", plot_correlation),
        ("fine_coarse_difference.pdf", "Fine/coarse difference", plot_difference),
    ]
    for filename, title, plot_function in individual_plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_function(ax)
        configure_x_axis(ax)
        ax.set_title(f"{title}, n={n_samples}")
        fig.tight_layout()
        path = subfigs_dir / f"{prefix}{filename}"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written_paths.append(path)

    diagnostics_csv = output_dir / f"{prefix}fine_coarse_mlmc_diagnostics.csv"
    diagnostics = pd.DataFrame(
        {
            "time": t,
            "fine_var": fine_var,
            "fine_var_q25": fine_var_q25,
            "fine_var_q75": fine_var_q75,
            "coarse_var": coarse_var,
            "coarse_var_q25": coarse_var_q25,
            "coarse_var_q75": coarse_var_q75,
            "diff_var": diff_var,
            "diff_var_q25": diff_var_q25,
            "diff_var_q75": diff_var_q75,
            "coarse_fine_var_ratio": ratio,
            "fine_diff_var_reduction": reduction,
            "fine_coarse_corr": corr,
            "mean_diff": bias,
            "diff_q25": diff_q25,
            "diff_q75": diff_q75,
        }
    )
    diagnostics.to_csv(diagnostics_csv, index=False)
    written_paths.append(diagnostics_csv)
    return written_paths


def plot_mlmc_paired_diagnostics(
    storage: SampleStorageHDF,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """
    Write paired fine/coarse diagnostics matching the sequential Saltelli collector plots.
    """
    written_paths: list[Path] = []
    for result_spec, level_id, values, _output_labels, sample_ids in _split_paired_level_blocks(storage):
        if values.shape[-1] < 2:
            logging.info("Skipping paired plot for level %s without coarse side.", level_id)
            continue
        level_metadata = _paired_level_metadata(metadata, level_id, sample_ids)
        fine_values = values[:, :, 0].T
        coarse_values = values[:, :, 1].T
        fine_times = level_metadata["fine_eval_time"].to_numpy(dtype=float)
        coarse_times = level_metadata["coarse_eval_time"].to_numpy(dtype=float)
        written_paths.extend(
            _plot_mlmc_paired_summary(
                result_name=result_spec.name,
                level_id=level_id,
                fine_values=fine_values,
                coarse_values=coarse_values,
                times=[float(time) for time in result_spec.times],
                fine_times=fine_times,
                coarse_times=coarse_times,
                output_dir=output_dir,
            )
        )
    return written_paths


def run_mlmc_paired_analysis(storage: SampleStorageHDF, output_dir: Path) -> None:
    """
    Read paired MLMC HDF and Zarr metadata and write diagnostics.
    """
    df = mlmc_paired_diagnostics(storage)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mlmc_paired_diagnostics.csv"
    df.to_csv(csv_path, index=False)

    metadata = read_mlmc_paired_zarr_metadata()
    if metadata.empty:
        raise ValueError("Paired MLMC analysis requires Zarr metadata, but no rows were found.")
    metadata_path = output_dir / "mlmc_paired_zarr_metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    logging.info("Wrote MLMC paired Zarr metadata table: %s", metadata_path)

    pdf_paths = plot_mlmc_paired_diagnostics(storage, metadata, output_dir)
    logging.info("Wrote MLMC paired diagnostics table: %s", csv_path)
    for pdf_path in pdf_paths:
        logging.info("Wrote MLMC paired diagnostic plot: %s", pdf_path)
