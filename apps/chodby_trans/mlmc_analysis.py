from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import zarr

from endorse.common import dotdict

import chodby_trans.job as job
from chodby_trans import ot_sa
from chodby_trans import transport_simulation

from mlmc.quantity.sobol import SaltelliSchema
from mlmc.sample_storage_hdf import SampleStorageHDF


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _variance(values: np.ndarray) -> np.ndarray:
    """
    Return sample variance over the last axis, using NaN for undersampled data.
    """
    if values.shape[-1] < 2:
        return np.full(values.shape[:-1], np.nan)
    return np.var(values, axis=-1, ddof=1)


def _stored_output_labels(result_spec) -> list[str]:
    """
    Build labels for the non-Saltelli output axis in the stored term-major result.
    """
    tail_shape = tuple(result_spec.shape[1:])
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


def _term_labels(schema: SaltelliSchema) -> list[str]:
    return (
        ["A"]
        + [f"AB_{i_param}" for i_param in range(schema.n_parameters)]
        + [f"BA_{i_param}" for i_param in range(schema.n_parameters)]
        + ["B"]
    )


def _second_order_labels(schema: SaltelliSchema) -> list[str]:
    return [
        f"S2_num_{i_param}_{j_param}"
        for i_param in range(schema.n_parameters)
        for j_param in range(i_param + 1, schema.n_parameters)
    ]


def _split_level_blocks(storage: SampleStorageHDF, schema: SaltelliSchema) -> Iterable[tuple]:
    """
    Yield stored Saltelli result blocks as ``(spec, level_id, values, labels)``.
    """
    result_format = storage.load_result_format()
    level_pairs = storage.sample_pairs()
    for level_id, sample_pairs in enumerate(level_pairs):
        if len(sample_pairs) == 0:
            continue
        offset = 0
        for result_spec in result_format:
            if tuple(result_spec.shape[:1]) != (schema.n_terms,):
                raise ValueError(
                    f"Expected first result axis to contain {schema.n_terms} Saltelli terms, "
                    f"got shape {result_spec.shape} for {result_spec.name!r}."
                )
            output_labels = _stored_output_labels(result_spec)
            n_outputs = len(output_labels)
            block_size = schema.n_terms * n_outputs
            block = sample_pairs[offset:offset + block_size]
            if block.shape[0] != block_size:
                raise ValueError(
                    f"Stored result for {result_spec.name!r} has too few values: "
                    f"expected {block_size}, got {block.shape[0]}."
                )
            values = block.reshape(schema.n_terms, n_outputs, block.shape[1], block.shape[2])
            yield result_spec, level_id, values, output_labels
            offset += block_size


def _sobol_mean_quantity(values: np.ndarray, schema: SaltelliSchema, side_id: int) -> np.ndarray:
    side = values[..., side_id]
    return 0.5 * (side[schema.a] + side[schema.b])


def _mlmc_mean_centers(level_blocks: list[tuple], schema: SaltelliSchema) -> dict[str, np.ndarray]:
    """
    Compute the MLMC mean value used to center the Sobol denominator quantity.
    """
    centers: dict[str, np.ndarray] = {}
    for result_spec, level_id, values, _labels in level_blocks:
        fine_mean = _sobol_mean_quantity(values, schema, side_id=0)
        if level_id == 0 or values.shape[-1] == 1:
            level_diff = fine_mean
        else:
            coarse_mean = _sobol_mean_quantity(values, schema, side_id=1)
            level_diff = fine_mean - coarse_mean
        centers.setdefault(result_spec.name, np.zeros(level_diff.shape[0], dtype=float))
        centers[result_spec.name] += np.mean(level_diff, axis=1)
    return centers


def _sobol_averaging_values(
    values: np.ndarray,
    schema: SaltelliSchema,
    center: np.ndarray,
    side_id: int,
) -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Construct per-sample values matching the averaging quantities in ``mlmc.quantity.sobol``.
    """
    side = values[..., side_id]
    a_values = side[schema.a]
    b_values = side[schema.b]
    ab_values = side[schema.ab]
    ba_values = side[schema.ba]
    center_values = center[:, None]

    mean_quantity = 0.5 * (a_values + b_values)
    denominator_quantity = 0.5 * ((a_values - center_values) ** 2 + (b_values - center_values) ** 2)
    first_order_quantity = (ab_values - a_values[None, ...]) * b_values[None, ...]
    total_order_quantity = 0.5 * (ab_values - a_values[None, ...]) ** 2

    first_terms = first_order_quantity
    second_order_terms = [
        ba_values[i_param] * ab_values[j_param]
        - a_values * b_values
        - first_terms[i_param]
        - first_terms[j_param]
        for i_param in range(schema.n_parameters)
        for j_param in range(i_param + 1, schema.n_parameters)
    ]
    if second_order_terms:
        second_order_quantity = np.asarray(second_order_terms)
    else:
        second_order_quantity = np.empty((0, values.shape[1], values.shape[2]))

    param_labels = [str(i_param) for i_param in range(schema.n_parameters)]
    return {
        "model_value": (side, _term_labels(schema)),
        "mean": (mean_quantity[None, ...], ["mean"]),
        "denominator": (denominator_quantity[None, ...], ["denominator"]),
        "first_order_numerator": (first_order_quantity, [f"S1_num_{label}" for label in param_labels]),
        "total_order_numerator": (total_order_quantity, [f"ST_num_{label}" for label in param_labels]),
        "second_order_numerator": (second_order_quantity, _second_order_labels(schema)),
    }


def _append_variance_rows(
    rows: list[dict],
    result_name: str,
    level_id: int,
    n_samples: int,
    output_labels: list[str],
    quantity_name: str,
    fine_values: np.ndarray,
    coarse_values: np.ndarray | None,
    component_labels: list[str],
) -> None:
    if coarse_values is None:
        diff_values = fine_values
        coarse_vars = np.full(fine_values.shape[:2], np.nan)
    else:
        diff_values = fine_values - coarse_values
        coarse_vars = _variance(coarse_values)

    fine_vars = _variance(fine_values)
    diff_vars = _variance(diff_values)

    for i_component, component_label in enumerate(component_labels):
        for i_output, output_label in enumerate(output_labels):
            coarse_var = coarse_vars[i_component, i_output]
            diff_var = diff_vars[i_component, i_output]
            rows.append(
                {
                    "result": result_name,
                    "level_id": level_id,
                    "n_samples": n_samples,
                    "quantity": quantity_name,
                    "component": component_label,
                    "output_index": i_output,
                    "output_label": output_label,
                    "fine_variance": fine_vars[i_component, i_output],
                    "coarse_variance": coarse_var,
                    "diff_variance": diff_var,
                    "diff_to_coarse": diff_var / coarse_var if coarse_var > 0 else np.nan,
                }
            )


def mlmc_variance_diagnostics(storage: SampleStorageHDF, schema: SaltelliSchema) -> pd.DataFrame:
    """
    Build variance diagnostics for stored MLMC Sobol sample pairs.
    """
    level_blocks = list(_split_level_blocks(storage, schema))
    if not level_blocks:
        raise ValueError("No collected MLMC samples found in the HDF storage.")

    centers = _mlmc_mean_centers(level_blocks, schema)
    rows: list[dict] = []
    for result_spec, level_id, values, output_labels in level_blocks:
        center = centers[result_spec.name]
        fine_quantities = _sobol_averaging_values(values, schema, center, side_id=0)
        coarse_quantities = None
        if level_id != 0 and values.shape[-1] > 1:
            coarse_quantities = _sobol_averaging_values(values, schema, center, side_id=1)

        for quantity_name, fine_data in fine_quantities.items():
            fine_values, component_labels = fine_data
            coarse_values = None
            if coarse_quantities is not None:
                coarse_values = coarse_quantities[quantity_name][0]
            _append_variance_rows(
                rows=rows,
                result_name=result_spec.name,
                level_id=level_id,
                n_samples=values.shape[2],
                output_labels=output_labels,
                quantity_name=quantity_name,
                fine_values=fine_values,
                coarse_values=coarse_values,
                component_labels=component_labels,
            )

    return pd.DataFrame(rows)


def plot_mlmc_variance_diagnostics(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """
    Write one multi-page PDF per diagnostic quantity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    value_columns = [
        ("fine_variance", "fine"),
        ("coarse_variance", "coarse"),
        ("diff_variance", "fine - coarse"),
    ]

    for quantity_name, quantity_df in df.groupby("quantity", sort=False):
        pdf_path = output_dir / f"mlmc_variance_{_safe_name(quantity_name)}.pdf"
        with PdfPages(pdf_path) as pdf:
            group_cols = ["result", "level_id", "component"]
            for (result_name, level_id, component), group_df in quantity_df.groupby(group_cols, sort=False):
                group_df = group_df.sort_values("output_index")
                fig, ax = plt.subplots(figsize=(10, 5))
                x_values = group_df["output_index"].to_numpy()
                for column, label in value_columns:
                    y_values = group_df[column].to_numpy(dtype=float)
                    if np.any(np.isfinite(y_values)):
                        ax.plot(x_values, y_values, marker="o", label=label)
                if np.any(group_df[[column for column, _label in value_columns]].to_numpy(dtype=float) > 0):
                    ax.set_yscale("log")
                ax.set_title(f"{quantity_name}: {result_name}, L{level_id}, {component}")
                ax.set_xlabel("output index")
                ax.set_ylabel("sample variance")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
                labels = group_df["output_label"].astype(str).to_list()
                ax.set_xticks(x_values)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        written_paths.append(pdf_path)
    return written_paths


def _split_paired_level_blocks(storage: SampleStorageHDF) -> Iterable[tuple]:
    """
    Yield paired-mode HDF result blocks as ``(spec, level_id, values, labels)``.
    """
    result_format = storage.load_result_format()
    level_pairs = storage.sample_pairs()
    for level_id, sample_pairs in enumerate(level_pairs):
        if len(sample_pairs) == 0:
            continue
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
            yield result_spec, level_id, values, output_labels
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


def mlmc_paired_diagnostics(storage: SampleStorageHDF) -> pd.DataFrame:
    """
    Build variance, correlation, and bias diagnostics for paired MLMC samples.
    """
    rows: list[dict] = []
    for result_spec, level_id, values, output_labels in _split_paired_level_blocks(storage):
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
    store_path = job.output.zarr_store_path
    if not store_path.exists():
        return pd.DataFrame()

    root = zarr.open_group(str(store_path), mode="r")
    if transport_simulation.MLMC_ZARR_GROUP not in root:
        return pd.DataFrame()

    rows: list[dict] = []
    mlmc_group = root[transport_simulation.MLMC_ZARR_GROUP]
    for group_name in sorted(mlmc_group.group_keys()):
        if not group_name.startswith("level_"):
            continue
        level_id = int(group_name.split("_", 1)[1])
        level_group = mlmc_group[group_name]
        param_names = [str(name) for name in np.asarray(level_group["param_name"])]
        i_sample = np.asarray(level_group["i_sample"])
        fine_rc = np.asarray(level_group["fine_return_code"])[:, 0]
        coarse_rc = np.asarray(level_group["coarse_return_code"])[:, 0]
        fine_time = np.asarray(level_group["fine_eval_time"])[:, 0]
        coarse_time = np.asarray(level_group["coarse_eval_time"])[:, 0]
        parameters = np.asarray(level_group["parameter"])[:, 0, :]
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


def plot_mlmc_paired_diagnostics(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """
    Write paired fine/coarse variance and correlation/bias diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for result_name, result_df in df.groupby("result", sort=False):
        pdf_path = output_dir / f"mlmc_paired_{_safe_name(result_name)}.pdf"
        with PdfPages(pdf_path) as pdf:
            for level_id, level_df in result_df.groupby("level_id", sort=False):
                level_df = level_df.sort_values("output_index")
                x_values = level_df["output_index"].to_numpy()

                fig, ax = plt.subplots(figsize=(10, 5))
                for column, label in [
                    ("fine_variance", "fine"),
                    ("coarse_variance", "coarse"),
                    ("diff_variance", "fine - coarse"),
                ]:
                    y_values = level_df[column].to_numpy(dtype=float)
                    if np.any(np.isfinite(y_values)):
                        ax.plot(x_values, y_values, marker="o", label=label)
                if np.any(level_df[["fine_variance", "coarse_variance", "diff_variance"]].to_numpy() > 0):
                    ax.set_yscale("log")
                ax.set_title(f"Paired variance: {result_name}, L{level_id}")
                ax.set_xlabel("output index")
                ax.set_ylabel("sample variance")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x_values, level_df["correlation"].to_numpy(dtype=float), marker="o", label="correlation")
                ax.plot(x_values, level_df["bias"].to_numpy(dtype=float), marker="o", label="mean(fine - coarse)")
                ax.set_title(f"Paired correlation and bias: {result_name}, L{level_id}")
                ax.set_xlabel("output index")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        written_paths.append(pdf_path)
    return written_paths


def run_mlmc_analysis(cfg: dotdict) -> None:
    """
    Read MLMC HDF samples and write variance diagnostic plots.
    """
    storage_path = job.output.mlmc_hdf_path
    if not storage_path.exists():
        raise FileNotFoundError(f"MLMC HDF storage does not exist: {storage_path}")

    output_dir = job.output.plots / "mlmc_analysis"
    storage = SampleStorageHDF(str(storage_path))
    sample_mode = str(cfg.mlmc.get("sample_mode", "saltelli"))

    if sample_mode == "paired":
        df = mlmc_paired_diagnostics(storage)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "mlmc_paired_diagnostics.csv"
        df.to_csv(csv_path, index=False)
        metadata = read_mlmc_paired_zarr_metadata()
        if not metadata.empty:
            metadata_path = output_dir / "mlmc_paired_zarr_metadata.csv"
            metadata.to_csv(metadata_path, index=False)
            logging.info("Wrote MLMC paired Zarr metadata table: %s", metadata_path)
        pdf_paths = plot_mlmc_paired_diagnostics(df, output_dir)
        logging.info("Wrote MLMC paired diagnostics table: %s", csv_path)
        for pdf_path in pdf_paths:
            logging.info("Wrote MLMC paired diagnostic plot: %s", pdf_path)
        return

    if sample_mode != "saltelli":
        raise ValueError(
            f"Unsupported cfg.mlmc.sample_mode={sample_mode!r}; expected 'saltelli' or 'paired'."
        )

    sa_obj = ot_sa.SensitivityAnalysis.from_cfg(cfg.ot_sensitivity)
    schema = SaltelliSchema.make(n_parameters=len(sa_obj.groups))
    df = mlmc_variance_diagnostics(storage, schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mlmc_variance_diagnostics.csv"
    df.to_csv(csv_path, index=False)
    pdf_paths = plot_mlmc_variance_diagnostics(df, output_dir)

    logging.info("Wrote MLMC variance diagnostics table: %s", csv_path)
    for pdf_path in pdf_paths:
        logging.info("Wrote MLMC variance diagnostic plot: %s", pdf_path)
