from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from endorse.common import dotdict

from chodby_trans import ot_sa
from chodby_trans.mlmc_var_analysis import run_mlmc_paired_analysis

from mlmc.quantity.sobol import SaltelliSchema
from mlmc.sample_storage_hdf import SampleStorageHDF

import chodby_trans.job as job


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


def _split_level_blocks(storage: SampleStorageHDF, schema: SaltelliSchema) -> Iterable[tuple]:
    """
    Yield stored Saltelli result blocks as ``(spec, level_id, values, labels)``.
    """
    result_format = storage.load_result_format()
    for level_id, sample_pairs, _sample_ids in _collected_level_pairs(storage):
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
        run_mlmc_paired_analysis(storage, output_dir)
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
