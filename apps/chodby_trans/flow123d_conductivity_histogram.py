from __future__ import annotations

import argparse
import csv
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RockMarker:
    """Highlight one reference conductivity value in the histogram."""

    name: str
    log10_value: float
    color: str


@dataclass(frozen=True)
class HistogramConfig:
    """Parameters that reproduce the ParaView conductivity-histogram pipeline."""

    region_id_min: int = 1
    region_id_max: int = 101
    bin_count: int = 20
    total_volume: float = 64000.0
    default_bar_color: str = "crimson"
    default_bar_edge: str = "black"
    default_bar_linewidth: float = 0.6


@dataclass(frozen=True)
class HistogramResult:
    """Machine-readable histogram values for later postprocess automation."""

    dataset_path: Path
    time_value: float | None
    bin_left: np.ndarray
    bin_right: np.ndarray
    bin_centers: np.ndarray
    volume_percent: np.ndarray
    cell_count: np.ndarray
    conductivity_min: float
    conductivity_max: float
    selected_cell_count: int


DEFAULT_ROCK_MARKERS = (
    RockMarker("Backfill", -12.101, "forestgreen"),
    RockMarker("Bulk", -12.87, "dodgerblue"),
    RockMarker("EDZ", -12.5, "crimson"),
)


def require_pyvista():
    """Import pyvista only at the feature boundary."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "pyvista is required for Flow123d conductivity histograms. "
            "Use the chodby_trans virtual environment."
        ) from exc
    return pv


def require_matplotlib():
    """Import matplotlib only when rendering the figure."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for Flow123d conductivity histogram plotting. "
            "Use the chodby_trans virtual environment."
        ) from exc
    return plt, Patch


def require_numpy():
    """Import numpy only when numeric processing is needed."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for Flow123d conductivity histogram processing. "
            "Use the chodby_trans virtual environment."
        ) from exc
    return np


def resolve_pvd_dataset_file(pvd_path: Path, time_index: int = 0) -> tuple[Path, float | None]:
    """Resolve one dataset file referenced by a `.pvd` collection file."""
    root = ET.parse(pvd_path).getroot()
    collection = root.find("Collection")
    if collection is None:
        raise ValueError(f"Missing Collection element in '{pvd_path}'.")

    datasets = collection.findall("DataSet")
    if not datasets:
        raise ValueError(f"No DataSet entries found in '{pvd_path}'.")

    selected = datasets[time_index]
    file_name = selected.get("file")
    if not file_name:
        raise ValueError(f"DataSet entry at index {time_index} has no file attribute.")

    time_value = selected.get("timestep")
    resolved = (pvd_path.parent / file_name).resolve()
    return resolved, None if time_value is None else float(time_value)


def read_flow_fields_dataset(pvd_path: Path, time_index: int = 0):
    """Load one Flow123d dataset referenced from a `.pvd` file."""
    pv = require_pyvista()
    dataset_path, time_value = resolve_pvd_dataset_file(pvd_path=pvd_path, time_index=time_index)
    dataset = pv.read(dataset_path)
    if isinstance(dataset, pv.MultiBlock):
        dataset = dataset.combine(merge_points=False)
    LOGGER.info("Loaded dataset '%s' for time index %s.", dataset_path, time_index)
    return dataset, dataset_path, time_value


def compute_histogram(
    dataset,
    *,
    dataset_path: Path,
    time_value: float | None,
    config: HistogramConfig,
) -> HistogramResult:
    """Reproduce the ParaView threshold, volume weighting, and histogram extraction."""
    np = require_numpy()
    thresholded = dataset.threshold(
        value=(config.region_id_min, config.region_id_max),
        scalars="region_id",
        preference="cell",
        all_scalars=True,
        continuous=False,
    )
    selected_cell_count = thresholded.n_cells
    if selected_cell_count == 0:
        raise ValueError("The selected region_id interval produced an empty dataset.")

    conductivity = np.asarray(thresholded.cell_data["conductivity"], dtype=float)
    if np.any(conductivity <= 0.0):
        raise ValueError("Conductivity contains non-positive values, log10 is undefined.")

    with_sizes = thresholded.compute_cell_sizes(
        length=False,
        area=False,
        volume=True,
        vertex_count=False,
    )
    volume = np.abs(np.asarray(with_sizes.cell_data["Volume"], dtype=float))
    log_cond = np.log10(conductivity)
    volume_percent = volume / config.total_volume * 100.0

    bin_totals, bin_edges = np.histogram(log_cond, bins=config.bin_count, weights=volume_percent)
    cell_count, _ = np.histogram(log_cond, bins=bin_edges)
    bin_left = bin_edges[:-1]
    bin_right = bin_edges[1:]
    bin_centers = 0.5 * (bin_left + bin_right)

    LOGGER.info(
        "Histogram computed from %s selected cells, conductivity range [%g, %g].",
        selected_cell_count,
        conductivity.min(),
        conductivity.max(),
    )
    return HistogramResult(
        dataset_path=dataset_path,
        time_value=time_value,
        bin_left=bin_left,
        bin_right=bin_right,
        bin_centers=bin_centers,
        volume_percent=bin_totals,
        cell_count=cell_count.astype(int),
        conductivity_min=float(conductivity.min()),
        conductivity_max=float(conductivity.max()),
        selected_cell_count=selected_cell_count,
    )


def plot_histogram(
    result: HistogramResult,
    output_path: Path,
    *,
    rock_markers: Sequence[RockMarker] = DEFAULT_ROCK_MARKERS,
    config: HistogramConfig = HistogramConfig(),
    width_px: int = 1600,
    height_px: int = 900,
) -> None:
    """Render the weighted conductivity histogram with the ParaView view styling."""
    plt, Patch = require_matplotlib()
    np = require_numpy()
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
        }
    )

    x = result.bin_centers
    y = result.volume_percent
    fig, ax = plt.subplots(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)

    if len(x) > 1:
        dx = np.median(np.diff(np.sort(x)))
        bar_width = 0.9 * dx if np.isfinite(dx) and dx > 0.0 else 0.8
    else:
        bar_width = 0.8

    bars = ax.bar(x, y, width=bar_width)
    texts = ax.bar_label(bars, fmt="%.2g", padding=5)
    for text in texts:
        text.set_rotation(90)
        text.set_ha("center")
        text.set_va("bottom")

    for bar in bars:
        bar.set_facecolor(config.default_bar_color)
        bar.set_edgecolor(config.default_bar_edge)
        bar.set_linewidth(config.default_bar_linewidth)

    def value_to_bin_index(value: float) -> int:
        return int(np.nanargmin(np.abs(np.asarray(x, dtype=float) - value)))

    for marker in rock_markers:
        index = value_to_bin_index(marker.log10_value)
        if 0 <= index < len(bars):
            bars[index].set_facecolor(marker.color)
            bars[index].set_edgecolor("black")
            bars[index].set_linewidth(0.8)

    handles = [Patch(facecolor=marker.color, edgecolor="black") for marker in rock_markers]
    labels = [marker.name for marker in rock_markers]
    ax.legend(
        handles,
        labels,
        loc="best",
        title="Regions",
        frameon=True,
        borderaxespad=1.5,
        borderpad=1.2,
        labelspacing=0.75,
        handletextpad=0.8,
    )

    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.25)
    ax.set_xlabel("hydraulic conductivity (log10)")
    ax.set_ylabel("Volume (%)" if np.nanmax(y) <= 100.0 else "ElVolume_total")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved histogram figure to '%s'.", output_path)


def write_histogram_csv(result: HistogramResult, output_path: Path) -> None:
    """Persist the histogram table for downstream sample postprocess steps."""
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "bin_left",
                "bin_right",
                "bin_center",
                "volume_percent",
                "cell_count",
            ]
        )
        for row in zip(
            result.bin_left,
            result.bin_right,
            result.bin_centers,
            result.volume_percent,
            result.cell_count,
            strict=True,
        ):
            writer.writerow(row)
    LOGGER.info("Saved histogram table to '%s'.", output_path)


def export_conductivity_histogram(
    pvd_path: Path,
    *,
    output_path: Path,
    csv_output_path: Path | None = None,
    time_index: int = 0,
    config: HistogramConfig = HistogramConfig(),
    rock_markers: Sequence[RockMarker] = DEFAULT_ROCK_MARKERS,
    width_px: int = 1600,
    height_px: int = 900,
) -> HistogramResult:
    """Load one Flow123d result and export the histogram figure and CSV table."""
    dataset, dataset_path, time_value = read_flow_fields_dataset(
        pvd_path=pvd_path,
        time_index=time_index,
    )
    result = compute_histogram(
        dataset,
        dataset_path=dataset_path,
        time_value=time_value,
        config=config,
    )

    resolved_csv_output = csv_output_path
    if resolved_csv_output is None:
        resolved_csv_output = output_path.with_suffix(".csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_csv_output.parent.mkdir(parents=True, exist_ok=True)
    plot_histogram(
        result,
        output_path=output_path,
        rock_markers=rock_markers,
        config=config,
        width_px=width_px,
        height_px=height_px,
    )
    write_histogram_csv(result, output_path=resolved_csv_output)

    np = require_numpy()
    LOGGER.info(
        "Histogram summary: dataset=%s time=%s selected_cells=%s total_volume_percent=%g",
        result.dataset_path,
        result.time_value,
        result.selected_cell_count,
        np.sum(result.volume_percent),
    )
    return result


def parse_rock_marker(raw_value: str) -> RockMarker:
    """Parse `Name:log10_value:color` CLI markers."""
    name, log10_value, color = raw_value.split(":", maxsplit=2)
    return RockMarker(name=name, log10_value=float(log10_value), color=color)


def parse_args() -> argparse.Namespace:
    """Create the command-line interface for conductivity histogram export."""
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the conductivity histogram from "
            "workdir_41e_test_ot_2/cond_histogram.pvsm without ParaView state."
        )
    )
    parser.add_argument("pvd_path", type=Path, help="Path to Flow123d flow_fields.pvd.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cond_hist.pdf"),
        help="Output figure path. The suffix controls the format, e.g. .pdf or .png.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path for the histogram table. Defaults to <output>.csv.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Dataset index inside the PVD collection. Default matches the ParaView state.",
    )
    parser.add_argument("--bin-count", type=int, default=20, help="Number of histogram bins.")
    parser.add_argument(
        "--total-volume",
        type=float,
        default=64000.0,
        help="Normalization volume used in ElVolume = abs(Volume) / total_volume * 100.",
    )
    parser.add_argument(
        "--region-id-min",
        type=int,
        default=1,
        help="Lower inclusive threshold for the cell-data array region_id.",
    )
    parser.add_argument(
        "--region-id-max",
        type=int,
        default=76,
        help="Upper inclusive threshold for the cell-data array region_id.",
    )
    parser.add_argument(
        "--rock-marker",
        action="append",
        default=None,
        help="Repeatable marker definition in the form Name:log10_value:color.",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=1600,
        help="Figure width in pixels used for matplotlib rendering.",
    )
    parser.add_argument(
        "--figure-height",
        type=int,
        default=900,
        help="Figure height in pixels used for matplotlib rendering.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = HistogramConfig(
        region_id_min=args.region_id_min,
        region_id_max=args.region_id_max,
        bin_count=args.bin_count,
        total_volume=args.total_volume,
    )
    rock_markers = DEFAULT_ROCK_MARKERS
    if args.rock_marker:
        rock_markers = tuple(parse_rock_marker(raw_value) for raw_value in args.rock_marker)

    export_conductivity_histogram(
        pvd_path=args.pvd_path,
        output_path=args.output,
        csv_output_path=args.csv_output,
        time_index=args.time_index,
        config=config,
        rock_markers=rock_markers,
        width_px=args.figure_width,
        height_px=args.figure_height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
