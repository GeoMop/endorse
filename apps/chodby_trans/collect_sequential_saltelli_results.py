from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/endorse_matplotlib")

import numpy as np

from endorse import common
from endorse.fullscale_transport import output_times


DEFAULT_OUTPUT_DIR = "sequential_saltelli"
DEFAULT_GATHER_DIR = "sequential_saltelli_gather"
DEFAULT_CONFIG_NAME = "transport_mlmc.yaml"
SAMPLE_GATHER_FILES = ("input.json", "status.json", "result.npz")


def sample_sort_key(sample_dir: Path) -> tuple[int, int, str]:
    """Sort sample directories by level, Saltelli index, and term name."""
    parts = sample_dir.name.split("_", 2)
    if len(parts) != 3 or not parts[0].startswith("L") or not parts[1].startswith("S"):
        return (10**9, 10**9, sample_dir.name)
    return (int(parts[0][1:]), int(parts[1][1:]), parts[2])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_config_path(workdir: Path, output_dir: Path, config_name: str) -> Path:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        summary_config = summary.get("config")
        if summary_config:
            config_path = Path(summary_config)
            if config_path.exists():
                return config_path

    return workdir / "input_data" / config_name


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def output_time_values(config_path: Path) -> list[float] | None:
    if not config_path.exists():
        return None

    cfg = common.config.load_config(str(config_path))
    return [float(time) for time in output_times(cfg.transport_fullscale)]


def result_time_values(output_dir: Path, config_path: Path, result_size: int) -> list[float] | None:
    gathered_times_path = output_dir / "output_times.json"
    if gathered_times_path.exists():
        times = [float(time) for time in load_json(gathered_times_path)["output_times"]]
    else:
        times = output_time_values(config_path)

    if times is None:
        return None

    if len(times) != result_size:
        raise ValueError(
            f"Result size {result_size} does not match {len(times)} output times from {config_path}."
        )
    return times


def time_columns(output_dir: Path, config_path: Path, result_size: int) -> list[str]:
    times = result_time_values(output_dir, config_path, result_size)
    if times is None:
        return [f"t_{idx}" for idx in range(result_size)]
    return [f"{time:g}" for time in times]


def iter_completed_sample_dirs(output_dir: Path) -> Iterable[Path]:
    for sample_dir in sorted(output_dir.iterdir(), key=sample_sort_key):
        if not sample_dir.is_dir():
            continue
        if (sample_dir / "input.json").exists() and (sample_dir / "result.npz").exists():
            yield sample_dir


def load_result_vector(npz: np.lib.npyio.NpzFile, key: str, sample_dir: Path) -> np.ndarray:
    values = np.asarray(npz[key], dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Empty {key!r} result in {sample_dir}")
    return values


def gather_result_files(
    workdir: Path,
    output_dir_name: str = DEFAULT_OUTPUT_DIR,
    gather_dir_name: str = DEFAULT_GATHER_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    overwrite: bool = False,
) -> Path:
    """
    Copy lightweight completed-sample files into a separate directory for local postprocessing.
    """
    workdir = workdir.absolute()
    output_dir = workdir / output_dir_name
    gather_dir = workdir / gather_dir_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Sequential output directory does not exist: {output_dir}")
    if gather_dir.exists() and not overwrite:
        raise FileExistsError(f"Gather directory already exists: {gather_dir}")
    if gather_dir.exists():
        shutil.rmtree(gather_dir)
    gather_dir.mkdir(parents=True)

    copied_samples = 0
    for sample_dir in iter_completed_sample_dirs(output_dir):
        target_dir = gather_dir / sample_dir.name
        target_dir.mkdir()
        for file_name in SAMPLE_GATHER_FILES:
            source_file = sample_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_dir / file_name)
        copied_samples += 1

    if copied_samples == 0:
        raise ValueError(f"No completed sequential samples with result.npz found in {output_dir}")

    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        shutil.copy2(summary_path, gather_dir / "summary.json")

    config_path = resolve_config_path(workdir, output_dir, config_name)
    times = output_time_values(config_path)
    if times is not None:
        write_json(gather_dir / "output_times.json", {"output_times": times})

    return gather_dir


def collect_results(
    workdir: Path,
    output_dir_name: str = DEFAULT_OUTPUT_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    csv_prefix: str = "",
) -> tuple[Path, Path, Path]:
    """
    Collect completed sequential Saltelli samples into parameter, fine-result, and coarse-result CSV files.
    """
    workdir = workdir.absolute()
    output_dir = workdir / output_dir_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Sequential output directory does not exist: {output_dir}")

    rows: list[tuple[str, dict[str, float], np.ndarray, np.ndarray]] = []
    parameter_names: list[str] | None = None
    result_size: int | None = None

    for sample_dir in iter_completed_sample_dirs(output_dir):
        payload = load_json(sample_dir / "input.json")
        parameters = {name: float(value) for name, value in payload["parameters"].items()}
        if parameter_names is None:
            parameter_names = list(parameters)
        elif list(parameters) != parameter_names:
            raise ValueError(f"Parameter order mismatch in {sample_dir}")

        with np.load(sample_dir / "result.npz") as npz:
            fine = load_result_vector(npz, "fine", sample_dir)
            coarse = load_result_vector(npz, "coarse", sample_dir)

        if fine.shape != coarse.shape:
            raise ValueError(f"Fine/coarse result shape mismatch in {sample_dir}: {fine.shape} != {coarse.shape}")
        if result_size is None:
            result_size = fine.size
        elif fine.size != result_size:
            raise ValueError(f"Result size mismatch in {sample_dir}: {fine.size} != {result_size}")

        rows.append((sample_dir.name, parameters, fine, coarse))

    if not rows or parameter_names is None or result_size is None:
        raise ValueError(f"No completed sequential samples with result.npz found in {output_dir}")

    config_path = resolve_config_path(workdir, output_dir, config_name)
    result_columns = time_columns(output_dir, config_path, result_size)

    parameters_csv = output_dir / f"{csv_prefix}parameters.csv"
    fine_csv = output_dir / f"{csv_prefix}fine_results.csv"
    coarse_csv = output_dir / f"{csv_prefix}coarse_results.csv"

    with parameters_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *parameter_names])
        for sample_id, parameters, _fine, _coarse in rows:
            writer.writerow([sample_id, *(parameters[name] for name in parameter_names)])

    with fine_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *result_columns])
        for sample_id, _parameters, fine, _coarse in rows:
            writer.writerow([sample_id, *fine.tolist()])

    with coarse_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", *result_columns])
        for sample_id, _parameters, _fine, coarse in rows:
            writer.writerow([sample_id, *coarse.tolist()])

    return parameters_csv, fine_csv, coarse_csv


def load_result_matrix(output_dir: Path, result_key: str) -> tuple[list[str], np.ndarray]:
    sample_ids = []
    rows = []
    result_size = None
    for sample_dir in iter_completed_sample_dirs(output_dir):
        with np.load(sample_dir / "result.npz") as npz:
            result = load_result_vector(npz, result_key, sample_dir)
        if result_size is None:
            result_size = result.size
        elif result.size != result_size:
            raise ValueError(f"Result size mismatch in {sample_dir}: {result.size} != {result_size}")
        sample_ids.append(sample_dir.name)
        rows.append(result)

    if not rows:
        raise ValueError(f"No completed sequential samples with result.npz found in {output_dir}")

    return sample_ids, np.vstack(rows)


def dataset_for_distribution_plot(sample_ids: list[str], values: np.ndarray, times: list[float]):
    import xarray as xr

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected result matrix with shape (samples, times), got {values.shape}")
    if values.shape[1] != len(times):
        raise ValueError(f"Result matrix has {values.shape[1]} columns, but got {len(times)} times.")

    return xr.Dataset(
        data_vars={
            "log10_conc_q99": (("QMC", "IID"), np.nanmax(values, axis=1)[:, None]),
            "log10_conc_q99_XYZ": (("QMC", "IID", "sim_time"), values[:, None, :]),
        },
        coords={
            "QMC": sample_ids,
            "IID": [0],
            "sim_time": np.asarray(times, dtype=float),
        },
    )


def result_time_axis(times: list[float]) -> tuple[np.ndarray, bool]:
    t_raw = np.asarray(times, dtype=float) / 1000.0
    t_pos = t_raw[t_raw > 0]
    if t_pos.size == 0:
        return np.arange(len(times), dtype=float), False
    return np.where(t_raw > 0, t_raw, float(t_pos.min()) * 0.999), True


def load_paired_results(
    workdir: Path,
    output_dir_name: str,
    config_name: str,
) -> tuple[Path, list[str], np.ndarray, np.ndarray, list[float]]:
    workdir = workdir.absolute()
    output_dir = workdir / output_dir_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Sequential output directory does not exist: {output_dir}")

    fine_ids, fine_values = load_result_matrix(output_dir, "fine")
    coarse_ids, coarse_values = load_result_matrix(output_dir, "coarse")
    if fine_ids != coarse_ids:
        raise ValueError("Fine and coarse sample ids do not match.")
    if fine_values.shape != coarse_values.shape:
        raise ValueError(f"Fine/coarse shapes do not match: {fine_values.shape} != {coarse_values.shape}")

    config_path = resolve_config_path(workdir, output_dir, config_name)
    times = result_time_values(output_dir, config_path, fine_values.shape[1])
    if times is None:
        times = [float(idx) for idx in range(fine_values.shape[1])]
    return output_dir, fine_ids, fine_values, coarse_values, times


def plot_result_distribution(
    workdir: Path,
    output_dir_name: str = DEFAULT_OUTPUT_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    result_key: str = "fine",
    plot_dir_name: str = "plots",
    plot_all_lines: bool = False,
) -> Path:
    workdir = workdir.absolute()
    output_dir = workdir / output_dir_name
    if not output_dir.exists():
        raise FileNotFoundError(f"Sequential output directory does not exist: {output_dir}")

    sample_ids, values = load_result_matrix(output_dir, result_key)
    config_path = resolve_config_path(workdir, output_dir, config_name)
    times = result_time_values(output_dir, config_path, values.shape[1])
    if times is None:
        times = [float(idx) for idx in range(values.shape[1])]

    import matplotlib

    matplotlib.use("Agg", force=True)
    try:
        from chodby_trans.plots import plot_conc_timeseries_distribution1
    except ModuleNotFoundError:
        from plots import plot_conc_timeseries_distribution1

    ds = dataset_for_distribution_plot(sample_ids, values, times)
    fig = plot_conc_timeseries_distribution1(
        ds,
        n_slices=len(sample_ids),
        max_extreme_lines=len(sample_ids),
        plot_all_lines=plot_all_lines,
    )
    plot_dir = output_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"{result_key}_timeseries_distribution.pdf"
    fig.savefig(out_path, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def plot_fine_coarse_comparison(
    workdir: Path,
    output_dir_name: str = DEFAULT_OUTPUT_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    plot_dir_name: str = "plots",
) -> Path:
    output_dir, _sample_ids, fine_values, coarse_values, times = load_paired_results(
        workdir, output_dir_name, config_name
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t, use_log_x = result_time_axis(times)

    fig, ax = plt.subplots(figsize=(14, 7))
    for fine, coarse in zip(fine_values, coarse_values):
        ax.plot(t, fine, color="tab:blue", lw=0.8, alpha=0.45)
        ax.plot(t, coarse, color="tab:orange", lw=0.8, alpha=0.45)

    ax.plot(t, np.nanmedian(fine_values, axis=0), color="tab:blue", lw=2.2)
    ax.plot(t, np.nanmedian(coarse_values, axis=0), color="tab:orange", lw=2.2)
    if use_log_x:
        ax.set_xscale("log")
        ax.set_xlabel("Time from 50y pulse (ky)")
    else:
        ax.set_xlabel("Output time index")
    ax.set_ylabel("Log10(conc)")
    ax.set_title("Fine and coarse sequential Saltelli time series")
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", lw=2.2, label="fine"),
            Line2D([0], [0], color="tab:orange", lw=2.2, label="coarse"),
        ],
        loc="lower right",
    )
    fig.tight_layout()

    plot_dir = output_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "fine_coarse_timeseries.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_mlmc_diagnostics(
    workdir: Path,
    output_dir_name: str = DEFAULT_OUTPUT_DIR,
    config_name: str = DEFAULT_CONFIG_NAME,
    plot_dir_name: str = "plots",
) -> Path:
    output_dir, _sample_ids, fine_values, coarse_values, times = load_paired_results(
        workdir, output_dir_name, config_name
    )
    diff_values = fine_values - coarse_values
    n_samples = fine_values.shape[0]

    fine_var = np.nanvar(fine_values, axis=0, ddof=1)
    coarse_var = np.nanvar(coarse_values, axis=0, ddof=1)
    diff_var = np.nanvar(diff_values, axis=0, ddof=1)
    ratio = np.divide(coarse_var, fine_var, out=np.full_like(fine_var, np.nan), where=fine_var > 0)
    reduction = np.divide(fine_var, diff_var, out=np.full_like(fine_var, np.nan), where=diff_var > 0)
    bias = np.nanmean(diff_values, axis=0)
    diff_q25, diff_q75 = np.nanquantile(diff_values, [0.25, 0.75], axis=0)

    corr = np.empty(fine_values.shape[1], dtype=float)
    for i_time in range(fine_values.shape[1]):
        fine_t = fine_values[:, i_time]
        coarse_t = coarse_values[:, i_time]
        mask = np.isfinite(fine_t) & np.isfinite(coarse_t)
        if np.count_nonzero(mask) < 2:
            corr[i_time] = np.nan
        elif np.nanstd(fine_t[mask]) == 0.0 or np.nanstd(coarse_t[mask]) == 0.0:
            corr[i_time] = np.nan
        else:
            corr[i_time] = np.corrcoef(fine_t[mask], coarse_t[mask])[0, 1]

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t, use_log_x = result_time_axis(times)
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True)

    ax = axes[0]
    ax.plot(t, fine_var, label="Var(fine)", color="tab:blue")
    ax.plot(t, coarse_var, label="Var(coarse)", color="tab:orange")
    ax.plot(t, diff_var, label="Var(fine - coarse)", color="tab:green")
    ax.set_yscale("log")
    ax.set_ylabel("Variance")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, ratio, label="Var(coarse) / Var(fine)", color="tab:purple")
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.set_ylabel("Variance ratio")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t, corr, label="Corr(fine, coarse)", color="tab:red")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Correlation")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[3]
    ax.fill_between(t, diff_q25, diff_q75, color="tab:green", alpha=0.2, label="IQR(fine - coarse)")
    ax.plot(t, bias, label="Mean(fine - coarse)", color="tab:green")
    ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
    ax.set_ylabel("Difference")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    if use_log_x:
        axes[-1].set_xscale("log")
        axes[-1].set_xlabel("Time from 50y pulse (ky)")
    else:
        axes[-1].set_xlabel("Output time index")

    fig.suptitle(f"Fine/coarse MLMC diagnostics, n={n_samples}")
    fig.tight_layout()

    plot_dir = output_dir / plot_dir_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "fine_coarse_mlmc_diagnostics.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    diagnostics_csv = plot_dir / "fine_coarse_mlmc_diagnostics.csv"
    with diagnostics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "fine_var",
                "coarse_var",
                "diff_var",
                "coarse_fine_var_ratio",
                "fine_diff_var_reduction",
                "fine_coarse_corr",
                "mean_diff",
                "diff_q25",
                "diff_q75",
            ]
        )
        for row in zip(t, fine_var, coarse_var, diff_var, ratio, reduction, corr, bias, diff_q25, diff_q75):
            writer.writerow(row)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect result.npz files from sequential Saltelli samples into three CSV files.",
    )
    parser.add_argument("workdir", type=Path, help="Workdir passed to sequential_saltelli_samples.py.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Sequential output subdirectory.")
    parser.add_argument("--gather-dir", default=DEFAULT_GATHER_DIR, help="Output directory for --gather-only.")
    parser.add_argument("--gather-only", action="store_true", help="Only gather lightweight result files.")
    parser.add_argument("--overwrite-gather", action="store_true", help="Replace an existing gather directory.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME, help="Config name inside workdir/input_data.")
    parser.add_argument("--csv-prefix", default="", help="Optional prefix for generated CSV file names.")
    parser.add_argument("--plot", action="store_true", help="Write distribution plots from collected result.npz files.")
    parser.add_argument(
        "--plot-result",
        choices=("fine", "coarse", "both"),
        default="fine",
        help="Result series to plot when using --plot.",
    )
    parser.add_argument("--plot-dir", default="plots", help="Plot output subdirectory below --output-dir.")
    parser.add_argument("--plot-all-lines", action="store_true", help="Plot every time series in the bottom panel.")
    parser.add_argument(
        "--plot-fine-coarse",
        action="store_true",
        help="Write a simple comparison plot with all fine and coarse series.",
    )
    parser.add_argument(
        "--plot-diagnostics",
        action="store_true",
        help="Write MLMC diagnostics for fine/coarse variance, ratio, correlation, and bias.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gather_only:
        gather_dir = gather_result_files(
            workdir=args.workdir,
            output_dir_name=args.output_dir,
            gather_dir_name=args.gather_dir,
            config_name=args.config_name,
            overwrite=args.overwrite_gather,
        )
        print(gather_dir)
        return 0

    did_plot = False
    if args.plot_fine_coarse:
        print(
            plot_fine_coarse_comparison(
                workdir=args.workdir,
                output_dir_name=args.output_dir,
                config_name=args.config_name,
                plot_dir_name=args.plot_dir,
            )
        )
        did_plot = True

    if args.plot_diagnostics:
        print(
            plot_mlmc_diagnostics(
                workdir=args.workdir,
                output_dir_name=args.output_dir,
                config_name=args.config_name,
                plot_dir_name=args.plot_dir,
            )
        )
        did_plot = True

    if args.plot:
        result_keys = ("fine", "coarse") if args.plot_result == "both" else (args.plot_result,)
        for result_key in result_keys:
            print(
                plot_result_distribution(
                    workdir=args.workdir,
                    output_dir_name=args.output_dir,
                    config_name=args.config_name,
                    result_key=result_key,
                    plot_dir_name=args.plot_dir,
                    plot_all_lines=args.plot_all_lines,
                )
            )
        did_plot = True

    if did_plot:
        return 0

    paths = collect_results(
        workdir=args.workdir,
        output_dir_name=args.output_dir,
        config_name=args.config_name,
        csv_prefix=args.csv_prefix,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
