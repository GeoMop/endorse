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

from sequential_saltelli_samples import read_sample_times


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


def time_columns(output_dir: Path, config_path: Path, result_size: int) -> list[str]:
    gathered_times_path = output_dir / "output_times.json"
    if gathered_times_path.exists():
        times = [float(time) for time in load_json(gathered_times_path)["output_times"]]
    else:
        times = output_time_values(config_path)

    if times is None:
        return [f"t_{idx}" for idx in range(result_size)]

    if len(times) != result_size:
        raise ValueError(
            f"Result size {result_size} does not match {len(times)} output times from {config_path}."
        )
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
    times: list[tuple[str, float, float]] = []

    for sample_dir in iter_completed_sample_dirs(output_dir):
        payload = load_json(sample_dir / "input.json")
        parameters = {name: float(value) for name, value in payload["parameters"].items()}
        if parameter_names is None:
            parameter_names = list(parameters)
        elif list(parameters) != parameter_names:
            raise ValueError(f"Parameter order mismatch in {sample_dir}")

        if "fine_time" not in payload:
            fine_time, coarse_time = read_sample_times(sample_dir)
        else:
            fine_time = payload["fine_time"]
            coarse_time = payload["coarse_time"]
        times.append((sample_dir.name, fine_time, coarse_time))

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
    times_csv = output_dir / f"{csv_prefix}times.csv"

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

    with times_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "fine_time", "coarse_time"])
        for sample_id, fine_t, coarse_t in times:
            writer.writerow([sample_id, fine_t, coarse_t])

    return parameters_csv, fine_csv, coarse_csv, times_csv


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
