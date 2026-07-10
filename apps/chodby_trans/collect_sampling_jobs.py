from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import yaml
import zarr


def _zarr_metadata(store_path: Path) -> dict:
    return json.loads((store_path / "zarr.json").read_text(encoding="utf-8"))


def _store_layout(store_path: Path) -> dict:
    return _zarr_metadata(store_path)["consolidated_metadata"]["metadata"]


def _merge_array_names(store_path: Path) -> list[str]:
    layout = _store_layout(store_path)
    return [
        name
        for name, node in layout.items()
        if node.get("node_type") == "array"
        and node.get("dimension_names")
        and node["dimension_names"][0] == "i_sample"
    ]


def _job_sample_range(job_dir: Path) -> tuple[int, int]:
    cfg = yaml.safe_load((job_dir / "input_data" / "_ot_sensitivity.yaml").read_text(encoding="utf-8"))
    start, stop = cfg["limit_samples"]
    return int(start), int(stop)


def _copy_template_store(source_store: Path, dest_store: Path, force: bool) -> None:
    if dest_store.exists():
        if not force:
            raise FileExistsError(f"Destination store already exists: {dest_store}")
        shutil.rmtree(dest_store)
    shutil.copytree(source_store, dest_store)


def _validate_store_layout(reference_store: Path, candidate_store: Path) -> None:
    reference_layout = _store_layout(reference_store)
    candidate_layout = _store_layout(candidate_store)
    if candidate_layout != reference_layout:
        raise ValueError(f"Zarr store layout mismatch: {candidate_store}")


def list_job_dirs(case_dir: Path) -> list[Path]:
    return sorted(
        job_dir
        for job_dir in case_dir.glob("job_*")
        if (job_dir / "transport_sampling").is_dir()
    )


def count_samples_by_rc(job_dir: Path, rc_select: list[int]) -> int:
    import chodby_trans.job as job
    import chodby_trans.sensitivity_sampling as sampling

    start, stop = _job_sample_range(job_dir)
    job.set_workdir(job_dir)
    tags, _parameters, _rc_stats = sampling.read_parameters_by_rc(rc_select, make_plots=False)
    return sum(start <= int(i_sample) < stop for _i_eval, i_sample, _i_saltelli in tags)


def job_return_code_stats(job_dir: Path) -> dict:
    from chodby_trans.exception_wrapper import ReturnCode
    import chodby_trans.job as job
    import chodby_trans.sensitivity_sampling as sampling

    start, stop = _job_sample_range(job_dir)
    job.set_workdir(job_dir)
    tags, _parameters, rc_stats = sampling.read_parameters_by_rc(ReturnCode.to_list(), make_plots=False)
    eval_to_sample = {int(i_eval): int(i_sample) for i_eval, i_sample, _i_saltelli in tags}
    counts = OrderedDict()
    for code in ReturnCode.to_list():
        ids = rc_stats.get(code, [])
        counts[code] = sum(start <= eval_to_sample.get(int(i_eval), -1) < stop for i_eval in ids)
    return {
        "job_dir": job_dir.resolve(),
        "limit_samples": (start, stop),
        "limit_range_size": stop - start,
        "n_results_non_none": sum(
            count
            for code, count in counts.items()
            if code != ReturnCode.NONE
        ),
        "counts": counts,
    }


def _return_code_labels() -> dict[int, str]:
    from chodby_trans.exception_wrapper import ReturnCode

    return {code: name for name, code in ReturnCode.to_dict().items()}


def _job_stats_lines(stats: dict) -> list[str]:
    labels = _return_code_labels()
    start, stop = stats["limit_samples"]
    lines = [
        f"{Path(stats['job_dir']).name}",
        f"  limit_samples: [{start}, {stop}]",
        f"  limit_range_size: {stats['limit_range_size']}",
        f"  n_results_non_none: {stats['n_results_non_none']}",
    ]
    for code, count in stats["counts"].items():
        lines.append(f"  {labels.get(code, str(code))} [{code}]: {count}")
    return lines


def write_job_return_code_report(case_dir: Path, output_path: Path | None = None) -> Path:
    case_dir = case_dir.resolve()
    if output_path is None:
        output_path = case_dir / "job_return_code_stats.txt"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_by_job = [job_return_code_stats(job_dir) for job_dir in list_job_dirs(case_dir)]
    lines = [f"Case: {case_dir}", f"Jobs: {len(stats_by_job)}", ""]
    for stats in stats_by_job:
        lines.extend(_job_stats_lines(stats))
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def collect_job_stores(case_dir: Path, output_store: Path | None = None, force: bool = False) -> Path:
    case_dir = case_dir.resolve()
    job_dirs = list_job_dirs(case_dir)
    if not job_dirs:
        raise FileNotFoundError(f"No job directories with transport_sampling found in {case_dir}")

    if output_store is None:
        output_store = case_dir / "transport_sampling"
    output_store = output_store.resolve()

    template_store = job_dirs[0] / "transport_sampling"
    merge_arrays = _merge_array_names(template_store)
    _copy_template_store(template_store, output_store, force=force)

    for job_dir in job_dirs:
        source_store = job_dir / "transport_sampling"
        _validate_store_layout(template_store, source_store)
        start, stop = _job_sample_range(job_dir)
        print(f"{job_dir.name}: i_sample[{start}:{stop}]")
        for array_name in merge_arrays:
            source_array = zarr.open(source_store / array_name, mode="r")
            dest_array = zarr.open(output_store / array_name, mode="r+")
            dest_array[start:stop, ...] = source_array[start:stop, ...]

    return output_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect chunked sampling job Zarr stores into one case store.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "CASE_0_32k",
        help="Case directory containing job_* subdirectories.",
    )
    parser.add_argument(
        "--output-store",
        type=Path,
        default=None,
        help="Destination Zarr store path. Defaults to <case-dir>/transport_sampling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination store if it already exists.",
    )
    parser.add_argument(
        "--job-stats",
        action="store_true",
        help="Inspect existing job directories and write a return-code summary text report.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Output path for --job-stats. Defaults to <case-dir>/job_return_code_stats.txt.",
    )
    args = parser.parse_args(argv)

    if args.job_stats:
        report_path = write_job_return_code_report(args.case_dir, output_path=args.report_path)
        print(report_path)
        return 0

    output_store = collect_job_stores(args.case_dir, output_store=args.output_store, force=args.force)
    print(output_store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
