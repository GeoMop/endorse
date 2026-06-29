from __future__ import annotations

import argparse
import json
import shutil
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
    args = parser.parse_args(argv)

    output_store = collect_job_stores(args.case_dir, output_store=args.output_store, force=args.force)
    print(output_store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
