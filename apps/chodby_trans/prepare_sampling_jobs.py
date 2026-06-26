from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def chunk_ranges(n_samples: int, samples_per_job: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + samples_per_job, n_samples))
        for start in range(0, n_samples, samples_per_job)
    ]


def _replace_yaml_line(text: str, key: str, replacement: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"{key}:") and not stripped.startswith("#"):
            indent = line[: len(line) - len(stripped)]
            lines[idx] = f"{indent}{replacement}"
            break
    else:
        raise KeyError(f"Missing {key} line")
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def write_job_input_data(
    job_dir: Path,
    source_input_data: Path,
    limit_samples: tuple[int, int],
    job_index: int,
) -> None:
    dest_input_data = job_dir / "input_data"
    shutil.copytree(source_input_data, dest_input_data)

    cfg_path = dest_input_data / "_ot_sensitivity.yaml"
    text = cfg_path.read_text(encoding="utf-8")
    text = _replace_yaml_line(
        text,
        "limit_samples",
        f"limit_samples: [{limit_samples[0]}, {limit_samples[1]}]",
    )
    cfg_path.write_text(text, encoding="utf-8")

    mesh_cfg_path = dest_input_data / "trans_mesh_config.yaml"
    text = mesh_cfg_path.read_text(encoding="utf-8")
    text = _replace_yaml_line(
        text,
        "pbs_name",
        f"pbs_name: trans_case_0_{job_index:02d}",
    )
    mesh_cfg_path.write_text(text, encoding="utf-8")


def prepare_job_dirs(case_dir: Path, samples_per_job: int, jobs_root: Path | None = None) -> list[Path]:
    case_dir = Path(case_dir).resolve()
    source_input_data = case_dir / "input_data"
    if jobs_root is None:
        jobs_root = case_dir / f"sampling_jobs_{samples_per_job}"
    jobs_root = Path(jobs_root).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)

    cfg_path = source_input_data / "_ot_sensitivity.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    n_samples = int(cfg["n_samples"])
    ranges = chunk_ranges(n_samples=n_samples, samples_per_job=samples_per_job)

    job_dirs = []
    for i, sample_range in enumerate(ranges):
        job_dir = jobs_root / f"job_{i:02d}_{sample_range[0]:04d}_{sample_range[1]:04d}"
        write_job_input_data(job_dir, source_input_data, sample_range, i)
        job_dirs.append(job_dir)
    return job_dirs


def run_job(job_dir: Path) -> None:
    app_dir = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "sensitivity_sampling.py", str(job_dir), "submit", "meta"],
        cwd=app_dir,
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split sensitivity sampling into chunked jobs.")
    parser.add_argument("samples_per_job", type=int, help="Number of samples per job chunk.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "CASE_0_32k",
        help="Case directory containing input_data.",
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=None,
        help="Directory where job subdirectories will be created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare job directories but do not launch sensitivity_sampling.py.",
    )
    args = parser.parse_args(argv)

    job_dirs = prepare_job_dirs(args.case_dir, args.samples_per_job, jobs_root=args.jobs_root)
    for job_dir in job_dirs:
        print(job_dir)
        if not args.dry_run:
            run_job(job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
