from __future__ import annotations

import argparse
import re
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


def _append_yaml_value_suffix(text: str, key: str, suffix: str) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^#\n]+)(.*)$")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        match = pattern.match(line)
        if match and not line.lstrip().startswith("#"):
            prefix, value, trailer = match.groups()
            lines[idx] = f"{prefix}{value.strip()}{suffix}{trailer}"
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
    text = _append_yaml_value_suffix(text, "pbs_name", f"_{job_index:02d}")
    mesh_cfg_path.write_text(text, encoding="utf-8")


def ensure_job_input_data(
    job_dir: Path,
    source_input_data: Path,
    limit_samples: tuple[int, int],
    job_index: int,
) -> None:
    if job_dir.exists():
        return
    write_job_input_data(job_dir, source_input_data, limit_samples, job_index)


def prepare_job_dirs(case_dir: Path, samples_per_job: int, jobs_root: Path | None = None) -> list[Path]:
    case_dir = Path(case_dir).resolve()
    source_input_data = case_dir / "input_data"
    if jobs_root is None:
        jobs_root = case_dir
    jobs_root = Path(jobs_root).resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)

    cfg_path = source_input_data / "_ot_sensitivity.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    n_samples = int(cfg["n_samples"])
    ranges = chunk_ranges(n_samples=n_samples, samples_per_job=samples_per_job)

    job_dirs = []
    for i, sample_range in enumerate(ranges):
        job_dir = jobs_root / f"job_{i:02d}_{sample_range[0]:05d}_{sample_range[1]:05d}"
        ensure_job_input_data(job_dir, source_input_data, sample_range, i)
        job_dirs.append(job_dir)
    return job_dirs


def run_job(job_dir: Path, app_cmd: str = "meta") -> None:
    app_dir = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "sensitivity_sampling.py", str(job_dir), "submit", app_cmd],
        cwd=app_dir,
        check=True,
    )


def _archived_path(path: Path) -> Path:
    for i in range(1, 1000):
        archived = path.with_name(f"{path.name}.rerun_{i:02d}")
        if not archived.exists():
            return archived
    raise RuntimeError(f"Unable to find archive name for {path}")


def archive_job_submission_files(job_dir: Path) -> list[Path]:
    log_paths = sorted(job_dir.glob("*.out"))
    if len(log_paths) > 1:
        raise RuntimeError(f"Expected at most one .out file in {job_dir}, found {len(log_paths)}")

    rename_paths = [
        job_dir / "logs.tar.gz",
        job_dir / "sensitivity_sampling.pbs",
        *log_paths,
    ]
    archived = []
    for path in rename_paths:
        if not path.exists():
            continue
        archived_path = _archived_path(path)
        path.rename(archived_path)
        archived.append(archived_path)
    return archived


def count_none_samples(job_dir: Path) -> int:
    import chodby_trans.job as job
    import chodby_trans.sensitivity_sampling as sampling
    from chodby_trans.exception_wrapper import ReturnCode

    job.set_workdir(job_dir)
    job.output.plots.mkdir(parents=True, exist_ok=True)
    tags, _parameters = sampling.read_parameters_by_rc([ReturnCode.NONE], make_plots=False)
    return len(tags)


def rerun_incomplete_jobs(job_dirs: list[Path], dry_run: bool = False) -> list[Path]:
    rerun_jobs = []
    for job_dir in job_dirs:
        n_none = count_none_samples(job_dir)
        if n_none == 0:
            print(f"{job_dir}: skip")
            continue
        print(f"{job_dir}: continue ({n_none} NONE samples)")
        rerun_jobs.append(job_dir)
        if dry_run:
            continue
        archive_job_submission_files(job_dir)
        run_job(job_dir, app_cmd="continue")
    return rerun_jobs


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
    parser.add_argument(
        "--rerun-none",
        action="store_true",
        help="Resubmit only jobs that still contain samples with NONE return code.",
    )
    args = parser.parse_args(argv)

    job_dirs = prepare_job_dirs(args.case_dir, args.samples_per_job, jobs_root=args.jobs_root)
    if args.rerun_none:
        rerun_incomplete_jobs(job_dirs, dry_run=args.dry_run)
        return 0

    for job_dir in job_dirs:
        print(job_dir)
        if not args.dry_run:
            run_job(job_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
