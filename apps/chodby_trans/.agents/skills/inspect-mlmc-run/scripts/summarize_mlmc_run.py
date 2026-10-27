#!/usr/bin/env python3
"""Summarize saved Chodby MLMC run artifacts without modifying them."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any


NONE_CODE = -2000
RETURN_CODE_NAMES = {
    0: "OK",
    -1000: "UNKNOWN_ERROR",
    -1001: "BGEM_GEOM_ERROR",
    -1002: "BGEM_GMSH_ERROR",
    -1003: "BGEM_HEAL_ERROR",
    -1010: "FLOW123_ERROR",
    -1020: "SAMPLE_ERROR",
    -1021: "FINE_TRANSPORT_ERROR",
    -1022: "COARSE_TRANSPORT_ERROR",
    -1030: "HOMOGENIZATION_ERROR",
    -1100: "ZARR_ERROR",
    -1999: "SKIP",
    NONE_CODE: "NONE",
}
SAMPLE_RE = re.compile(r"L(?P<level>\d+)_S(?P<sample>\d+)")
TERM_FAILURE_RE = re.compile(
    r"Transport term failed: level=(?P<level>\d+) sample=(?P<sample>\d+) "
    r"term=(?P<term>\d+) fine_code=(?P<fine>-?\d+) coarse_code=(?P<coarse>-?\d+)"
)
TERM_START_RE = re.compile(r"Starting Saltelli term (?P<term>\d+)/\d+.*?(L\d+_S\d+)")
TERM_DONE_RE = re.compile(r"Completed Saltelli term (?P<term>\d+).*?(L\d+_S\d+)")
CONVERGENCE_RE = re.compile(r"Failed to converge:\s+(?P<reason>-?\d+)")


def decode(value: Any) -> str:
    """Convert HDF scalar fields to displayable text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def row_field(row: Any, index: int) -> Any:
    """Read a positional field from structured or ordinary HDF rows."""
    try:
        return row[index]
    except (IndexError, TypeError):
        if index == 0:
            return row
        raise


def sample_sort_key(sample_id: str) -> tuple[int, int, str]:
    match = SAMPLE_RE.search(sample_id)
    if match is None:
        return 10**9, 10**9, sample_id
    return int(match.group("level")), int(match.group("sample")), sample_id


def code_label(code: int) -> str:
    return f"{code}({RETURN_CODE_NAMES.get(code, 'UNMAPPED')})"


def summarize_hdf(run_dir: Path) -> None:
    candidates = [run_dir / "transport_mlmc.hdf", *sorted(run_dir.glob("*.hdf5"))]
    hdf_path = next((path for path in candidates if path.is_file()), None)
    print("\n[MLMC HDF]")
    if hdf_path is None:
        print("not found")
        return

    try:
        import h5py
    except ImportError as error:
        print(f"cannot inspect {hdf_path}: {error}")
        return

    print(hdf_path)
    with h5py.File(hdf_path, "r") as storage:
        levels = storage.get("Levels")
        if levels is None:
            print("missing Levels group")
            return
        for level_name in sorted(levels, key=lambda value: int(value)):
            group = levels[level_name]
            scheduled = {
                decode(row_field(row, 0)) for row in group.get("scheduled", [])
            }
            collected = {
                decode(row_field(row, 0)) for row in group.get("collected_ids", [])
            }
            failed_rows = list(group.get("failed", []))
            failed = {decode(row_field(row, 0)) for row in failed_rows}
            unfinished = scheduled - collected - failed
            print(
                f"level {int(level_name):02d}: scheduled={len(scheduled)} "
                f"collected={len(collected)} failed={len(failed)} unfinished={len(unfinished)}"
            )
            if collected:
                print("  collected: " + ", ".join(sorted(collected, key=sample_sort_key)))
            if failed:
                print("  failed: " + ", ".join(sorted(failed, key=sample_sort_key)))
            if unfinished:
                print("  unfinished: " + ", ".join(sorted(unfinished, key=sample_sort_key)))

            signatures: Counter[str] = Counter()
            for row in failed_rows:
                try:
                    message = decode(row_field(row, 1))
                except (IndexError, TypeError):
                    continue
                signatures[classify_error_text(message)] += 1
            for signature, count in signatures.most_common():
                print(f"  failed signature x{count}: {signature}")


def classify_error_text(text: str) -> str:
    """Choose a stable root-cause signature from a traceback string."""
    checks = [
        (r"trans_mesh_homogenization\.msh", "missing homogenization mesh"),
        (r"Missing cached file: ([^\n]+)", "missing cached file"),
        (r"SIGSEGV\(-11\)", "native SIGSEGV"),
        (r"HomogenizationException: ([^\n]+)", "homogenization exception"),
        (r"FineTransportException: ([^\n]+)", "fine transport exception"),
        (r"CoarseTransportException: ([^\n]+)", "coarse transport exception"),
        (r"TerminatedWorkerError: ([^\n]+)", "terminated loky worker"),
        (r"prepare_fine_input", "fine input preparation traceback"),
        (r"FileNotFoundError: ([^\n]+)", "missing file"),
    ]
    for pattern, label in checks:
        match = re.search(pattern, text)
        if match:
            detail = match.group(1).strip() if match.lastindex else ""
            return f"{label}: {detail}" if detail else label
    exception_lines = [line.strip() for line in text.splitlines() if "Error:" in line]
    return exception_lines[-1] if exception_lines else "unclassified traceback"


def discover_zarr_levels(store: Path) -> list[int]:
    level_root = store / "mlmc"
    levels = []
    if level_root.is_dir():
        for path in level_root.glob("level_*"):
            try:
                levels.append(int(path.name.removeprefix("level_")))
            except ValueError:
                continue
    return sorted(levels)


def summarize_zarr(run_dir: Path) -> None:
    store = run_dir / "transport_sampling"
    print("\n[ZARR RETURN CODES]")
    if not store.is_dir():
        print("not found")
        return
    try:
        import numpy as np
        import xarray as xr
    except ImportError as error:
        print(f"cannot inspect {store}: {error}")
        return

    levels = discover_zarr_levels(store)
    if not levels:
        print(f"no mlmc/level_* groups under {store}")
        return

    for level in levels:
        try:
            dataset = xr.open_zarr(
                str(store),
                group=f"mlmc/level_{level:02d}",
                consolidated=False,
            )
            fine = np.asarray(dataset["fine_return_code"].to_numpy(), dtype=int)
            coarse = np.asarray(dataset["coarse_return_code"].to_numpy(), dtype=int)
        except Exception as error:
            print(f"level {level:02d}: unable to read return codes: {error}")
            continue

        print(f"level {level:02d}: shape={fine.shape}")
        for stage, values in (("fine", fine), ("coarse", coarse)):
            counts = Counter(int(value) for value in values.flat if int(value) != NONE_CODE)
            summary = ", ".join(
                f"{code_label(code)}={count}" for code, count in sorted(counts.items())
            )
            print(f"  {stage}: {summary or 'no started terms'}")

        for sample_index in range(fine.shape[0]):
            active = (fine[sample_index] != NONE_CODE) | (coarse[sample_index] != NONE_CODE)
            if not active.any():
                continue
            fine_ok = int((fine[sample_index] == 0).sum())
            coarse_ok = int((coarse[sample_index] == 0).sum())
            paired_ok = int(((fine[sample_index] == 0) & (coarse[sample_index] == 0)).sum())
            failures = [
                f"t{term}:fine={code_label(int(fine[sample_index, term]))},"
                f"coarse={code_label(int(coarse[sample_index, term]))}"
                for term in range(fine.shape[1])
                if (
                    int(fine[sample_index, term]) < 0
                    and int(fine[sample_index, term]) != NONE_CODE
                )
                or (
                    int(coarse[sample_index, term]) < 0
                    and int(coarse[sample_index, term]) != NONE_CODE
                )
            ]
            sample_id = f"L{level:02d}_S{sample_index:07d}"
            detail = f" failures=[{'; '.join(failures)}]" if failures else ""
            print(
                f"  {sample_id}: paired_ok={paired_ok} fine_ok={fine_ok} "
                f"coarse_ok={coarse_ok}{detail}"
            )


def iter_log_files(run_dir: Path) -> list[Path]:
    patterns = (
        "*.out",
        "*.log",
        "scratch_*/logs/*.log",
        "logs*/*.log",
    )
    return sorted({path for pattern in patterns for path in run_dir.glob(pattern) if path.is_file()})


def add_example(
    examples: dict[str, list[tuple[Path, int, str]]],
    key: str,
    path: Path,
    line_no: int,
    line: str,
    maximum: int,
) -> None:
    if len(examples[key]) < maximum:
        examples[key].append((path, line_no, line.strip()))


def summarize_logs(run_dir: Path, max_examples: int) -> None:
    print("\n[LOG EVENTS]")
    logs = iter_log_files(run_dir)
    print(f"files={len(logs)}")
    if not logs:
        return

    counts: Counter[str] = Counter()
    examples: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)
    convergence: Counter[int] = Counter()
    failures: set[tuple[int, int, int, int, int]] = set()
    progress: dict[str, dict[str, set[int]]] = defaultdict(
        lambda: {"started": set(), "completed": set(), "failed": set()}
    )

    signatures = {
        "walltime": re.compile(r"walltime .*exceeded|job killed: walltime", re.I),
        "scheduler SIGTERM": re.compile(r"Received signal SIGTERM"),
        "shutdown communication": re.compile(
            r"CancelledError|CommClosedError|Lost all workers|heartbeat|ConnectionRefusedError"
        ),
        "missing cached file": re.compile(r"Missing cached file:"),
        "loky terminated worker": re.compile(r"TerminatedWorkerError"),
        "native SIGSEGV": re.compile(r"SIGSEGV\(-11\)"),
        "fine stage exception": re.compile(r"FineTransportException:"),
        "coarse stage exception": re.compile(r"CoarseTransportException:"),
        "homogenization exception": re.compile(r"HomogenizationException:"),
    }

    for path in logs:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                for key, pattern in signatures.items():
                    if pattern.search(line):
                        counts[key] += 1
                        add_example(examples, key, path, line_no, line, max_examples)

                convergence_match = CONVERGENCE_RE.search(line)
                if convergence_match:
                    convergence[int(convergence_match.group("reason"))] += 1

                failure_match = TERM_FAILURE_RE.search(line)
                if failure_match:
                    event = tuple(int(failure_match.group(name)) for name in (
                        "level", "sample", "term", "fine", "coarse"
                    ))
                    failures.add(event)
                    sample_id = f"L{event[0]:02d}_S{event[1]:07d}"
                    progress[sample_id]["failed"].add(event[2])

                start_match = TERM_START_RE.search(line)
                if start_match:
                    sample_id = start_match.group(2)
                    progress[sample_id]["started"].add(int(start_match.group("term")))
                done_match = TERM_DONE_RE.search(line)
                if done_match:
                    sample_id = done_match.group(2)
                    progress[sample_id]["completed"].add(int(done_match.group("term")))

    for key, count in counts.most_common():
        suffix = " (traceback layers may repeat)" if key not in {
            "walltime", "scheduler SIGTERM", "shutdown communication"
        } else ""
        print(f"{key}: {count}{suffix}")
        for path, line_no, line in examples[key]:
            print(f"  {path}:{line_no}: {line[:240]}")

    if convergence:
        print("Flow convergence messages: " + ", ".join(
            f"reason {reason}={count}" for reason, count in sorted(convergence.items())
        ))

    if failures:
        print("Unique structured transport failures:")
        for level, sample, term, fine, coarse in sorted(failures):
            print(
                f"  L{level:02d}_S{sample:07d} term={term} "
                f"fine={code_label(fine)} coarse={code_label(coarse)}"
            )

    if progress:
        print("Worker-log Saltelli progress:")
        for sample_id in sorted(progress, key=sample_sort_key):
            state = progress[sample_id]
            print(
                f"  {sample_id}: started={sorted(state['started'])} "
                f"completed={sorted(state['completed'])} failed={sorted(state['failed'])}"
            )


def summarize_artifacts(run_dir: Path) -> None:
    print("\n[SHARED AND NODE-LOCAL ARTIFACTS]")
    common_files = sorted(run_dir.glob("homogenization/*"))
    node_files = sorted(run_dir.glob("scratch_*/workdir/trans_mesh_homogenization*"))
    core_files = sorted(run_dir.glob("scratch_*/workdir/output/**/core.*"))
    if common_files:
        print("shared homogenization:")
        for path in common_files:
            print(f"  {path} ({path.stat().st_size} bytes)")
    if node_files:
        print("legacy/node-local homogenization:")
        for path in node_files:
            print(f"  {path} ({path.stat().st_size} bytes)")
    if core_files:
        print("core dumps:")
        for path in core_files:
            print(f"  {path} ({path.stat().st_size} bytes)")
    if not common_files and not node_files and not core_files:
        print("none found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="MLMC run work directory")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=2,
        help="maximum source locations shown per log signature",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    print(f"MLMC run: {run_dir}")
    summarize_hdf(run_dir)
    summarize_zarr(run_dir)
    summarize_logs(run_dir, max(0, args.max_examples))
    summarize_artifacts(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
