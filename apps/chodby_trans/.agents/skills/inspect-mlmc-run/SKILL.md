---
name: inspect-mlmc-run
description: >-
  Inspect and diagnose Chodby transport MLMC sampling runs from a work directory containing PBS output, copied
  node scratch directories, Dask logs, MLMC HDF storage, Zarr return-code storage, and retained sample artifacts.
  Use when asked to evaluate an MLMC run, explain failed or stuck samples, separate model failures from walltime or
  Dask shutdown noise, compare node-local behavior, or determine what a continuation run will reuse.
---

# Inspect MLMC Run

Inspect run artifacts read-only. Establish stored sample state first, then correlate it with logs and retained files.
Do not change configuration, storage, or sample directories unless the user separately asks for a fix.

## Start with the summarizer

From the repository root, run:

```bash
apps/chodby_trans/venv/bin/python \
  apps/chodby_trans/.agents/skills/inspect-mlmc-run/scripts/summarize_mlmc_run.py \
  <run-directory>
```

Use `--max-examples N` to change the number of reported log examples. Treat the summary as an index into the
artifacts, not as a replacement for reading the relevant traceback context.

If an optional Python dependency is unavailable, continue with the sections the script can inspect and report the
missing dependency. Do not install packages during diagnosis.

## Diagnostic workflow

1. Resolve the run directory and list PBS output, HDF, Zarr, copied workdirs from nodes scratches, logs, and failed-sample trees.
2. Read MLMC HDF state:
   - `scheduled` is every persisted work item.
   - `failed` is terminal in MLMC's finished-sample count.
   - `collected_ids` contains successful complete samples only.
   - `scheduled - failed - collected` is unfinished at the last durable HDF update.
3. Read per-level Zarr fine and coarse return codes. Report state per sample and Saltelli term.
4. Correlate failed HDF IDs with `Transport term failed` and `Saltelli term ... failed` worker events.
5. Inspect context around one representative traceback for each distinct root cause.
6. Compare copied node scratch trees when failures are node-specific. Identical `$SCRATCHDIR` strings refer to
   different local filesystems on different nodes.
7. Inspect retained solver output, mesh files, and core dumps for exceptional samples.
8. Read the PBS and scheduler tail to establish whether shutdown errors followed walltime or preceded it.
9. Report conclusions with evidence and clearly separate confirmed causes from inferences.

Prefer targeted `rg` searches. If `rg` is unavailable, use scoped `grep`; never dump every worker log in full.

## Interpretation rules

- One MLMC sample is one complete Saltelli row. Successful individual terms do not create a collected HDF sample.
- A failed sample is finished for scheduling-count purposes even though it is not collected as a valid result.
- Return code `-2000` means `NONE` or not run; do not count it as a model failure.
- Preserve a successful fine result when the following coarse stage fails. Report fine and coarse codes separately.
- Codes `-1021`, `-1022`, and `-1030` identify fine-stage, coarse-stage, and homogenization failures respectively.
- Chained loky tracebacks repeat the same exception. Count structured transport-failure events or HDF failed IDs,
  not raw `Traceback` lines.
- `SIGTERM`, `CancelledError`, `CommClosedError`, lost workers, and heartbeat failures after the PBS walltime marker
  are shutdown consequences unless earlier evidence shows otherwise.
- `TerminatedWorkerError` with `SIGSEGV(-11)` and a core dump is a native-process crash, not a Python exception.
- Dask memory warnings support an OOM diagnosis; the generic loky message alone does not.
- `Failed to converge` with Flow123d exit status zero is nonfatal in the current code. Report its frequency and
  avoid declaring the sample failed unless its return code or output proves failure.
- HDF may lag Zarr and scratch artifacts if the driver is killed before receiving or persisting a task result.
- A continuation reuses persisted `scheduled_inputs`, but unfinished samples normally restart as whole MLMC tasks.

## Inspect traceback context

For each signature reported by the script, read a bounded region around its first occurrence:

```bash
sed -n '<start>,<end>p' <reported-log>
```

For a retained core dump, first use `file`. Use a bounded batch GDB backtrace only when it materially helps locate
which native library crashed. Warn that backtraces can be incomplete when runtime libraries differ from the node.

## Report format

Lead with a compact run-state summary:

- scheduled, collected, failed, and unfinished IDs per level;
- successful paired terms and fine-only terms from Zarr;
- primary root causes, affected samples, nodes, stages, and return codes;
- independent model or native crashes;
- walltime-interrupted samples and shutdown-only noise;
- notable nonfatal warnings;
- what a continuation will rerun or preserve.

Then recommend the smallest next action. Do not implement fixes during an inspection-only request.
