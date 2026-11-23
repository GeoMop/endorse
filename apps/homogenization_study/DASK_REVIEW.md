# Dask wrapper review, 2026-11-20

Scope: repository `dask/` helpers and preloads, both applications' thin shell wrappers, the homogenization
sampling driver, PBS shell regressions, homogenization driver contracts, Chodby MLMC tests, and the older
`chodby_trans/dask_test` demonstration. `src/endorse/homogenisation.py` contains numerical averaging code,
with no Dask orchestration; no numerical changes are proposed here.

This review adds descriptions and comments only. The behavioral changes below remain recommendations.

## Current contract

1. Application wrappers select the application, heavy worker module, and lightweight scheduler task module.
2. The PBS template stages inputs on each node. The shared PBS helper creates runtime/log directories, starts
   the scheduler, and launches one single-threaded worker process per usable slot. Local launch uses the same
   process model but stores runtime files and logs under the supplied workdir.
3. Scheduler preload imports task definitions. Worker preload imports the physical stack and initializes job paths
   before reporting completion. Stable aliases in `sys.modules` let lightweight status probes inspect both preloads.
4. Shell runners publish `DASK_EXPECTED_WORKERS`; the driver waits for those registrations and checks preloads.
   Direct `Client(None)` runs omit this barrier and currently skip preload validation.
5. The driver owns HDF scheduling/restart state; workers write scientific fields to fixed sample slots in Zarr.
   MLMC creates each task's workspace and changes the worker process cwd before calling the calculation.
6. PBS shutdown stops workers, collects logs, collects application workspaces, and only then permits scratch cleanup.
   Transfer failure retains scratch. A directory-existence check does not prove a pre-existing copy is complete.

## Findings

### P1: task workspaces still use the driver's scratch root on every node

Location: `sensitivity_sampling.py`, `run_mlmc_sampling`, pool construction; `job.py`, `set_workdir`;
`problem_simulation.py`, `run_homogenization_sample`.

The driver resolves `job.scratch.dir_path` on the head node and passes that absolute path as the pool's `work_dir`.
The installed MLMC `SamplingPoolDask.schedule_sample` passes its derived `output` directory unchanged to
`SamplingPool.calculate_sample`; `handle_sim_files` creates it and changes cwd on the executing worker.
`dask worker --local-directory` controls Dask runtime/spill files and does not override this task argument.

For example, a head-node `/scratch/local.ssd/job123/workdir/output` is also used on a node whose own scratch is
`/scratch.ssd/job123`. Task setup can fail before the physical calculation, or write outside the workspace collected
by PBS. The worker preload's correct node-local `job.scratch` cannot repair the already supplied task path.
This is established from the installed MLMC call path, not reproduced on a real heterogeneous allocation.

Recommendation: resolve the sample workspace on the executing worker, and keep pool cleanup/move paths consistent
with that choice. Test the actual MLMC workspace boundary with different simulated head/worker roots, including
an error before calculation and output collection. Fixing only `job.set_workdir` is insufficient.

### P1: direct local sampling can execute concurrent cwd changes in one process

Location: `sensitivity_sampling.py`, `resolve_subcmd`, `Client(scheduler)`.

With no scheduler address, Dask creates a default LocalCluster. Its installed implementation chooses worker/thread
counts from CPU availability; it can create workers with multiple computation threads. MLMC's workspace setup uses
process-global `os.chdir`, and both sample functions derive sample identity from `Path.cwd()`.
Overlapping tasks can therefore use each other's workspace or sample id, potentially writing the wrong Zarr row.
The shell launchers explicitly use one thread per process; the direct CLI path lacks that guarantee.

Recommendation: construct the direct local cluster with explicit single-threaded process workers. Keep daemon=False
for nested mesh subprocesses. Verify simultaneous real MLMC task workspace setup, without substituting a model.
The existing ClientProbe test checks routing/configuration only and cannot detect this concurrency problem.

### P1: local stop can terminate another run's cluster

Location: `../../dask/dask_cluster_local.sh`, `stop_cluster`.

After processing PID files, cleanup calls `pkill -f` with generic worker and scheduler patterns. Those patterns do
not include this run's runtime directory, scheduler address, or recorded PID, so another cluster belonging to the
same user can be stopped too. With nullglob disabled, an empty PID glob also enters the loop as a literal filename.

Recommendation: stop only the recorded processes/process groups, verify their identity, and wait for their exit.
Use nullglob or an explicit file guard. Test two independent runs with mocked process commands and an empty PID set.
No live shutdown probe was performed during review because the current command can affect unrelated processes.

### P2: Chodby preload tests still import a removed application module

Location: `../chodby_trans/test/test_mlmc_sampling.py`, module imports and
`test_dask_worker_preload_initializes_job`.

The test imports `chodby_trans.dask_worker_preload`, but that source file was removed when preloads moved to `dask/`.
The shared preload installs that name as a runtime alias only when a Dask process loads the shared file.
It is not an importable replacement for ordinary pytest collection.

Recommendation: load the real shared preload in a fresh subprocess with explicit application environment variables.
Test import failure, missing input/output variables, setup completion, and status aliases for both applications.
Keep preload import side effects out of the pytest process. Collection verification is recorded below.

### P2: exhausted sampling returns success even with failed samples

Location: `sensitivity_sampling.py`, `run_mlmc_sampling`, pool-draining loop and return.

When the last running task finishes, the loop returns the sampler even if successes are below the requested count.
`resolve_subcmd` ignores that result, so a PBS job prints `FINISHED SAMPLING` and can finish successfully with failures.
Continuing after individual failures is useful; the final process status currently does not express that distinction.

Recommendation: retain all results, log a final successful/failed/unfinished summary, and choose an explicit policy
for the final exit status. Do not restore failure on the first bad sample or schedule beyond fixed Zarr capacity.
Open policy question: should a fully collected run containing failed samples return a nonzero driver status?

### P2: the client is not explicitly closed

Location: `sensitivity_sampling.py`, `resolve_subcmd`.

The function creates a Client but never closes it on normal return or an exception. In an embedded/repeated call,
connections and an automatically created local cluster can outlive the sampling operation.

Recommendation: use the Client context manager, and explicitly own a LocalCluster when creating one. Closing a
client connected to an external scheduler should preserve the shell runner's ownership of that scheduler.
Extend ClientProbe to verify cleanup on successful sampling and exceptions.

### P2: generated PBS paths are inserted as shell syntax

Location: `sensitivity_sampling.py`, `PBS_SCRIPT_TEMPLATE`, `output_dir`, `SCRIPT_PATH`, and `PROJECT_DIR` assignments.

Workdir/script paths containing spaces break the unquoted substitutions. Shell metacharacters in interpolated paths
are interpreted by Bash. Recommendation: quote substituted shell values at generation time, then quote expansions.
Validate PBS directive paths separately; Bash quoting alone does not define PBS directive parsing.
Add script-generation cases with spaces and literal shell metacharacters.

## Duplication and simplification opportunities

| Repeated responsibility | Suggested simplification |
| --- | --- |
| Scheduler polling in local/PBS helpers | One sourced readiness function with log, PID, timeout args. |
| Local/PBS worker commands, loops, PID files | One worker launcher parameterized by runtime/log dirs. |
| Four similar application wrappers | Retain thin wrappers; avoid another abstraction for a few assignments. |
| Preload status decoding in both applications | One lightweight shared probe taking an alias. |
| Worker registration barrier in both drivers | One shared barrier; retain application-specific validation policy. |
| PBS staging/log/workspace rsync | One checked transfer primitive retaining node-local expansion and context. |
| Preload timing boilerplate | Keep scheduler/worker entry points distinct; extract only reused metrics. |

Remove unused `NCPUS` calculations and unused PBS formatting entries (`python`, `n_workers`, `workdir`) in a later
cleanup. The `submit_pbs` worker formula is unused and differs from the launcher's actual reservation rule.
`ENDORSE_DASK_WORK_DIRNAME` is exported by the homogenization wrapper but its only shared consumer is commented out.
Rename `stage_inputs` to describe directory preparation and `run_example` to describe running the driver.
Move environment/argument validation before top-level `$SCRATCHDIR`/`$PBS_NODEFILE` access so missing variables and
`help` produce useful messages. Validate positive worker counts before starting a scheduler.

The driver log's `scheduled` value is captured before new samples are submitted; update or relabel it for accurate
post-mortem counts. Log every worker's preload PID, import duration, and peak RSS, as the Chodby driver already does.
Prefer these explicit diagnostics to more generic startup messages.

## Test review and verification

- PBS shell regressions: **20 passed**. These exercise heterogeneous scratch staging/collection, remote completion
  diagnostics, and failure propagation. They mock the launch wrapper and transfers; they do not execute a sample.
- Focused homogenization driver/serialization contracts: **7 passed, 9 deselected** using
  `-k 'scheduler or pbs or resolve_subcmd or main_passes'`.
- Chodby MLMC test collection: **failed** at the removed `dask_worker_preload` import, even with `PYTHONPATH=apps`.
  PyCharm selects the homogenization venv for this module. The first attempt without that path also lacked the
  Chodby package itself; adding the source path exposed the stale import described above.
- Bash syntax and `git diff --check` pass. IDE build reports success with limited build diagnostics.
- Missing focused coverage: direct-local thread count and client lifecycle; worker barrier timeout/invalid values;
  incomplete scheduler/worker preloads; scheduler startup death/timeout; worker registration failure after launch;
  real MLMC sample workspace routing; two local runs during shutdown; existing-but-incomplete copy destinations.
- `test_pbs_startup.py` is currently untracked in this checkout. Include it in the eventual reviewed change so the
  passing shell regressions are retained in version control. No staging or commit was performed.
- `run_helper` cuts shell source at the command dispatcher and rewrites its path discovery. Extracting sourceable
  helper functions would remove this test dependency on source spelling; retain a small real CLI dispatch check.
- `chodby_trans/dask_test` uses its own scheduler-file/SSH demonstration and does not exercise the current wrappers.
  Keep it labeled as a manual demonstration, or replace its duplicated orchestration with current wrapper smoke tests.
- No real PBS allocation, Flow123d calculation, storage conversion, or live cluster shutdown was run for this review.

The installed MLMC source was inspected to establish workspace and finished-count semantics. No dependency files
were changed. The current `sample_counts` subtraction is correct: this MLMC version counts successes and failures
together in `n_finished_samples`.
