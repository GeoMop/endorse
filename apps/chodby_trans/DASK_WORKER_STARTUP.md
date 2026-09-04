# Dask Worker Startup, Imports, And Heartbeats

This note records the worker-startup issue observed during MLMC sampling on Metacentrum and the implemented
preload design. It deliberately does not propose changes to the loky subprocess used for Gmsh and mesh preparation.

## Process Layout

The sampling run has three process roles:

```text
PBS allocation
├── Dask scheduler process
└── Dask worker processes
    └── temporary loky child processes for mesh preparation
```

Each Dask worker is a separate Python process with one computation thread. Workers are launched without a nanny and
with `DASK_DISTRIBUTED__WORKER__DAEMON=False`, because a daemonic Python process may not start the loky child needed
by mesh preparation.

## Worker Startup Before Preloading

Previously, a worker started in this order:

1. Start Python and import Dask.
2. Create the Dask worker event loop.
3. Connect to the scheduler and register as available.
4. Run the lightweight project job-directory and logging initialization.
5. Accept the first MLMC task.
6. Import the full transport and mesh stack while executing that first task.

Dask therefore considered the worker ready before modules such as `transport_simulation`, `fullscale_transport`,
PyVista, SciPy, BGEM, Xarray, and Zarr had been imported.

The scheduled task uses the lightweight callable `mlmc_worker.calculate_transport_saltelli`. This prevents task
deserialization from reconstructing the full `sensitivity_sampling.py` driver. The callable deliberately imports
the heavy transport modules only when it begins executing.

Python caches successfully imported modules in `sys.modules`. The first task on every worker pays the import cost;
later tasks in the same worker reuse the modules. Imports are not shared between independent worker processes.
Starting many workers at once can amplify the delay because they concurrently read modules from shared storage.

## Event Loop And Heartbeats

The worker event loop performs Dask control-plane work:

- communicating with the scheduler;
- sending periodic heartbeats and resource metrics;
- receiving tasks;
- reporting task completion and failure;
- responding to cancellation and shutdown.

Task computation and the event loop use threads in the same Python process. Both threads need Python's Global
Interpreter Lock when they execute Python code. Large module imports and module-level initialization can retain or
repeatedly reacquire that lock long enough to delay the event-loop thread.

An `Event loop was unresponsive` warning means that a periodic event-loop callback ran later than expected. It does
not by itself mean that the worker died. Prolonged delays are still undesirable because they postpone heartbeats,
task-state updates, cancellation, and other scheduler communication.

In the 2026-07-24 `workdir_test` run, every worker reported an initial event-loop delay of about 17 to 18 seconds.
For worker 0:

- the worker started at `11:44:48.930`;
- it registered at `11:44:49.332`;
- project job initialization ran at `11:46:35.271`;
- its first Saltelli task was announced at `11:47:31.554`;
- Dask reported a 17.89-second event-loop delay;
- transport execution was announced at `11:48:16.196`.

About 45 seconds elapsed between announcing the first task and reaching transport execution. Existing logs do not
separate module import time from sensitivity-parameter construction, so the preload records import timing directly.

The warning timestamp records when Dask detects the delay, not necessarily when the blocking operation began.

## Implemented Preload Sequence

Workers now start with the `chodby_trans.dask_worker_preload` Dask preload module:

```text
worker process starts
→ preload imports the transport stack and records time and peak RSS
→ worker finishes Dask startup and registers
→ the MLMC driver verifies preload state on every connected worker
→ real samples are scheduled
```

The preload does not remove the import cost. It moves the cost into explicit worker startup, before the worker is
used for MLMC samples. Import failures therefore prevent normal sampling instead of failing or delaying the first
sample on that worker.

The driver-side check reports each worker's preload time, process ID, and peak resident memory. It fails before
sampling if any connected worker has not completed the preload.

The scheduler starts with the separate `chodby_trans.dask_scheduler_preload` module. It imports the lightweight
`mlmc_worker` task module before registering workers or accepting the sampling client. This moves the scheduler's
first task-module import out of graph submission. The driver checks and logs scheduler preload time, process ID, and
peak resident memory before checking workers.

Worker launch scripts also set these native-library thread limits:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Each Dask worker has one computation thread, so allowing numerical libraries to create additional thread pools
would oversubscribe allocated CPUs and could increase startup memory.

## Loky And Gmsh Boundary

`fullscale_transport.run_in_subprocess` creates a fresh loky executor for decorated mesh-preparation calls. The loky
child is another Python interpreter and can have its own startup and import cost. Preloading the Dask parent does
not guarantee that these imports are inherited by the loky child.

The fresh-child design also provides useful isolation:

- fresh Gmsh and geometry-library state;
- no retained mesh data between calls;
- no stale child-process working directory;
- child memory is returned when the child exits.

The current implementation intentionally leaves this lifecycle unchanged. Any future loky optimization should
first measure child startup separately and consider Gmsh state, working-directory correctness, and retained memory.

## Reading Startup Logs

A healthy scheduler log should show its preload before the scheduler starts accepting connections. A healthy worker
log should show a successful transport-preload message with elapsed seconds and peak RSS before normal Dask
registration. It should later show the `Initialized MLMC process context` message and the first `Evaluating Saltelli
MLMC sample` message without a new cold-import delay. The driver log separately confirms the preload state reported
by the scheduler and every worker.

Preload failures should be treated as cluster-startup failures. A scheduler or worker without preload completion is
rejected by the MLMC driver's readiness validation.
