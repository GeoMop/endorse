# PLAN

## Current Goals

Goal 1: Introduce TransportSimulation for factory of level simulations and calculation of a
        single fine - coarse pair of model evaluations.
        Make a test to this end which restricts the end time of the simulation (use specific
        test config file). With a wise setup we can make it run under 5 minutes.
        The single pair calculation should replace the single_sample function from
        sensitivity_sampling.py. For the dask pool we don't need data_schema_key and tags,
        which are used in current implementation to write the samples to a ZARR storage. We
        instead want to pass results through Dask's means back to the master process where it
        is saved into a HDF5 storage maintained by MLMC. MLMC also prepares and provides the
        sample specific workdir.

Goal 2: Modify entry points in sensitivity_analysis to setup MLMC HDF storage and sampler with dask pool.

Goal 3: Use the sampler to evaluate first N samples of the finest level (l=L) and them M
        samples of the coarse level (l=0) (see separate configuration ofr the levels).
        Assuming just two levels for the moment. Tricky part is that level l-1 depends on the
        homogenized fields on input of the coarse model at level l. Due to lack of support for
        this on the MLMC part we assume to write the coarse fields at level l to the numpy
        files in <level l dir>/coarse_fields and we can read them from level l-1. We only
        need a main sample planing to guarantee few at least 10 samples on level l before
        planning samples on level l-1. To make evaluation of samples reproducible we will
        pass the number of l-level samples at the time of sample planning as the first item of
        the sample input vector.

Goal 4: Side by side to MLMC, we still need to save the results in ZARR storage.
        For this, we need to create the storage node for each level (sofar identified the place into
        `TransportSaltelliSimulation:level_instance`),
        and write the results (sofar identified the place into `TransportSimulation:calculate`).

Goal 5: Add a low-impact paired-sampling mode for coarse-model setup and variance diagnostics.
        One scheduled MLMC sample shall evaluate exactly one shared parameter vector with the
        fine and coarse models, without constructing Saltelli A/B/cross terms. Keep the working
        Saltelli mode unchanged for later Sobol analysis. Use the MLMC HDF and Zarr results to
        replace the auxiliary sequential sampling and collection workflow. After equivalent
        sampling, restart, failure-diagnostic, and plotting functionality is available through
        MLMC, remove `sequential_saltelli_samples.py` and
        `collect_sequential_saltelli_results.py`.

AGENT: Review the goals and provided materials. Report if you have lack of information or any
details of the goals that you need to specify better or my specification is ambiguous.
Resolved: Reviewed `AGENTS.md`, `apps/chodby_trans/README.md`, `apps/chodby_trans/SA_USAGE.md`,
`python_coding.md`, current entry points, config, and relevant tests. Main ambiguities are recorded below in
`AGENT Questions And Remarks`.

## Current Repository State

AGENT: Summarize here key properties of the current state relevant to the goals.
Resolved:
- `sensitivity_sampling.py` is still the main driver. It prepares OpenTURNS/Saltelli
  input rows, schedules `single_sample` over Dask, and persists sample outputs to Zarr.
- `single_sample` currently owns sample directory discovery/creation, config patching
  from sampled parameters, `transport_wrapper.Wrapper` execution, and return-code
  handling. This is the primary replacement target for Goal 1.
- `transport_wrapper.Wrapper.get_observations()` still returns one result slice and
  writes it into Zarr through `job.output.zarr_store_path`. That storage side effect
  has to move out of the worker path for MLMC.
- `job.py` already separates `input`, `scratch`, and `output` directories. That should
  remain the path abstraction used by any MLMC simulation wrapper.
- `fullscale_transport.transport_run()` already has fine plus macro/coarse preparation
  in one call path. The coarse branch is incomplete and currently not returned as a
  paired MLMC fine/coarse result.
- `input_data/trans_mesh_config.yaml` already contains two `mlmc.levels`, so there is
  existing config structure for a two-level setup even though the current runtime is
  still OpenTURNS/Zarr-oriented.
- `input_data/_ot_sensitivity.yaml` still defines the active Sobol sampling
  configuration, including grouping and QMC sampling assumptions. MLMC integration has
  to preserve parameter mapping compatibility.
- `SA_USAGE.md` expects one MLMC scheduled sample to represent one complete Saltelli
  row, with outputs stored in HDF5 and later exposed through `Quantity` with the
  leading Saltelli-term axis preserved.
- Existing tests in `test/test_openturns.py` and `test/test_ot_sa.py` cover OpenTURNS
  design and Sobol processing, but there is no current test around an MLMC
  `Simulation`, storage, or Dask-backed fine/coarse pair evaluation.

## Work Plan
AGENT:  write a particular steps for each goal following your understanding. But still
make detailed goal plan very compact.
Resolved:
- Goal 1:
  1. Introduce a new MLMC-facing `TransportSimulation` module instead of
     patching MLMC concerns into `transport_wrapper.Wrapper`.
  2. Route the real transport branch through `fullscale_transport.transport_run`
     and keep transport-side pair assembly inside that module rather than in a
     new adapter-specific helper.
  3. Treat MLMC `level_parameters` as transport level selectors. Current
     implementation supports explicit level selectors and mesh-step-like values.
  3. Move sampled-parameter application into a reusable helper so both the
     future MLMC entry point and any local single-pair debug call use the same
     config patching path.
  4. Replace worker-side Zarr persistence with direct fine/coarse return values.
     The worker output for Goal 1 is a compact time series:
     `q99_XYZ(log10(concentration))`.
  5. Keep `calculate()` independent of config-file reads in the worker. The
     adapter passes level/root config through `config_dict` and only uses
     `job.set_workdir(...)` to restore input/output path conventions.
  6. For the current state of `transport_run`, the fine result is real and the
     coarse result remains an adapter placeholder until transport-side pair
     outputs are exposed directly.
  6. Add a focused MLMC test config and a unit/integration-style test that runs
     one pair through the new adapter using `test_random_data`.
  DONE.
- Goal 2:
  1. Keep the current OpenTURNS/Zarr path intact for now and add a parallel
     MLMC entry path in `sensitivity_sampling.py` instead of replacing the old
     commands in one step.
  2. Add explicit workdir paths for MLMC HDF storage and an MLMC output
     directory in `job.py`.
  3. Build the MLMC forward simulation as:
     `TransportSimulation -> TransportSaltelliSimulation -> Sampler`.
  4. Use `SamplingPoolDask` with the caller-owned Dask `Client` and
     `SampleStorageHDF` persisted inside the app workdir.
  5. Keep logging explicit around planned level counts, scheduled inputs, HDF
     file path, and pool/workdir selection.
  6. Preserve grouped Sobol sampling by generating Saltelli matrices in group
     space and expanding group uniforms to full parameter vectors only inside
     `TransportSimulation`.
  7. Inline the Saltelli wrapper behavior directly into
     `TransportSaltelliSimulation` so it inherits from MLMC `Simulation`
     without depending on `mlmc.sim.saltelli_simulation.SaltelliSchemaSimulation`.
- Goal 3:
  1. Pass the number of avaliable fine samples to the coarser level as the
     first value in input_vec.
  2. Use a custom `LevelSimulation.prepare_samples()` wrapper on the transport
     simulation so the master can prepend the current finer-level collected
     sample count to each scheduled parameter vector.
  3. Implement staged planning in the entry point: first schedule the finest
     level only, wait until a minimum number of completed samples is available,
     then schedule the coarser level.
  4. Leave the cross-level `coarse_fields` file handoff as a later transport
     concern; the current Goal 3 implementation only establishes the planning
     contract and propagated leading input value.
  5. Persist scheduled work items in HDF and re-submit unfinished samples on
     restart because `SamplingPoolDask` itself does not hold permanent tasks.

- Goal 4:
  1. Keep MLMC Zarr storage separate from the legacy OpenTURNS storage layout,
     but reuse the same axis conventions where possible.
  2. Create one fixed-capacity Zarr group per MLMC level from
     `TransportSaltelliSimulation.level_instance()`.
  3. Size the per-level storage as:
     `i_sample = cfg.ot_sensitivity.n_samples`,
     `i_saltelli = schema.n_terms`.
  4. Store both fine and coarse outputs and metadata in dedicated variables:
     `fine_conc`, `coarse_conc`, `fine_return_code`, `coarse_return_code`,
     `fine_eval_time`, `coarse_eval_time`.
  5. Move MLMC sample-path parsing from `sensitivity_sampling.py` into
     `transport_simulation.py` so `TransportSimulation.calculate()` can derive
     `(level_id, i_sample)` from the sample workspace path.
  6. Extend the per-term forward input tail to carry both `i_saltelli` and the
     finer-level collected-sample count, and let
     `TransportSimulation.calculate()` decompose that tail before writing.

Goal 5: Verify and correct `MacroTetra.interact` kernel weights.

1. Add lightweight regression coverage using a reference tetrahedron and micro-element
   barycentres, without invoking the full transport model.
2. Change the core implementation to compute barycentric coordinates in the tetrahedron
   scaled about its centre, then calculate the piecewise-linear radial kernel from the
   smallest barycentric coordinate.
3. Keep the point-coordinate calculations array-oriented so batching micro barycentres can
   reuse the same geometry algebra when the API is extended.

- Goal 6:
  1. Add an explicit `mlmc.sample_mode` switch with `saltelli` as the
     backward-compatible default and `paired` as the coarse-model diagnostic mode.
  2. Add a small paired simulation wrapper beside `TransportSaltelliSimulation`.
     Generate one grouped QMC/MC parameter row per MLMC sample, use
     `n_saltelli=1`, and reuse the existing lightweight worker with one term.
  3. Keep `TransportSimulation.calculate()`, fine/coarse execution, return codes,
     persistent `00` workspaces, Dask initialization, and concurrent Zarr writes
     unchanged. Treat Zarr `i_saltelli=0` as a compatibility index only.
  4. Expose the paired fine/coarse compact time series in HDF without a logical
     Saltelli axis. Schedule the desired tens or hundreds of pairs through the
     finest level's `min_samples`.
  5. Make the paired sample target and Zarr `i_sample` capacity consistent.
     Initially document the required configuration relationship; make one field
     authoritative only if the implementation review shows that this is safer.
  6. Add an MLMC paired-results collector that reads compact collected pairs from
     HDF and return codes, parameters, and timings from Zarr, then reuses the
     existing variance, correlation, bias, cost, bootstrap, and plotting logic.
  7. Cover paired input generation, deterministic scheduling, singleton Zarr
     indexing, result shape, and unchanged Saltelli behavior with focused tests.
     Once MLMC has equivalent operational and diagnostic coverage, remove
     `sequential_saltelli_samples.py` and
     `collect_sequential_saltelli_results.py`.

- MLMC failure diagnostics:
  1. Keep fine and coarse failure state separate in Zarr, retaining a completed
     fine result when the following coarse calculation fails.
  2. Add explicit fine, coarse, and homogenization failure codes and exception
     boundaries with stage-specific logging.
  3. Execute every Saltelli term in a persistent two-digit subdirectory below
     its MLMC sample workspace.
  4. Seed the OpenTURNS matrix generator explicitly and cover repeatability and
     per-term workspace behavior with focused tests.

## AGENT Log
- 2026-08-10: Fixed `MacroTetra.interact` to use barycentric coordinates of the
  center-scaled tetrahedron and added `interaction_weights` for batched micro-element
  barycentres. Added focused core-suite regression tests in
  `tests/homogenization/test_homogenisation.py` for outside, fractional interior, and
  batched kernel weights; all three pass. Refined the geometry implementation to scale the
  Jacobian directly while retaining the centroid-preserving scaled reference vertex.
- 2026-07-28: Added Goal 6 paired MLMC sample mode beside Saltelli. Paired mode schedules one grouped
  parameter row per MLMC sample, keeps singleton Zarr term metadata, exposes HDF results without a
  logical Saltelli axis, and adds paired MLMC variance/correlation/bias diagnostics.
- 2026-07-29: Fixed paired MLMC analysis for staged HDF files by skipping levels without collected values.
  Zarr metadata is read directly from local Zarr v3 chunks, avoiding the hanging `zarr.open_*` path, and the
  paired plot now matches the sequential `fine_coarse_mlmc_diagnostics.pdf` layout with individual subfigures.
  Paired variance diagnostics live in `mlmc_var_analysis.py`; `mlmc_analysis.py` only dispatches paired mode.
- 2026-07-30: Resolved the paired variance-analysis source note to log the five samples with largest absolute
  fine/coarse differences, including sample id, time, and signed difference value.
- 2026-08-13: Added paired MLMC fine/coarse distribution PDFs using `plot_conc_timeseries_distribution1`.
- 2026-07-27: Initialized `job` paths in every Dask worker preload, propagated shared input/output paths from both
  cluster launchers, and added an expected-worker registration barrier before MLMC scheduling.
- 2026-07-27: Added a reusable read-only PBS run inspector subagent specification, with general PBS output,
  scheduler, worker, node-scratch, timeline, and failure-summary guidance plus an MLMC-skill handoff.
- 2026-07-27: Added the repository-local `inspect-mlmc-run` skill with a read-only summarizer for PBS logs,
  MLMC HDF state, Zarr return codes, copied node scratch artifacts, and native core dumps.
- 2026-07-27: Moved the run-global homogenization mesh from node-local scratch to the shared
  `<job.output.dir_path>/homogenization` directory and covered the location with a focused test.
- 2026-07-24: Separated fine, coarse, and homogenization failures in MLMC Zarr
  metadata, retained completed fine data across coarse failures, and added full
  stage and loky traceback logging. Saltelli terms now keep two-digit workdirs,
  and production OpenTURNS matrices are explicitly seeded for repeatable fresh
  runs with the same configuration and scheduling sequence.
- 2026-07-24: Added timed scheduler preloading of the lightweight MLMC task
  module, with peak-RSS reporting and driver-side readiness validation, to
  move the remaining first task-module import out of graph submission.
- 2026-07-24: Added explicit Dask worker preloading for the heavy transport
  module stack, including import time and peak-RSS logging, driver-side
  readiness validation, and single-thread limits for native numerical
  libraries. Documented worker startup, heartbeat delays, and the unchanged
  loky/Gmsh process boundary in `DASK_WORKER_STARTUP.md`.
- 2026-07-24: Persisted MLMC transport failures to the per-level Zarr store
  before re-raising them to MLMC. Known wrapper exceptions retain their
  geometry, mesh, healing, or Flow123d return code; unexpected exceptions use
  `UNKNOWN_ERROR`, with zero-valued concentration placeholders.
- 2026-07-24: Made the transport result grid an explicit `grid_size`
  configuration value. MLMC startup validates it against the generated data
  schema, while worker postprocessing no longer depends on a run-specific
  `data_schema_key`.
- 2026-07-23: Replaced bound `TransportSaltelliSimulation` Dask task
  callables with functions from the importable lightweight `mlmc_worker`
  module. Worker deserialization no longer reconstructs the directly executed
  `sensitivity_sampling.py` module or its driver-side sample-planning closure;
  transport root configs cross Dask as plain built-in containers. Added a
  focused cloudpickle regression test for the worker boundary.
- 2026-07-23: Configured Metacentrum Dask workers as non-daemonic before
  launch so MLMC tasks can start the loky subprocess used for mesh preparation.
  Added process name/PID/PPID/daemon logging at Dask worker initialization and
  immediately before loky submission for post-mortem verification.
- 2026-07-21: Added an explicit MLMC Zarr capacity guard in
  `transport_simulation.py` so writes now fail with a direct
  `sample_id/saltelli_id outside storage extent` error instead of an opaque
  xarray dimension-mismatch when the planned MLMC sample count exceeds the
  fixed-capacity Zarr layout.
- 2026-07-21: Fixed two MLMC Zarr write regressions found in
  `workdir_test/logs/worker_0.log`: per-level storage creation now checks for
  the specific `mlmc/level_XX` group instead of the root store, and the
  Dask-region lock helper now derives chunk lengths from the `i_sample` and
  `i_saltelli` dimensions directly.
- 2026-07-21: Moved MLMC mesh-level patching into
  `chodby_trans/mlmc_levels.py` and reused it from both
  `fullscale_transport.py` and `mesh/run_create_mesh.py`. While rewiring the
  helper, fixed the remaining old-order coarse/fine neighbor direction in
  `fullscale_transport.py` to match the new `cfg.mlmc.levels` order.
- 2026-07-21: Updated the MLMC level-order contract to follow
  `cfg.mlmc.levels` directly as `coarse -> fine`: removed the reverse mapping
  from `mlmc_level_parameters()`, switched staged scheduling and finer-sample
  gating to computed coarse/fine indices, and aligned local fixtures/helpers
  that still assumed the older `fine -> coarse` config order or explicit
  per-level `id` fields.
- 2026-07-17: Added continue-mode handling for
  `sequential_saltelli_samples.py`: existing `status.json` files are treated
  as authoritative for restart, unfinished sample dirs without status are
  rerun, and `summary.json` is rebuilt from on-disk statuses at the end.
- 2026-07-15: Made `ensure_mlmc_level_zarr_storage()` recreate stale local
  MLMC level groups when their shape or expanded `param_name` axis no longer
  matches the current transport config, avoiding reuse of incompatible
  development-era Zarr schemas across reruns.
- 2026-07-15: Fixed MLMC Zarr parameter-axis sizing in
  `transport_simulation.py` to use the expanded `SensitivityAnalysis`
  parameter names rather than the raw config keys, matching the actual stored
  full parameter vector for DFN population-derived parameters.
- 2026-07-15: Realigned `sequential_saltelli_samples.py` with the current
  `TransportSimulation.calculate()` contract by passing the per-term Saltelli
  index and `n_saltelli` through `sample_input` and `config_dict`.
- 2026-07-13: Simplified `read_parameters_by_rc()` for the development MLMC
  Zarr path only. It now assumes `mlmc/level_*` groups exist, reads them
  through `zarr` directly, and emits separate fine/coarse diagnostics per
  level without any legacy-layout fallback or local metadata repair.
- 2026-07-13: Implemented Goal 4 MLMC-side Zarr integration across
  `transport_simulation.py`, `sensitivity_sampling.py`, and
  `fullscale_transport.py`: added fixed-capacity per-level MLMC groups,
  per-term `i_saltelli` propagation, worker-side writes of fine/coarse
  concentration blocks plus return/time metadata, and a focused synthetic
  storage test in `test/test_mlmc_sampling.py`.
- 2026-06-30: Inlined the Saltelli wrapper logic into
  `TransportSaltelliSimulation`, so the chodby_trans MLMC path now subclasses
  MLMC `Simulation` directly and no longer imports
  `mlmc.sim.saltelli_simulation.SaltelliSchemaSimulation`.
- 2026-06-23: Added bootstrap IQR uncertainty bands for fine, coarse, and
  fine-coarse variance estimates to the sequential Saltelli MLMC diagnostics
  plot and CSV output.
- 2026-06-23: Added `diagnostics_summary.md` next to the sequential Saltelli
  plots, summarizing current fine/coarse MLMC diagnostics and physical
  interpretation.
- 2026-06-23: Added `--plot-diagnostics` to
  `collect_sequential_saltelli_results.py`, producing fine/coarse variance,
  variance-ratio, correlation, and difference-bias diagnostics plus CSV data.
- 2026-06-23: Added `--plot-fine-coarse` to
  `collect_sequential_saltelli_results.py` for a simple paired fine/coarse
  time-series comparison figure without histograms.
- 2026-06-23: Added a `plot_all_lines` option to
  `plot_conc_timeseries_distribution1` and exposed it as `--plot-all-lines`
  for sequential Saltelli result plots.
- 2026-06-23: Added `--plot` support to
  `collect_sequential_saltelli_results.py`, building the minimal xarray
  dataset needed by `plot_conc_timeseries_distribution1` from gathered
  `result.npz` files and writing fine/coarse distribution PDFs.
- 2026-06-23: Extended `collect_sequential_saltelli_results.py` with a
  `--gather-only` mode that copies lightweight per-sample result files into a
  separate gather directory while preserving sample subdirectories.
- 2026-06-22: Added `collect_sequential_saltelli_results.py` to collect
  completed sequential Saltelli `result.npz` files into aligned parameter,
  fine-result, and coarse-result CSV tables.
- 2026-06-22: Added a subprocess-region quick fix in `mesh/create_mesh.py`:
  before meshing, BGEM `Region._max_reg_id` is bumped above the unpickled
  fracture region ids to avoid collisions with newly created boundary regions.
- 2026-06-22: Added `sequential_saltelli_samples.py` to run real
  `input_data/transport_mlmc.yaml` Saltelli terms sequentially without
  MLMC/Dask storage, writing generated group rows and transformed parameters.
- 2026-06-21: Made `job.set_workdir(..., input_dir=...)` export the input
  directory through `ENDORSE_INPUT_DIR`, so loky subprocesses that call
  `job.set_workdir(workdir)` inherit the correct shared `input_data` path.
- 2026-06-20: Fixed two PBS MLMC worker issues from `driver_174251`: worker
  launch now passes `ENDORSE_DISABLE_MEMOIZE` explicitly through `pbsdsh`, and
  fine mesh subprocess setup preserves the configured `input_data` directory.
- 2026-06-20: Updated `dask_cluster.sh` to reserve scheduler-node worker slots
  through `DASK_HEAD_WORKER_RESERVE` (default 2), reducing CPU starvation of
  the Dask scheduler and driver during MLMC sample bursts.
- 2026-06-20: Added `ENDORSE_DISABLE_MEMOIZE=1` support in
  `src/endorse/common/memoize.py` so cluster MLMC runs can bypass all existing
  `@memoize` decorators without manually editing hot-path functions.
- 2026-06-20: Changed `ot_sa.SensitivityAnalysis.from_cfg` caching to use a
  deterministic JSON key built from `dotdict.serialize(...)`, with the cached
  reconstruction moved into `_from_cfg_cached`. This avoids `TypeError:
  unhashable type: 'dotdict'` when memoizing scheduler-side deserialization.
- 2026-06-12: Added an independent `mlmc_analysis` subcommand that reads the
  existing MLMC HDF storage, computes fine/coarse/difference variance
  diagnostics for Sobol averaging quantities, and writes CSV plus PDF plots.
- 2026-06-12: Made the MLMC driver instantiate the forward simulation class
  named by `cfg.mlmc.sim_class` from `transport_simulation.py`. Added
  `RandomTransportSimulation` for lightweight synthetic concentration runs and
  removed the random-data branch from `TransportSimulation.calculate`.
- 2026-06-13: Isolated Dask MLMC sample execution in per-sample subprocesses
  so Gmsh signal handling and process cwd changes do not run in worker
  threads. Tightened MLMC sampling waits and tests to require collected HDF
  samples and fail on stored worker failures.
- 2026-06-11: Fixed the first MLMC sampling blockers in the local app path:
  reversed `mlmc_level_parameters()` into MLMC order, read Goal 3 sample
  targets from `mlmc.levels[*].min_samples` and `min_finer_samples`, made the
  finer-sample planning count read live sampler state, and changed
  `TransportSaltelliSimulation` to persist one scheduled Saltelli matrix per
  sample so HDF restarts do not corrupt `scheduled_inputs`. Updated
  `test/test_mlmc_sampling.py` to use a fresh copied workdir config and to
  overwrite the actual runtime `transport_mlmc.yaml` with
  `test_random_data=True`.
- 2026-06-08: Reviewed planning and integration context for MLMC Sobol work;
  summarized current OpenTURNS/Zarr pipeline, MLMC expectations, and missing
  design details.
- 2026-06-08: Preserved the simplified `transport_simulation.py` structure and
  tuned the blocking bits only: removed the obsolete local output-times helper,
  fixed the MLMC sample-input contract handling, kept per-sample config
  isolation, enforced explicit `task_size` in the Goal 1 test config, and
  aligned the focused test to the current leading sample-size input convention.
  Verification passed with `apps/chodby_trans/venv/bin/python -m py_compile
  apps/chodby_trans/transport_simulation.py
  apps/chodby_trans/test/test_transport_simulation.py` and
  `apps/chodby_trans/venv/bin/python -m pytest
  apps/chodby_trans/test/test_transport_simulation.py -vv`.
- 2026-06-08: Implemented Goal 1 transport-side MLMC adapter in
  `transport_simulation.py` and a focused synthetic MLMC test config with
  `test/test_transport_simulation.py`. After review of in-code instructions,
  the adapter was aligned to call `fullscale_transport.transport_run`
  directly and to keep worker execution driven by `config_dict` plus the MLMC
  sample workspace. Verification passed with
  `apps/chodby_trans/venv/bin/python -m py_compile ...` and
  `apps/chodby_trans/venv/bin/python -m pytest
  apps/chodby_trans/test/test_transport_simulation.py -vv`.
- 2026-06-08: Implemented Goal 2 and the planning part of Goal 3 in
  `sensitivity_sampling.py`: added the MLMC/HDF/Dask driver, grouped-Saltelli
  matrix generation, staged fine-then-coarse scheduling, and restart
  re-submission of unfinished HDF-planned samples. Extended
  `transport_simulation.py` so MLMC grouped inputs expand to the full
  transport parameter vector inside the worker. Added focused tests in
  `test/test_mlmc_sampling.py`.
- 2026-06-14: Added `plot_scripts/dfn_trace_matrix.py` for deterministic DFN
  trace visualization from the main transport config. It resolves the fracture
  template locally, samples a large fracture cloud, clips traces to the problem
  box, and renders XY/XZ/YZ projections for a geometric progression of
  `r_limit` values.
- 2026-06-15: Extended `plot_scripts/dfn_trace_matrix.py` to export each
  thresholded fracture set as a full 3D Gmsh mesh named `mesh_{r_limit}.msh`
  alongside the figure output.

## AGENT Questions And Remarks

- 2026-08-10: Goal 5 needs an edit to `src/endorse/homogenisation.py`, but the current
  repository instruction restricts edits to `apps/chodby_trans`. The app-level regression
  test can be added now; explicit authorization is needed before changing the core module.
  Resolved: The user authorized edits to `homogenisation.py` and its test on this branch.

- 2026-07-24: `AGENTS.md` requires `python_coding.md`, but that file is not
  present in the repository workspace and could not be reviewed.

- 2026-06-22: Investigation note: the real fine `box_drilled` mesh path still
  drops boundary physical groups such as `.side_x0`, `.tunnel_head_y0`, and
  `.fractures_out` before `input_fields.msh2` reaches Flow123d. This is now
  separate from the lightweight MLMC sampling test, which should not run Flow.

- `SA_USAGE.md` defines the generic MLMC contracts, but the exact
  Dask-compatible MLMC sampling-pool class expected in this project is still
  unspecified. The implementation plan assumes such a pool already exists or
  will be provided by the installed MLMC branch.
  AGENT: Full MLMC sources provided for reference SamplingPool Dask ready to use.
- Goal 1 implementation assumption: MLMC `level_parameters` may come either as
  explicit positive level selectors or as mesh-step-like values. The adapter
  currently accepts both and maps them onto `cfg.mlmc.levels`.
- The exact MLMC output quantity to store is still not stated. The current
  transport path yields one concentration slice plus return metadata; MLMC
  Sobol estimation will need a stable fine/coarse result shape and
  result-format definition before implementation.
  AGENT: The model output is the concentration array with coords: x,y,z,
  simulation_time. To make the output passed by Dask small please add part of
  postprocessing to the end of the model (both coarse and fine): a) log10 fo
  concentration b) quantile over xyz coordinates
  Then the QuantitySpec is float dependent on the time coord.
- Goal 3 depends on a cross-level `coarse_fields` handoff. The naming, file
  layout, and invalidation rules for those files should be fixed early to avoid
  non-reproducible reuse.
  AGENT: I agree, so provide your suggestions adn quastions to the steps for goal 3 I will give you a feedback.
- Suggestion for Goal 3 `coarse_fields` handoff:
  use `<sample_dir>/coarse_fields/L{fine_level_id:02d}/fields.npz` plus a
  sibling `meta.json` storing the fine sample id, transport level id, mesh
  name, and a hash of the parameter vector. Then level `l-1` can read only
  files whose metadata matches the current sample and invalidate stale
  homogenized fields deterministically.
- The current repository contains unrelated user changes, including an existing
  modification in `apps/chodby_trans/sensitivity_sampling.py`. Any later
  implementation must be rebased carefully onto that local state rather than
  overwrite it.
  AGENT: Just stage files with changes.
- 2026-06-13 machine_config compatibility review: `submit_pbs` still reads
  `cfg.machine_config.pbs` directly and will need the resolved machine block
  under the new `machine_config.__resolved__` mechanism.
- 2026-06-13 machine_config compatibility review: legacy
  `src/endorse/scripts/endorse_mlmc.py:create_sampler` passes the full
  `cfg.machine_config` host map to `create_sampling_pool`, which expects PBS
  keys on the selected machine block.
- 2026-07-13 local verification note: `py_compile` passes for the Goal 4
  changes, but the focused local MLMC Zarr probe still hangs during runtime
  execution in this environment before a clean passing test result is reached.
- 2026-07-21 investigation note: current MLMC scheduling count and MLMC Zarr
  capacity are driven by different config fields. `run_mlmc_sampling()`
  schedules from `cfg.mlmc.levels[*].min_samples` (currently 10 in
  `transport_mlmc.yaml`), while `ensure_mlmc_level_zarr_storage()` allocates
  `i_sample` from `cfg.ot_sensitivity.n_samples` (currently 4 in
  `_ot_sensitivity.yaml`). These values must match, or one side must be made
  authoritative.
- 2026-07-28 local verification note: direct MLMC Zarr storage creation still
  hangs locally in `zarr.open_group(...)` during existing Zarr-backed sampling
  tests. Goal 6 tests patch the Zarr boundary and verify singleton write
  arguments instead of exercising the local Zarr backend.
