# PLAN

## Current Goals

Goal 1: Introduce TransportSimulation for factory of level simulations and calculation of a
        single fine - coarse pair of model evaluations.
        Make a test to this end which restricts the end time of the simulation (use specific
        test config file). With a wise setup we can make it run under 5 minutes.
        The single pair calculation should replace the single_sample function from
        sensitivity_sampling.py. For the dask pool we doesn't need data_schema_key and tags,
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
        
        
AGENT: Review the goals and provided materials. Report if you have lack of information or any
details of the goals that you need to specify better or my specification is ambiguous.
Resolved: Reviewed `AGENTS.md`, `apps/chodby_trans/README.md`, `apps/chodby_trans/SA_USAGE.md`, `python_coding.md`, 
current entry points, config, and relevant tests. Main ambiguities are recorded below in `AGENT Questions And Remarks`.

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
     `TransportSimulation -> SaltelliSchemaSimulation -> Sampler`.
  4. Use `SamplingPoolDask` with the caller-owned Dask `Client` and
     `SampleStorageHDF` persisted inside the app workdir.
  5. Keep logging explicit around planned level counts, scheduled inputs, HDF
     file path, and pool/workdir selection.
  6. Preserve grouped Sobol sampling by generating Saltelli matrices in group
     space and expanding group uniforms to full parameter vectors only inside
     `TransportSimulation`.
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

-  Goal 4 (PE work):
   transport_macro: homogenization fo fixed mesh and interpolation to macro mesh


## AGENT Log
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

## AGENT Questions And Remarks


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
