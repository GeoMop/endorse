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
  1. Introduce a `TransportSimulation` adapter that matches the MLMC
     forward-simulation contract and builds per-level simulation instances from
     current config and `job` paths.
  2. Refactor current `single_sample` logic into a pair-evaluation path that
     accepts MLMC-provided sample input and workdir, runs fine/coarse
     evaluations, and returns results without Zarr writes.
  3. Add a focused test config with shortened transport horizon and a test that
    exercises one pair evaluation end to end.
- Goal 2:
  1. Replace Zarr-oriented sampling entry points with MLMC sampler/bootstrap
     setup while keeping existing CLI/workdir conventions where possible.
  2. Create HDF5 storage initialization, Saltelli schema simulation wiring, and
     Dask sampling-pool setup in `sensitivity_sampling.py`.
  3. Keep logging explicit around scheduled inputs, sample ids, worker
     workdirs, and persisted HDF records for postmortem analysis.
- Goal 3:
  1. Define the two-level sample input so reproducibility data and Saltelli rows
     are both explicit in the MLMC sample payload.
  2. Implement staged planning: schedule enough finest-level samples first,
     then allow coarse-level scheduling once coarse-field dependencies are
     available.
  3. Store transferred coarse fields in a deterministic per-level location and
     make level `l-1` read only the fields produced by the intended level-`l`
     planning state.



## AGENT Log
- 2026-06-08: Reviewed planning and integration context for MLMC Sobol work;
  summarized current OpenTURNS/Zarr pipeline, MLMC expectations, and missing
  design details.

## AGENT Questions And Remarks


- `SA_USAGE.md` defines the generic MLMC contracts, but the exact
  Dask-compatible MLMC sampling-pool class expected in this project is still
  unspecified. The implementation plan assumes such a pool already exists or
  will be provided by the installed MLMC branch.
  AGENT: Full MLMC sources provided for reference SamplingPool Dask ready to use.
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
- The current repository contains unrelated user changes, including an existing
  modification in `apps/chodby_trans/sensitivity_sampling.py`. Any later
  implementation must be rebased carefully onto that local state rather than
  overwrite it.
  AGENT: Just stage files with changes.
