# Goal

- Validate `src/endorse/homogenisation.py` blob upscaling on a controlled 3D single-fracture problem.
- Compare direct estimate, existing blob homogenization, and the MLMC-DFM CNN surrogate on the same 5 x 5 x 5
  macro grid. Keep the expensive per-block Flow123d reference as a later fallback only.

## Setup

- macro domain (0,16)^3 is extended by 8m border to (-8, 24)^3
- macro elements have centers in grid [0, 4, 8, 12, 16]^3 = 125 grid points; each macro block is cube of
  size 8m so the blob of the micro elements will cover this cube
- The macro domain (0,16)^3 has a fixed bulk conductivity 1e-10 and single fracture with apperture 1e-4 and
  conductivity 1e-3.
  Proper bulk value is probably around 1e-11 or 1e-12, but we need a higher value covered by the CNN surogate.
- The effective tensors on the macro grid points will be evaluated for the following 24 fracture configurations,
  cartesian product of:
  - threee fracture sizes : [4, 6, 8] meters
  - four normals n = (0, 0, 1), (0, 1, 1), (0, 1, 2), (1, 1, 1)
  - two shape rotations: 0 deg, 45 deg


## Methods to compare
1. direct estimation, estimate the homogenized tensor as:
   k_bulk ((n \otimes n)) + k_homo * (I - (n \otimes n)); k_homo ~ 1e-3 *1e-4 / 8 ~ 1e-8
   the homogenized tensors should have eigen values in the range (k_bulk, k_homo), while getting close to
   both extremes depending on
   the "area of macro element  - fracture intersection"
   
2. blob homogenization with implementation referenced above

3. CNN surrogate using code from <endorse root>/apps/chodby_trans/MLMC-DFM 
   the approximation should go like:
   1. input: matrix tensors in give points; fractures + their conductivities
   2. interpolate matrix tensors to the 64^3 grid
   3. voxelize fractures to the grid (simple per voxel homogenization equivalent to antialising)
   4. compute avarage conductivity per block K_avg -> divide by it
   5. compute average fracture conductivity Kf and average matix conductivity Km, determine Kf/Km
      -> choose surrogate A, B, C
   6. apply surrogate
   7. multiply by K_avg
   
   Idea is to apply this procedure to each macro element block of size 8m^3, i.e. for each block set matrix
   conductivities and select
   preselect fractures that could intersect the block -> voxelization will do exact selection

4. compute direct homogenization calling Flow123d 3 times for each of the 125 blocks
   We will omit this option from first try, and only apply it if we get contradictory results from methods 1. 2. 3.

- Macro domain `(0,16)^3` is extended by an 8 m border to `(-8,24)^3`.
- Macro centers are `[0, 4, 8, 12, 16]^3`; every averaging block is an 8 m cube centered on a macro point.
- Boundary blocks extend into the buffer domain.
- Bulk conductivity is `1e-10`, fracture aperture is `1e-4`, and fracture conductivity is `1e-3`.
- The 24 cases are the Cartesian product of:
  - square side lengths `[4, 6, 8]` m;
  - normals `(0,0,1)`, `(0,1,1)`, `(0,1,2)`, `(1,1,1)`, normalized before use;
  - shape rotations `0` and `45` degrees using the bgem convention.
- The fracture center is fixed at `(8, 8, 8)`.

## Plan

1. Study setup, geometry, and Method 1 direct estimate
   Status: IMPLEMENTED first pass in `setup.py`, `direct_study.py`, `output.py`, `config.yaml`, and
   `env_check.py`.
   - YAML config owns domain, grid, material, fracture, VTK, surrogate, and Flow123d wrapper settings.
   - Generated study and environment-check artifacts use the fixed `single_fr_study/workdir` layout.
   - Study containers use `attrs`; fracture cases use `bgem.stochastic.Fracture` and `FractureSet`.
   - Method 1 uses numerical square-fracture/cube clipping, aperture-volume fraction, full tensor output,
     sorted eigenvalues, eigenvectors as diagnostics, summary CSV, xarray/Zarr, and pyvista VTK.

2. Method 2: existing blob homogenization
   Status: IMPLEMENTED first pass in `method2.py`; full 24-case run still needs cluster-side review.
   - Method 2 is enabled by `direct_study.py --include-method2`.
   - The app builds a structured extended-domain micro mesh with a coarse voxelized fracture conductivity field.
   - The app builds a macro mesh whose element centers correspond to the 5 x 5 x 5 planned block centers.
   - Blob averaging uses `endorse.homogenisation.Subproblems` with an app-local 8 m cube macro shape.
   - Flow123d runs through `endorse.common.call_flow` and `environment.flow_call` from `config.yaml`.
   - Method 2 writes Flow123d inputs/outputs, macro/micro meshes, and VTK diagnostics under `workdir/study`.

3. Method 3: CNN surrogate
   Status: NEXT.
   - Use trained models under `apps/chodby_trans/MLMC-DFM/optuna_runs`.
   - With `k_fracture / k_bulk = 1e7`, select the surrogate trained for ratio `1e7`.
   - For each 8 m macro block, prepare a 64^3 raster with isotropic matrix tensors and the clipped square
     fracture.
   - Follow `MLMC-DFM/homogenization/sim_sample_3d.py#L2531` and the README Dataset Generation path as the
     first reference.
   - Prefer an in-memory prediction path; write intermediate Zarr only if the existing API requires it.
   - Save predicted tensors in the same component order and result structure as methods 1 and 2.

4. Comparison and reports
   Status: PARTIAL for Method 1 CSV/VTK and xarray/Zarr code; local Zarr write is still an environment issue.
   - Normalize tensor component order at method boundaries.
   - Compare methods primarily by sorted eigenvalue relative error.
   - Use eigenvector direction cosines as diagnostics, with sign ignored.
   - Add plots of eigenvalues over the 5 x 5 x 5 grid and comparison plots for Method 2/3 against Method 1.
   - Mark blocks where Method 1 should be a reliable equivalent tensor separately from blocks where larger
     errors are expected.

5. Method 4 stub only
   Status: TODO.
   - Add a placeholder for direct per-block Flow123d homogenization.
   - Do not schedule or implement the 125-block x 3-load run in the first pass.
   - Document which contradictory Method 1/2/3 results would justify enabling this reference later.

6. Verification
   Status: PARTIAL.
   - Run syntax checks for touched study modules.
   - Run `env_check.py` for imports, xarray/Zarr write, and Flow123d through `endorse.common.call_flow`.
   - Run the Method 1 driver under a timeout for smoke verification.
   - Full Flow123d verification is expected on the target Chodby/cluster environment.

## AGENT log

- 2026-06-19: Refactored `direct_study.py` so the main execution loop iterates by fracture case and dispatches
  all enabled methods per case while reusing the existing direct/blob method implementations. Replaced the
  macro diagnostics multiblock `.vtm` with a single macro-grid `.vtu` per case and kept the Method 2
  micro-mesh output as a separate file.
- 2026-06-19: Compressed resolved Q&A and implemented steps into the current plan. Resolved decisions now live in
  `Setup`, `Plan`, and this log instead of the full historical Q&A text.
- 2026-06-19: Implemented Method 2 first pass using app-generated macro/micro meshes, endorse
  `Subproblems`, `homogenize_batch`, and Flow123d via `call_flow`; verified one Method 2 case locally.
- 2026-06-19: Updated VTK diagnostics to include Method 2 micro mesh output, basic-domain macro tensor points,
  and whole tensor/eigenvalue arrays instead of per-component tensor arrays.
- 2026-06-19: Removed output path parametrization from `config.yaml`; study and environment checks now write
  fixed filenames under `single_fr_study/workdir`.
- 2026-06-19: Flow123d wrapper probe passes with the configured `endorse_fterm --no-term run` command and a
  fixed tiny Darcy input under `workdir/env_check/flow123d`.
- 2026-06-19: xarray/Zarr probe writes successfully to `workdir/env_check/xarray_zarr_probe.zarr`.
- 2026-06-19: Moved Flow123d environment config to `environment.flow_call` and changed `env_check.py` to call
  Flow123d through `endorse.common.call_flow`.
- 2026-06-19: Split first-pass implementation into `setup.py`, `direct_study.py`, and `output.py`; added
  `env_check.py` for package, xarray/Zarr-write, and Flow123d wrapper checks.
- 2026-06-19: Replaced local fracture orientation helpers with `bgem.stochastic.Fracture` plus `FractureSet`;
  output now uses xarray for Zarr and pyvista for VTK.
- 2026-06-19: Local environment verification found package imports pass, but `xarray.Dataset.to_zarr` times out
  and `flow123d` is not on PATH.
- 2026-06-18: Implemented points 1-3 first pass: YAML setup, deterministic cases, numerical direct tensor
  estimates, CSV summary, diagnostic VTK, and later xarray/Zarr output wiring.
- 2026-06-18: Converted answered Q&A into a staged first-pass plan. No code implementation performed.

## AGENT Questions And Remarks

- USER: No open questions after the compression pass.
- Local Flow123d and xarray/Zarr environment checks pass with the fixed workdir probe paths.

AGENT: The Vtk output issues:
  - no micro mesh output (from method 2)
  - the gird for effective tensors is not the extended domain, but it should be only on the basic domain (0, 16)^3
  - Output the eigen values and whole tensor in the grid points, no component only arrays.
Resolved: VTK now adds `method2_micro_mesh`, keeps macro tensors on the 125 basic-domain grid centers, and writes
`direct_tensor`, `direct_eigenvalues`, `blob_tensor`, and `blob_eigenvalues` arrays.
