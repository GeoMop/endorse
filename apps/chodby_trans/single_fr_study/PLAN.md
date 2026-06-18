# Goal

- src/endorse/homogenisation.py implements a "blob upscaling" where we compute three velocity fields for three
  pressure loads on the whole domain and then compute macro element conductivity tensor from avarages of velocities
  and pressure gradients on the irregular blob of micro elements arround the macro element
- goal is to validate this approach on a simple problem with single fracture

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

## PLAN

AGENT: Plan for the first implementation pass. Keep all new study code under
`apps/chodby_trans/single_fr_study`; call existing core/app homogenization code
without copying it. Ask explicitly before any change outside `apps/chodby_trans`.

Resolved: User answers in `QaA` were incorporated into this plan on 2026-06-18.

1. Study package and configuration
   - Add a small app-local study driver, configuration module, and result writer
     under `single_fr_study`.
   - Store all run outputs under a study output directory, with one subdirectory
     per fracture configuration.
   - Represent the 24 cases as the Cartesian product of:
     fracture square side length `[4, 6, 8]`, normalized normals, and shape
     rotations `[0, 45]`.
   - Use a fixed fracture center `(8, 8, 8)`, aperture `1e-4`, fracture
     conductivity `1e-3`, and bulk conductivity `1e-10`.
   - Use the macro grid centers `[0, 4, 8, 12, 16]^3`; every macro block is an
     8 m cube centered on the grid point and may extend into the buffer domain.

  AGENT: put the setup into a yaml config   
     
2. Geometry and tensor conventions
   - Generate the square fracture using the same orientation convention as bgem:
     rotate the square counterclockwise in the XY plane by the shape rotation,
     then rotate the Z axis to the normalized fracture normal.
   - Use the direct tensor convention:
     `K = k_normal * n n^T + k_tangent * (I - n n^T)`.
   - Treat the direct estimate as low normal conductivity and high tangential
     conductivity, with tangential strength scaled by fracture aperture-volume
     fraction in the 8 m block.
   - Write a small geometry diagnostic per case: normal, tangent basis, fracture
     corners, block/fracture intersection volumes or areas used by method 1.
   
   AGENT:
   - for each case create a VTK file with:
     - the basic and extended domain boxes, fracture polygon
     - the micro mesh with bolean incidence arrays for each block named by the block center like 'block_[0,1,1]'
       indicating the averaging blob of that block. 
     - tensor values in the grid points for each of the compared methods
   
3. Method 1: direct estimate
   - Implement an analytical/numerical direct estimator for every macro block.
   - Clip/intersect the square fracture with each 8 m cube and compute an
     aperture-volume fraction.
   - Convert the volume fraction to the tangential effective contribution, using
     `k_homo = k_fracture * aperture / 8` as an upper scale.
   - Save tensors, sorted eigenvalues, eigenvectors, and the scalar intersection
     diagnostics for all 125 block centers.
   
     
4. Method 2: existing blob homogenization
   - Build the single-fracture micro mesh on the extended domain and prepare
     pressure-load fields compatible with the existing homogenization path.
   - Call the existing blob path, primarily `macro_conductivity` /
     `Subproblems`, without modifying or copying core code.
   - Use a macro mesh whose volumetric element centers correspond to the
     planned 5 x 5 x 5 block centers, or add only app-local glue if such a mesh
     has to be generated specially for this study.
   - Persist Flow123d return codes, mesh paths, field paths, and homogenization
     logs so a failed cluster run can be inspected after the fact.

   AGENT: make sure you can run Flow123d  
     
5. Method 3: CNN surrogate
   - Use trained models under `apps/chodby_trans/MLMC-DFM/optuna_runs`.
   - With `k_fracture / k_bulk = 1e7`, select the surrogate trained for the
     `1e7` fracture-to-matrix conductivity ratio.
   - For each 8 m macro block, create a 64^3 raster with isotropic matrix
     tensor values and the clipped/intersecting square fracture.
   - Reuse the existing MLMC-DFM rasterization and prediction code where it is
     callable; otherwise add a thin app-local adapter that prepares its expected
     Zarr inputs and calls the model-loading/prediction functions.
   - Save predicted tensors in the same component order and file structure as
     methods 1 and 2.
     
   AGENT: comment from colleaguue:
   "Asi nejjednodussi bude vygenerovat data a zarr soubor, podle navodu v readme - Dataset Generation. Mozna by to slo i nejak zjednodusit pouzitim jenom casti kodu napr. v homogenization/sim_sample_3d.py#L2531"
   Follow that code path, but I prefer not to write the interrmediate ZARR just for surrogate evaluation. In future we want to eavaluate the 
   surrogate for large numbers of inputs just in memory without file writes.
   

6. Comparison and reports
   - Normalize tensor component order at the boundary between methods.
   - For each method and block, compute sorted eigenvalues and eigenvectors.
   - Compare methods by relative eigenvalue error and eigenvector direction
     cosine, treating eigenvector sign as irrelevant.
   - Produce CSV/NPZ tables for all raw tensors and diagnostics.
   - Produce first plots of eigenvalues over the 5 x 5 x 5 grid and comparison
     plots for method 2 vs method 1 and method 3 vs method 1.
   - Mark cases where method 1 is expected to be a reliable equivalent tensor
     separately from cases where larger error is expected.

7. Method 4 stub only
   - Add a placeholder interface and documentation for direct per-block Flow123d
     homogenization.
   - Do not schedule or implement the 125-block x 3-load reference run in the
     first pass.
   - The stub should make clear which contradictory results would justify
     enabling this method later.

8. Verification for the first implementation pass
   - Run syntax/import checks for the new study modules.
   - Run lightweight geometry/unit checks for normals, rotations, tensor
     ordering, block centers, and method-1 bounds.
   - Do not require a full Flow123d run locally; keep logging and persisted
     intermediate data sufficient for cluster-side inspection.

## AGENT log

- 2026-06-18: Converted answered Q&A into a staged first-pass plan. No code
  implementation performed.


   
## QaA

Put here any questions if you are not sure that you understand what exactly is my intent or how some code works.

 Questions I can already ask because the spec is otherwise plannable:

  1. For the single fracture, should the fracture be centered at (8, 8, 8) for all 24 configurations?
     AGENT: yes
     
  2. In “fracture size” [4, 6, 8], is that diameter, radius, side length, or some other characteristic length of
     the fracture shape?
     AGENT: In bgem it is named 'radius" but it is in fact side length for the square shaped fractures
     
     
  3. What is the fracture shape for this study: square, disk/ellipse, rectangle, or the same shape class used by
     the DFN code?
     AGENT: square
     
  4. Are the listed vectors (0,0,1), (0,1,1), (0,1,2), (1,1,1) fracture plane normals? I read the direct tensor
     formula as low conductivity in the normal direction and high conductivity in the tangent
     plane.

     AGENT: Yes to both.However the listed vectors are not unit normals, must be normalized first.
     
  5. What exactly should “shape rotation 0 deg / 45 deg” rotate around: the fracture normal within the fracture
     plane, or a global axis?
  
     AGENT: that is called probably shape rotation in bgem, the shape is first rotated conter clockwise in XY
     plane and then Z axis is rotated to the normal.
     
  6. For macro grid points on the boundary, e.g. center 0 or 16, should the 8 m block be centered on the point
     and extend into the (-8,24)^3 buffer, or clipped to (0,16)^3 for direct/reference
     calculations?
     AGENT: block is centered on the grid points extending into buffer out of the basic (0,16)^3 domain.
      
  7. Should method 1 use only the fracture area inside each 8 m block as a scalar multiplier of k_homo, or
     should it use a more explicit aperture-volume fraction / projected-area formula?
     
     AGENT: OK. Use aperture-volume fraction; for the 8m fracture and (0,0,1) both should be equivalent, but
     for other sizes and rotations the volume fraction would possibly give more precise results, with k_homo be
     a upper bound.
  
  8. For method 2, should the validation call the existing macro_conductivity/Subproblems path as-is, even
     though current repo rules allow near-term edits only under apps/chodby_trans, or should we
     create app-local study code that only imports core functions?

     AGENT: I do not usnderstand. You are supposed to call existing code, which doesn't mean you need to
     modify it. In current state it should be ready to use for this purpose. So ask me explicitely about
     changes you need in the code out of chodby_trans, but definitely do not copy that code just if you need
     the modification.
     
     
  9. For comparing tensors, which outputs do you want first: eigenvalues/eigenvectors, full tensor components,
     relative error to direct estimate, plots over the 5x5x5 grid, or all of these?
     
     AGENT: First eigen values; for comparison of two tensors use relative error in eigenvalues and e.g. cos
     norm for the eigen vector diffs.
     
  10. What is the acceptance criterion for “validate”: qualitative agreement, eigenvalue ranges only, correlation
      with fracture-block intersection area, or numeric tolerances?
      AGENT: I will see, for the method 1 cases with reliable equivalent tensor we should be in cloase
      agreement, for other cases there could be much larger error.
      
      
  11. For the CNN surrogate, do you already have the trained model directories for the three ratios, and should
      the study expect paths in a config file under single_fr_study?
  
      AGENT: Yes the terined models are under MLMC-DFM/optuna_runs
  
  12. Here Kf/Km = 1e-3 / 1e-13 = 1e10, while the MLMC-DFM README mentions trained ratios 1e3, 1e5, 1e7.
     Should the surrogate use the nearest available ratio (1e7), should we rescale to fit, or should
     this benchmark use a different matrix conductivity for surrogate comparison?
     
     AGENT: I see, so consider the bulk conductivity 1e-10 in this study (fixed in the spec.)

  13. Should method 4 remain completely out of the first executable plan, or should the plan include a
      stub/checkpoint section describing when to run the 125-block Flow123d reference?
     AGENT:make a stub, but no real implementation for now.

 Additional questions before implementation:

  14. The first setup bullet says macro domain `(0,8)^3`, but later bullets and
      the grid `[0, 4, 8, 12, 16]^3` imply `(0,16)^3`. Should the first bullet
      be corrected to `(0,16)^3`?
      
      AGENT: Yes. macro domain is (0,16)^3 so that there are both blocks without and blocks with whole fracture

  15. Should the method-1 aperture-volume fraction use exact polygon/cube
      clipping, or is high-resolution numerical clipping acceptable for this
      validation study?
      AGENT: Numerical clipping would be ok.

  16. For eigenvector comparison, should nearly repeated eigenvalues be grouped
      into eigenspaces before comparing directions? This matters for isotropic
      tangential eigenpairs.
      
      No. Just provide eigne values and compare them sorted from the largest to smallest, there are only 3 of them.

  17. What file format do you prefer as the primary result artifact:
      CSV tables, NPZ arrays, Xarray/Zarr, or a combination with CSV summaries
      and NPZ for full tensor arrays?
      
      AGENT: Organise the numerical results in to a zarr store, but 
      then produce a summary CSV table.
