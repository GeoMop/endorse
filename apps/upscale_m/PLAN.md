# PLAN: apps/upscale_m — numerical upscaling of the effective elastic tensor

Author: Ronald Siddall (TUL / FSv CVUT internship, EURAD-2), with AI-assisted implementation.
Based on instructions of J. Brezina (2026-07-07) and the theory of R. Siddall's internship report
"Numerical upscaling of mechanical processes in 3D fractured media" (supervisor P. Kabele).

## Goal

Volume homogenisation of the elastic tensor over irregular interior subdomains, mirroring the
conductivity pipeline `macro_conductivity` (src/endorse/macro_flow_model.py), with:

- own micro mesh: outer cube edge `L_ext = 2 L` carrying the BCs, inner concentric cube edge `L = 1`
  as the averaging volume (buffer shell of thickness `L/2` per face; Saint-Venant decay of BC
  artifacts replaces the support-tetrahedra construction needed on an unbuffered SUBC domain),
- discrete fracture network (explicit list first, stochastic sampling later), fractures modeled as
  thin elastic 2D elements (Flow123d mixed-dimension mechanics; aperture delta = 1e-3 * r),
- 6 independent load states per BC type; switch KUBC / SUBC / both (KUBC implemented first),
- effective tensor from volume-averaged stress/strain over the inner cube; exact 6-case inversion
  `C = Sigma @ E^-1` initially, symmetric least-squares fit (21 unknowns, >=36 equations) later,
- `MacroCube(MacroShapeBase)` for the generic homogenisation machinery, analogous to `MacroSphere`.

## Staged plan

1. [x] Skeleton: PLAN.md, config.yaml, module layout.
2. [x] `micro_mesh.py`: buffered cube + explicit fracture list, conforming inner box (OCC fragment),
       regions `rock_inner`, `rock_outer`, `fractures`, `.surface_[xyz]_[minus|plus]`; bgem healing.
3. [x] Mesh verification on a tiny single-fracture case (inspectable in gmsh GUI).
4. [x] Stochastic DFN sampling adapted to current bgem API (bgem@JB_homo: DiscShape removed,
       use EllipseShape; verify Population/Fracture attribute drift). See stochastic_dfn.py.
5. [x] `MacroCube` (developed here, candidate for src/endorse/homogenisation.py after review).
6. [x] KUBC: mechanics Flow123d templates (`<placeholder>` + `_tmpl.yaml` convention),
       `micro_problem` via `common.call_flow`, volume-integration postprocess over the inner cube
       (stress read from VTU cell data; matrix strain via Hooke `eps = M sigma`; fracture elements
       as thin plates weighted by aperture). Unit-tested without the solver.
7. [x] Tensor assembly + formatted report; interface shaped for the later LS swap-in.
8. [x] Validation: `L_ext = L` (no buffer) single fracture r = 0.2 L against R. Siddall's existing
       numerical results and the Oda analytical solution; then enable the buffer and study its effect.
       First pass done at coarse step 0.15 (see log); quantitative pass at reference step 0.05 pending.
9. [ ] Later: SUBC (needs rigid-body decision, see questions), symmetric LS fit, PSD projection.

## Instructions from J. Brezina, 10. 7. 2026 (meeting review) — NEXT WORK PHASE

Verbatim intent, translated and itemized. Overall verdict: physics accepted, template+formula
substitution explicitly approved ("dobre"); the mesher and DFN glue must integrate bgem and
endorse much more deeply instead of hand-rolled gmsh code.

A. Environment / agent workflow:
   1. [ ] Run the AI agent from the endorse REPO ROOT (not from an external directory), so the
          whole endorse + bgem codebase is visible; PLAN.md states that main changes live in
          `apps/upscale_m`.
   2. [ ] Create the environment with `bin/setup_env` -> creates `/venv`; run all tests with
          `venv/bin/python` (Windows: venv/Scripts/python.exe).
   3. [ ] Agent must verify it has FUNCTIONAL bgem and endorse libraries in that venv before work.

B. make_micro_mesh rework (bgem-first):
   4. [ ] REMOVE the inner grid of macro-element cells (geometry.subdomains imprinting added
          2026-07-10) — averaging volumes are endorse-side windows, not imprinted geometry.
   5. [ ] Rebuild the mesher on bgem: outer cube via bgem; fractures via
          FractureSet -> make_fractures -> bgem fragment ... — follow `chodby_trans`
          make_micro_mesh as the reference implementation.
   6. [ ] Replace `stochastic_fracture_set`: change the config schema to EXACTLY what
          `Population.from_cfg` accepts (no custom translation layer); use
          `bgem.fr_mesh.geometry_gmsh` for fracture polygon creation; control disc approximation
          via bgem gmsh options (MinimumCirclePoints).
   7. [ ] Audit the remaining code for duplication with existing endorse functionality and
          replace duplicates with endorse calls (root cause of duplication: the agent previously
          worked without the full endorse repo in its working directory — fixed by A.1).

## Design decisions taken

- Raw gmsh OCC calls (port of R. Siddall's proven mesher) rather than the bgem factory wrapper;
  entity classification via the `occ.fragment` output map instead of geometric heuristics.
  Possible later refactor to the bgem factory if preferred upstream.
- The inner (averaging) cube is embedded conformally by fragmenting the domain with it: every
  bulk and fracture element then lies entirely inside or outside the averaging volume, so
  region/barycenter selection is exact and no partial-element weights are needed for the
  single-cube case. `MacroCube.interact` remains the general mechanism for the macro-mesh setting.
- Boundary-face region names match R. Siddall's existing Flow123d templates
  (`.surface_x_minus`, ...) to make template conversion 1:1.
- Fractures may cross the inner-cube boundary freely (that is the point of the buffer). Parts
  protruding outside the OUTER cube are clipped away (fragment + removal), as in the original code.

## AGENT Questions And Remarks

- USER/JB: rigid-body modes under SUBC with the buffer: pure-traction outer BC is singular; keep the
  support-tetrahedra trick on the outer cube, or is a Flow123d null-space/other mechanism preferred?
- USER/JB: fracture strain contribution in volume averaging: is the thin-elastic-plate treatment
  (strain from Hooke with fracture compliance, weighted by aperture) an accepted approximation of the
  displacement-jump term J, or should the explicit jump integral over fracture surfaces be
  implemented (report sec. 3.4.3 / 5.1)?
- RESOLVED (R. Siddall, 9. 7.): the result pair is measured <sigma> vs measured <eps>, both
  volume-averaged over the inner cube. With fractures intersecting the outer boundary the average
  strain theorem fails (boundary displacement jumps), so prescribed load values are not valid as
  results; they remain boundary conditions only. This also fixes the LS input pair (measured
  both, consistent with the `get_load_data` philosophy).
- USER/JB: should fractures populate the whole buffered domain (statistically homogeneous buffer,
  current default) or only the inner cube?
- USER/JB (partly resolved 9. 7.): averaging weights. The conforming inner cube makes the
  sharp-indicator average exact (w_e = V_e / V_inner, sum = 1). For the non-conforming macro-mesh
  setting, MacroCube.interact now returns the VOLUME FRACTION of the element inside the window
  (via refine_barycenters, 8^level equal-volume sub-tets; decision R. Siddall) instead of the 0/1
  any-node indicator that MacroSphere uses. Remaining question: is a genuinely smooth kernel
  (weight tapering, cf. phi_V of the report sec. 2.4.2) wanted on top, and should MacroSphere get
  the same volume-fraction upgrade upstream?
- JB: `tests/homogenisation` fine_mesh proof-of-concept seems to ignore its mesh-size field with
  current bgem@JB_homo (produces ~8M elements regardless of requested steps) — bitrot?
- JB: `MacroSphere.rel_radius` is stored but never used (`_center_radius` returns the plain mean
  vertex distance). `MacroCube.rel_size` is applied as documented; intentional difference?
- JB: `homogenisation.py::homogenize_batch` appears to mix Voigt off-diagonal orderings between the
  LS assembly ([YZ, XZ, XY]) and `voigt_to_tensor` ([XY, XZ, YZ]) used for the PSD projection;
  harmless when no eigenvalue is clamped, distorting otherwise. Also the per-subdomain loop result
  in `equivalent_tensor_field` is immediately overwritten by `homogenize_batch` (dead work).
- JB: `common.workdir.copy` builds the destination from `src.stem`, silently dropping the file
  extension of copied inputs (e.g. `mesh.msh2` arrives as `mesh`). Bug? upscale_m copies inputs
  itself as a workaround.
- JB: `common.File` stores absolute host paths; on Windows these are invalid inside the Flow123d
  container. Solved locally by `flow_win_wrapper.py` (translates `C:\...` args and CWD to `/C/...`
  and mirrors the mounts of the original .bat). A portable alternative would be for call_flow to
  pass the main input path relative to the work dir.
- Windows note: development machine runs Flow123d 4.0.3dev via Docker Desktop
  (`machine_config` hostname `LAPTOP-MM6A3LRF`, executable = venv python + flow_win_wrapper.py);
  `mlmc` package deliberately not installed; `attrs` is required by src/endorse but missing from
  setup.py install_requires.

## AGENT log

- 2026-07-08: venv created (Python 3.11), endorse editable install, bgem@JB_homo, machine_config
  entry for LAPTOP-MM6A3LRF added to tests/homogenisation/input/config.yaml. Skeleton created.
- 2026-07-08: micro_mesh.py implemented and verified. Demo (r=0.3 fracture, factor 2, step 0.25):
  673 tets rock_inner / 3587 rock_outer / 14 fracture triangles / 6 boundary face groups; healed
  msh2 round-trips through GmshIO. Conformity check with a boundary-crossing fracture (r=0.7):
  52 fracture elements inside / 29 outside the averaging cube, 0 straddling.
- 2026-07-08: macro_cube.py implemented (max-norm variant of MacroSphere, weight 1/0), unit tests
  in test/test_macro_cube.py all pass, incl. the corner probe distinguishing cube from sphere.
- 2026-07-08: KUBC stage built: single parametrized template mechanics_kubc_tmpl.yaml (replaces six
  per-case files; formula u = E_q x generated by kubc.formula_from_matrix, extensible to arbitrary
  loads for the LS fit), kubc.py runner (call_flow per case, mesh copied to case workdir),
  postprocess.py (port of GeneralComputationClass restricted to rock_inner + in-cube fracture
  elements, V_ref = L^3), upscale_tensor.py (exact inversion for both prescribed and measured E,
  report file), run_upscale.py driver. Unit tests (test_kubc_units.py) pass: formulas, template
  rendering, synthetic-mesh averaging against hand-computed values (buffer-region contamination
  excluded), Hooke round-trip, effective-tensor recovery of the analytic isotropic stiffness.
  Not yet executed against Flow123d (requires Docker; validation = stage 8).
- 2026-07-08: first full Flow123d execution on Windows. Two integration fixes: explicit input copy
  (workdir.copy drops extensions) and flow_win_wrapper.py (host->container path translation; the
  original .bat could not receive absolute Windows paths from common.File). Full KUBC pipeline then
  ran end-to-end (~2 s/solve at step 0.15). Validation vs Oda (xi = 0.008): correct transverse
  isotropy about z (C11 = C22, C13 = C23, C44 = C55), C66* = 1.000 exactly as theory demands,
  symmetric tensor, prescribed-E and measured-E variants agree to ~5 digits. Numerical values sit
  2.8-4 % ABOVE Oda, consistent with the report's own finding that both HRV bounds lie above the
  non-interacting Oda estimate, amplified here by the coarse step 0.15 (reference step is 0.05).
  Results: sandbox_validation/{effective_tensor_kubc.txt, oda_comparison.txt, *.npy}.
- 2026-07-08: convergence + buffer isolation. Three runs, single XY fracture r=0.2, xi=0.008:
    C33*: Oda 0.9542 | step 0.15 unbuffered 0.98105 | step 0.15 BUFFERED 0.98109 | step 0.05 0.96517
    C44*: Oda 0.9563 | 0.99472 | 0.99461 | 0.98762
  (a) Refinement 0.15 -> 0.05 moves the tensor markedly toward Oda (C33* gap 2.8% -> 1.15%);
  (b) buffer effect at equal step is ~3e-5, i.e. NEGLIGIBLE for KUBC with an interior fracture —
  the buffer's value is expected for SUBC, boundary-crossing fractures (DFN) and stronger
  heterogeneity, which is its design regime. (c) R. Siddall's stored one_fracture_volume
  kinematic result (step 0.05) shows C33 = 2.6e9 > C_m33 = 1.2e9 and strong asymmetry — unphysical
  (fractures cannot stiffen; report's own theorem), most likely from the era of the UNFINISHED
  volume-integration fracture term (report sec. 5.1) — not a usable reference. The new pipeline's
  volume integration yields C* <= 1 everywhere, i.e. it resolves that report item. USER: confirm
  which stored dataset (if any) is a trusted numerical reference beyond Oda.
- 2026-07-08: stochastic DFN ported (stochastic_dfn.py + config_dfn.yaml). Semantics note: current
  bgem EllipseShape is a disc of UNIT AREA, so bgem's Fracture.r = sqrt(area) = rho * sqrt(pi);
  upscale_m config and code use GEOMETRIC radii rho throughout and convert at the bgem boundary
  (this also disambiguates the diameter/radius muddle of the original code). Population built via
  Population.from_cfg (Fisher + power law + p_32), center placement with boundary pads ported
  verbatim incl. the horizontal fixed-normal z-pad special case. Sampling unit tests pass (seed
  reproducibility, radii in range, unit normals, fixed-normal override, pads, intensity scaling);
  end-to-end DFN mesh smoke: 10 fractures, buffered domain, all regions, healed, ~10k elements.
- 2026-07-08: layout restructured on R. Siddall's request (source/data separation as in his
  original project): configs/ holds study definitions, runs/ (git-ignored via app .gitignore)
  holds ALL run products; run_upscale.py resolves bare config names in configs/ and places
  RUN_NAME under runs/ automatically. Tests and demos write to runs/ as well.
- 2026-07-08: run-dir structure aligned with the original result layout: mesh/geometry at the run
  root; per-load-case dirs named pure_normal_E_11 ... pure_shear_E_12, each with its own rendered
  input (pure_normal_E_11.yaml, via a local per-case template copy), solver stdout/stderr, and the
  Flow123d data in output/ (name kept: FlowOutput expects it and the case dir already carries the
  case identity). Verified: tensor values identical before/after the restructure.
- 2026-07-08: final layout polish per R. Siddall: core logic moved to src/ (7 modules), app root
  keeps only the entry point run_upscale.py, flow_win_wrapper.py, docs, configs/, templates,
  test/, runs/ (emptied — all prior run data and caches deleted for a clean-slate test case).
  Redundant derived configs removed (validation_fine, buffered_015 — recreate by editing step).
  All unit tests pass after the restructure.
- 2026-07-09: MAPPING.md rewritten as the verification plan requested by J. Brezina (mail 9. 7.):
  assignment -> implementation table, ORIGINAL -> upscale_m pipeline map with rationale per row,
  conventions pinned by named unit tests, an explicit "deliberate deviations" list (buffer, no
  support tets, single template, config aperture, log-based convergence check, dual C reports,
  unchanged J-term), and a ~30-min coarse review checklist ending at the Oda validation numbers.
- 2026-07-09: Oda solution removed from the numerical pipeline on R. Siddall's request:
  src/oda_reference.py deleted, driver no longer prints/writes any Oda comparison. The Oda closed
  forms remain a report-side analytical reference only (REPORT eqs. 2.131-2.134); historical log
  entries above that quote Oda numbers are kept as a record of the validation already performed.
- 2026-07-09: CODE_WALKTHROUGH.txt added — plain-language, line-referenced explanation of every
  class/function in the app (for R. Siddall's code review; complements MAPPING.md).
- 2026-07-09: .npy matrix outputs removed on R. Siddall's request; the formatted tensor report is
  the single result artifact of a run.
- 2026-07-09: results restricted to MEASURED quantities (R. Siddall): with boundary-crossing
  fractures the average strain theorem fails, so the prescribed-E tensor variant was removed.
  assemble_matrices returns (E, Sigma) measured; write_report now mirrors the ORIGINAL report
  layout exactly (sections Sigma / E / C_k (with dash) / S_k, same prefixes and rulers); report
  renamed effective_tensor_kubc.txt -> effective_tensor_C_k_kinematic_bc.txt. This also resolves
  the LS input-pair question (measured both). LoadCaseResult keeps E_prescribed_voigt as an input
  record (not reported).
- 2026-07-09: test/ directory removed by R. Siddall (deliberate: "core of the program only");
  MAPPING.md convention section now points at code anchors instead of the deleted unit tests.
- 2026-07-09: MacroCube.interact upgraded from the 0/1 any-node indicator to a VOLUME-FRACTION
  weight (R. Siddall's request, motivated by non-conforming macro-mesh use): exact 1/0 shortcuts
  for fully-inside/disjoint elements, otherwise fraction of 8^level equal-volume sub-tet
  barycenters inside the window (endorse refine_barycenters; level attr, default 2). Verified
  against Monte-Carlo volume fractions (cut-by-face, cut-by-corner, conforming face-touch cases;
  agreement within the 2^-level bound). Conforming meshes still get exactly 1.0/0.0.
- 2026-07-10: config_dfn.yaml fixed (R. Siddall reported an apparently empty DFN run): p_32 was
  defined over population_radius_range [0.02, 100] while sampling only [0.15, 0.3] — the power law
  concentrates intensity in small radii, so the sample window caught ~1 fracture. Population range
  now equals the sample range (p_32 0.3 -> 15 fractures at seed 1234); mesh step relaxed 0.1 ->
  0.2 (15-fracture buffered mesh ~14k elements, laptop-sized). Mesh stage verified end-to-end.
- 2026-07-10: subdomain grid added on R. Siddall's request (mirrors macro_conductivity's
  per-subdomain tensors, tailored to the conforming setting): geometry.subdomains [nx, ny, nz]
  splits the averaging cube into cells, each IMPRINTED into the mesh (still conforming -> exact
  per-cell averages, no interact weights needed), all cells read from the SAME six solutions,
  one report per cell (subdomain_ix_iy_iz/effective_tensor_C_k_kinematic_bc.txt). Implementation:
  micro_mesh cell boxes as fragment tools + MicroMesh.subdomain_boxes/subdivision;
  postprocess.average_boxes (VTU read once, per-box barycenter selection, V_ref = cell volume);
  LoadCaseResult carries (n_sub, 6) arrays; assemble_matrices(results, i_sub); driver loops
  subdomains. Verified: synthetic per-box averaging exact to 1e-12; [2,2,2] mesh yields 8 cells
  of volume 0.125 exactly with 0 straddling elements; full Docker run (unbuffered, central
  z-fracture, step 0.15) gives 8 mirror-symmetric cells agreeing to <4e-4 relative with C33
  softened and C66 intact; [1,1,1] regression reproduces the previous validation exactly
  (C33 = 1.177e9 at step 0.15). Default [1,1,1] = previous behaviour, old configs unaffected.
- 2026-07-10: per-cell selection made half-open ([lo, hi) on interior interfaces, inclusive on
  the outer rim): a fracture element lying EXACTLY on a cell interface (R. Siddall's canonical
  centered-fracture + even-grid case) was previously counted in BOTH adjacent cells. Verified:
  interface fracture counted exactly once; rerun of the [2,2,2] central-fracture case now gives
  the four fracture-free cells the intact isotropic tensor EXACTLY (C11 = C33 = 1.2000e9,
  C66 = 4.0000e8) and softens only the four fracture-carrying cells — numerically confirming
  that homogenization identifies the material, not the loading (Hooke-reconstructed strain makes
  fracture-free cells return C_intact identically, regardless of the perturbed stress field).
- 2026-07-10 (evening, post-meeting): J. Brezina's review feedback recorded as the new work-phase
  section above. Summary: physics and the template/formula design accepted; mesher + DFN glue to
  be rebuilt bgem-first (chodby_trans make_micro_mesh as reference, Population.from_cfg-native
  config, bgem.fr_mesh.geometry_gmsh, MinimumCirclePoints via bgem options); the 2026-07-10
  subdomain-cell imprinting to be REMOVED (averaging via endorse windows instead — the per-cell
  physics results above remain valid as validation data); agent to run from the endorse repo root
  with the bin/setup_env venv and to verify functional bgem/endorse imports first. R. Siddall's
  independent numerical result kept on record: fracture-free subdomains return exactly C_intact.
