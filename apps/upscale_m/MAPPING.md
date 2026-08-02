# MAPPING — how the original code maps onto the endorse structure

Purpose (per J. Brezina, 7. 7. 2026): AI-produced code must be checkable, so this document is the
verification plan. It traces every piece of `apps/upscale_m` back to (a) the supervisor's assignment and
(b) R. Siddall's original, validated code (`Python_scripts/src_internship_cvut`), states what changed and
why, and gives a concrete way to check each claim. Read section 6 for the 30-minute coarse review.

Sources referenced below:
- ORIGINAL = `Python_scripts/src_internship_cvut` (R. Siddall's internship code, validated in the report)
- REPORT = R. Siddall, "Numerical upscaling of mechanical processes in 3D fractured media", FSv CVUT 2026
- ENDORSE = `src/endorse` on branch `ronald_upscale_mech` (fork of GeoMop/endorse @ PE_trans_mlmc)

## 1. Assignment -> implementation (the coarse map)

Each row is one item of the supervisor's instruction (7. 7.) and where it landed.

| Assignment item                                        | Implementation                                  | Status |
|--------------------------------------------------------|--------------------------------------------------|--------|
| New app `apps/upscale_m` analogous to `macro_conductivity` (`src/endorse/macro_flow_model.py`) | `run_upscale.py` + `src/` — same stage structure, see section 2 | done for ONE micro problem (single REV); the macro-mesh loop over `Subproblems` is the next phase |
| Replace `make_micro_mesh` with own mesher with buffer, L_ext = 2, L = 1 | `src/micro_mesh.make_micro_mesh` (function name kept on purpose); `L_ext_factor: 2.0` in config = buffer shell L/2 per face | done |
| Micro problem split: input prep / `call_flow` / postprocess (mirroring `subproblem_input` / `call_flow` / `micro_postprocess` in `macro_flow_model.py`) | `kubc.run_kubc`: params dict + template (= subproblem_input), `endorse.common.call_flow` (verbatim reuse), `postprocess.average_inner_cube` (= micro_postprocess) | done |
| KUBC / SUBC / both switch                               | `loads.bc_type` in config; driver asserts `kubc` | KUBC done; SUBC blocked on rigid-body question (PLAN.md) |
| `MacroCube(MacroShapeBase)` analogous to `MacroSphere` (`sum(nc*nc) < r*r` -> cube test) | `src/macro_cube.py`: max-norm window; `interact` returns the VOLUME FRACTION inside (endorse `refine_barycenters`), not the 0/1 indicator | done; not yet exercised inside `Subproblems` |
| Least-squares symmetric equivalent tensor               | reserved slot `upscale_tensor.equivalent_tensor(method='ls_symmetric')` raises NotImplementedError | deferred by agreement; input pair decided: measured (sigma, eps) — see section 5 item 6 |

## 2. Pipeline map: ORIGINAL -> upscale_m

Stages in execution order; the driver `run_upscale.py` replaces ORIGINAL `main.py` + `run_all.sh`.
The endorse analogy: `run_upscale.py` plays the role `macro_conductivity` plays for flow, restricted to
a single micro problem (no macro mesh yet).

| ORIGINAL (what R. Siddall wrote and validated)          | upscale_m                                       | What changed and why |
|---------------------------------------------------------|--------------------------------------------------|----------------------|
| `master_config_template.yaml` + `Logic_classes/ConfigManager.py` (one getter per key, ~440 lines) | `configs/*.yaml` + `endorse.common.load_config` | Config loads into a `dotdict`: `cfg.materials.young_modulus_rock` instead of a getter method. No path-getter side effects; directories are created by `common.workdir`. `machine_config` (hostname-resolved solver path) replaces `dir_of_flow123d_executable_bat_file`. |
| `dfn_cvut_static.GenerateFractures3D_Universal.generator_of_fracture_set` (bgem Population sampling + pad placement) | `src/stochastic_dfn.stochastic_fracture_set` | Ported to bgem@JB_homo API: `Population.from_cfg` instead of `add_family`; `DiscShape` is gone, `EllipseShape` has UNIT AREA so bgem sizes are `rho*sqrt(pi)` — config and outputs stay in GEOMETRIC radii, conversion only at the bgem boundary. Pad placement logic (pad_xy = rho + boundary_distance, relaxed pad_z for fixed horizontal normal) ported verbatim. |
| `surface.py` / `GeneralComputationClassSurfaceArch.py` (cube + disks, OCC fragment, classify, mesh, heal) | `src/micro_mesh.make_micro_mesh` | Same gmsh OCC calls and all mesh options (Algorithm 6, MeshSizeMin = step/10, msh v2.2). New: (1) outer buffer cube `L_ext = factor * L`, inner averaging cube embedded CONFORMINGLY by the same `occ.fragment` call; (2) entities classified via the fragment input->output map instead of bounding-box heuristics (axis-aligned fractures cannot be confused with the inner-cube interface); (3) region `rock` -> `rock_inner` (tag 1) + `rock_outer` (tag 2), fractures tag 3, boundary faces keep the ORIGINAL names `.surface_x_minus` etc.; (4) NO support tetrahedra — KUBC needs none, SUBC decision pending. Healing identical (`bgem.gmsh.heal_mesh`). |
| `Templates_dont_delete/Templates_kinematic_boundary_conditions/` (6 yaml files differing in one formula line) | `flow123d_templates/mechanics_kubc_tmpl.yaml` (1 file) | endorse substitutes named `<placeholders>`, so six files collapse into one; the BC formula `u = E_q x` is generated by `kubc.formula_from_matrix(E)` for ANY symmetric load matrix (ready for extra LS load states). Solver structure inside the template is copied from the ORIGINAL kinematic template. |
| `GenerateKinematicYaml.generate_kinematic_yamls` (string replacement lists) | params dict in `kubc.run_kubc` + `endorse.common.substitute_placeholders` | Same idea, framework version: named placeholders, fails visibly on missing values. Per-case rendered inputs keep the ORIGINAL names (`pure_normal_E_11.yaml`, ...) in per-case directories (`pure_normal_E_11/`, ...). |
| `GenerateKinematicVtuFiles.run_kinematic_simulations` (subprocess on the .bat, check 6 VTUs exist) | `kubc.run_kubc` loop + `endorse.common.call_flow` | `call_flow` renders the template, runs the solver (Docker via `flow_win_wrapper.py` on Windows, resolved from `machine_config` by hostname), captures stdout/stderr, and PARSES THE LOG for the convergence reason — stronger check than "VTU exists". `@memoize`/CallCache skips already-finished simulations on rerun. |
| `Logic_classes/GeneralComputationClass.py` (pyvista VTU read, analytic Hooke strain, volume averaging) | `src/postprocess.py` | Function-per-function port: `build_compliance_matrix_voigt`, `strain_from_stress`, `parse_stress_field` keep the ORIGINAL math (M[3][3] = 2(1+nu)/E convention included). The averaging became `average_inner_cube`: bulk elements by region tag 1 (`rock_inner`), fracture elements by region tag 3 AND barycenter inside the inner box (exact thanks to the conforming mesh), fracture volume = area * aperture, reference volume = L^3. Aperture comes from config (`aperture_per_r * r`) instead of the hardcoded 2e-4. |
| `GenerateKinematicEffectiveElasticTensor` (`C_k = Sigma E^-1`, formatted txt report) | `src/upscale_tensor.py` | Same exact inversion, from the MEASURED inner-cube-averaged `E` and `Sigma` only (R. Siddall, 9. 7.: with boundary-crossing fractures the average strain theorem fails, so prescribed values are not usable as results). Report sections and file name (`effective_tensor_C_k_kinematic_bc.txt`) mirror the ORIGINAL txt files. |
| `main.py` + `run_all.sh` (bash + sed placeholder sweep)  | `run_upscale.py`                                | One driver: mesh -> 6 runs -> tensors -> report(s), everything inside a disposable `runs/<name>/` dir. `geometry.subdomains [nx,ny,nz]` yields one tensor per grid cell from the SAME six solves (macro_conductivity's per-subdomain loop, exact here because cells are imprinted/conforming); default [1,1,1] = single tensor. Sweeps will later use endorse's config-variant mechanism instead of sed. |
| `NumericalOda.py` / Oda formulas in the report           | (removed on R. Siddall's request, 9. 7.)        | The Oda analytical solution stays a REPORT-side reference (eqs. 2.131-2.134); it is not part of the numerical pipeline or its outputs. |
| (no ORIGINAL equivalent)                                 | `src/macro_cube.py`                             | New per assignment: cubic averaging window for endorse's general macro-mesh machinery (max-norm variant of `MacroSphere`). Upgraded past MacroSphere's 0/1 any-node indicator: `interact` returns the volume fraction of the element inside the window (equal-volume subdivision via endorse `refine_barycenters`), correct on non-conforming meshes; on conforming ones it returns exactly 1/0. Not used by the single-cube pipeline — the conforming inner cube makes weights degenerate to exact V_e/V_inner there. |

## 3. endorse framework pieces reused (all `src/endorse/common/`)

- `load_config` / `dotdict` — YAML -> attribute-style config; `machine_config` resolved by
  `socket.gethostname()` (same config file runs on the laptop and on the cluster).
- `workdir(dir, inputs=[...])` — context manager: create dir, copy inputs, chdir. NOTE: its `copy`
  drops file extensions (`src.stem` bug, PLAN.md), so the mesh is copied explicitly in `kubc.py`.
- `substitute_placeholders(template, out, params)` — the `<name>` template engine, `_tmpl.yaml` convention.
- `call_flow(machine_cfg, template, params)` — render, run Flow123d, capture output, check convergence.
- `@memoize` / `CallCache` — disk cache; finished simulations are never recomputed.

## 4. Conventions — identical to the REPORT throughout

Each convention is fixed in exactly ONE code location (the review anchor to check):

- Voigt order sigma = [11, 22, 33, 23, 13, 12]; strain shear rows carry ENGINEERING shear gamma = 2 eps;
  prescribed KUBC strain matrix stacks to diag(b, b, b, 2b, 2b, 2b) (REPORT eq. 2.66).
  Anchors: `kubc.py` `VOIGT_IJ` + `prescribed_strain_voigt`; `postprocess.py` `strain_to_voigt`.
- Averaging math: weights V_e / L^3, fracture volume = area * aperture, buffer region excluded.
  Anchor: `postprocess.py::average_inner_cube`.
- Radius semantics: GEOMETRIC radii everywhere; the bgem conversion rho*sqrt(pi) is confined to
  `stochastic_dfn.py` (`_to_bgem_size` / `_to_geometric_radius`).

(The unit tests that pinned these were removed together with the test/ directory on R. Siddall's
request, 9. 7. 2026 — "core of the program only"; the anchors above are the review points now.)

## 5. Deliberate deviations — the places where a human should look twice

These are the points where the port does NOT reproduce the original 1:1, each a conscious decision:

1. **Buffer cube (new physics-relevant geometry).** ORIGINAL applied BCs and averaged on the SAME cube;
   upscale_m applies BCs on `[0, L_ext]^3` and averages only over the inner `L^3`. `L_ext_factor: 1.0`
   reproduces the original unbuffered setup exactly (used for validation).
2. **No support tetrahedra.** ORIGINAL static (SUBC) meshes carried statically-determinate supports;
   KUBC does not need them, so the mesher does not build them. Must be revisited for SUBC.
3. **Single template instead of six.** Rendered per-case inputs are byte-comparable to what the six
   ORIGINAL templates would produce (up to the placeholder values) — compare any rendered
   `pure_*/pure_*.yaml` of a run with the ORIGINAL counterpart to verify.
4. **Aperture from config**, `delta = aperture_per_r * mean(r)`; ORIGINAL hardcoded 2e-4 (= 1e-3 * 0.2).
   For multiple distinct radii a mean-radius aperture is a temporary simplification (warning printed;
   per-fracture regions pending, PLAN.md).
5. **Convergence checked from the solver log** (`FlowOutput.check_conv_reasons`), not from VTU presence.
6. **Only MEASURED (sigma, eps) enter the results, exactly as in the ORIGINAL** (reaffirmed by
   R. Siddall, 9. 7.): with fractures intersecting the outer boundary the average strain theorem fails,
   so prescribed load values are not valid as results — they are boundary conditions only. (An interim
   version of upscale_m reported a prescribed-E variant alongside; removed. On interior-fracture cases
   the two coincide to ~5 digits, so the interim validation numbers remain valid.)
7. **J-term (displacement-jump contribution to <eps>) unchanged**: like the ORIGINAL, fractures enter the
   strain average via thin-plate Hooke on 2D elements (REPORT sec. 5.1 open item; question to J. Brezina).

## 6. Coarse review checklist (~30 min, no code archaeology needed)

1. `configs/config.yaml` — is the study definition (geometry, materials, loads) what you expect? All
   physics numbers live here and nowhere else.
2. `src/micro_mesh.py` module docstring + `flow123d_templates/mechanics_kubc_tmpl.yaml` — regions and BC
   structure; compare the template with any ORIGINAL kinematic template side by side.
3. `src/kubc.py` `VOIGT_IJ` / `kubc_load_matrices` / `formula_from_matrix` — the six load states
   (REPORT eq. 2.62/2.66).
4. `src/postprocess.py::average_inner_cube` — the averaging definition (one screen of numpy).
5. Run the pipeline on the single-central-fracture study and read the report
   (`effective_tensor_C_k_kinematic_bc.txt`):
   `cd apps/upscale_m && ../../venv/Scripts/python.exe run_upscale.py check config.yaml`.
   Expected: transverse isotropy exact (C11 = C22, C13 = C23, C44 = C55), C66 equal to the matrix
   shear stiffness to ~1e-5, every diagonal entry <= its intact-matrix counterpart, values
   decreasing under mesh refinement (REPORT sec. 4).
6. Open questions and known endorse quirks are consolidated in PLAN.md, section "AGENT Questions And
   Remarks".
