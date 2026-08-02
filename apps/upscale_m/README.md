# upscale_m — mechanical upscaling

Numerical homogenisation of the effective elastic tensor of a fractured rock cube (REV) with a
Saint-Venant buffer: boundary conditions act on an outer cube `L_ext = 2 L`, averaging runs over the
inner concentric cube `L`. Six independent KUBC/SUBC load states are solved by Flow123d
(mixed-dimension mechanics, fractures as thin 2D elastic elements) and volume-averaged into the
macroscopic 6x6 stress/strain matrices, from which the effective tensor follows.

See `PLAN.md` for the staged plan, design decisions, and open questions.

Theory reference: R. Siddall, "Numerical upscaling of mechanical processes in 3D fractured media",
internship report, FSv CVUT 2026. Voigt convention: sigma = [11, 22, 33, 23, 13, 12], engineering
shear (gamma = 2 eps) on the strain shear rows.

## Layout (source and data strictly separated)

- `src/` — the mechanics-upscaling logic: `micro_mesh.py` + `stochastic_dfn.py` (geometry/DFN),
  `kubc.py` (load cases + solver calls), `postprocess.py` (inner-cube averaging),
  `upscale_tensor.py` (tensor assembly + report), `macro_cube.py` (general-framework window)
- `run_upscale.py` — the entry point / driver (root level, run this)
- `flow_win_wrapper.py` — Windows/Docker plumbing (referenced by machine_config, not physics)
- `flow123d_templates/` — mechanics input templates, `<placeholder>` + `_tmpl.yaml` convention
- `configs/` — study definitions (geometry, mesh, materials, loads, machine_config); one file = one study
- `runs/` — ALL run products (meshes, Flow123d outputs, tensor reports); git-ignored, disposable

## Usage

    cd apps/upscale_m
    ../../venv/Scripts/python.exe run_upscale.py RUN_NAME CONFIG.yaml   # e.g. myrun config_dfn.yaml

CONFIG is looked up in `configs/`; outputs land in `runs/RUN_NAME/`.

`geometry.subdomains: [nx, ny, nz]` splits the averaging cube into a grid and writes one tensor
report per cell (`subdomain_ix_iy_iz/...`), all computed from the same six solutions — the
conforming, exact analogue of `macro_conductivity`'s per-subdomain tensors. `[1, 1, 1]` (default)
gives the classic single report.

## Environment

Run with the repo venv (`endorse/venv`); requires a working Flow123d, resolved per hostname via
`machine_config` in `config.yaml` (Windows dev machine uses a Docker Desktop wrapper .bat).
