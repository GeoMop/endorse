"""
KUBC (kinematic uniform boundary conditions) load cases and the micro-problem runner.

STRUCTURE mirrors the endorse conductivity pipeline (src/endorse/macro_flow_model.py):
    run_kubc            ≈ macro_conductivity's load loop (gen_load_responses / micro_load_response)
    micro_problem       ≈ macro_flow_model.micro_problem — per-case workdir + call_flow +
                          micro_postprocess; endorse's `subproblem_input` step is intentionally
                          ABSENT: bulk and fracture materials are given by mesh REGIONS
                          substituted into the template, no per-element field file is needed.
    micro_postprocess   ≈ macro_flow_model.micro_postprocess — load the solver output mesh
                          (endorse mesh_class) and average the measured (load, response) pair,
                          here (eps, sigma), over the averaging windows.

Six independent load states q = 1..6 prescribe u = E_q x on the whole outer boundary
(report sec. 2.5.2 / 3.2.1). Voigt ordering (report convention): [11, 22, 33, 23, 13, 12],
engineering shear (gamma = 2 eps) on the strain shear rows.

Each case runs Flow123d via endorse.common.call_flow with a single parametrized template
(flow123d_templates/, filename from config key loads.input_template); the displacement formula
string is generated from E_q, so additional (e.g. mixed) load states for a least-squares
identification need no new template.
"""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from bgem.upscale import tn_to_voigt, voigt_to_tn
from bgem.upscale.fem import Grid
from endorse.common import dotdict, workdir, call_flow, report, FlowOutput
from endorse.homogenisation import Subproblems
from endorse.mesh_class import load_mesh

from macro_cube import MacroCube
from micro_mesh import MicroMesh
from postprocess import ENG_SHEAR, average_windows, windows_macro_mesh

_src_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.dirname(_src_dir)


@dataclass
class LoadCaseResult:
    name: str
    E_prescribed_voigt: np.ndarray   # prescribed strain (Voigt, engineering shear) — input record
    eps_voigt: np.ndarray            # measured strain per window, shape (n_windows, 6)
    sigma_voigt: np.ndarray          # measured stress per window, shape (n_windows, 6)
    spatial_file: str                # Flow123d output fields file the averages were read from


@report
def micro_postprocess(mats: dotdict, micro: MicroMesh,
                      subproblems: Subproblems, grid: Grid, level: int,
                      aperture_by_region_id: Dict[int, float], micro_model: FlowOutput):
    """
    Mirror of endorse macro_flow_model.micro_postprocess for the mechanics pair: load the
    solver's mechanics output (endorse mesh_class; stress, region_id — the FLOW/hydro sub-
    equation output is never read: its only job is carrying the cross_section FieldFE
    declaration required by Flow123d's schema, see PLAN.md), write a ParaView-viewable .vtu
    preview next to the GMSH output (endorse Mesh.write_fields_vtu — the same pyvista-backed
    method endorse fields_file already uses for the cross_section field; a visualization
    convenience only, the computation itself stays GMSH-only), and return the per-window
    averaged (eps, sigma) 3x3 pairs (endorse Subproblems machinery).
    """
    print("loading mesh:", micro_model.mechanic.spatial_file)
    output_mesh = load_mesh(micro_model.mechanic.spatial_file)
    output_mesh.write_fields_vtu("output/mechanics_fields.vtu", dict(
        stress=output_mesh.get_p0_values("stress", 0.0),
        region_id=output_mesh.get_p0_values("region_id", 0.0),
    ))
    return average_windows(
        output_mesh, aperture_by_region_id, subproblems, grid,
        young_rock=float(mats.young_modulus_rock),
        poisson_rock=float(mats.poisson_ratio_rock),
        young_fracture=float(mats.young_modulus_fracture),
        poisson_fracture=float(mats.poisson_ratio_fracture),
        bulk_region_id=micro.bulk_region_id,
        fracture_region_ids=micro.fracture_region_ids,
        level=level,
    )


def micro_problem(cfg: dotdict, tag: str, load_matrix: np.ndarray,
                  micro: MicroMesh, subproblems: Subproblems, grid: Grid, level: int,
                  aperture_by_region_id: Dict[int, float]):
    """
    Mirror of endorse macro_flow_model.micro_problem: run one load case in its own work dir
    (template substitution + call_flow) and post-process. No `subproblem_input` — materials
    are region-wise template parameters and the per-fracture aperture travels as a SEPARATE
    cross_section field file (endorse fields_file + FieldFE), distinct from the mesh_file.
    Returns (per_window averages, output fields file path).
    """
    mats = cfg.materials
    mesh_abs_path = os.path.abspath(micro.mesh_file.path)
    cross_section_abs_path = os.path.abspath(micro.cross_section_file.path)
    boundary_regions = list(cfg.geometry.boundary_regions)
    missing = [r for r in boundary_regions if r not in micro.regions]
    assert not missing, f"geometry.boundary_regions not found in the mesh: {missing}"
    template_path = os.path.join(_app_dir, "flow123d_templates", cfg.loads.input_template)
    with workdir(tag, inputs=[]):
        # NOTE: mesh copied explicitly, not via workdir(inputs=...) — workdir.copy uses src.stem
        # and silently drops the file extension (recorded in PLAN.md questions).
        shutil.copy2(mesh_abs_path, os.path.basename(mesh_abs_path))
        shutil.copy2(cross_section_abs_path, os.path.basename(cross_section_abs_path))
        # local per-case template copy so the rendered input carries the case name
        # (pure_normal_E_11.yaml, ...), mirroring the original per-case yaml files
        local_template = f"{tag}_tmpl.yaml"
        shutil.copy2(template_path, local_template)
        params = dict(
            description=f"upscale_m KUBC load {tag}",
            mesh_file=os.path.basename(mesh_abs_path),
            cross_section_file=os.path.basename(cross_section_abs_path),
            bulk_regions="box",
            fracture_regions=f"[{', '.join(micro.fracture_regions)}]",
            boundary_regions=f"[{', '.join(boundary_regions)}]",
            # endorse micro-template idiom: "<matrix> @ [x,y,z]" in the template
            bc_displacement_matrix=str(load_matrix.tolist()),
            rock_young_modulus=f"{float(mats.young_modulus_rock):g}",
            rock_poisson_ratio=f"{float(mats.poisson_ratio_rock):g}",
            fracture_young_modulus=f"{float(mats.young_modulus_fracture):g}",
            fracture_poisson_ratio=f"{float(mats.poisson_ratio_fracture):g}",
        )
        micro_output = call_flow(cfg.machine_config, Path(local_template), params)
        if not micro_output.check_conv_reasons():
            raise ValueError(f"Load case {tag}: Flow123d simulation failed.")
        per_box = micro_postprocess(mats, micro, subproblems, grid, level,
                                    aperture_by_region_id, micro_output)
        return per_box, os.path.abspath(micro_output.mechanic.spatial_file.path)


def run_kubc(cfg: dotdict, micro: MicroMesh) -> List[LoadCaseResult]:
    """
    Run the six KUBC load states on the given micro mesh (in the CURRENT work dir; one
    subdirectory per case) — the load loop of endorse macro_conductivity (gen_load_responses),
    returning per-case (load, response) = (eps, sigma) window averages.
    """
    beta = float(cfg.loads.deformation_parameter_beta)
    case_names = list(cfg.loads.case_names)
    case_dir_names = list(cfg.loads.case_dir_names)
    assert len(case_names) == 6 and len(case_dir_names) == 6, \
        "loads.case_names/case_dir_names must list exactly the 6 KUBC cases in Voigt order"
    # sub-element refinement for elements cut by a window boundary — one knob shared by the
    # bulk (MacroCube.interact) and fracture (postprocess._window_weights) estimates
    level = int(cfg.geometry.get("window_refine_level", 2))

    # averaging windows: bgem Grid over the inner cube (NOT imprinted in the mesh), presented
    # to the endorse Subproblems machinery as the synthetic windows macro mesh — the exact
    # structure of macro_flow_model.macro_conductivity (incl. its subdivision = [1, 1, 1])
    lo, hi = micro.inner_box
    grid = Grid(np.full(3, hi - lo), np.asarray(cfg.geometry.get("subdomains", [1, 1, 1])),
                origin=np.full(3, lo))
    subproblems = Subproblems.create(
        windows_macro_mesh(grid), list(range(grid.n_elements)), load_mesh(micro.mesh_file),
        MacroCube(rel_size=1.0, level=level), np.array([1, 1, 1]))

    # UNDEFORMED per-fracture aperture (report sec. 3.1), CONSTANT per fracture region — built
    # directly from the fracture radii, keyed by region id (NOT element id: Flow123d renumbers
    # elements internally in its own output — e.g. it drops boundary-only faces from its own
    # element count entirely — so element ids from the pre-solve mesh do NOT line up with the
    # solver output's element ids; region ids, being explicit DATA rather than an internal
    # numbering scheme, ARE preserved correctly and are all this needs, since aperture never
    # varies within one fracture region anyway)
    aperture_by_region_id = dict(zip(
        micro.fracture_region_ids, float(cfg.materials.aperture_per_r) * micro.fracture_radii))

    # the six elementary prescribed macro-strain matrices E_q (report eq. 2.62) are exactly
    # the Voigt unit vectors mapped to tensors by bgem voigt_to_tn (report order)
    results = []
    for name, dir_name, E in zip(case_names, case_dir_names, voigt_to_tn(beta * np.eye(6))):
        print(f"[upscale_m kubc] case {name}: u = {E.tolist()} @ [x, y, z]")
        per_box, spatial_file = micro_problem(cfg, dir_name, E, micro, subproblems, grid, level,
                                              aperture_by_region_id)
        # Voigt vectors via bgem tn_to_voigt (report order [11, 22, 33, 23, 13, 12]);
        # engineering shear gamma = 2 eps on the strain shear rows (report eq. 2.66)
        results.append(LoadCaseResult(
            name=name,
            E_prescribed_voigt=tn_to_voigt(E[None])[0] * ENG_SHEAR,
            eps_voigt=tn_to_voigt(np.array([e for e, s in per_box])) * ENG_SHEAR,
            sigma_voigt=tn_to_voigt(np.array([s for e, s in per_box])),
            spatial_file=spatial_file,
        ))
        n_sub = len(per_box)
        sig0 = results[-1].sigma_voigt[0]
        suffix = f" (window 0 of {n_sub})" if n_sub > 1 else ""
        print(f"[upscale_m kubc] case {name}: <sigma>{suffix} = {sig0}")
    return results
