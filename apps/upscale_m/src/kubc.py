"""
KUBC (kinematic uniform boundary conditions) load cases and the micro-problem runner.

Six independent load states q = 1..6 prescribe u = E_q x on the whole outer boundary
(report sec. 2.5.2 / 3.2.1). Voigt ordering (report convention): [11, 22, 33, 23, 13, 12],
engineering shear (gamma = 2 eps) on the strain shear rows.

Each case runs Flow123d via endorse.common.call_flow with the single parametrized template
flow123d_templates/mechanics_kubc_tmpl.yaml; the displacement formula string is generated from E_q,
so additional (e.g. mixed) load states for a least-squares identification need no new template.
"""
import os
import shutil
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List

import numpy as np

from endorse import common
from endorse.common import dotdict, workdir, call_flow

from micro_mesh import MicroMesh
from postprocess import average_boxes, stress_to_voigt, strain_to_voigt

_src_dir = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.dirname(_src_dir)
KUBC_TEMPLATE = os.path.join(_app_dir, "flow123d_templates", "mechanics_kubc_tmpl.yaml")

# Voigt slot -> (i, j) of the symmetric load matrix; report order [11, 22, 33, 23, 13, 12].
VOIGT_IJ = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
CASE_NAMES = ["E_11", "E_22", "E_33", "E_23", "E_13", "E_12"]
# per-case directory names, matching R. Siddall's original result layout
CASE_DIR_NAMES = ["pure_normal_E_11", "pure_normal_E_22", "pure_normal_E_33",
                  "pure_shear_E_23", "pure_shear_E_13", "pure_shear_E_12"]


def kubc_load_matrices(beta: float) -> List[np.ndarray]:
    """The six elementary prescribed macro-strain matrices E_q (report eq. 2.62)."""
    matrices = []
    for i, j in VOIGT_IJ:
        E = np.zeros((3, 3))
        E[i, j] = E[j, i] = beta
        matrices.append(E)
    return matrices


def formula_from_matrix(E: np.ndarray) -> str:
    """
    Flow123d FieldFormula string for u_i = E_ij x_j, e.g. "[0.01*x, 0.0, 0.0]".
    Works for any (not only elementary) load matrix.
    """
    variables = ("x", "y", "z")
    components = []
    for i in range(3):
        terms = [f"{E[i, j]:g}*{variables[j]}" for j in range(3) if E[i, j] != 0.0]
        components.append(" + ".join(terms) if terms else "0.0")
    return "[" + ", ".join(components) + "]"


def prescribed_strain_voigt(E: np.ndarray) -> np.ndarray:
    """Prescribed macro strain in Voigt notation with engineering shear (report eq. 2.66)."""
    return np.array([E[0, 0], E[1, 1], E[2, 2], 2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1]])


@dataclass
class LoadCaseResult:
    name: str
    E_prescribed_voigt: np.ndarray   # prescribed strain (Voigt, engineering shear) — input record
    eps_voigt: np.ndarray            # measured strain per subdomain, shape (n_subdomains, 6)
    sigma_voigt: np.ndarray          # measured stress per subdomain, shape (n_subdomains, 6)
    vtu_file: str


def _find_mechanics_vtu(output_dir: str = "output") -> str:
    """
    Locate the mechanics result VTU. Preference for '*000000.vtu' follows R. Siddall's validated
    pipeline (the steady mechanics solution is written at the first output time).
    """
    candidates = sorted(glob(os.path.join(output_dir, "mechanics", "*.vtu")))
    assert candidates, f"No mechanics VTU found under {os.path.abspath(output_dir)}"
    preferred = [f for f in candidates if f.endswith("000000.vtu")]
    return preferred[0] if preferred else candidates[0]


def run_kubc(cfg: dotdict, micro: MicroMesh) -> List[LoadCaseResult]:
    """
    Run the six KUBC load states on the given micro mesh (in the CURRENT work dir; one
    subdirectory per case) and return per-case averaged fields over the inner cube.
    """
    mats = cfg.materials
    beta = float(cfg.loads.deformation_parameter_beta)
    radii = [fr.r for fr in micro.fractures]
    aperture = float(mats.aperture_per_r) * float(np.mean(radii)) if radii else 0.0
    if radii and len(set(radii)) > 1:
        print("[upscale_m kubc] WARNING: multiple fracture radii, using mean-radius aperture "
              "(per-fracture regions pending, see PLAN.md)")

    mesh_abs_path = os.path.abspath(micro.mesh_file.path)
    bulk_regions = "[rock_inner, rock_outer]" if micro.has_buffer else "rock_inner"
    results = []
    for name, dir_name, E in zip(CASE_NAMES, CASE_DIR_NAMES, kubc_load_matrices(beta)):
        # NOTE: mesh copied explicitly, not via workdir(inputs=...) — workdir.copy uses src.stem
        # and silently drops the file extension (recorded in PLAN.md questions).
        with workdir(dir_name, inputs=[]):
            shutil.copy2(mesh_abs_path, os.path.basename(mesh_abs_path))
            # local per-case template copy so the rendered input carries the case name
            # (pure_normal_E_11.yaml, ...), mirroring the original per-case yaml files
            local_template = f"{dir_name}_tmpl.yaml"
            shutil.copy2(KUBC_TEMPLATE, local_template)
            params = dict(
                description=f"upscale_m KUBC load {name}",
                mesh_file=os.path.basename(mesh_abs_path),
                bulk_regions=bulk_regions,
                bc_displacement_formula=formula_from_matrix(E),
                rock_young_modulus=f"{float(mats.young_modulus_rock):g}",
                rock_poisson_ratio=f"{float(mats.poisson_ratio_rock):g}",
                fracture_young_modulus=f"{float(mats.young_modulus_fracture):g}",
                fracture_poisson_ratio=f"{float(mats.poisson_ratio_fracture):g}",
                cross_section=f"{aperture:g}",
            )
            print(f"[upscale_m kubc] case {name}: u = {params['bc_displacement_formula']}")
            flow_out = call_flow(cfg.machine_config, Path(local_template), params)
            if not flow_out.check_conv_reasons():
                raise RuntimeError(f"KUBC case {name}: Flow123d did not converge.")

            vtu = _find_mechanics_vtu()
            # one (eps, sigma) pair per subdomain cell, all read from this case's solution
            # (macro_conductivity's per-subdomain averaging; exact here — cells are imprinted)
            per_box = average_boxes(
                vtu_file=vtu,
                boxes=micro.subdomain_boxes,
                young_rock=float(mats.young_modulus_rock),
                poisson_rock=float(mats.poisson_ratio_rock),
                young_fracture=float(mats.young_modulus_fracture),
                poisson_fracture=float(mats.poisson_ratio_fracture),
                aperture=aperture,
            )
            results.append(LoadCaseResult(
                name=name,
                E_prescribed_voigt=prescribed_strain_voigt(E),
                eps_voigt=np.array([strain_to_voigt(e) for e, s in per_box]),
                sigma_voigt=np.array([stress_to_voigt(s) for e, s in per_box]),
                vtu_file=os.path.abspath(vtu),
            ))
            n_sub = len(per_box)
            sig0 = results[-1].sigma_voigt[0]
            suffix = f" (subdomain 0 of {n_sub})" if n_sub > 1 else ""
            print(f"[upscale_m kubc] case {name}: <sigma>{suffix} = {sig0}")
    return results
