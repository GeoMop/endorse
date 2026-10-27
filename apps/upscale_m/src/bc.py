"""
KUBC prescribes a strain:   u = E_q x valid on the whole boundary.
SUBC prescribes a traction: t = sigma_q @ n_face, and additionally needs the support
                            tetrahedra in the mesh to pin the six rigid-body modes.


Voigt ordering throughout: [11, 22, 33, 23, 13, 12], engineering shear (gamma = 2 eps)
"""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bgem.upscale import voigt_to_tn
from endorse.common import dotdict, workdir, call_flow

from micro_mesh import MicroMesh

_app_dir = Path(__file__).parents[1]

# config key scaling the six unit load states, per BC type
_SCALE_PARAM = {"kubc": "deformation_parameter_beta", "subc": "stress_parameter_alpha"}


@dataclass
class LoadCaseResult:
    name: str
    spatial_file: str


def load_cases(cfg: dotdict, bc_type: str) -> List[Tuple[str, str, np.ndarray]]:
    # The six prescribed load tensors with their case and directory names, in Voigt order
    scale = float(cfg.loads[_SCALE_PARAM[bc_type]])
    names_key, dirs_key = f"case_names_{bc_type}", f"case_dir_names_{bc_type}"
    for key in (names_key, dirs_key):
        assert key in cfg.loads, f"missing config key loads.{key}"
    case_names = list(cfg.loads[names_key])
    case_dir_names = list(cfg.loads[dirs_key])
    assert len(case_names) == 6 and len(case_dir_names) == 6, \
        f"loads.{names_key}/{dirs_key} must list exactly the 6 load cases in Voigt order"
    return list(zip(case_names, case_dir_names, voigt_to_tn(scale * np.eye(6))))


def bc_params(cfg: dotdict, bc_type: str, load_matrix: np.ndarray, micro: MicroMesh
              ) -> Tuple[str, str, Dict[str, str]]:

    if bc_type == "kubc":
        boundary_regions = list(cfg.geometry.boundary_regions)
        missing = [r for r in boundary_regions if r not in micro.regions]
        assert not missing, f"geometry.boundary_regions not found in the mesh: {missing}"
        extra_params = dict(
            boundary_regions=f"[{', '.join(boundary_regions)}]",
            # endorse micro-template: "<matrix> @ [x,y,z]" in the template
            bc_displacement_matrix=str(load_matrix.tolist()),
        )
        return cfg.loads.input_template_kubc, micro.bulk_region, extra_params

    # SUBC additionally requires the mesh to have been built with subc_support=True
    assert bc_type == "subc", f"bc_type must be 'kubc' or 'subc' (got {bc_type!r})"
    # traction t = sigma @ n per outer face, n the outward unit normal: face <axis><side>, with
    extra_params = {
        f"traction_{axis}{side}": str((load_matrix @ (sign * np.eye(3)[i])).tolist())
        for i, axis in enumerate("xyz") for side, sign in ((0, -1.0), (1, 1.0))
    }
    # the support tetrahedra go in with the bulk, so they get the rock material
    support = cfg.geometry.support_region
    return cfg.loads.input_template_subc, f"[{micro.bulk_region}, {support}]", extra_params


def micro_problem(cfg: dotdict, tag: str, load_matrix: np.ndarray,
                  micro: MicroMesh, bc_type: str) -> str:
    """
    Runs one load case in its own work dir and return the path to its Flow123d output
    """
    mats = cfg.materials
    mesh_abs_path = os.path.abspath(micro.mesh_file.path)
    cross_section_abs_path = os.path.abspath(micro.cross_section_file.path)
    template_name, bulk_regions, bc_extra_params = bc_params(cfg, bc_type, load_matrix, micro)
    template_path = _app_dir / "flow123d_templates" / template_name
    case_name = os.path.basename(tag)

    with workdir(tag, inputs=[]):
        shutil.copy2(mesh_abs_path, os.path.basename(mesh_abs_path))
        shutil.copy2(cross_section_abs_path, os.path.basename(cross_section_abs_path))
        local_template = f"{case_name}_tmpl.yaml"
        shutil.copy2(template_path, local_template)
        params = dict(
            description=f"upscale_m {bc_type.upper()} load {tag}",
            mesh_file=os.path.basename(mesh_abs_path),
            cross_section_file=os.path.basename(cross_section_abs_path),
            bulk_regions=bulk_regions,
            fracture_regions=f"[{', '.join(micro.fracture_regions)}]",
            rock_young_modulus=f"{float(mats.young_modulus_rock):g}",
            rock_poisson_ratio=f"{float(mats.poisson_ratio_rock):g}",
            fracture_young_modulus=f"{float(mats.young_modulus_fracture):g}",
            fracture_poisson_ratio=f"{float(mats.poisson_ratio_fracture):g}",
            **bc_extra_params,
        )
        micro_output = call_flow(cfg.machine_config, Path(local_template), params)

        os.remove(local_template)
        if not micro_output.check_conv_reasons():
            raise ValueError(f"Load case {tag}: Flow123d simulation failed.")

        expected = str(cfg.loads.output_fields_file)
        produced = os.path.abspath(micro_output.mechanic.spatial_file.path)
        assert produced.endswith(os.path.normpath(expected)), \
            f"loads.output_fields_file={expected!r} disagrees with what Flow123d wrote: " \
            f"{produced}"
        return produced


def run_bc(cfg: dotdict, micro: MicroMesh, bc_type: str) -> List[LoadCaseResult]:
    label = "u" if bc_type == "kubc" else "sigma"
    results = []
    for name, dir_name, load in load_cases(cfg, bc_type):
        tag = f"{bc_type}/{dir_name}"
        print(f"[upscale_m {bc_type}] case {name}: {label} = {load.tolist()}")
        spatial_file = micro_problem(cfg, tag, load, micro, bc_type)
        results.append(LoadCaseResult(name=name, spatial_file=spatial_file))
    return results
