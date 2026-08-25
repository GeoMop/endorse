"""
Per-macro-element effective elastic tensor field used for the homogenized half of the macro vs DNS
comparison.

Runs against an completed micro_mechanics_model.py run: it re-reads the six cached Flow123d
solutions per BC type and needs no Docker. Output is a field file with one 6x6 Voigt tensor per
macro element, written via endorse fields_file (msh2 + vtu).

The averaging window for each macro element is the element itself (endorse MacroTetra): a micro
element belongs to it iff its barycenter lies inside the tetrahedron. That makes the field a true
partition of the micro mesh -- no micro element counted twice, none lost -- which a box- or
sphere-shaped window around the element cannot give.

The macro mesh carries NO fractures; their influence enters only through the tensors.

"""
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np

from bgem.gmsh import gmsh
from bgem.upscale import tn_to_voigt
from endorse import common
from endorse.common import dotdict, File
from endorse.equivalent_field import eq_tensor
from endorse.homogenisation import MacroTetra, Subdomain
from endorse.macro_flow_model import fields_file
from endorse.mesh_class import Mesh, _load_mesh, load_mesh

from bc import bc_params, load_cases
from export_vtu import export_macro_case
from micro_mesh import MicroMesh, make_geometry, make_micro_mesh, meshing
from postprocess import strain_stress_fields, subdomain_measures, weighted_sum

_app_dir = Path(__file__).parents[1]

def equivalent_tensor_6x6(loads: np.ndarray, responses: np.ndarray) -> np.ndarray:
    assert loads.shape[0] == 6 and loads.shape == responses.shape, \
        f"need 6-row Voigt column matrices, got {loads.shape} / {responses.shape}"
    eq = eq_tensor(dim=6)
    return eq.to_full_tn(eq.flat(loads.T, responses.T)).reshape(6, 6)

def kelvin_columns_3x3(C_voigt: np.ndarray) -> np.ndarray:
    """
    The six 3x3 tensors Flow123d reads as stiffness_tensor_0 .. stiffness_tensor_5

    Returns (6, 9): one row per column index k, row-major 3x3
    """

    # Kelvin scaling of the three shear rows/columns:
    # same ordering as Voigt, i.e. [xx, yy, zz, yz, xz, xy]
    KELVIN = np.diag([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
    assert C_voigt.shape == (6, 6), f"need a 6x6 Voigt tensor, got {C_voigt.shape}"
    C_kelvin = KELVIN @ C_voigt @ KELVIN
    out = np.zeros((6, 3, 3))
    s = np.sqrt(2.0)
    for k in range(6):
        c = C_kelvin[:, k]
        out[k] = [[c[0], c[5] / s, c[4] / s],
                  [c[5] / s, c[1], c[3] / s],
                  [c[4] / s, c[3] / s, c[2]]]
    return out.reshape(6, 9)


def isotropic_voigt(young: float, poisson: float) -> np.ndarray:
    """6x6 Voigt tensor of an isotropic material, our ordering [xx, yy, zz, yz, xz, xy]."""
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.diag_indices(3)] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def homogenized_elements(macro_mesh: Mesh, support_region: str) -> Tuple[List[int], List[int]]:
    """
    Split the macro mesh's dim-3 elements into (homogenized, support).

    Support tetrahedra exist only in a SUBC macro mesh and must not be averaged: they protrude
    from the cube, so a Subdomain over one of them captures no micro element at all.
    """
    support_id = macro_mesh.gmsh_io.physical.get(support_region, (None, None))[0]
    bulk = macro_mesh.el_dim_slice(dim=3)
    homogenized, support = [], []
    for iel in range(bulk.start, bulk.stop):
        target = support if macro_mesh.elements[iel].tags[0] == support_id else homogenized
        target.append(iel)
    return homogenized, support


def make_macro_mesh(cfg_geometry: dotdict, cfg_macro: dotdict, cfg_mesh: dotdict,
                    work_dir: str = ".", subc_support: bool = False) -> Mesh:
    """
    Coarse fracture-free tet mesh of the inner cube, element size cfg_macro.mesh_step.

    Built through micro_mesh.make_geometry with an EMPTY fracture set, so it comes out with the
    same named outer faces (.side_x0 ...) and, for SUBC, the same rigid-body support tetrahedra
    as the micro mesh. That is what lets the macro problem take the same bc_type and the same six
    load cases -- the only difference between the two problems stays the material description.

    Centered at the origin, the same bgem convention as the micro mesh, so the two share
    coordinates.
    """
    mesh_path = Path(work_dir) / str(cfg_macro.mesh_file)
    if not mesh_path.exists():
        # the macro mesh is the inner cube: the buffer shell belongs to the micro problem alone,
        # so L_ext_factor is forced to 1 whatever the micro problem uses
        macro_geometry = dotdict(**{**dotdict.serialize(cfg_geometry), "L_ext_factor": 1.0})
        # make_geometry/meshing read the element size under the micro problem's key name; the
        # tolerances and curve options have no macro counterpart and come from cfg.mesh unchanged
        macro_mesh_cfg = dotdict(**{
            **dotdict.serialize(cfg_mesh),
            "mesh_name": mesh_path.stem,
            "fracture_mesh_step": float(cfg_macro.mesh_step),
            "mesh_size_min_fraction": float(cfg_macro.mesh_size_min_fraction),
        })
        factory = gmsh.GeometryOCC(mesh_path.stem, verbose=False)
        factory.geom_options.Tolerance = float(cfg_mesh.tolerance)
        factory.geom_options.ToleranceBoolean = float(cfg_mesh.tolerance_boolean)
        # no fractures here: make_geometry only tests len(), so an empty sequence is enough
        geometry = make_geometry(factory, macro_geometry, macro_mesh_cfg, [], subc_support)
        # the micro mesh pins its element size on the fracture group; with no fractures the box
        # needs that size field itself, otherwise gmsh drifts towards CharacteristicLengthMin
        geometry.mesh_step(float(cfg_macro.mesh_step))
        meshing(factory, [geometry], str(mesh_path), macro_mesh_cfg)
        factory.close()
    return load_mesh(File(str(mesh_path)))


def macro_element_averages(output_mesh: Mesh, macro_mesh: Mesh, micro: MicroMesh,
                           aperture_by_region_id: Dict[int, float],
                           young_rock: float, poisson_rock: float,
                           young_fracture: float, poisson_fracture: float,
                           stress_field: str, region_id_field: str,
                           el_indices: List[int]
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    One (eps_avg, sigma_avg) 3x3 pair per homogenized macro element, for ONE solved load case.

    Each element gets its own independent Subdomain

    The average is normalized by the CAPTURED bulk volume, not by the macro element's declared
    volume. The two differ element by element: membership is decided per whole micro element by
    its barycenter.

    el_indices lists the macro elements to average, in order. SUBC support tetrahedra are NOT in
    it: they stick out of the cube, so no micro element falls inside them.
    """
    rid, eps9, stress9 = strain_stress_fields(
        output_mesh, micro.bulk_region_id, micro.fracture_region_ids,
        young_rock, poisson_rock, young_fracture, poisson_fracture,
        stress_field=stress_field, region_id_field=region_id_field)

    pairs = []
    for iel in el_indices:
        sub = Subdomain.create(MacroTetra(rel_radius=1.0), output_mesh, macro_mesh, iel,
                               dims=(2, 3))
        sub, measures = subdomain_measures(sub, output_mesh, rid, micro.bulk_region_id,
                                           micro.fracture_region_ids, aperture_by_region_id)
        bulk_mask = rid[sub.el_indices] == micro.bulk_region_id
        weighted = measures * np.asarray(sub.intersect_weights)
        V_ref = float(np.sum(weighted[bulk_mask]))
        assert V_ref > 0.0, f"macro element {iel}: no bulk micro elements captured"
        sigma_avg = weighted_sum(sub, stress9, measures) / V_ref
        eps_avg = weighted_sum(sub, eps9, measures) / V_ref
        pairs.append((eps_avg.reshape(3, 3), sigma_avg.reshape(3, 3)))
    return pairs


def macro_elastic_tensor(cfg: dotdict, micro: MicroMesh, macro_mesh: Mesh,
                         results, bc_type: str) -> File:
    """
    Assembles the tensor field for one BC type from its six already solved load cases

    Per macro element: six measured (eps, sigma) Voigt pairs -> a 6x6 fit -> the six 3x3 Kelvin
    columns Flow123d reads as stiffness_tensor_0 .. stiffness_tensor_5
    """
    mats = cfg.materials
    aperture_by_region_id = micro.aperture_by_region_id(float(mats.aperture_per_r))
    # engineering shear on the Voigt strain shear rows: gamma = 2 eps
    eng_shear = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    homogenized, support = homogenized_elements(macro_mesh, cfg.geometry.support_region)

    eps_cases, sigma_cases = [], []
    for r in results:
        print(f"[upscale_m macro/{bc_type}] averaging case {r.name}: {r.spatial_file}")
        output_mesh = _load_mesh(File(r.spatial_file), None)
        pairs = macro_element_averages(
            output_mesh, macro_mesh, micro, aperture_by_region_id,
            young_rock=float(mats.young_modulus_rock),
            poisson_rock=float(mats.poisson_ratio_rock),
            young_fracture=float(mats.young_modulus_fracture),
            poisson_fracture=float(mats.poisson_ratio_fracture),
            stress_field=str(cfg.loads.stress_field_name),
            region_id_field=str(cfg.loads.region_id_field_name),
            el_indices=homogenized)
        eps_cases.append(tn_to_voigt(np.array([e for e, s in pairs])) * eng_shear)
        sigma_cases.append(tn_to_voigt(np.array([s for e, s in pairs])))


    eps_all = np.stack(eps_cases, axis=-1)
    sigma_all = np.stack(sigma_cases, axis=-1)

    # one 9-component field per Kelvin column
    col_fields = np.zeros((6, len(macro_mesh.elements), 9))
    for i, iel in enumerate(homogenized):
        C = equivalent_tensor_6x6(eps_all[i], sigma_all[i])
        col_fields[:, iel, :] = kelvin_columns_3x3(C)

    if support:
        # SUBC support tetrahedra carry no homogenized rock -- they are there to pin the six
        # rigid-body modes.
        rock = kelvin_columns_3x3(isotropic_voigt(float(mats.young_modulus_rock),
                                                  float(mats.poisson_ratio_rock)))
        col_fields[:, support, :] = rock[:, None, :]
        print(f"[upscale_m macro/{bc_type}] {len(support)} support element(s): isotropic rock")

    field_path = Path(bc_type) / cfg.mechanics_macroscale.tensor_field_file
    prefix = str(cfg.mechanics_macroscale.tensor_field_name_prefix)
    os.makedirs(field_path.parent, exist_ok=True)
    fields_file(macro_mesh, {f"{prefix}{k}": col_fields[k] for k in range(6)},
                file_name=str(field_path))
    print(f"[upscale_m macro/{bc_type}] tensor field written: {field_path} "
          f"({len(homogenized)} homogenized + {len(support)} support elements)")
    return File(str(field_path))


def macro_problem(cfg: dotdict, tag: str, load_matrix: np.ndarray, macro_mesh: Mesh,
                  mesh_path: str, tensor_file: File, bc_type: str) -> str:
    """
    Runs one load case of the macro problem and returns its Flow123d output path

    The micro counterpart is bc.micro_problem, the only difference is the material...
    macro problem has no input young_modulus/poisson_ratio, but the six
    stiffness_tensor_i fields read from tensor_file
    """
    macro_cfg = cfg.mechanics_macroscale
    # bc_params needs only .regions and .bulk_region, so the macro mesh describes itself
    mesh_desc = SimpleNamespace(regions=dict(macro_mesh.gmsh_io.physical),
                                bulk_region=str(cfg.geometry.bulk_region))
    _micro_template, bulk_regions, bc_extra_params = bc_params(
        cfg, bc_type, load_matrix, mesh_desc)
    template_name = str(macro_cfg[f"input_template_{bc_type}"])
    template_path = _app_dir / "flow123d_templates" / template_name
    case_name = os.path.basename(tag)

    mesh_abs_path = os.path.abspath(mesh_path)
    tensor_abs_path = os.path.abspath(tensor_file.path)
    prefix = str(macro_cfg.tensor_field_name_prefix)

    with common.workdir(tag, inputs=[]):
        shutil.copy2(mesh_abs_path, os.path.basename(mesh_abs_path))
        shutil.copy2(tensor_abs_path, os.path.basename(tensor_abs_path))
        local_template = f"{case_name}_tmpl.yaml"
        shutil.copy2(template_path, local_template)
        params = dict(
            description=f"upscale_m MACRO {bc_type.upper()} load {tag}",
            mesh_file=os.path.basename(mesh_abs_path),
            bulk_regions=bulk_regions,
            tensor_field_file=os.path.basename(tensor_abs_path),
            **{f"stiffness_field_{k}": f"{prefix}{k}" for k in range(6)},
            **bc_extra_params,
        )
        macro_output = common.call_flow(cfg.machine_config, Path(local_template), params)

        os.remove(local_template)
        if not macro_output.check_conv_reasons():
            raise ValueError(f"Macro load case {tag}: Flow123d simulation failed.")
        return os.path.abspath(macro_output.mechanic.spatial_file.path)


def run_macro(cfg: dotdict, macro_mesh: Mesh, mesh_path: str, tensor_file: File,
              bc_type: str) -> List[SimpleNamespace]:
    """The six macro load cases for one BC type, run under macro_<bc_type>/."""
    label = "u" if bc_type == "kubc" else "sigma"
    results = []
    for name, dir_name, load in load_cases(cfg, bc_type):
        tag = f"macro_{bc_type}/{dir_name}"
        print(f"[upscale_m macro {bc_type}] case {name}: {label} = {load.tolist()}", flush=True)
        spatial_file = macro_problem(cfg, tag, load, macro_mesh, mesh_path, tensor_file, bc_type)
        results.append(SimpleNamespace(name=name, spatial_file=spatial_file))
        case_dir = os.path.dirname(os.path.dirname(os.path.abspath(spatial_file)))
        export_macro_case(cfg, macro_mesh, spatial_file, case_dir)
    return results


def main(run_name="default", config_file="config_master.yaml"):
    app_dir = Path(__file__).parents[1]
    cfg = common.load_config(app_dir / "configs" / config_file)
    run_dir = app_dir / "runs" / run_name
    assert run_dir.is_dir(), \
        f"{run_dir} does not exist -- run `python src/micro_mechanics_model.py {run_name}` first"

    bc_type_cfg = str(cfg.loads.bc_type).strip().lower()
    subc_wanted = bc_type_cfg in ("subc", "both")
    # solving the macro problem is optional: without it the run just produces the tensor fields
    solve_macro = bool(cfg.mechanics_macroscale.get("solve", True))
    with common.workdir(run_dir, inputs=[]):
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures,
                                cfg.materials.aperture_per_r, work_dir=".",
                                subc_support=subc_wanted)
        macro_mesh = make_macro_mesh(cfg.geometry, cfg.mechanics_macroscale, cfg.mesh,
                                     work_dir=".", subc_support=subc_wanted)
        macro_mesh_path = str(cfg.mechanics_macroscale.mesh_file)

        tensor_files, macro_results = {}, {}
        for bc_type in [t for t in ("kubc", "subc") if bc_type_cfg in (t, "both")]:
            results = []
            for name, dir_name, _load in load_cases(cfg, bc_type):
                spatial_file = os.path.abspath(os.path.join(bc_type, dir_name, str(cfg.loads.output_fields_file)))
                assert os.path.isfile(spatial_file), f"missing cached output: {spatial_file}"
                results.append(SimpleNamespace(name=name, spatial_file=spatial_file))
            tensor_files[bc_type] = macro_elastic_tensor(cfg, micro, macro_mesh, results, bc_type)
            if solve_macro:
                print(f"=== upscale_m: MACRO {bc_type.upper()} load cases ===", flush=True)
                macro_results[bc_type] = run_macro(cfg, macro_mesh, macro_mesh_path,
                                                   tensor_files[bc_type], bc_type)
        return tensor_files, macro_results


if __name__ == "__main__":
    main(*sys.argv[1:])
