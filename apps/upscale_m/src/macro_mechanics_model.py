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
from endorse.mesh import mesh_tools
from endorse.mesh_class import Mesh, load_mesh

from bc import load_cases
from micro_mesh import MicroMesh, make_micro_mesh
from postprocess import strain_stress_fields, subdomain_measures, weighted_sum

def equivalent_tensor_6x6(loads: np.ndarray, responses: np.ndarray) -> np.ndarray:
    assert loads.shape[0] == 6 and loads.shape == responses.shape, \
        f"need 6-row Voigt column matrices, got {loads.shape} / {responses.shape}"
    eq = eq_tensor(dim=6)
    return eq.to_full_tn(eq.flat(loads.T, responses.T)).reshape(6, 6)


def make_macro_mesh(cfg_geometry: dotdict, cfg_macro: dotdict, work_dir: str = ".") -> Mesh:
    """
    Coarse fracture-free tet mesh of the inner cube, element size cfg_macro.mesh_step.

    Centered at the origin, the same bgem convention as the micro mesh, so the two share
    coordinates.
    """
    mesh_path = Path(work_dir) / str(cfg_macro.mesh_file)
    if not mesh_path.exists():
        L = float(cfg_geometry.L_inner)
        step = float(cfg_macro.mesh_step)
        min_fraction = float(cfg_macro.mesh_size_min_fraction)
        factory = gmsh.GeometryOCC("macro_mesh", verbose=False)
        factory.mesh_options.CharacteristicLengthMin = step / min_fraction
        factory.mesh_options.CharacteristicLengthMax = step
        box = factory.box([L, L, L]).set_region(cfg_geometry.bulk_region)
        box.mesh_step(step)
        mesh_tools.edz_meshing(factory, [box], str(mesh_path))
        factory.close()
    return load_mesh(File(str(mesh_path)))


def macro_element_averages(output_mesh: Mesh, macro_mesh: Mesh, micro: MicroMesh,
                           aperture_by_region_id: Dict[int, float],
                           young_rock: float, poisson_rock: float,
                           young_fracture: float, poisson_fracture: float,
                           stress_field: str, region_id_field: str
                           ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    One (eps_avg, sigma_avg) 3x3 pair per macro element, for ONE solved load case.

    Each element gets its own independent Subdomain

    The average is normalized by the CAPTURED bulk volume, not by the macro element's declared
    volume. The two differ element by element: membership is decided per whole micro element by
    its barycenter.
    """
    rid, eps9, stress9 = strain_stress_fields(
        output_mesh, micro.bulk_region_id, micro.fracture_region_ids,
        young_rock, poisson_rock, young_fracture, poisson_fracture,
        stress_field=stress_field, region_id_field=region_id_field)

    macro_bulk = macro_mesh.el_dim_slice(dim=3)
    pairs = []
    for iel in range(macro_bulk.start, macro_bulk.stop):
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
    Assemble the tensor field for one BC type from its six already-solved load cases.

    Per macro element: six measured (eps, sigma) Voigt pairs -> a 6x6 fit -> 36 row-major
    components of the P0 field. Written under the bc_type/ directory as msh2 + vtu.
    results needs only .name and .spatial_file per case.
    """
    mats = cfg.materials
    aperture_by_region_id = micro.aperture_by_region_id(float(mats.aperture_per_r))
    # engineering shear on the Voigt strain shear rows: gamma = 2 eps
    eng_shear = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    eps_cases, sigma_cases = [], []
    for r in results:
        print(f"[upscale_m macro/{bc_type}] averaging case {r.name}: {r.spatial_file}")
        output_mesh = load_mesh(File(r.spatial_file))
        pairs = macro_element_averages(
            output_mesh, macro_mesh, micro, aperture_by_region_id,
            young_rock=float(mats.young_modulus_rock),
            poisson_rock=float(mats.poisson_ratio_rock),
            young_fracture=float(mats.young_modulus_fracture),
            poisson_fracture=float(mats.poisson_ratio_fracture),
            stress_field=str(cfg.loads.stress_field_name),
            region_id_field=str(cfg.loads.region_id_field_name))
        eps_cases.append(tn_to_voigt(np.array([e for e, s in pairs])) * eng_shear)
        sigma_cases.append(tn_to_voigt(np.array([s for e, s in pairs])))


    eps_all = np.stack(eps_cases, axis=-1)
    sigma_all = np.stack(sigma_cases, axis=-1)

    macro_bulk = macro_mesh.el_dim_slice(dim=3)
    C_field = np.zeros((len(macro_mesh.elements), 36))
    for i, iel in enumerate(range(macro_bulk.start, macro_bulk.stop)):
        C = equivalent_tensor_6x6(eps_all[i], sigma_all[i])
        C_field[iel, :] = C.reshape(36)

    field_path = Path(bc_type) / cfg.mechanics_macroscale.tensor_field_file
    field_name = str(cfg.mechanics_macroscale.tensor_field_name)
    os.makedirs(field_path.parent, exist_ok=True)
    fields_file(macro_mesh, {field_name: C_field}, file_name=str(field_path))
    print(f"[upscale_m macro/{bc_type}] tensor field written: {field_path} "
          f"({macro_bulk.stop - macro_bulk.start} macro elements)")
    return File(str(field_path))


def main(run_name="default", config_file="config_master.yaml"):
    app_dir = Path(__file__).parents[1]
    cfg = common.load_config(app_dir / "configs" / config_file)
    run_dir = app_dir / "runs" / run_name
    assert run_dir.is_dir(), \
        f"{run_dir} does not exist -- run `python src/micro_mechanics_model.py {run_name}` first"

    bc_type_cfg = str(cfg.loads.bc_type).strip().lower()
    with common.workdir(run_dir, inputs=[]):
        micro = make_micro_mesh(cfg.geometry, cfg.mesh, cfg.fractures,
                                cfg.materials.aperture_per_r, work_dir=".",
                                subc_support=bc_type_cfg in ("subc", "both"))
        macro_mesh = make_macro_mesh(cfg.geometry, cfg.mechanics_macroscale, work_dir=".")

        tensor_files = {}
        for bc_type in [t for t in ("kubc", "subc") if bc_type_cfg in (t, "both")]:
            results = []
            for name, dir_name, _load in load_cases(cfg, bc_type):
                spatial_file = os.path.abspath(os.path.join(bc_type, dir_name, str(cfg.loads.output_fields_file)))
                assert os.path.isfile(spatial_file), f"missing cached output: {spatial_file}"
                results.append(SimpleNamespace(name=name, spatial_file=spatial_file))
            tensor_files[bc_type] = macro_elastic_tensor(cfg, micro, macro_mesh, results, bc_type)
        return tensor_files


if __name__ == "__main__":
    main(*sys.argv[1:])
