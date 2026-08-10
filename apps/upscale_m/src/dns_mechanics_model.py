"""
DNS: the cube solved directly on the explicitly fractured mesh, with no buffer zone

This is the reference solution the homogenized model of macro_mechanics_model is meant to
reproduce. Its product is the per-element stress and strain FIELDS as .vtu, one file per load
case and BC type, not effective tensors.
"""
import os
import sys
from pathlib import Path
from typing import List

import numpy as np

from endorse import common
from endorse.common import dotdict, File
from endorse.macro_flow_model import fields_file
from endorse.mesh_class import load_mesh

from bc import run_bc
from micro_mesh import MicroMesh, make_micro_mesh
from postprocess import strain_stress_fields


def dns_geometry(cfg_geometry: dotdict) -> dotdict:
    # cfg.geometry with L_ext_factor forced to 1.0, so the meshed domain is the L_inner cube.
    geom = dotdict.create(dict(dotdict.serialize(cfg_geometry)))
    geom["L_ext_factor"] = 1.0
    return geom


def dns_mesh_cfg(cfg_mesh: dotdict, cfg_dns: dotdict) -> dotdict:
    mesh_cfg = dict(dotdict.serialize(cfg_mesh))
    mesh_cfg["mesh_name"] = str(cfg_dns.get("mesh_name", "dns_mesh"))
    step = cfg_dns.get("mesh_step", None)
    if step is not None:
        mesh_cfg["fracture_mesh_step"] = float(step)
    return dotdict.create(mesh_cfg)


def fracture_sampling_box(cfg_geometry: dotdict) -> List[float]:
    # The box the DFN is sampled in: the BUFFERED cube, not the bare cube the DNS meshes
    return [float(cfg_geometry.get("L_ext_factor", 2.0)) * float(cfg_geometry.L_inner)] * 3


def make_dns_mesh(cfg: dotdict, work_dir: str = ".", subc_support: bool = False) -> MicroMesh:
    # Bare cube meshed over the buffered problem's own fracture realization
    cfg_dns = cfg.get("dns", dotdict.create({}))
    return make_micro_mesh(
        dns_geometry(cfg.geometry), dns_mesh_cfg(cfg.mesh, cfg_dns), cfg.fractures,
        cfg.materials.aperture_per_r, work_dir=work_dir, subc_support=subc_support,
        fracture_box=fracture_sampling_box(cfg.geometry))


def write_dns_fields(cfg: dotdict, micro: MicroMesh, spatial_file: str, out_dir: str,
                     basename: str = "dns_fields") -> File:
    # One solved load case -> per-element stress and strain over the whole mesh, as .vtu / .msh2.
    mats = cfg.materials
    output_mesh = load_mesh(File(spatial_file))
    rid, eps9, stress9 = strain_stress_fields(
        output_mesh, micro.bulk_region_id, micro.fracture_region_ids,
        float(mats.young_modulus_rock), float(mats.poisson_ratio_rock),
        float(mats.young_modulus_fracture), float(mats.poisson_ratio_fracture),
        stress_field=str(cfg.loads.stress_field_name),
        region_id_field=str(cfg.loads.region_id_field_name))

    fields = {
        "stress": np.asarray(stress9, dtype=float),
        "strain": np.asarray(eps9, dtype=float),
        # write_fields wants an (N, 1) column here, not a flat vector
        "region_id": np.asarray(rid, dtype=float).reshape(-1, 1),
    }

    output_mesh.gmsh_io.node_data.clear()
    output_mesh.gmsh_io.element_data.clear()
    output_mesh.gmsh_io.element_node_data.clear()

    os.makedirs(out_dir, exist_ok=True)
    return fields_file(output_mesh, fields, file_name=os.path.join(out_dir, basename + ".msh2"))


def run_dns(cfg: dotdict, micro: MicroMesh, bc_type: str) -> List[File]:
    # Solves the six load cases of one BC type on the DNS mesh, one field file per case
    results = run_bc(cfg, micro, bc_type)
    written = []
    for r in results:
        case_dir = os.path.dirname(os.path.dirname(os.path.abspath(r.spatial_file)))
        print(f"[upscale_m dns/{bc_type}] fields for case {r.name}: {case_dir}")
        written.append(write_dns_fields(cfg, micro, r.spatial_file, case_dir))
    return written


def main(run_name="dns", config_file="config_master.yaml"):
    # Build the bare cube, solve every BC type loads.bc_type asks for, write the fields
    app_dir = Path(__file__).parents[1]
    cfg_path = app_dir / "configs" / config_file
    if not cfg_path.is_file():
        cfg_path = app_dir / config_file
    cfg = common.load_config(cfg_path)

    run_dir = Path(run_name) if os.path.isabs(run_name) else app_dir / "runs" / run_name
    bc_type_cfg = str(cfg.loads.bc_type).strip().lower()
    assert bc_type_cfg in ("kubc", "subc", "both"), \
        f"loads.bc_type must be 'kubc', 'subc' or 'both' (got {bc_type_cfg!r})"
    bc_types = [t for t in ("kubc", "subc") if bc_type_cfg in (t, "both")]

    print(f"[upscale_m dns] config: {cfg_path}")
    print(f"[upscale_m dns] run dir: {run_dir}")
    print(f"[upscale_m dns] cube edge {cfg.geometry.L_inner} m, NO buffer (L_ext_factor = 1.0)")

    os.makedirs(run_dir, exist_ok=True)
    written = {}
    with common.workdir(run_dir, inputs=[]):
        print("=== upscale_m dns: mesh ===", flush=True)
        micro = make_dns_mesh(cfg, work_dir=".", subc_support="subc" in bc_types)
        for bc_type in bc_types:
            print(f"=== upscale_m dns: {bc_type.upper()} load cases ===", flush=True)
            written[bc_type] = run_dns(cfg, micro, bc_type)
    print("\n=== upscale_m dns: DONE ===")
    for bc_type, files in written.items():
        print(f"  {bc_type}: {len(files)} field file(s), *.vtu next to each *.msh2")
    return written


if __name__ == "__main__":
    main(*sys.argv[1:])
