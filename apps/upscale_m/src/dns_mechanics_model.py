"""
DNS: the cube solved directly on the explicitly fractured mesh, with no buffer zone

This is the reference solution the homogenized model of macro_mechanics_model is meant to
reproduce. Its product is the solved fields themselves (Flow123d .msh + a plain .vtu conversion),
one per load case and BC type, not effective tensors.
"""
import os
from typing import List

from endorse.common import dotdict

from bc import run_bc
from export_vtu import export_dns_case
from micro_mesh import MicroMesh, make_micro_mesh


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


def run_dns(cfg: dotdict, micro: MicroMesh, bc_type: str) -> List[str]:
    # Solves the six load cases of one BC type on the DNS mesh, one Flow123d output per case
    results = run_bc(cfg, micro, bc_type)
    written = []
    for r in results:
        case_dir = os.path.dirname(os.path.dirname(os.path.abspath(r.spatial_file)))
        print(f"[upscale_m dns/{bc_type}] fields for case {r.name}: {case_dir}")
        written.append(r.spatial_file)
        export_dns_case(cfg, r.spatial_file, case_dir)
    return written
