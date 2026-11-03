import shutil
from pathlib import Path
import logging

import numpy as np

from endorse import common
from endorse.mesh import mesh_tools, fracture_tools

import homogenization_study.exception_wrapper as exp
from homogenization_study.create_mesh import (
    fragment, geometry_finalize,
    make_geometry_homo,
    make_mesh
)
from homogenization_study import job
from homogenization_study.model import StudyMeshFiles
from homogenization_study.problem import prepare_conductivity_homogenization, run_homogenization_problem


def validate_conductivity_field(result) -> None:
    """Check positive definiteness of all homogenized conductivity tensors.

    The Voigt order is ``[Kxx, Kyy, Kzz, Kxy, Kxz, Kyz]``. A symmetric
    conductivity tensor should be positive definite for the three pressure loads.
    """
    xx, yy, zz, xy, xz, yz = result.conductivity.T
    tensors = np.empty((result.n_elements, 3, 3), dtype=float)
    tensors[:, 0, 0] = xx
    tensors[:, 1, 1] = yy
    tensors[:, 2, 2] = zz
    tensors[:, 0, 1] = tensors[:, 1, 0] = xy
    tensors[:, 0, 2] = tensors[:, 2, 0] = xz
    tensors[:, 1, 2] = tensors[:, 2, 1] = yz

    eigenvalues = np.linalg.eigvalsh(tensors)
    invalid = np.flatnonzero(np.min(eigenvalues, axis=1) <= 0.0)
    assert invalid.size == 0, (
        "Homogenized conductivity tensors must be positive definite; "
        f"invalid result rows: {invalid[:10].tolist()}"
    )

@exp.rethrow_as(exp.GeomException, "Geometry exception")
def make_geometry_micro(factory, cfg_mesh:'dotdict', fracture_set):
    cfg_geom = cfg_mesh.geometry

    # Prepare objects
    vol_dict = {
        "fractures_group": None,
        "box_A": None,
        "box_B": None
    }
    bnd_dict = {}

    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions, cfg_geom.box_center)
    bnd_dict = {**bnd_dict, **box_sides_dict}

    dims = cfg_geom.box_dimensions
    rect_A = factory.rectangle([dims[0] / 2, dims[1]], center=[-dims[0] / 2, 0, -dims[2]/2])
    rect_B = factory.rectangle([dims[0] / 2, dims[1]], center=[ dims[0] / 2, 0, -dims[2]/2])
    box_A = rect_A.extrude(vector=[0, 0, dims[2]])[3]
    box_B = rect_B.extrude(vector=[0, 0, dims[2]])[3]

    vol_dict["box_A"] = box_A.set_region("box_A")
    vol_dict["box_B"] = box_B.set_region("box_B")

    if "fractures" in cfg_geom.include:
        fractures = fracture_tools.create_fractures_rectangles(factory, fracture_set, cfg_geom.box_center, factory.rectangle())
        vol_dict["fractures_group"] = factory.group(*fractures).intersect(box)
        vol_dict["fractures_group"].mesh_step(cfg_mesh.fracture_mesh_step) # .set_region("fractures")

    [print(k, v) for k, v in vol_dict.items()]
    factory.synchronize()
    # factory.show()
    # exit(0)

    # Step 5: Map results back to their labels
    fr_dict, fr_bnd_dict = fragment(factory, vol_dict, bnd_dict)

    [print(k, v) for k, v in fr_dict.items()]
    [print(k, v) for k, v in fr_bnd_dict.items()]

    # include all volumetric fragments
    geometry_set = list(fr_dict.values())

    # get box surface
    for side_name in box_sides_dict.keys():
        fr_bnd_dict[side_name+"_fr"] \
            .set_region('.'+side_name) \
            .mesh_step(cfg_mesh.boundary_mesh_step)
        geometry_set.append(fr_bnd_dict[side_name+"_fr"])

    if "fractures" in cfg_geom.include:
        # fractures and its boundary
        b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        b_fractures_out = b_fractures_fr \
            .select_by_intersect(box.get_boundary().deepcopy()) \
            .set_region(".fractures_out") \
            .mesh_step(cfg_mesh.boundary_mesh_step)
        geometry_set.append(b_fractures_out)

    return geometry_finalize(factory, geometry_set)



def test_mesh() -> None:
    """Build both meshes and prepare the averaging domains for conductivity homogenization."""
    app_dir = Path(__file__).absolute().parents[1]
    work_dir = app_dir / "workdir_msh"
    job.set_workdir(work_dir)
    logging.info(job.to_str())

    cfg_path = app_dir / "test_small.py"
    cfg = common.config.load_config(str(cfg_path))

    with common.workdir(work_dir, clean=False):

        homogenization_mesh = make_mesh(
            cfg.mesh,
            fracture_set=[],
            mesh_seed=1.0,
            geometry_func=make_geometry_homo,
        )

        micro_mesh = make_mesh(
            cfg.mesh,
            fracture_set,
            mesh_seed=1.0,
            geometry_func=make_geometry_micro,
        )

        meshes = StudyMeshFiles(
                micro_mesh=micro_mesh.path,
                homogenization_mesh=homogenization_mesh.path)
        prepared = prepare_conductivity_homogenization(cfg, meshes, fractures=fracture_set)

        result = run_homogenization_problem(cfg, work_dir, meshes, prepared)
        validate_conductivity_field(result)

    assert prepared.homogenized_elements.size > 0
    assert prepared.subproblems.n_subdomains == prepared.homogenized_elements.size
    assert prepared.empty_macro_elements.size == 0
    micro_bulk = prepared.micro_mesh.el_dim_slice(dim=3)
    assert (prepared.micro_fields["cross_section"][micro_bulk] == 1.0).all()
