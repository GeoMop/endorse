"""Mesh construction boundary for the deterministic homogenization study.

USER implementation: build the microscale box with several conductivity regions
and its fracture network, and build the fixed homogenization mesh used to report
the six-component tensor field.
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Callable

import homogenization_study.job as job
from bgem.gmsh.gmsh import ObjectSet
from endorse import common
from endorse.common import dotdict, File, memoize
from endorse.mesh import mesh_tools, fracture_tools

from bgem.gmsh import gmsh, options, gmsh_io, heal_mesh, field
from bgem.stochastic import Population

import homogenization_study.exception_wrapper as exp


def preserve_region_counter_after_unpickle(fracture_set) -> None:
    """ Quick Fix
        Keep BGEM-generated regions unique after fracture regions cross a subprocess boundary.
    """
    if not fracture_set:
        return

    max_fracture_region_id = max(fr.region.id for fr in fracture_set)
    gmsh.Region._max_reg_id = max(gmsh.Region._max_reg_id, max_fracture_region_id)


def fragment(factory, object_dict: dict[str, ObjectSet], boundary_object_dict: dict[str, ObjectSet])\
        -> tuple[dict[str, ObjectSet], dict[str, ObjectSet]]:
    """
    :param factory: bgem gmsh factory
    :param object_dict: "volumetric" objects (of arbitrary dimensions)
    :param boundary_object_dict: "boundary" objects to determine boundary regions later
    :return:
    """
    # Merge the dictionaries (order preserved since Python 3.7+)
    merged_inputs = {**object_dict, **boundary_object_dict}

    # determine ObjectSets which are not None
    enabled_keys = [key for key, value in merged_inputs.items() if value is not None]
    # create list of ObjectSets for fragmentation
    input_list = [merged_inputs[key] for key in enabled_keys]

    output_list = factory.fragment(*input_list)

    # Map outputs to "_fr" keys
    fragmented_all = {f"{key}_fr": val for key, val in zip(enabled_keys, output_list)}

    # Split back into two output dictionaries
    fragmented_1 = {f"{key}_fr": fragmented_all[f"{key}_fr"]
                    for key in object_dict if f"{key}_fr" in fragmented_all}

    fragmented_2 = {f"{key}_fr": fragmented_all[f"{key}_fr"]
                    for key in boundary_object_dict if f"{key}_fr" in fragmented_all}

    return fragmented_1, fragmented_2


def safe_list(source_dict: dict[str, ObjectSet], keys: list[str]) -> list[ObjectSet]:
    """
    Creates a new list with values from `source_dict` for the given `keys`,
    skipping any that are None.
    """
    new_list = [source_dict[k] for k in keys if source_dict[k] is not None]
    return new_list

def geometry_finalize(factory, geometry_set):
    print("Finalize geometry...")
    geometry_final = factory.group(*geometry_set)
    factory.synchronize()
    # need to keep tunnel lines due to refinement fields
    factory.keep_only(geometry_final)
    factory.synchronize()
    factory.remove_duplicate_entities()
    factory.synchronize()
    print("Geometry finished...")
    return geometry_final


@exp.rethrow_as(exp.GeomException, "Geometry exception")
def make_geometry_homo(factory, cfg_mesh:'dotdict', fracture_set):
    cfg_geom = cfg_mesh.geometry

    # Prepare objects
    vol_dict = {
        "box": None,
    }
    bnd_dict = {}

    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions, cfg_geom.box_center)
    # bnd_dict = {**bnd_dict, **box_sides_dict}

    vol_dict["box"] = box.set_region("box").mesh_step(cfg_mesh.boundary_mesh_step)

    # include all volumetric fragments
    geometry_set = list(vol_dict.values())

    return geometry_finalize(factory, geometry_set)


@exp.rethrow_as(exp.GeomException, "Geometry exception")
def make_geometry_micro(factory, cfg_mesh:'dotdict', fracture_set):
    cfg_geom = cfg_mesh.geometry

    # Prepare objects
    vol_dict = {
        # "box": None,
        "fractures_group": None,
        "box_A": None,
        "box_B": None,
        "box_thin": None,
        "box_thick": None,
    }
    bnd_dict = {}

    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions, cfg_geom.box_center)
    bnd_dict = {**bnd_dict, **box_sides_dict}

    dims = cfg_geom.box_dimensions
    # create large side boxes of 1/3 width
    rect_A = factory.rectangle([dims[0] / 3, dims[1]], center=[-dims[0] / 3, 0, -dims[2]/2])
    rect_B = factory.rectangle([dims[0] / 3, dims[1]], center=[ dims[0] / 3, 0, -dims[2]/2])
    box_A = rect_A.extrude(vector=[0, 0, dims[2]])[3]
    box_B = rect_B.extrude(vector=[0, 0, dims[2]])[3]

    # create the middle boxes, width 1/3
    # n pairs in y axis: each pair consist of 1/4 high
    # --------------
    # |            |
    # |            | 3/4
    # |            |
    # --------------
    # |            | 1/4
    # --------------
    n=10
    dy = dims[1] / n / 4
    rects_m = []
    rects_M = []
    for i in range(n):
        rm = factory.rectangle([dims[0] / 3, dy], center=[0, (4*i + 0.5) * dy - dims[1]/2, -dims[2]/2])
        rects_m.append(rm.extrude(vector=[0, 0, dims[2]])[3])
        rM = factory.rectangle([dims[0] / 3, 3*dy], center=[0, (4*i + 2.5) * dy - dims[1]/2, -dims[2]/2])
        rects_M.append(rM.extrude(vector=[0, 0, dims[2]])[3])


    # drill the box, so later we do not have fractures in drilled volume
    # vol_dict["box"] = box.deepcopy()
    # vol_dict["box"].set_region("box")
    vol_dict["box_A"] = box_A.set_region("box_A")
    vol_dict["box_B"] = box_B.set_region("box_B")
    vol_dict["box_thin"] = factory.group(*rects_m).set_region("box_thin").mesh_step(dy)
    vol_dict["box_thick"] = factory.group(*rects_M).set_region("box_thick").mesh_step(dy)

    if "fractures" in cfg_geom.include:
        fractures = fracture_tools.create_fractures_rectangles(factory, fracture_set, cfg_geom.box_center, factory.rectangle())
        # vol_dict["fractures_group"] = factory.group(*fractures).intersect(vol_dict["box"])
        vol_dict["fractures_group"] = factory.group(*fractures).intersect(box)
        vol_dict["fractures_group"].mesh_step(cfg_mesh.fracture_mesh_step) # .set_region("fractures")
        # determine fracture outer boundary
        # b_fractures_outer = vol_dict["fractures_group"].get_boundary()[1]
        # bnd_dict["b_fractures_outer"] = b_fractures_outer \
        #     .select_by_intersect(box.get_boundary().deepcopy()) \
        #     .set_region(".fractures_outer") \
        #     .mesh_step(cfg_mesh.boundary_mesh_step)

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


@exp.rethrow_as(exp.MeshException, "Meshing exception")
def meshing(factory, objects, mesh_filename):
    """
    Common EDZ and transport domain meshing setup.
    """
    print("Meshing...", flush=True)
    logging.info("Writing Gmsh BREP before 3D mesh generation: %s.brep", factory.model_name)
    factory.write_brep()
    #factory.mesh_options.CharacteristicLengthMin = cfg.get("min_mesh_step", cfg.boreholes_mesh_step)
    #factory.mesh_options.CharacteristicLengthMax = cfg.boundary_mesh_step
    # factory.mesh_options.CharacteristicLengthMin = 0.001
    # factory.mesh_options.CharacteristicLengthMax = 15
    factory.mesh_options.MinimumCirclePoints = 6
    factory.mesh_options.MinimumCurvePoints = 3
    #factory.mesh_options.Algorithm = options.Algorithm3d.MMG3D

    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    # mesh.Algorithm = options.Algorithm2d.Delaunay
    # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay

    factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay
    #mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    #mesh.CharacteristicLengthFromPoints = True
    #factory.mesh_options.CharacteristicLengthFromCurvature = False
    #factory.mesh_options.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
    #mesh.CharacteristicLengthMin = min_el_size
    # mesh.CharacteristicLengthMax = max_el_size
    # factory.mesh_options.CharacteristicLengthMax = 1

    # factory.show()
    #factory.remove_duplicate_entities()
    logging.info("Starting Gmsh 3D mesh generation for %s", factory.model_name)
    factory.make_mesh(objects, dim=3, eliminate=False)
    print("Meshing finished.", flush=True)
    factory.write_mesh(filename=mesh_filename, format=gmsh.MeshFormat.msh2)
    print("Mesh written.", flush=True)

def make_gmsh(cfg_mesh:'dotdict', fracture_set, mesh_seed, geometry_func:Callable):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    final_mesh_filename = cfg_mesh.mesh_name + ".msh2"

    factory = gmsh.GeometryOCC(cfg_mesh.mesh_name, verbose=False)
    import gmsh as gmsh_orig
    gmsh_orig.option.setNumber("General.Terminal", 1)     # headless
    gmsh_orig.option.setNumber("General.NumThreads", 1)   # avoid hidden threading

    factory.get_logger().start()
    # gopt = options.Geometry()
    # gopt.Tolerance = 0.0001
    # gopt.ToleranceBoolean = 0.001

    # factory.show()
    preserve_region_counter_after_unpickle(fracture_set)

    geometry_set = geometry_func(factory, cfg_mesh, fracture_set)

    # factory.show()
    # exit(0)

    logging.info(f"Setting GMSH Mesh.RandomSeed: {mesh_seed}")
    logging.info(
        "Gmsh mesh sizes for box %s: boundary=%s, fractures=%s",
        cfg_mesh.geometry.box_dimensions,
        cfg_mesh.boundary_mesh_step,
        cfg_mesh.fracture_mesh_step,
    )
    factory.mesh_options.RandomSeed = mesh_seed
    meshing(factory, [geometry_set], final_mesh_filename)
    # factory.show()
    del factory
    return File(final_mesh_filename)


@exp.rethrow_as(exp.HealException, "Meshing exception")
def make_heal_mesh(mesh_name, mesh_file: File):
    mesh_file_healed = Path(mesh_name + "_healed.msh2")
    if not Path(mesh_file_healed).exists():
        print("HEAL MESH")

        # use mesh_seed for heal_mesh randomization (elements, nodes permutation)
        hm = heal_mesh.HealMesh.read_mesh(mesh_file.path, node_tol=1e-4)
        hm.heal_mesh(gamma_tol=0.002)
        # hm.stats_to_yaml(cfg_mesh.mesh_name + "_heal_stats.yaml")
        hm.write(file_name=mesh_file_healed.name)
    return mesh_file_healed


@memoize
def make_fractures(cfg_mesh, dfn_seed):

    fr_pop = Population.from_cfg(cfg_mesh.fractures.population, cfg_mesh.geometry.box_dimensions)

    if "fractures" in cfg_mesh.geometry.include:
        fracture_set, n_large = fracture_tools.fracture_set(cfg_mesh, fr_pop, dfn_seed)
    else:
        fracture_set, n_large = None, 0

    return fr_pop, fracture_set, n_large


@memoize
def make_mesh(cfg_mesh, fracture_set, mesh_seed, geometry_func:Callable):

    if not Path(cfg_mesh.mesh_name + ".msh2").exists():
        mesh_file = make_gmsh(cfg_mesh, fracture_set, mesh_seed, geometry_func)    # use mesh_seed for gmsh randomization
    else:
        mesh_file = File(cfg_mesh.mesh_name + ".msh2")

    # the number of elements written by factory logger does not correspond to actual count
    # reader = gmsh_io.GmshIO(mesh_file.path)
    # print("N Elements: ", len(reader.elements))

    # heal mesh
    mesh_file_healed = make_heal_mesh(cfg_mesh.mesh_name, mesh_file)

    print("Final mesh file: ", mesh_file_healed)
    return File(mesh_file_healed.name)

def main(cfg, workdir, dfn_seed, mesh_seed):

    import chodby_trans.input_data as input_data
    # common.EndorseCache.instance().expire_all()

    with common.workdir(workdir, clean=False):
        fr_pop, fracture_set, n_large = make_fractures(cfg.mesh, dfn_seed)
        return make_mesh(cfg, fracture_set, mesh_seed)


# if __name__ == '__main__':
#
#     app_dir = Path(__file__).absolute()
#     work_dir = app_dir / "workdir_msh"
#     job.set_workdir(work_dir)
#     shutil.copytree(app_dir / job.input.dir_path.name, job.input.dir_path, dirs_exist_ok=True)
#     logging.info(job.to_str())
#
#     if not job.input.dir_path.exists():
#         raise Exception(f"Input data '{job.input.dir_path}' not found in workdir '{work_dir}'")
#     cfg_path = job.input.transport_cfg_path
#     cfg = common.config.load_config(str(cfg_path))
#
#     main(cfg, work_dir, dfn_seed=1, mesh_seed=1)
