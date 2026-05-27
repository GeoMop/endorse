from typing import *
import os
import logging
import numpy as np
from bgem.gmsh import field, options, gmsh
from bgem.stochastic import Fracture, Population
from endorse.common import dotdict


def box_with_sides(factory, dimensions, center=[0,0,0]):
    """
    Make a box and dictionary of its sides named: 'side_[xyz][01]'
    :return: box, sides_dict
    """
    box = factory.box(dimensions).translate(center).set_region("box")
    side_z = factory.rectangle([dimensions[0], dimensions[1]])
    side_y = factory.rectangle([dimensions[0], dimensions[2]])
    side_x = factory.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.translate(center).modify_regions(name)
    return box, sides


def legacy_seed_from_hash(hash_value):
    return abs(hash_value) % (2**32)

def generate_fractures(pop:Population, range: Tuple[float, float], fr_limit, box,  seed,
                       id_offset=0) -> List['RegionFracture']:
    """
    Generate set of stochastic fractures.
    """
    from endorse.mesh.fracture_tools import RegionFracture
    # legacy_seed = legacy_seed_from_hash(seed)
    # print(f"seed: {seed}, legacy_seed: {legacy_seed}")
    np.random.seed(seed)
    # rng = np.random.default_rng(seed)
    family_ranges = np.array([f.size.sample_range for f in pop.families], dtype=float)
    current_range = tuple(np.median(family_ranges, axis=0))
    lower, upper = range
    sample_range = (
        current_range[0] if lower is None else lower,
        current_range[1] if upper is None else upper,
    )
    if fr_limit is not None and (lower is None or upper is None):
        free_bound = 0 if lower is None else 1
        sample_range = pop.common_range_for_sample_size(fr_limit, free_bound=free_bound, initial_range=sample_range)
    pop = pop.set_sample_range(sample_range)
    logging.info(f"fr set range: {sample_range}, fr_lim: {fr_limit}, mean population size: {pop.mean_size()}")

    fractures = [fr for fr in pop.sample(keep_nonempty=True)]
    for i, fr in enumerate(fractures):
        reg = gmsh.Region.get(f"fr_{id_offset+i}")
        fractures[i] = RegionFracture(fr, reg)

    # fracture.fr_intersect(fractures)

    #used_families = set((f.region for f in fractures))
    #for model in ["transport_params"]:
        #model_dict = config_dict[model]
        #model_dict["fracture_regions"] = list(used_families)
        #model_dict["boreholes_fracture_regions"] = [".{}_boreholes".format(f) for f in used_families]
        #model_dict["main_tunnel_fracture_regions"] = [".{}_main_tunnel".format(f) for f in used_families]
    return fractures




def edz_refinement_field(factory: "GeometryOCC", line,
                         r: List[float],
                         step: List[float],
                         q: float, n_sampling = 10) -> field.Field:
    """
    Refinement mesh step field for resolution of the EDZ.
    :param line: gmsh factory.line instance
    :param r: (r_tunnel, r_edz, r_outer_boundary)
    :param step: (step_tunnel, step_edz, step_outer_boundary)
    :param q: quotient of polynomial field progression
    n_sampling : number of sample points for construction of the distance field
    """
    r_tunnel, r_edz, r_outer_boundary = r
    step_tunnel, step_edz, step_outer_boundary = step

    dist = field.distance(line, sampling = n_sampling)
    inner = field.geometric(dist,
                            a=(r_tunnel, step_tunnel),
                            b=(r_edz, step_edz))
    outer = field.polynomial(dist,
                             a=(r_edz, step_edz),
                             b=(r_outer_boundary, step_outer_boundary),
                             q=q)
    return field.maximum(inner, outer)


def edz_meshing(factory, objects, mesh_file):
    """
    Common EDZ and transport domain meshing setup.
    """
    factory.write_brep()
    #factory.mesh_options.CharacteristicLengthMin = cfg.get("min_mesh_step", cfg.boreholes_mesh_step)
    #factory.mesh_options.CharacteristicLengthMax = cfg.boundary_mesh_step
    factory.mesh_options.MinimumCirclePoints = 6
    factory.mesh_options.MinimumCurvePoints = 6
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
    #mesh.CharacteristicLengthMax = max_el_size

    #factory.keep_only(*objects)
    #factory.remove_duplicate_entities()
    factory.make_mesh(objects, dim=3)
    #factory.write_mesh(me gmsh.MeshFormat.msh2) # unfortunately GMSH only write in version 2 format for the extension 'msh2'
    factory.write_mesh(format=gmsh.MeshFormat.msh2)
    os.rename(factory.model_name + ".msh2", mesh_file)


def container_period(cfg):
    cont = cfg.containers
    return cont.length + cont.spacing


def container_x_pos(cfg, i_pos):
    cont = cfg.containers
    return cont.offset + i_pos * container_period(cfg)
