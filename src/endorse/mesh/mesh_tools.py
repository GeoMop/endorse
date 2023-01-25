from typing import *
import os
import logging
import numpy as np
from bgem.gmsh import field, options, gmsh
from bgem.stochastic import fracture
from endorse.common import dotdict



def generate_fractures(pop:fracture.Population, range: Tuple[float, float], fr_limit, box,  seed) -> List[fracture.Fracture]:
    """
    Generate set of stochastic fractures.
    """
    np.random.seed(seed)
    max_fr_size = np.max(box)
    r_min, r_max = range
    if r_max is None:
        r_max = max_fr_size
    if r_min is None:
        # smallest size range
        n_frac_lim = fr_limit
    else:
        # prescribed fracture range
        n_frac_lim = None
    pop.domain = [b if d > 0 else 0.0 for d, b in zip(pop.domain, box)]
    pop.set_sample_range([r_min, r_max], sample_size=n_frac_lim)
    logging.info(f"fr set range: {[r_min, r_max]}, fr_lim: {n_frac_lim}, mean population size: {pop.mean_size()}")


    pos_gen = fracture.UniformBoxPosition(pop.domain)
    fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    for i, fr in enumerate(fractures):
        reg = gmsh.Region.get(f"fr_{i}")
        fr.region = reg

    # fracture.fr_intersect(fractures)

    #used_families = set((f.region for f in fractures))
    #for model in ["transport_params"]:
        #model_dict = config_dict[model]
        #model_dict["fracture_regions"] = list(used_families)
        #model_dict["boreholes_fracture_regions"] = [".{}_boreholes".format(f) for f in used_families]
        #model_dict["main_tunnel_fracture_regions"] = [".{}_main_tunnel".format(f) for f in used_families]
    return fractures




def edz_refinement_field(factory: "GeometryOCC", cfg_geom: "dotdict", cfg_mesh: "dotdict") -> field.Field:
    """
    Refinement mesh step field for resolution of the EDZ.
    :param cfg_geom:
    """
    b_cfg = cfg_geom.borehole
    bx, by, bz = cfg_geom.box_dimensions
    edz_radius = cfg_geom.edz_radius
    center_line = factory.line([0,0,0], [b_cfg.length, 0, 0]).translate([0, 0, b_cfg.z_pos])


    n_sampling = int(b_cfg.length / 2)
    dist = field.distance(center_line, sampling = n_sampling)
    inner = field.geometric(dist, a=(b_cfg.radius, cfg_mesh.edz_mesh_step * 0.9), b=(edz_radius, cfg_mesh.edz_mesh_step))
    outer = field.polynomial(dist, a=(edz_radius, cfg_mesh.edz_mesh_step), b=(by / 2, cfg_mesh.boundary_mesh_step), q=1.7)
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