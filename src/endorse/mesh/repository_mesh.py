import logging
from typing import *
import os
import math

import yaml

import bgem
from bgem.gmsh import gmsh, options
from bgem.stochastic.fracture import Population
from bgem.stochastic.fracture import FisherOrientation
import numpy as np
from endorse.common import dotdict, File, report, memoize
from endorse.mesh import mesh_tools


def create_fractures_rectangles(gmsh_geom, fractures, shift, base_shape: 'ObjectSet'):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    if len(fractures) == 0:
        return []


    shapes = []
    for i, fr in enumerate(fractures):
        shape = base_shape.copy()
        print("fr: ", i, "tag: ", shape.dim_tags)
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=[0,0,1], angle=fr.shape_angle) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.center + shift).set_region(fr.region)

        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


# @attrs.define
# class ThreeTunnelGeom:
#     main_tunnel: TunnelParams
#     lateral_tunnel: TunnelParams

def tunnel(factory, tunnel_dict):
    """
    A box with rounded "roof", basic box dimensions:
    hight= radius, width, length

    The crosscut is formed by the height x width rectangle with the dick segment roof.
    Result is translated to have [0,0,0] at the boundary of the floor rectangle.
    At the center of the 'width' side.
    """
    radius = tunnel_dict.radius
    height = tunnel_dict.height
    width = tunnel_dict.width
    length = tunnel_dict.length
    box = factory.box([width, length, height])
    z_shift = math.sqrt(radius * radius - 0.25 * width * width) - height / 2
    cylinder = factory.cylinder(radius, axis=[0, length, 0]).translate([0, -length / 2, -z_shift])
    roof = cylinder.intersect(box.copy().translate([0, 0, height]))
    return box.fuse(roof).translate([0,0,+height / 2])

def box_with_sides(factory, dimensions):
    """
    Make a box and dictionary of its sides named: 'side_[xyz][01]'
    :return: box, sides_dict
    """
    box = factory.box(dimensions).set_region("box")
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
        side.modify_regions(name)
    return box, sides

def make_access_tunnels(factory, geom_dict):

    lateral_length = geom_dict.lateral_tunnel.length
    main_tunnel_cylinder = tunnel(
        factory, geom_dict.main_tunnel,
        ).translate([-lateral_length, 0, 0])
    lateral_tunnel_1 = tunnel(
        factory, geom_dict.lateral_tunnel,
    ).rotate([0,0,1], math.pi/2).translate([-lateral_length/2, 0, 0])

    laterals = [lateral_tunnel_1]

    borehole_distance = geom_dict.borehole.y_spacing
    for i_shift in range(geom_dict.borehole.n_explicit):
        laterals.append(lateral_tunnel_1.copy().translate([0, borehole_distance * i_shift, 0]))
        laterals.append(lateral_tunnel_1.copy().translate([0, -borehole_distance * i_shift, 0]))


    return main_tunnel_cylinder.fuse(*laterals)

def boreholes_full(factory, geom_dict):
    lateral_length = geom_dict.lateral_tunnel.length
    b_cfg = geom_dict.borehole
    borehole_radius = b_cfg.radius
    borehole_length = b_cfg.length
    borehole_distance = b_cfg.y_spacing

    b_1 = factory.cylinder(borehole_radius, axis=[borehole_length, 0, 0])
    boreholes = [b_1]
    for i_shift in range(geom_dict.borehole.n_explicit):
        boreholes.append(b_1.copy().translate([0, borehole_distance * i_shift, 0]))
        boreholes.append(b_1.copy().translate([0, -borehole_distance * i_shift, 0]))

    return factory.group(*boreholes)

def outer_box_shift(geom_dict):
    return [geom_dict.borehole.length / 2, 0, 0]


def basic_shapes(factory, geom_dict):
    bh_z_pos = geom_dict.borehole.z_pos

    box, sides = box_with_sides(factory, geom_dict.box_dimensions)
    box = box.translate(outer_box_shift(geom_dict))
    access_tunnels = make_access_tunnels(factory, geom_dict) #.translate([-bh_length / 2, 0, 0])
    boreholes = boreholes_full(factory, geom_dict).translate([0, 0, bh_z_pos])
    tunnels = boreholes.copy().fuse(access_tunnels.copy())
    box_drilled = box.copy().cut(tunnels).set_region("box")
    return box_drilled, box, access_tunnels, boreholes


def make_geometry(factory, cfg_geom, fractures, cfg_mesh):
    box_drilled, box, access_tunnels, boreholes = basic_shapes(factory, cfg_geom)
    #box_drilled, box, tunnels = basic_shapes_simple(factory, geom_dict)

    fractures = create_fractures_rectangles(factory, fractures, outer_box_shift(cfg_geom), factory.rectangle())
    fractures_group = factory.group(*fractures).intersect(box_drilled)

    #b_rec = box_drilled.get_boundary()#.set_region(".sides")

    box_fr, fractures_fr = factory.fragment(box_drilled, fractures_group)
    fractures_fr.mesh_step(cfg_mesh.fracture_mesh_step) #.set_region("fractures")

    b_box_fr = box_fr.get_boundary().split_by_dimension()[2]
    b_fractures_fr = fractures_fr.get_boundary().split_by_dimension()[1]

    # select outer boundary
    boundary_mesh_step = cfg_mesh.boundary_mesh_step
    b_box = b_box_fr.select_by_intersect(box.get_boundary().copy()).set_region(".box_outer").mesh_step(boundary_mesh_step)
    b_fractures = b_fractures_fr.select_by_intersect(box.get_boundary().copy()).set_region(".fr_outer").mesh_step(boundary_mesh_step)

    # select inner boreholes boundary
    boreholes_step = cfg_mesh.boreholes_mesh_step
    select = boreholes.get_boundary().copy()
    b_box_boreholes = b_box_fr.select_by_intersect(select)\
                  .set_region(".box_boreholes").mesh_step(boreholes_step)
    b_fr_boreholes = b_fractures_fr.select_by_intersect(select)\
                 .set_region(".fr_boreholes").mesh_step(boreholes_step)

    tunnel_mesh_step = cfg_mesh.main_tunnel_mesh_step
    select = access_tunnels.get_boundary().copy()
    b_box_tunnel = b_box_fr.select_by_intersect(select)\
                  .set_region(".box_tunnel").mesh_step(tunnel_mesh_step)
    b_fr_tunnel = b_fractures_fr.select_by_intersect(select)\
                  .set_region(".fr_tunnel").mesh_step(tunnel_mesh_step)


    boundary = factory.group(b_box, b_fractures,
                             b_box_boreholes, b_fr_boreholes,
                             b_box_tunnel, b_fr_tunnel)
    bulk_geom = factory.group(box_fr, fractures_fr, boundary)
    edz_refined = factory.group(b_box_boreholes, b_fr_boreholes, b_box_tunnel, b_fr_tunnel)
    #boundary = factory.group(b_box)

    # Following makes some mesing issues:
    #factory.group(b_box_inner, b_fr_inner).mesh_step(geom_dict['main_tunnel_mesh_step'])
    #boundary.select_by_intersect(boreholes.get_boundary()).mesh_step(geom_dict['boreholes_mesh_step'])

    return bulk_geom, edz_refined


def make_geometry_2d(factory, cfg_geom, fractures, cfg_mesh):
    #box_drilled, box, access_tunnels, boreholes = basic_shapes(factory, cfg_geom)
    #box_drilled, box, tunnels = basic_shapes_simple(factory, geom_dict)
    bh_top = 2 * cfg_geom.borehole.radius
    x, y, z = cfg_geom.box_dimensions
    XZ_rot = ([1, 0, 0], np.pi / 2)
    x_shift_vec = outer_box_shift(cfg_geom)
    box = factory.rectangle([x, z/2]).rotate(*XZ_rot).translate([x_shift_vec[0], 0, z/4 + bh_top]).set_region("box")
    borehole_side = factory.line([-x/2,  -z/4, 0],  [x/2, -z/4, 0] ).rotate(*XZ_rot).translate([x_shift_vec[0], 0, z/4 + bh_top])

    base_shape = factory.line([-0.5, 0, 0], [0.5, 0, 0])
    fr_shapes = create_fractures_rectangles(factory, fractures, [0,0,0], base_shape)

    fr_shapes = factory.group(*fr_shapes).rotate(*XZ_rot).translate(x_shift_vec)
    fractures_group = fr_shapes.intersect(box.copy())

    #b_rec = box_drilled.get_boundary()#.set_region(".sides")

    box_fr, fractures_fr = factory.fragment(box.copy(), fractures_group)
    fractures_fr.mesh_step(cfg_mesh.fracture_mesh_step) #.set_region("fractures")

    b_box_fr = box_fr.get_boundary().split_by_dimension()[1]
    b_fractures_fr = fractures_fr.get_boundary().split_by_dimension()[0]

    # select outer boundary
    select = borehole_side
    boundary_mesh_step = cfg_mesh.boundary_mesh_step
    outer = box.get_boundary().copy().cut(select)
    b_box = b_box_fr.select_by_intersect(outer).set_region(".box_outer").mesh_step(boundary_mesh_step)
    b_fractures = b_fractures_fr.select_by_intersect(box.get_boundary().copy()).set_region(".fr_outer").mesh_step(boundary_mesh_step)

    # select inner boreholes boundary
    boreholes_step = cfg_mesh.boreholes_mesh_step

    b_box_boreholes = b_box_fr.select_by_intersect(select)\
                  .set_region(".box_boreholes").mesh_step(boreholes_step)
    b_fr_boreholes = b_fractures_fr.select_by_intersect(select)\
                 .set_region(".fr_boreholes").mesh_step(boreholes_step)


    boundary = factory.group(b_box, b_fractures,
                             b_box_boreholes, b_fr_boreholes)
    bulk_geom = factory.group(box_fr, fractures_fr, boundary)
    edz_refined = factory.group(b_box_boreholes, b_fr_boreholes)
    #boundary = factory.group(b_box)

    # Following makes some mesing issues:
    #factory.group(b_box_inner, b_fr_inner).mesh_step(geom_dict['main_tunnel_mesh_step'])
    #boundary.select_by_intersect(boreholes.get_boundary()).mesh_step(geom_dict['boreholes_mesh_step'])

    return bulk_geom, edz_refined


def one_borehole(cfg_geom:dotdict, fractures:List['Fracture'], cfg_mesh:dotdict, geom_fn):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    mesh_file = "one_borehole.msh2"
    base, ext = os.path.splitext(os.path.basename(mesh_file))
    factory = gmsh.GeometryOCC(base, verbose=True)
    factory.get_logger().start()
    #factory = gmsh.GeometryOCC(mesh_name)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    bulk, refined = geom_fn(factory, cfg_geom, fractures, cfg_mesh)


    factory.set_mesh_step_field(mesh_tools.edz_refinement_field(factory, cfg_geom, cfg_mesh))
    mesh_tools.edz_meshing(factory, [bulk], mesh_file)
    # factory.show()
    del factory
    return File(mesh_file)



def fr_dict_repr(fr):
    return dict(r=float(fr.r), normal=fr.normal.tolist(), center=fr.center.tolist(),
                aspect=float(fr.aspect), shape_angle=float(fr.shape_angle), region=fr.region.name)


def fracture_set(cfg, fr_population:Population, seed):
    main_box_dimensions = cfg.geometry.box_dimensions

    # Fixed large fractures
    fix_seed = cfg.fractures.fixed_seed
    large_min_r = cfg.fractures.large_min_r
    large_box_dimensions = cfg.fractures.large_box
    fr_limit = cfg.fractures.n_frac_limit
    logging.info(f"Large fracture seed: {fix_seed}")
    max_large_size = max([fam.size.diam_range[1] for fam in fr_population.families])
    fractures = mesh_tools.generate_fractures(fr_population, (large_min_r, max_large_size), fr_limit, large_box_dimensions, fix_seed)

    large_fr_dict=dict(seed=fix_seed, fr_set=[fr_dict_repr(fr) for fr in fractures])
    with open(f"large_Fr_set.yaml", "w") as f:
        yaml.dump(large_fr_dict, f, sort_keys=False)
    n_large = len(fractures)
    #if n_large == 0:
    #    raise ValueError()
    # random small scale fractures
    small_fr = mesh_tools.generate_fractures(fr_population, (None, large_min_r), fr_limit, main_box_dimensions, seed)
    fractures.extend(small_fr)
    logging.info(f"Generated fractures: {n_large} large, {len(small_fr)} small.")
    return fractures, n_large

@report
@memoize
def fullscale_transport_mesh_3d(cfg, fr_population, seed):
    fractures, n_large = fracture_set(cfg, fr_population, seed)
    return one_borehole(cfg.geometry, fractures, cfg.mesh, make_geometry), fractures, n_large


@report
@memoize
def fullscale_transport_mesh_2d(cfg, fr_population, seed):
    fractures, n_large = fracture_set(cfg, fr_population, seed)
    return one_borehole(cfg.geometry, fractures, cfg.mesh, make_geometry_2d), fractures, n_large
