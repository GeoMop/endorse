"""
Functions for preparation of the Bukov2, L5 min-by experiment
"""
import os,math
from endorse.common import dotdict, File
from endorse.mesh import mesh_tools, repository_mesh
from bgem.gmsh import gmsh, options




def basic_shapes(factory, geom_dict):
    bh_z_pos = geom_dict.borehole.z_pos
    lateral_length = geom_dict.lateral_tunnel.length

    box, sides = repository_mesh.box_with_sides(factory, geom_dict.box_dimensions)
    box = box.translate([lateral_length/2, 0, 0])
    tunnel = repository_mesh.tunnel

    main_tunnel = tunnel(
        factory, geom_dict.main_tunnel,
        ).translate([-lateral_length, 0, 0])
    lateral_tunnel_1 = tunnel(
        factory, geom_dict.lateral_tunnel,
    ).rotate([0,0,1], math.pi/2).translate([-lateral_length/2, 0, 0])

    #laterals = [lateral_tunnel_1]

    # borehole_distance = geom_dict.borehole.y_spacing
    # for i_shift in range(geom_dict.borehole.n_explicit):
    #     laterals.append(lateral_tunnel_1.copy().translate([0, borehole_distance * i_shift, 0]))
    #     laterals.append(lateral_tunnel_1.copy().translate([0, -borehole_distance * i_shift, 0]))
    tunnels = main_tunnel.copy().fuse(lateral_tunnel_1.copy())
    #access_tunnels = repository_mesh.make_access_tunnels(factory, geom_dict) #.translate([-bh_length / 2, 0, 0])
    #boreholes = repository_mesh.boreholes_full(factory, geom_dict).translate([0, 0, bh_z_pos])

    #tunnels = boreholes.copy().fuse(access_tunnels.copy())
    box_drilled = box.copy().cut(tunnels.copy()).set_region("box")
    return box_drilled, box, main_tunnel, lateral_tunnel_1


def l5_xk_gemoetry(factory, cfg_geom:dotdict, cfg_mesh:dotdict):
    box_drilled, box, access_tunnels, boreholes = basic_shapes(factory, cfg_geom)
    box_fr = box_drilled
    #box_drilled, box, tunnels = basic_shapes_simple(factory, geom_dict)

    #fractures = create_fractures_rectangles(factory, fractures, outer_box_shift(cfg_geom), factory.rectangle())
    #fractures_group = factory.group(*fractures).intersect(box_drilled)

    #b_rec = box_drilled.get_boundary()#.set_region(".sides")

    #box_fr, fractures_fr = factory.fragment(box_drilled, fractures_group)
    #fractures_fr.mesh_step(cfg_mesh.fracture_mesh_step) #.set_region("fractures")

    b_box_fr = box_fr.get_boundary().split_by_dimension()[2]
    #b_fractures_fr = fractures_fr.get_boundary().split_by_dimension()[1]

    # select outer boundary
    boundary_mesh_step = cfg_mesh.boundary_mesh_step
    b_box = b_box_fr.select_by_intersect(box.get_boundary().copy()).set_region(".box_outer").mesh_step(boundary_mesh_step)
    #b_fractures = b_fractures_fr.select_by_intersect(box.get_boundary().copy()).set_region(".fr_outer").mesh_step(boundary_mesh_step)

    # select inner boreholes boundary
    boreholes_step = cfg_mesh.boreholes_mesh_step
    select = boreholes.get_boundary().copy()
    b_box_boreholes = b_box_fr.select_by_intersect(select)\
                  .set_region(".box_boreholes").mesh_step(boreholes_step)
    #b_fr_boreholes = b_fractures_fr.select_by_intersect(select)\
    #             .set_region(".fr_boreholes").mesh_step(boreholes_step)

    tunnel_mesh_step = cfg_mesh.main_tunnel_mesh_step
    select = access_tunnels.get_boundary().copy()
    b_box_tunnel = b_box_fr.select_by_intersect(select)\
                  .set_region(".box_tunnel").mesh_step(tunnel_mesh_step)
    #b_fr_tunnel = b_fractures_fr.select_by_intersect(select)\
    #              .set_region(".fr_tunnel").mesh_step(tunnel_mesh_step)


    boundary = factory.group(b_box,
                             b_box_boreholes,
                             b_box_tunnel)
    bulk_geom = factory.group(box_fr,  boundary)
    edz_refined = factory.group(b_box_boreholes,  b_box_tunnel)
    #boundary = factory.group(b_box)

    # Following makes some mesing issues:
    #factory.group(b_box_inner, b_fr_inner).mesh_step(geom_dict['main_tunnel_mesh_step'])
    #boundary.select_by_intersect(boreholes.get_boundary()).mesh_step(geom_dict['boreholes_mesh_step'])

    return bulk_geom, edz_refined

def l5_zk_mesh(cfg_geom:dotdict, cfg_mesh:dotdict):
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

    bulk, refined = l5_xk_gemoetry(factory, cfg_geom, cfg_mesh)


    factory.set_mesh_step_field(mesh_tools.edz_refinement_field(factory, cfg_geom, cfg_mesh))
    mesh_tools.edz_meshing(factory, [bulk], mesh_file)
    # factory.show()
    del factory
    return File(mesh_file)
