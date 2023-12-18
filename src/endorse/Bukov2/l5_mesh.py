"""
Functions for preparation of the Bukov2, L5 min-by experiment
"""
import os,math
from endorse import common
from endorse.mesh import mesh_tools, repository_mesh
from bgem.gmsh import gmsh, options, field as gmsh_field

def tunnel_profile(factory, tunnel_dict):
    radius = tunnel_dict.radius
    height = tunnel_dict.height
    width = tunnel_dict.width



def tunnel_smooth(factory, tunnel_dict):
    """
    A box with rounded "roof", basic box dimensions:
    hight= radius, width, length

    The crosscut is formed by the height x width rectangle with the dick segment roof.
    Result is translated to have [0,0,0] at the boundary of the floor rectangle.
    At the center of the 'width' side.
    """
    #radius = tunnel_dict.radius
    vertical_side = tunnel_dict.vertical_side

    height = tunnel_dict.height
    width = tunnel_dict.width
    length = tunnel_dict.length
    box = factory.box([width, length, vertical_side]).translate([0, 0, vertical_side/2])
    edge = factory.line([0, -length/2, 0], [0, length/2, 0])
    edges = gmsh.ObjectSet.group(
                edge.copy().translate([-width / 2, 0, 0]),
                edge.copy().translate([width / 2, 0, 0])
            )
    box_edges = box.get_boundary().get_boundary().unique()
    f_box_edges = box_edges.select_by_intersect(edges)
    fill_box = box.fillet(f_box_edges, 0.5)
    #z_shift = math.sqrt(radius * radius - 0.25 * width * width) - height / 2
    hat_height = height - vertical_side
    cylinder = (factory.cylinder(width / 2, axis=[0, length, 0])
                .scale([1, 1, hat_height / (width / 2)])    # ellipse
                .translate([0, -length / 2, vertical_side]))
    #roof = cylinder.intersect(box.copy().translate([0, 0, height]))
    tunnel = fill_box.fuse(cylinder.copy())  #.translate([0,0,+height / 2])
    center_line = factory.line([0, 0, 0], [0, length, 0]).translate([0,-length / 2,height/2])
    return tunnel, center_line

def basic_shapes(factory, geom_dict):
    """
    Coordinate system:
    X - in direction of laterals, positive in direction to L6
    Y - in direction of the L5 main shaft, all positive, increasing towards L5 end
    Z - vertical, positive upwards

    - center between two laterals approximately in the center of domain
    - to break central point symmetry, the main shaft is artificially near the box boundary
    - marked boundary regions:
        - outer box boundary
        - main shaft walls
        - postmeshing 5 excavation zones (parametrized) for each lateral,
        - refinement around given lateral
    :param factory:
    :param geom_dict:
    :return:
    """
    bh_z_pos = geom_dict.borehole.z_pos
    lateral_length = geom_dict.lateral_tunnel.length
    main_width = geom_dict.main_tunnel.width
    laterals_distance = geom_dict.laterals_distance

    box, sides = repository_mesh.box_with_sides(factory, geom_dict.box_dimensions)

    # make main tunnel assymetric, possitive end inside the box
    # to force geometry orientation
    main_tunnel = tunnel_smooth(
        factory, geom_dict.main_tunnel,
        )
    main_tunnel, main_line = gmsh.translate(main_tunnel, [0, 0, 0])

    lateral_cfg = common.dotdict(geom_dict.lateral_tunnel)
    lateral_cfg.length = lateral_cfg.length + main_width / 2
    laterals_pos = [
        [(main_width/2 + lateral_length)/2, -laterals_distance/2, 0],
        [-(main_width/2 + lateral_length) / 2, laterals_distance / 2, 0]
    ]
    laterals = [
        gmsh.translate(
        gmsh.rotate(tunnel_smooth(factory, lateral_cfg),
        [0,0,1], math.pi / 2),
        shift)
        for shift in laterals_pos
        ]
    laterals, lateral_lines = zip(*laterals)
    #laterals = [lateral_tunnel_1]
    b_laterals = gmsh.copy(gmsh.get_boundary(laterals))

    # borehole_distance = geom_dict.borehole.y_spacing
    # for i_shift in range(geom_dict.borehole.n_explicit):
    #     laterals.append(lateral_tunnel_1.copy().translate([0, borehole_distance * i_shift, 0]))
    #     laterals.append(lateral_tunnel_1.copy().translate([0, -borehole_distance * i_shift, 0]))

    tunnels = main_tunnel.copy().fuse(*laterals)
    #access_tunnels = repository_mesh.make_access_tunnels(factory, geom_dict) #.translate([-bh_length / 2, 0, 0])
    #boreholes = repository_mesh.boreholes_full(factory, geom_dict).translate([0, 0, bh_z_pos])

    #tunnels = boreholes.copy().fuse(access_tunnels.copy())
    box_drilled = box.copy().cut(tunnels.copy()).set_region("box")
    return box_drilled, box, tunnels, lateral_lines


def l5_xk_gemoetry(factory, cfg_geom:'dotdict', cfg_mesh:'dotdict'):
    box_drilled, box, access_tunnels, l_lines = basic_shapes(factory, cfg_geom)
    box_fr = box_drilled

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
    # lateral regions
    # marked_laterals = [
    #     b_box_fr.select_by_intersect(l)
    #         .set_region(f".lateral_{il}")
    #         .mesh_step(boreholes_step)
    #     for il, l in enumerate(l_bc)
    # ]
    #b_fr_boreholes = b_fractures_fr.select_by_intersect(select)\
    #             .set_region(".fr_boreholes").mesh_step(boreholes_step)

    tunnel_mesh_step = cfg_mesh.main_tunnel_mesh_step
    select = access_tunnels.get_boundary().copy()
    b_box_tunnel = b_box_fr.select_by_intersect(select)\
                  .set_region(".box_tunnel").mesh_step(tunnel_mesh_step)
    #b_fr_tunnel = b_fractures_fr.select_by_intersect(select)\
    #              .set_region(".fr_tunnel").mesh_step(tunnel_mesh_step)


    boundary = factory.group(b_box,
                             b_box_tunnel)
    bulk_geom = factory.group(box_fr,  boundary)

    # distance function scales linearly with number of points
    # therefore having points distributed on the surfaces is prohibitively slow
    # We rather construct lines in centers of lateral tunnels and measure distance from them.
    refined_lines = l_lines
    #boundary = factory.group(b_box)

    # Following makes some mesing issues:
    #factory.group(b_box_inner, b_fr_inner).mesh_step(geom_dict['main_tunnel_mesh_step'])
    #boundary.select_by_intersect(boreholes.get_boundary()).mesh_step(geom_dict['boreholes_mesh_step'])

    return bulk_geom, refined_lines

def l5_zk_mesh(cfg_geom:'dotdict', cfg_mesh:'dotdict'):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    mesh_file = "Bukov_both.msh2"
    base, ext = os.path.splitext(os.path.basename(mesh_file))
    factory = gmsh.GeometryOCC(base, verbose=True)
    factory.get_logger().start()
    #factory = gmsh.GeometryOCC(mesh_name)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    bulk, refined = l5_xk_gemoetry(factory, cfg_geom, cfg_mesh)

    line_fields = (mesh_tools.line_distance_edz(factory, line, cfg_mesh.line_refinement)
        for line in refined)
    common_field = gmsh_field.minimum(*line_fields)
    factory.set_mesh_step_field(common_field)
    mesh_tools.edz_meshing(factory, [bulk], mesh_file)
    # factory.show()
    del factory
    return common.File(mesh_file)


def make_mesh(workdir):
    conf_file = os.path.join(workdir, "./Bukov2_mesh.yaml")
    cfg = common.config.load_config(conf_file)
    mesh_file = l5_zk_mesh(cfg.geometry, cfg.mesh)
    print("Mesh file: ", mesh_file)


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    make_mesh(script_dir)