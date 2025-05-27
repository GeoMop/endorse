import math
import os
import numpy as np

from endorse import common
from endorse.mesh import mesh_tools

from bgem.gmsh import gmsh, options, gmsh_io, heal_mesh, field
# import gmsh as gmsh_api
from boreholes import Boreholes

script_dir = os.path.dirname(os.path.realpath(__file__))


def tunnel_center_line(factory, tunnel_dict):
    length = tunnel_dict.length
    height = tunnel_dict.height
    center_line = factory.line([0, 0, 0], [0, length, 0]).translate([0, -length / 2, height / 2])
    return center_line

def tunnel_lines(factory, geom_dict):
    """
    Coordinate system:
    X - in direction of laterals, positive in direction to L6
    Y - in direction of the L5 main shaft, all positive, increasing towards L5 end
    Z - vertical, positive upwards

    :param factory:
    :param geom_dict:
    :return:
    """
    lateral_length = geom_dict.lateral_tunnel.length
    main_width = geom_dict.main_tunnel.width
    laterals_distance = geom_dict.laterals_distance

    # main_tunnel_line = tunnel_center_line(factory, geom_dict.main_tunnel)

    lateral_cfg = common.dotdict(geom_dict.lateral_tunnel)
    lateral_cfg.length = lateral_cfg.length + main_width / 2
    laterals_pos = [
        [(main_width/2 + lateral_length)/2, -laterals_distance/2, 0],
        [-(main_width/2 + lateral_length) / 2, laterals_distance / 2, 0]
    ]
    lateral_lines = [
        tunnel_center_line(factory, lateral_cfg)
            .rotate([0,0,1], math.pi / 2)
            .translate(shift)
        for shift in laterals_pos
        ]

    factory.synchronize()
    return lateral_lines


def line_distance_edz(factory: "GeometryOCC", line, cfg_mesh: "dotdict") -> field.Field:
    """
    :param factory:
    :param line:
    :param cfg_mesh:
    :return:
    """
    cfg = cfg_mesh
    line_length = line.get_mass()
    n_sampling = int(line_length / cfg.r_inner)
    dist = field.distance(line, sampling = n_sampling)
    inner = field.geometric(dist, a=(cfg.r_inner, cfg.h_inner), b=(cfg.r_outer, cfg.h_outer))
    outer = field.polynomial(dist, a=(cfg.r_outer, cfg.h_outer), b=(cfg.r_inf, cfg.h_inf), q=cfg.q_outer)
    return field.maximum(inner, outer)


def make_geometry(factory, cfg_geom:'dotdict', cfg_mesh:'dotdict', tunnel_laser_scan):
    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions)
    box_sides_group = factory.group(*list(box_sides_dict.values())).copy() # keep the original

    # print("tunnel_laser_scan:\n", tunnel_laser_scan.dim_tags)
    # print(tunnel_laser_scan.regions)
    print("box:\n", box.dim_tags)
    print("box_sides_group:\n", box_sides_group)

    # create center lines for meshing field
    tunnel_center_lines = tunnel_lines(factory, cfg_geom)
    print("tunnel_center_lines:\n", tunnel_center_lines)

    # create borehole chamber lines for meshing field
    bhs = Boreholes(cfg_mesh.borehole_data_file)
    borehole_chamber_lines = bhs.make_gmsh_lines(factory)

    # factory.show()

    tunnel = tunnel_laser_scan.split_by_dimension()[3]
    tunnel_boundary = tunnel_laser_scan.split_by_dimension()[2]

    # fragment
    print("Fragmenting...")
    box_fr, box_sides_fr, tunnel_fr, tunnel_boundary_fr = factory.fragment(box, box_sides_group, tunnel, tunnel_boundary)
    print("Fragmenting finished.")

    # get boundary of fragmented volumes
    b_box_fr = box_fr.get_boundary().split_by_dimension()[2]
    b_tunnel_fr = tunnel_fr.get_boundary().split_by_dimension()[2]

    # CHECK
    print("Checking fragments...")
    res = box_fr.dt_intersection(tunnel_fr)     # = tunnel_fr
    assert res.dt_equal(tunnel_fr)
    # res = box_sides_fr.dt_intersection(tunnel_boundary_fr) # is empty
    res1 = b_box_fr.dt_intersection(tunnel_boundary_fr)  # 81 dimtags
    res2 = b_tunnel_fr.dt_intersection(tunnel_boundary_fr)  # 81 dimtags
    assert res1.dt_equal(res2)

    print("Get final geometry objects.")
    # GET box minus tunnel volume
    box_fr.dt_drop(tunnel_fr)
    box_fr.set_region("box")
    print("box minus tunnel:\n", box_fr)

    # GET box sides
    # check whether boundary of fragmented box volume includes all dimtags of fragmented boundary box
    print("box_sides_fr: \n", box_sides_fr)
    box_sides_no_tunnel = b_box_fr.dt_intersection(box_sides_fr)  # = box_sides_fr
    assert box_sides_no_tunnel.dt_equal(box_sides_fr)
    # get tunnel head surfaces (create by box fragment)
    tunnel_heads = b_tunnel_fr.dt_intersection(box_sides_fr)
    print("tunnel_heads: \n", tunnel_heads)
    # get rid of tunnel head surfaces
    box_sides_no_tunnel.dt_drop(tunnel_heads)
    print("box_sides_no_tunnel: \n", box_sides_no_tunnel)

    # factory.show()

    # GET tunnel walls
    print("Get tunnel walls.")
    tunnel_walls = b_box_fr.dt_intersection(b_tunnel_fr) # = b_tunnel_fr
    assert tunnel_walls.dt_equal(b_tunnel_fr)
    tunnel_walls.dt_drop(tunnel_heads)
    tunnel_walls.set_region(".tunnel")

    # SET final geometry set
    print("Set regions to box sides.")
    geometry_set = []
    for side_name, side_obj in box_sides_dict.items():
        b_side = box_sides_no_tunnel.select_by_intersect(side_obj)
        b_side.set_region('.'+side_name).mesh_step(cfg_mesh.boundary_mesh_step)
        geometry_set.append(b_side)
    geometry_set.append(tunnel_walls)
    geometry_set.append(box_fr)

    # create refinement fields around drifts
    line_fields = [line_distance_edz(factory, line, cfg_mesh.line_refinement)
                   for line in tunnel_center_lines]
    line_fields.extend( [line_distance_edz(factory, line, cfg_mesh.borehole_refinement)
                   for line in borehole_chamber_lines] )
    common_field = field.minimum(*line_fields)
    factory.set_mesh_step_field(common_field)

    print("Finalize geometry...")
    geometry_final = factory.group(*geometry_set)
    factory.synchronize()
    # need to keep tunnel lines due to refinement fields
    all_lines = tunnel_center_lines + borehole_chamber_lines
    factory.keep_only(geometry_final, *all_lines)
    factory.synchronize()
    factory.remove_duplicate_entities()
    factory.synchronize()

    print("Geometry finished...")
    return geometry_final


def meshing(factory, objects, mesh_filename):
    """
    Common EDZ and transport domain meshing setup.
    """
    print("Meshing...")
    # factory.write_brep()
    #factory.mesh_options.CharacteristicLengthMin = cfg.get("min_mesh_step", cfg.boreholes_mesh_step)
    #factory.mesh_options.CharacteristicLengthMax = cfg.boundary_mesh_step
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

    #factory.keep_only(*objects)
    #factory.remove_duplicate_entities()
    factory.make_mesh(objects, dim=3)
    print("Meshing finished.")
    factory.write_mesh(filename=mesh_filename, format=gmsh.MeshFormat.msh2)
    print("Mesh written.")


def load_boundary_mesh(filename):
    # gmsh.initialize()
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    # gmsh.open(filename)
    gmsh.merge(filename)

def make_gmsh(cfg:'dotdict'):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    final_mesh_filename = os.path.join(cfg.output_dir, cfg.mesh_name + ".msh2")
    boundary_brep_filename = os.path.join(cfg.output_dir, cfg.boundary_brepfile)
    cfg.mesh.borehole_data_file = os.path.join(cfg.output_dir, cfg.mesh.borehole_data_file)

    factory = gmsh.GeometryOCC(cfg.mesh_name, verbose=True)
    factory.get_logger().start()
    # gopt = options.Geometry()
    # gopt.Tolerance = 0.0001
    # gopt.ToleranceBoolean = 0.001

    tunnel_laser_scan = factory.import_shapes(boundary_brep_filename, highestDimOnly=False)

    # print(tunnel_laser_scan.dim_tags)
    # print(tunnel_laser_scan.regions)

    tunnel_bulk = tunnel_laser_scan.split_by_dimension()[3]
    tunnel_boundary = tunnel_laser_scan.split_by_dimension()[2]
    tunnel_group = factory.group(tunnel_bulk, tunnel_boundary)

    # transform tunnel coordinate system
    # move to center to origin
    tunnel_group.translate(-np.array(cfg.geometry.center))
    # rotate
    oax = cfg.geometry.orig_x_axis
    # add 180 degrees to reorient Y-axis to follow L5
    angle = math.pi + math.atan(oax[0]/oax[1])
    tunnel_group.rotate(axis=[0,0,1], angle=angle)
    factory.synchronize()

    # factory.show()
    geometry_set = make_geometry(factory, cfg.geometry, cfg.mesh, tunnel_group)
    # factory.show()
    # exit(0)

    meshing(factory, [geometry_set], final_mesh_filename)
    # factory.show()
    del factory
    return common.File(final_mesh_filename)


def make_mesh(workdir, output_dir, cfg_file):
    conf_file = os.path.join(workdir, cfg_file)
    cfg = common.config.load_config(conf_file)
    cfg.output_dir = output_dir

    mesh_file = make_gmsh(cfg)

    # the number of elements written by factory logger does not correspond to actual count
    # reader = gmsh_io.GmshIO(mesh_file.path)
    # print("N Elements: ", len(reader.elements))

    # heal mesh
    mesh_file_healed = os.path.join(cfg.output_dir, cfg.mesh_name + "_healed.msh2")
    # if not os.path.exists(mesh_file_healed):
    print("HEAL MESH")
    hm = heal_mesh.HealMesh.read_mesh(mesh_file.path, node_tol=1e-4)
    hm.heal_mesh(gamma_tol=0.02)
    # hm.stats_to_yaml(os.path.join(output_dir, cfg.mesh_name + "_heal_stats.yaml"))
    hm.write(file_name=mesh_file_healed)

    # print("Mesh file: ", mesh_file_healed)
    return common.File(mesh_file_healed)

if __name__ == '__main__':
    # output_dir = None
    # len_argv = len(sys.argv)
    # assert len_argv > 1, "Specify input yaml file and output dir!"
    # if len_argv == 2:
    #     output_dir = os.path.abspath(sys.argv[1])
    output_dir = script_dir

    make_mesh(script_dir, output_dir, "./l5_mesh_config.yaml")

