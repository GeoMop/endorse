import math
import os
import numpy as np

from endorse import common, mesh_class

from bgem.gmsh import gmsh, options, gmsh_io, heal_mesh, field
# import gmsh as gmsh_api
from chodby_inv.hm_model.boreholes import Boreholes

from chodby_inv import input_data
from cfg import script_dir, workdir, input_dir
from fractures import *


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
        box_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        box_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        box_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        box_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        box_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        box_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    # not necessary - not final objects
    # for name, side in sides.items():
    #     side.modify_regions(name)
    factory.synchronize()
    return box, sides

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


def make_geometry(factory, cfg:'dotdict', tunnel_laser_scan):
    cfg_geom = cfg.geometry
    box, box_sides_dict = box_with_sides(factory, cfg_geom.box_dimensions)
    box_sides_group = factory.group(*list(box_sides_dict.values())).copy() # keep the original

    # print("tunnel_laser_scan:\n", tunnel_laser_scan.dim_tags)
    # print(tunnel_laser_scan.regions)
    print("box:\n", box.dim_tags)
    print("box_sides_group:\n", box_sides_group)

    # create center lines for meshing field
    tunnel_center_lines = tunnel_lines(factory, cfg_geom)
    print("tunnel_center_lines:\n", tunnel_center_lines)

    # create borehole chamber lines for meshing field
    bhs = Boreholes(cfg.boreholes)
    borehole_chamber_lines = bhs.make_gmsh_lines(factory)

    # factory.show()

    tunnel = tunnel_laser_scan.split_by_dimension()[3]
    tunnel_boundary = tunnel_laser_scan.split_by_dimension()[2]

    # create fractures
    line_objs = create_line_objs(cfg.line_objects, bhs)
    fracs = generate_fractures(cfg.generated_fractures, line_objs, cfg.boreholes.geometry.l5_azimuth)
    fractures_dict = create_planes_gmsh(factory, fracs, size=50)
    fractures = factory.group( *fractures_dict.values() )
    fractures_cut = fractures.copy().intersect(box).cut(tunnel)

    # fragment
    print("Fragmenting...")
    box_fr, box_sides_fr, tunnel_fr, tunnel_boundary_fr, fractures_fr \
        = factory.fragment(box, box_sides_group, tunnel, tunnel_boundary, fractures_cut)
    print("Fragmenting finished.")

    # get boundary of fragmented volumes
    b_box_fr = box_fr.get_boundary().split_by_dimension()[2]
    b_tunnel_fr = tunnel_fr.get_boundary().split_by_dimension()[2]
    b_fractures_fr = fractures_fr.get_boundary().split_by_dimension()[1]
    print(f"fractures_fr: {fractures_fr}")


    # CHECK
    print("Checking fragments...")
    res = box_fr.dt_intersection(tunnel_fr)     # = tunnel_fr
    print(f"res: {res} tunnel_fr: {tunnel_fr}")
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
        b_side.set_region('.'+side_name).mesh_step(cfg.mesh.boundary_mesh_step)
        geometry_set.append(b_side)

    # GET fracture-tunnel boundaries and set regions
    print("Set regions to fractures and their boundaries.")
    b_fractures_fr = b_fractures_fr.select_by_intersect(tunnel_walls)
    for fr_name,fr_obj in fractures_dict.items():
        fr = fractures_fr.select_by_intersect(fr_obj)
        fr.set_region(fr_name)
        geometry_set.append(fr)
        b_fr_tunnel = fr.get_boundary().split_by_dimension()[1].select_by_intersect(tunnel_walls)
        b_fr_tunnel.set_region(f".{fr_name}_tunnel")
        geometry_set.append(b_fr_tunnel)
        b_fr = fr.get_boundary().split_by_dimension()[1].dt_drop(b_fr_tunnel).select_by_intersect(box_sides_no_tunnel)
        b_fr.set_region(f".{fr_name}")
        geometry_set.append(b_fr)
        print(f"fracture {fr_name} objects: {fr} boundary: {b_fr} boundary_tunnel: {b_fr_tunnel}")

    geometry_set.append(tunnel_walls)
    geometry_set.append(box_fr)

    # create refinement fields around drifts
    line_fields = [line_distance_edz(factory, line, cfg.mesh.line_refinement)
                   for line in tunnel_center_lines]
    line_fields.extend( [line_distance_edz(factory, line, cfg.mesh.borehole_refinement)
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
    # factory.show()

    print("Geometry finished...")
    return geometry_final, fracs


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
    if not Path(filename).is_file():
        raise FileNotFoundError(filename)
    # gmsh.open(filename)
    gmsh.merge(filename)


def make_gmsh(cfg:'dotdict', real_geometry=True):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    final_mesh_filename = input_dir / (cfg.mesh_name + ".msh2")
    boundary_brep_filename = input_dir / cfg.boundary_brepfile

    factory = gmsh.GeometryOCC(cfg.mesh_name, verbose=True)
    factory.get_logger().start()
    # gopt = options.Geometry()
    # gopt.Tolerance = 0.0001
    # gopt.ToleranceBoolean = 0.001

    if real_geometry is True:
        tunnel_laser_scan = factory.import_shapes(str(boundary_brep_filename), highestDimOnly=False)

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
    else:
        # simplified tunnel shape (cylinders):
        cfg_geom = cfg.geometry
        box_dim = cfg_geom.box_dimensions
        r_l5 = cfg_geom.main_tunnel.height / 2
        r_zk = cfg_geom.lateral_tunnel.height / 2
        offset_zk = cfg_geom.laterals_distance / 2
        l_zk = cfg_geom.lateral_tunnel.length
        offset_z = r_l5/2 # elevation of tunnel axis
        l5 = factory.cylinder(r=r_l5, center=[0, -box_dim[1]/2, offset_z], axis=[0, box_dim[1], 0])
        zk51j = factory.cylinder(r=r_zk, center=[0, -offset_zk, offset_z], axis=[l_zk, 0, 0])
        zk51s = factory.cylinder(r=r_zk, center=[0, offset_zk, offset_z], axis=[-l_zk, 0, 0])
        tunnel_laser_scan = l5.fuse(l5, zk51s, zk51j)
        tunnel_bulk = tunnel_laser_scan.split_by_dimension()[3]
        tunnel_boundary = tunnel_laser_scan.get_boundary().split_by_dimension()[2]
        tunnel_group = factory.group(tunnel_bulk, tunnel_boundary)

    factory.synchronize()

    # factory.show()
    geometry_set, fracs = make_geometry(factory, cfg, tunnel_group)
    # factory.show()
    # exit(0)

    meshing(factory, [geometry_set], str(final_mesh_filename))
    # factory.show()
    del factory
    return common.File(final_mesh_filename), fracs

def set_fracture_regions(cfg:'dotdict', fracs, filename):

    # creates rotation matrix that maps a to b
    def rotation_matrix_from_vectors(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        if s == 0:
            return np.eye(3)
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        return R

    # create fractures
    bhs = Boreholes(cfg.boreholes)
    line_objs = create_line_objs(cfg.line_objects, bhs)

    # read mesh and fracture region ids
    mesh = mesh_class.Mesh.load_mesh(common.File(filename))
    reg_ids = mesh.gmsh_io.get_reg_ids_by_physical_names(fracs.keys())
    b_reg_ids = mesh.gmsh_io.get_reg_ids_by_physical_names([f".{s}" for s in fracs.keys()])
    bt_reg_ids = mesh.gmsh_io.get_reg_ids_by_physical_names([f".{s}_tunnel" for s in fracs.keys()])
    print(f"reg_ids: {reg_ids}")

    # process each fracture
    for idx,(fr_name,fr_data) in enumerate(fracs.items()):
        # get bulk fracture elements and boundary elements intersecting tunnel
        el_ids = mesh.gmsh_io.get_elements_of_regions([reg_ids[idx]])
        b_el_ids = mesh.gmsh_io.get_elements_of_regions([b_reg_ids[idx]])
        bt_el_ids = mesh.gmsh_io.get_elements_of_regions([bt_reg_ids[idx]])
        print(f"{fr_name}")
        # print(f"  el_ids: {el_ids} b_el_ids: {b_el_ids}")

        # get points on boundary elements intersecting tunnel
        bt_points = []
        for el in bt_el_ids:
            el_idx = mesh.el_indices[el]
            bt_points.append( mesh.elements[el_idx].barycenter() )
            # print(f" el {el} nodes: {mesh.elements[el_idx].node_indices}")
        # print(f" bt_points: {bt_points}")

        # get points on external boundary elements
        b_points = []
        for el in b_el_ids:
            el_idx = mesh.el_indices[el]
            b_points.append( mesh.elements[el_idx].barycenter() )
            # print(f" el {el} nodes: {mesh.elements[el_idx].node_indices}")
        # print(f" b_points: {b_points}")

        # find control points on fracture
        control_points = []
        fr_normal = fr_data[0]
        fr_offset = fr_data[1]
        for pt in cfg.generated_fractures[fr_name].points:
            l_start = np.array(line_objs[pt.line_object].start_pt)
            l_dir = np.array(line_objs[pt.line_object].direction)
            position = (fr_offset - np.dot(fr_normal,l_start)) / np.dot(fr_normal,l_dir)
            point = l_start + position*l_dir
            control_points.append(point)
            # print(f"  point: {point}")

        # transform points onto 2D plane
        all_points = np.asarray( b_points + bt_points + control_points )
        rot_matrix = rotation_matrix_from_vectors(fr_normal, np.array([0,0,1]) )
        all_points = all_points @ rot_matrix.T
        all_points = all_points[:,:2]
        # print(f"  all points: {all_points}")

        # create new fracture regions
        last_reg_id = max(mesh.gmsh_io.get_reg_ids_by_physical_names(mesh.gmsh_io.physical.keys()))
        mesh.gmsh_io.physical[f"{fr_name}"] = (last_reg_id + 1, 2)
        pt_to_reg_id = [last_reg_id + 1] * len(b_points)
        mesh.gmsh_io.physical[f"{fr_name}_tunnel"] = (last_reg_id + 2, 2)
        pt_to_reg_id.extend( [last_reg_id + 2] * len(bt_points) )
        for i, _ in enumerate(control_points):
            pt = cfg.generated_fractures[fr_name].points[i]
            mesh.gmsh_io.physical[f"{fr_name}_{pt.line_object}"] = (last_reg_id + 3 + i, 2)
            pt_to_reg_id.append(last_reg_id + 3 + i)

        # create KDTree for computing distance to control points
        from scipy.spatial import KDTree
        tree = KDTree(all_points)

        # assign new regions to fracture elements
        el_points = []
        for el in el_ids:
            el_idx = mesh.el_indices[el]
            el_points.append( (rot_matrix @ mesh.elements[el_idx].barycenter())[:2] )
        _, reg_indices = tree.query(el_points)
        for el,reg_idx in zip(el_ids,reg_indices):
            type, tags, nodes = mesh.gmsh_io.elements[el]
            # tags = list(tags)
            tags = [ pt_to_reg_id[reg_idx], pt_to_reg_id[reg_idx] ]
            mesh.gmsh_io.elements[el] = (type, tags, nodes)
    mesh.gmsh_io.write(filename + "_fr_regions.msh2", format="msh2", binary=False)




def make_mesh(cfg):
    mesh_file, fracs = make_gmsh(cfg, real_geometry=False)

    # the number of elements written by factory logger does not correspond to actual count
    # reader = gmsh_io.GmshIO(mesh_file.path)
    # print("N Elements: ", len(reader.elements))

    # heal mesh
    mesh_file_healed = cfg.mesh_name + "_healed.msh2"
    # if not os.path.exists(mesh_file_healed):
    print("HEAL MESH")
    hm = heal_mesh.HealMesh.read_mesh(mesh_file.path, node_tol=1e-4)
    hm.heal_mesh(gamma_tol=0.02)
    # hm.stats_to_yaml(cfg.mesh_name + "_heal_stats.yaml")
    hm.write(file_name=mesh_file_healed)

    set_fracture_regions(cfg, fracs, mesh_file_healed)

    # print("Mesh file: ", mesh_file_healed)
    return common.File(mesh_file_healed)


def main():

    # common.EndorseCache.instance().expire_all()

    conf_file = input_data.l5_mesh_cfg_yaml
    cfg = common.config.load_config(conf_file)

    with common.workdir(str(workdir), clean=False):
        make_mesh(cfg)


if __name__ == '__main__':
    main()
