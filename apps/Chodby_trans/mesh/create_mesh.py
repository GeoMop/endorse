import math
import os
import numpy as np
import pandas as pd

from apps.Chodby_inv.mesh.venv.share.doc.gmsh.tutorials.python.t4 import factory
from endorse import common
from endorse.mesh import mesh_tools, fracture_tools

from bgem.gmsh import gmsh, options, gmsh_io, heal_mesh, field
from bgem.stochastic.fracture import Population
# import gmsh as gmsh_api

script_dir = os.path.dirname(os.path.realpath(__file__))


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


def create_main_tunnel(factory, cfg_geom:'dotdict'):
    """
    Creates main tunnel by extrusion from cross-section points of the L5 tunnel.
    :param factory:
    :param cfg_geom:
    :return:
        tunnel (ObjectSet)
    """
    # Read points defining head of tunnel in XZ plane, Y=0
    df = pd.read_csv(os.path.join(script_dir, cfg_geom.main_tunnel.csv_points))
    main_tunnel_points = df[['x', 'y', 'z']].to_numpy()

    # create polygon
    tunnel_polygon = factory.make_polygon(points=main_tunnel_points)
    # compute center of polygon
    tunnel_center = factory.point(np.average(main_tunnel_points, axis=0))
    tunnel_polygon = factory.group(tunnel_polygon, tunnel_center)
    tunnel_polygon.translate(vector=[0, -cfg_geom.main_tunnel.length / 2, 0])
    tunnel_extrude = tunnel_polygon.extrude(vector=[0, cfg_geom.main_tunnel.length, 0])
    # extrude polygon and its center to get the tunnel and its central line
    tunnel = tunnel_extrude[3]
    tunnel_center_line = tunnel_extrude[1]
    tunnel.set_region("main_tunnel")

    # bottom coordinate of the main tunnel (needed by storage boreholes)
    tunnel_bottom_z = np.min(main_tunnel_points[:, 2], axis=0)

    return tunnel, tunnel_center_line, tunnel_bottom_z


def create_storage_boreholes(factory, cfg_geom:'dotdict', tunnel, tunnel_bottom_z):
    # make the boreholes longer, sticking out into the main tunnel
    # then cut it by the tunnel
    csb = cfg_geom.storage_borehole
    storage_boreholes = []
    plug, container = None, None
    d = cfg_geom.storage_borehole_distance
    s = -((cfg_geom.n_storage_boroholes - 1) / 2.0) * d  # y coordinate of the first storage

    for i in range(cfg_geom.n_storage_boroholes):
        if i is cfg_geom.damaged_storage_borehole:
            # damaged storage is split into plug and container
            plug = factory.cylinder(r=csb.radius, axis=[0, 0, -csb.plug + tunnel_bottom_z], center=[0, s + i * d, 0])
            plug = plug.cut(tunnel)
            plug.set_region(f"plug_{i}").mesh_step(csb.mesh_step)

            container = factory.cylinder(r=csb.radius, axis=[0, 0, -(csb.length-csb.plug) + tunnel_bottom_z],
                                         center=[0, s + i * d, -csb.plug + tunnel_bottom_z])
            container.set_region(f"container_{i}").mesh_step(csb.mesh_step)
        else:
            sbh = factory.cylinder(r=csb.radius, axis=[0, 0, -csb.length + tunnel_bottom_z], center=[0, s + i * d, 0])
            sbh = sbh.cut(tunnel)
            sbh.set_region(f"storage_{i}").mesh_step(csb.mesh_step)
            storage_boreholes.append(sbh)

        # # possibly create plug
        # # for some reason, tunnel does not get meshed this way
        # sbh = factory.cylinder(r=csb.radius, axis=[0, 0, -csb.length + tunnel_bottom_z], center=[0, s + i * d, 0])
        # sbh = sbh.cut(tunnel)
        # if i is cfg_geom.damaged_storage_borehole:
        #     plug_aux = factory.cylinder(r=csb.radius, axis=[0, 0, -csb.plug + tunnel_bottom_z], center=[0, s + i * d, 0])
        #     sbh_fr, plug_fr = factory.fragment(sbh, plug_aux)
        #     plug = plug_fr.dt_intersection(sbh_fr).set_region(f"plug_{i}").mesh_step(csb.mesh_step)
        #     container = sbh_fr.dt_drop(plug).set_region(f"container_{i}").mesh_step(csb.mesh_step)
        # else:
        #     sbh.set_region(f"storage_{i}").mesh_step(csb.mesh_step)
        #     storage_boreholes.append(sbh)

    return storage_boreholes, plug, container


def make_geometry(factory, cfg:'dotdict', fracture_population, seed):
    cfg_geom = cfg.geometry
    cfg_mesh = cfg.mesh

    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions)
    box_sides_group = factory.group(*list(box_sides_dict.values())).copy() # keep the original sides
    print("box:\n", box.dim_tags)
    print("box_sides_group:\n", box_sides_group)

    tunnel, tunnel_center_line, tunnel_bottom_z = create_main_tunnel(factory, cfg_geom)
    print("tunnel:\n", tunnel)

    storage_boreholes, plug, container = create_storage_boreholes(factory, cfg_geom, tunnel, tunnel_bottom_z)
    storage_boreholes_group = factory.group(*storage_boreholes)
    print("storage_boreholes:\n", storage_boreholes)
    print("plug:\n", plug)
    print("container:\n", container)
    # assert plug and container is not None

    # drill the box, so later we do not have fractures in drilled volume
    box_drilled = box.copy().cut(tunnel, plug, container, storage_boreholes_group)
    box_drilled.set_region("box")

    fracture_set, n_large = fracture_tools.fracture_set(cfg, fracture_population, seed)
    fractures = fracture_tools.create_fractures_rectangles(factory, fracture_set, [0,0,0], factory.rectangle())
    fractures_group = factory.group(*fractures).intersect(box_drilled)
    fractures_group.set_region("fractures").mesh_step(cfg_mesh.fracture_mesh_step)

    factory.synchronize()
    # factory.show()
    # exit(0)

    # fragment
    print("Fragmenting...")
    if (plug is None) or (container is None):
        box_fr, box_sides_fr, fractures_fr, tunnel_fr, storage_boreholes_group_fr \
            = factory.fragment(box_drilled, box_sides_group, fractures_group, tunnel, storage_boreholes_group)
        plug_fr, container_fr = None, None
    else:
        box_fr, box_sides_fr, fractures_fr, tunnel_fr, plug_fr, container_fr, storage_boreholes_group_fr\
            = factory.fragment(box_drilled, box_sides_group, fractures_group, tunnel, plug, container, storage_boreholes_group)
    # if (plug is None) or (container is None):
    #     box_fr, box_sides_fr, tunnel_fr, storage_boreholes_group_fr \
    #         = factory.fragment(box_drilled, box_sides_group, tunnel, storage_boreholes_group)
    #     plug_fr, container_fr = None, None
    # else:
    #     box_fr, box_sides_fr, tunnel_fr, plug_fr, container_fr, storage_boreholes_group_fr\
    #         = factory.fragment(box_drilled, box_sides_group, tunnel, plug, container, storage_boreholes_group)
    print("Fragmenting finished.")

    # get boundary of fragmented volumes
    b_box_fr = box_fr.get_boundary().split_by_dimension()[2]
    b_tunnel_fr = tunnel_fr.get_boundary().split_by_dimension()[2]

    b_fractures_fr = fractures_fr.get_boundary().split_by_dimension()[1]
    b_fractures = (b_fractures_fr.select_by_intersect(box.get_boundary().copy()).set_region(".fr_outer")
        .mesh_step(cfg_mesh.boundary_mesh_step))

    print("Determine box objects.")
    print("box_fr:\n", box_fr)
    print("tunnel_fr:\n", tunnel_fr)
    print("storage_boreholes_fr:\n", storage_boreholes_group_fr)
    print("plug_fr:\n", plug_fr)
    print("container_fr:\n", container_fr)

    # GET box sides
    # check whether boundary of fragmented box volume includes all dimtags of fragmented boundary box
    print("box_sides_fr: \n", box_sides_fr)
    box_sides_no_tunnel = b_box_fr.dt_intersection(box_sides_fr)  # = box_sides_fr
    # assert box_sides_no_tunnel.dt_equal(box_sides_fr)

    # get tunnel head surfaces (create by box fragment)
    tunnel_heads = b_tunnel_fr.dt_intersection(box_sides_fr)
    tunnel_heads.set_region("tunnel_heads")
    print("tunnel_heads: \n", tunnel_heads)
    box_sides_no_tunnel.dt_drop(tunnel_heads)
    print("box_sides_no_tunnel: \n", box_sides_no_tunnel)

    # factory.show()

    # SET final geometry set
    print("Set regions to box sides.")
    geometry_set = [box_fr, tunnel_fr, fractures_fr, storage_boreholes_group_fr]
    if plug is not None: geometry_set.append(plug_fr)
    if container is not None: geometry_set.append(container_fr)

    # if cfg_geom.clip_box_dimensions != cfg_geom.box_dimensions:
    #     clip_box, clip_box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions)
    #     geometry_group = factory.group(*geometry_set)
    #     clip_geometry = geometry_group.intersect(clip_box)
    #     geometry_set = [clip_geometry]
    #
    #     box_sides_dict = clip_box_sides_dict
    #     b_clip_box = clip_box.get_boundary().split_by_dimension()[2]

    for side_name, side_obj in box_sides_dict.items():
        b_side = box_sides_no_tunnel.select_by_intersect(side_obj)
        b_side.set_region('.'+side_name).mesh_step(cfg_mesh.boundary_mesh_step)
        geometry_set.append(b_side)

    geometry_set.append(b_fractures)
    geometry_final = factory.group(*geometry_set)

    # create refinement fields around drifts
    # tunnel_center_lines = []
    # line_fields = (line_distance_edz(factory, line, cfg_mesh.line_refinement)
    #                for line in tunnel_center_lines)
    line_fields = [line_distance_edz(factory, tunnel_center_line, cfg_mesh.main_line_refinement)]
    common_field = field.minimum(*line_fields)
    factory.set_mesh_step_field(common_field)

    # exit(0)
    print("Finalize geometry...")
    factory.synchronize()
    # need to keep tunnel lines due to refinement fields
    factory.keep_only(geometry_final, tunnel_center_line)
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

    factory.show()
    #factory.remove_duplicate_entities()
    factory.make_mesh(objects, dim=3, eliminate=False)
    print("Meshing finished.")
    factory.write_mesh(filename=mesh_filename, format=gmsh.MeshFormat.msh2)
    print("Mesh written.")

def make_gmsh(cfg:'dotdict', fracture_population, seed):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    final_mesh_filename = os.path.join(cfg.output_dir, cfg.mesh_name + ".msh2")

    factory = gmsh.GeometryOCC(cfg.mesh_name, verbose=True)
    factory.get_logger().start()
    # gopt = options.Geometry()
    # gopt.Tolerance = 0.0001
    # gopt.ToleranceBoolean = 0.001

    # factory.show()
    geometry_set = make_geometry(factory, cfg, fracture_population, seed)
    # factory.show()
    # exit(0)

    meshing(factory, [geometry_set], final_mesh_filename)
    # factory.show()
    del factory
    return common.File(final_mesh_filename)


def make_mesh(workdir, output_dir, cfg_file, seed):
    conf_file = os.path.join(workdir, cfg_file)
    cfg = common.config.load_config(conf_file)
    cfg.output_dir = output_dir

    fr_pop = Population.initialize_3d(cfg.fractures.population, cfg.geometry.box_dimensions)

    mesh_file = make_gmsh(cfg, fr_pop, seed)

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

    seed = 1
    make_mesh(script_dir, output_dir, "./trans_mesh_config.yaml", seed)

