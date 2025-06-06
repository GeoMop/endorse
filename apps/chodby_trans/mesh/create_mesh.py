import numpy as np
import pandas as pd
from pathlib import Path

from bgem.gmsh.gmsh import ObjectSet
from endorse import common
from endorse.common import dotdict, File, memoize
from endorse.mesh import mesh_tools, fracture_tools

from bgem.gmsh import gmsh, options, gmsh_io, heal_mesh, field
from bgem.stochastic.fracture import Population

import chodby_trans.input_data as input_data
input_dir = input_data.input_dir
work_dir = input_data.work_dir


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


def rescale_dimension(values, target_range):
    """Rescale 1D array to have the specified range, keep the center at origin."""
    min_val = values.min()
    max_val = values.max()
    center = (max_val + min_val) / 2
    current_range = max_val - min_val
    scale = target_range / current_range if current_range != 0 else 0  # handle constant values
    return (values - center) * scale # center at zero

def merge_along_sequence(points: np.ndarray, tol: float, use_centroid: bool = True) -> np.ndarray:
    """
    Merge consecutive 3D points in `points` that lie within `tol` of each other.

    Parameters
    ----------
    points : (N,3) array of float
        Your input point sequence, sorted so that “neighbors” in the array
        are also spatially neighbors around the ellipse.
    tol : float
        Distance threshold. If the gap between two consecutive points
        is <= tol, they go into the same merge‐cluster.
    use_centroid : bool, default True
        If True, each cluster is represented by its centroid.
        If False, by the first point in that cluster.

    Returns
    -------
    merged : (M,3) array of float
        The reduced set of points after merging.
    """
    if points.size == 0:
        return points.reshape(0, 3)

    clusters = []
    # start the first cluster
    current = [points[0]]

    # walk through the rest
    for p in points[1:]:
        dist = np.linalg.norm(p - current[-1])
        if dist <= tol:
            # still “close enough,” add to current cluster
            current.append(p)
        else:
            # finish the old cluster
            if use_centroid:
                clusters.append(np.mean(current, axis=0))
            else:
                clusters.append(current[0])
            # start a new one
            current = [p]

    # don’t forget the last cluster
    if use_centroid:
        clusters.append(np.mean(current, axis=0))
    else:
        clusters.append(current[0])

    return np.vstack(clusters)


def create_main_tunnel(factory, cfg:'dotdict'):
    """
    Creates main tunnel by extrusion from cross-section points of the L5 tunnel.
    :param factory:
    :param cfg_geom:
    :return:
        tunnel (ObjectSet)
    """
    cfg_mt = cfg.geometry.main_tunnel
    # Read points defining head of tunnel in XZ plane, Y=0
    df = pd.read_csv(input_dir / cfg_mt.csv_points)
    main_tunnel_points = df[['x', 'y', 'z']].to_numpy()

    # rescale points to intended tunnel dimensions (x-width, z-height)
    main_tunnel_points[:, 0] = rescale_dimension(main_tunnel_points[:, 0], cfg_mt.width)  # x
    main_tunnel_points[:, 2] = rescale_dimension(main_tunnel_points[:, 2], cfg_mt.height)  # z

    cluster_tol = (cfg_mt.width+cfg_mt.height)/2 / 15
    main_tunnel_points_clustered = merge_along_sequence(main_tunnel_points, cluster_tol)

    # create polygon
    tunnel_polygon = factory.make_polygon(points=main_tunnel_points_clustered)
    # compute center of polygon
    cfg_mt.center = np.average(main_tunnel_points_clustered, axis=0)
    tunnel_center = factory.point(cfg_mt.center)
    tunnel_polygon = factory.group(tunnel_polygon, tunnel_center)
    tunnel_polygon.translate(vector=[0, -cfg_mt.length / 2, 0])
    tunnel_extrude = tunnel_polygon.extrude(vector=[0, cfg_mt.length, 0])
    # extrude polygon and its center to get the tunnel and its central line
    tunnel = tunnel_extrude[3]
    tunnel_center_line = tunnel_extrude[1]
    tunnel.set_region("main_tunnel").mesh_step(cfg.mesh.main_tunnel_mesh_step)

    # bottom coordinate of the main tunnel (needed by storage boreholes)
    tunnel_bottom_z = np.min(main_tunnel_points_clustered[:, 2], axis=0)

    return tunnel, tunnel_center_line, tunnel_bottom_z


def create_storage_boreholes(factory, cfg_geom:'dotdict', tunnel, tunnel_bottom_z, mesh_step):
    # make the boreholes longer, sticking out into the main tunnel
    # then cut it by the tunnel
    csb = cfg_geom.storage_borehole
    storage_boreholes = []
    plug, container = None, None
    d = cfg_geom.storage_borehole_distance
    s = -((cfg_geom.n_storage_boreholes - 1) / 2.0) * d  # y coordinate of the first storage

    for i in range(cfg_geom.n_storage_boreholes):
        if i is cfg_geom.damaged_storage_borehole:
            # damaged storage is split into plug and container
            plug = factory.cylinder(r=csb.diameter/2, axis=[0, 0, -csb.plug + tunnel_bottom_z],
                                    center=[0, s + i * d, 0])
            plug = plug.cut(tunnel)
            plug.set_region(f"plug_{i}").mesh_step(mesh_step)

            container = factory.cylinder(r=csb.diameter/2, axis=[0, 0, -(csb.length-csb.plug)],
                                         center=[0, s + i * d, -csb.plug + tunnel_bottom_z])
            container.set_region(f"container_{i}").mesh_step(mesh_step)
        else:
            sbh = factory.cylinder(r=csb.diameter/2, axis=[0, 0, -csb.length + tunnel_bottom_z],
                                   center=[0, s + i * d, 0])
            sbh = sbh.cut(tunnel)
            sbh.set_region(f"storage_{i}").mesh_step(mesh_step)
            storage_boreholes.append(sbh)

        # # possibly create plug
        # # for some reason, tunnel does not get meshed this way
        # sbh = factory.cylinder(r=csb.diameter/2, axis=[0, 0, -csb.length + tunnel_bottom_z], center=[0, s + i * d, 0])
        # sbh = sbh.cut(tunnel)
        # if i is cfg_geom.damaged_storage_borehole:
        #     plug_aux = factory.cylinder(r=csb.diameter/2, axis=[0, 0, -csb.plug + tunnel_bottom_z], center=[0, s + i * d, 0])
        #     sbh_fr, plug_fr = factory.fragment(sbh, plug_aux)
        #     plug = plug_fr.dt_intersection(sbh_fr).set_region(f"plug_{i}").mesh_step(csb.mesh_step)
        #     container = sbh_fr.dt_drop(plug).set_region(f"container_{i}").mesh_step(csb.mesh_step)
        # else:
        #     sbh.set_region(f"storage_{i}").mesh_step(csb.mesh_step)
        #     storage_boreholes.append(sbh)

    return storage_boreholes, plug, container


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


def make_geometry(factory, cfg:'dotdict', fracture_set):
    cfg_geom = cfg.geometry
    cfg_mesh = cfg.mesh

    # Prepare objects
    vol_dict = {
        "box_drilled": None,
        "fractures_group": None,
        "tunnel": None,
        "plug": None,
        "container": None,
        "storage_boreholes_group": None,
    }
    bnd_dict = {
        "drill_surface_group": None,
        "b_storage_boreholes_group": None
    }

    vol_dict["tunnel"], tunnel_center_line, tunnel_bottom_z = create_main_tunnel(factory, cfg)

    if "boreholes" in cfg_geom.include:
        storage_boreholes, vol_dict["plug"], vol_dict["container"] \
            = create_storage_boreholes(factory, cfg_geom, vol_dict["tunnel"], tunnel_bottom_z, cfg_mesh.boreholes_mesh_step)
        vol_dict["storage_boreholes_group"] = factory.group(*storage_boreholes)

        # TODO: For unknown reason, we have to set storages mesh step in create_storage_boreholes
        # and not later to all_storage_boreholes or bnd_dict["b_storage_boreholes_group"]
        # if we drill the boreholes out, we need its boundary to prescribe mesh step
        all_storage_boreholes = factory.group(*safe_list(vol_dict, ["plug", "container", "storage_boreholes_group"]))
        # all_storage_boreholes.mesh_step(cfg_mesh.boreholes_mesh_step)
        bnd_dict["b_storage_boreholes_group"] = all_storage_boreholes.get_boundary().copy().split_by_dimension()[2]
        bnd_dict["b_storage_boreholes_group"].mesh_step(cfg_mesh.boreholes_mesh_step)

        # group and fuse everything to drill,
        # make copies to keep original objects for fragmentation
        drill_group = vol_dict["tunnel"].copy().fuse(all_storage_boreholes.copy())
    else:
        drill_group = vol_dict["tunnel"]

    # boundary of drilled volume
    bnd_dict["drill_surface_group"] = drill_group.get_boundary().copy().split_by_dimension()[2]

    box, box_sides_dict = mesh_tools.box_with_sides(factory, cfg_geom.box_dimensions)
    # bnd_dict["box_sides_group"] = factory.group(*list(box_sides_dict.values())).copy()  # keep the original sides
    bnd_dict = {**bnd_dict, **box_sides_dict}

    # drill the box, so later we do not have fractures in drilled volume
    vol_dict["box_drilled"] = box.copy().cut(drill_group)
    vol_dict["box_drilled"].set_region("box")

    if "fractures" in cfg_geom.include:
        fractures = fracture_tools.create_fractures_rectangles(factory, fracture_set, [0,0,0], factory.rectangle())
        vol_dict["fractures_group"] = factory.group(*fractures).intersect(vol_dict["box_drilled"])
        vol_dict["fractures_group"].mesh_step(cfg_mesh.fracture_mesh_step)
        # determine fracture outer boundary
        # b_fractures_outer = vol_dict["fractures_group"].get_boundary()[1]
        # bnd_dict["b_fractures_outer"] = b_fractures_outer \
        #     .select_by_intersect(box.get_boundary().copy()) \
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
    if "drilled_volume" in cfg_geom.include:
        geometry_set = list(fr_dict.values())
    else:
        geometry_set = [v for k, v in fr_dict.items() if k in ["box_drilled_fr", "fractures_group_fr"]]

    # get tunnel head surfaces
    tunnel_head_y0 = fr_bnd_dict["drill_surface_group_fr"] \
        .dt_intersection(fr_bnd_dict["side_y0_fr"]) \
        .set_region(".tunnel_head_y0")
    tunnel_head_y1 = fr_bnd_dict["drill_surface_group_fr"] \
        .dt_intersection(fr_bnd_dict["side_y1_fr"]) \
        .set_region(".tunnel_head_y1")
    if "drilled_volume" in cfg_geom.include:
        geometry_set.extend([tunnel_head_y0, tunnel_head_y1])

    # get box surface
    for side_name in box_sides_dict.keys():
        fr_bnd_dict[side_name+"_fr"] \
            .dt_drop(tunnel_head_y0, tunnel_head_y1) \
            .set_region('.'+side_name) \
            .mesh_step(cfg_mesh.boundary_mesh_step)
        geometry_set.append(fr_bnd_dict[side_name+"_fr"])

    if "drilled_volume" not in cfg_geom.include:
        if "boreholes" in cfg_geom.include:
            b_storages_fr = fr_bnd_dict["b_storage_boreholes_group_fr"].dt_intersection(fr_bnd_dict["drill_surface_group_fr"])
            b_storages_fr \
                .set_region(".storages") \
                .mesh_step(cfg_geom.storage_borehole.mesh_step)
            geometry_set.append(b_storages_fr)
            b_tunnel_fr = fr_bnd_dict["drill_surface_group_fr"].dt_copy() \
                        .dt_drop(b_storages_fr)
        else:
            b_tunnel_fr = fr_bnd_dict["drill_surface_group_fr"]

        b_tunnel_fr.dt_drop(tunnel_head_y0, tunnel_head_y1)
        # get drilled surface without tunnel heads and boreholes
        b_tunnel_fr \
            .set_region(".tunnel") \
            .mesh_step(cfg_mesh.main_tunnel_mesh_step)
        geometry_set.append(b_tunnel_fr)


    if "fractures" in cfg_geom.include:
        # fractures and its boundary
        b_fractures_fr = fr_dict.get("fractures_group_fr").get_boundary().split_by_dimension()[1]
        b_fractures_out = b_fractures_fr \
            .select_by_intersect(box.get_boundary().copy()) \
            .set_region(".fractures_out") \
            .mesh_step(cfg_mesh.boundary_mesh_step)
        b_fractures_in = b_fractures_fr \
            .select_by_intersect(fr_bnd_dict["drill_surface_group_fr"]) \
            .set_region(".fractures_in") \
            .mesh_step(cfg_mesh.main_tunnel_mesh_step)

        geometry_set.append(b_fractures_out)
        if "drilled_volume" not in cfg_geom.include:
            geometry_set.append(b_fractures_in)

    if "drilled_volume" in cfg_geom.include:
        fr_dict["tunnel_fr"].mesh_step(cfg_mesh.main_tunnel_mesh_step)
        fr_dict["storage_boreholes_group_fr"].mesh_step(cfg_mesh.boreholes_mesh_step)

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

def make_gmsh(cfg:'dotdict', fracture_set):
    """
    :param cfg_geom: repository mesh configuration cfg.repository_mesh
    :param fractures:  generated fractures
    :param mesh_file:
    :return:
    """
    final_mesh_filename = cfg.mesh_name + ".msh2"

    factory = gmsh.GeometryOCC(cfg.mesh_name, verbose=True)
    factory.get_logger().start()
    # gopt = options.Geometry()
    # gopt.Tolerance = 0.0001
    # gopt.ToleranceBoolean = 0.001

    # factory.show()
    geometry_set = make_geometry(factory, cfg, fracture_set)
    # factory.show()
    # exit(0)

    meshing(factory, [geometry_set], final_mesh_filename)
    # factory.show()
    del factory
    return File(final_mesh_filename)


@memoize
def make_mesh(cfg, fr_pop, seed):

    if "fractures" in cfg.geometry.include:
        fracture_set, n_large = fracture_tools.fracture_set(cfg, fr_pop, seed)
    else:
        fracture_set, n_large = None, 0

    mesh_file = make_gmsh(cfg, fracture_set)

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

    print("Final mesh file: ", mesh_file_healed)
    return File(mesh_file_healed), fracture_set, n_large


def main():

    # common.EndorseCache.instance().expire_all()

    conf_file = input_data.transport_config
    cfg = common.config.load_config(str(conf_file))

    seed = 101
    with common.workdir(str(work_dir), clean=False):
        fr_pop = Population.initialize_3d(cfg.fractures.population, cfg.geometry.box_dimensions)
        make_mesh(cfg, fr_pop, seed)


if __name__ == '__main__':
    main()
