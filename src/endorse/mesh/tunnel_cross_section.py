import os
import numpy as np

from endorse.common.config import dotdict

from bgem.gmsh import gmsh
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import heal_mesh

def setup_gmsh_options(cfg_geom:dotdict):
    tunnel_mesh_step = cfg_geom['tunnel_mesh_step']
    max_elem_size = cfg_geom["max_elem_size"]

    mesh = gmsh_options.Mesh()
    # mesh.Algorithm = gmsh_options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    mesh.Algorithm = gmsh_options.Algorithm2d.Delaunay
    # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
    # mesh.Algorithm3D = gmsh_options.Algorithm3d.Frontal
    # mesh.Algorithm3D = gmsh_options.Algorithm3d.Delaunay

    # mesh.Algorithm = gmsh_options.Algorithm2d.FrontalDelaunay
    # mesh.Algorithm3D = gmsh_options.Algorithm3d.HXT

    mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 2
    mesh.CharacteristicLengthMin = tunnel_mesh_step
    mesh.CharacteristicLengthMax = max_elem_size
    mesh.MinimumCirclePoints = 6
    mesh.MinimumCurvePoints = 2


def create_mesh(factory: gmsh.GeometryOCC, mesh_objects):
    mesh_name = factory.model_name
    mesh_file = mesh_name + ".msh"
    mesh_healed = mesh_name + "_healed.msh"
    mesh_healed2 = mesh_name + "_healed.msh2"

    factory.keep_only(*mesh_objects)
    # factory.remove_duplicate_entities()
    factory.write_brep(mesh_file)

    # factory.make_mesh(mesh_groups, dim=2)
    factory.make_mesh(mesh_objects)

    gmsh_logger = factory.get_logger()
    gmsh_log_msgs = gmsh_logger.get()
    gmsh_logger.stop()
    check_gmsh_log(gmsh_log_msgs)

    factory.write_mesh(format=gmsh.MeshFormat.msh2)
    os.rename(mesh_name + ".msh2", mesh_file)

    # self.make_mesh_B(config_dict, mesh_name, mesh_file, cut_tunnel=cut_tunnel)
    hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
    hm.heal_mesh(gamma_tol=0.01)
    hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
    # pass msh2 filename due to gmsh version 2 format
    hm.write(file_name=mesh_healed2)
    os.rename(mesh_name + "_healed.msh2", mesh_healed)
    return mesh_healed


def check_gmsh_log(lines):
    """
    Search for "No elements in volume" message -> could not mesh the volume -> empty mesh.
    # PLC Error:  A segment and a facet intersect at point (-119.217,65.5762,-40.8908).
    #   Segment: [70,2070] #-1 (243)
    #   Facet:   [3147,9829,13819] #482
    # Info    : failed to recover constrained lines/triangles
    # Info    : function failed
    # Info    : function failed
    # Error   : HXT 3D mesh failed
    # Error   : No elements in volume 1
    # Info    : Done meshing 3D (Wall 0.257168s, CPU 0.256s)
    # Info    : 13958 nodes 34061 elements
    # Error   : ------------------------------
    # Error   : Mesh generation error summary
    # Error   :     0 warnings
    # Error   :     2 errors
    # Error   : Check the full log for details
    # Error   : ------------------------------
    """
    empty_volume_error = "No elements in volume"
    res = [line for line in lines if empty_volume_error in line]
    if len(res) != 0:
        raise Exception("GMSH error - No elements in volume")


def make_tunnel_cross_section_mesh(cfg_geom:dotdict):
    mesh_name = cfg_geom.mesh_name

    tunnel_mesh_step = cfg_geom['tunnel_mesh_step']
    dimensions = cfg_geom["box_dimensions"]
    # tunnel_dims = np.array([cfg_geom["tunnel_dimX"], cfg_geom["tunnel_dimY"]]) / 2
    tunnel_dims = np.array([cfg_geom["radius"], cfg_geom["radius"]])
    tunnel_center = cfg_geom["tunnel_center"]

    print("load gmsh api")
    factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gmsh_logger = factory.get_logger()
    gmsh_logger.start()
    gopt = gmsh_options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001
    # gopt.MatchMeshTolerance = 1e-1
    gopt.OCCFixSmallEdges = True
    gopt.OCCFixSmallFaces = True

    # Main box
    box = factory.rectangle(dimensions).set_region("box")
    side = factory.line([-dimensions[0] / 2, 0, 0], [dimensions[0] / 2, 0, 0])
    sides = dict(
        bottom=side.copy().translate([0, -dimensions[1] / 2, 0]),
        top=side.copy().translate([0, +dimensions[1] / 2, 0]),
        left=side.copy().translate([0, +dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2),
        right=side.copy().translate([0, -dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2)
    )

    tunnel_disc = factory.disc(tunnel_center, *tunnel_dims)
    tunnel_select = tunnel_disc.copy()
    tunnel_ngh = factory.disc(tunnel_center, *(5 * tunnel_dims))

    print("cutting and fragmenting...")
    box_drilled = box.cut(tunnel_disc)
    ngh_drilled = tunnel_ngh.cut(tunnel_disc)
    box_fr, ngh_fr, tunnel_fr = factory.fragment(box_drilled, ngh_drilled, tunnel_disc)

    # b_ngh_fr = ngh_fr.get_boundary()
    # isec_ngh = ngh_fr.select_by_intersect(b_ngh_fr)
    # isec_ngh.modify_regions("tunnel_ngh").mesh_step(2*tunnel_mesh_step)
    # ngh_fr.modify_regions("tunnel_ngh").mesh_step(4 * tunnel_mesh_step)
    ngh_fr.modify_regions("box").mesh_step(6 * tunnel_mesh_step)
    box_all = [box_fr, ngh_fr]

    print("marking boundary regions...")
    b_box_fr = box_fr.get_boundary()
    for name, side_tool in sides.items():
        isec = b_box_fr.select_by_intersect(side_tool)
        box_all.append(isec.modify_regions("." + name))

    b_tunnel_select = tunnel_select.get_boundary()
    b_tunnel = b_box_fr.select_by_intersect(b_tunnel_select)
    b_tunnel.modify_regions(".tunnel").mesh_step(tunnel_mesh_step)
    box_all.extend([b_tunnel])

    mesh_objects = [*box_all]

    print("meshing...")
    setup_gmsh_options(cfg_geom)
    mesh_output_file = create_mesh(factory, mesh_objects)
    return mesh_output_file

